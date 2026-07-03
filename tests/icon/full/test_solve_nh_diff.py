"""Unit test for the ``mo_solve_nh_diff`` differential helpers.

The ICON binding-swap integration runs the stock Fortran ``solve_nh`` and the
SDFG ``solve_nh_dace_icon`` on the SAME input and compares the prognostic output
BIT-FOR-BIT (:file:`mo_solve_nh_diff.f90`).  Two properties must hold for that to
be sound, and this test pins both against gfortran with a small driver:

  * **deep copy is independent** -- ``clone_state_indep_prog`` must allocate FRESH
    prognostic targets (the fields are ``POINTER``, so a shallow ``dst = src``
    would alias the same storage and the reference run would clobber the SDFG
    run).  Mutating the original after cloning must NOT change the clone.

  * **compare is bit-exact** -- ``compare_prog_nnew`` reports 0 differing elements
    for identical state and a non-zero count after a single-bit perturbation
    (no tolerance).

The real ``t_nh_state`` / ``t_prepare_adv`` are replaced with minimal modules of
the same names carrying only the fields the helpers touch, so the test needs
just gfortran (no flang / SDFG build).
"""
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")

_HERE = Path(__file__).resolve().parent
_DIFF_F90 = _HERE / "mo_solve_nh_diff.f90"

# Minimal stand-ins for ICON's types -- SAME module + field names the helpers
# use, so mo_solve_nh_diff.f90 compiles unchanged against them.
_MIN_TYPES = """\
module mo_nonhydro_types
  implicit none
  type :: t_nh_prog
    real(kind=8), pointer, contiguous :: w(:,:,:), vn(:,:,:), rho(:,:,:), exner(:,:,:), theta_v(:,:,:)
  end type t_nh_prog
  ! Same field names + ranks the real ICON t_nh_diag carries, so
  ! mo_solve_nh_diff's clone_diag_indep / compare_diag compile unchanged.
  type :: t_nh_diag
    real(kind=8), pointer, contiguous :: exner_pr(:,:,:) => null(), mass_fl_e(:,:,:) => null(), &
      rho_ic(:,:,:) => null(), theta_v_ic(:,:,:) => null(), grf_tend_vn(:,:,:) => null(), &
      grf_tend_w(:,:,:) => null(), grf_tend_rho(:,:,:) => null(), grf_tend_mflx(:,:,:) => null(), &
      grf_bdy_mflx(:,:,:) => null(), grf_tend_thv(:,:,:) => null(), vn_ie_int(:,:,:) => null(), &
      vn_ie_ubc(:,:,:) => null(), w_int(:,:,:) => null(), w_ubc(:,:,:) => null(), &
      theta_v_ic_int(:,:,:) => null(), theta_v_ic_ubc(:,:,:) => null(), rho_ic_int(:,:,:) => null(), &
      rho_ic_ubc(:,:,:) => null(), mflx_ic_int(:,:,:) => null(), mflx_ic_ubc(:,:,:) => null(), &
      vn_incr(:,:,:) => null(), exner_incr(:,:,:) => null(), rho_incr(:,:,:) => null(), &
      vt(:,:,:) => null(), ddt_exner_phy(:,:,:) => null(), ddt_vn_phy(:,:,:) => null(), &
      exner_dyn_incr(:,:,:) => null(), vn_ie(:,:,:) => null(), w_concorr_c(:,:,:) => null(), &
      mass_fl_e_sv(:,:,:) => null(), ddt_vn_dyn(:,:,:) => null(), ddt_vn_dmp(:,:,:) => null(), &
      ddt_vn_adv(:,:,:) => null(), ddt_vn_cor(:,:,:) => null(), ddt_vn_pgr(:,:,:) => null(), &
      ddt_vn_phd(:,:,:) => null(), ddt_vn_iau(:,:,:) => null(), ddt_vn_ray(:,:,:) => null(), &
      ddt_vn_grf(:,:,:) => null()
    real(kind=8), pointer, contiguous :: ddt_vn_apc_pc(:,:,:,:) => null(), &
      ddt_vn_cor_pc(:,:,:,:) => null(), ddt_w_adv_pc(:,:,:,:) => null()
  end type t_nh_diag
  type :: t_nh_state
    type(t_nh_prog), allocatable :: prog(:)
    type(t_nh_diag) :: diag
  end type t_nh_state
end module mo_nonhydro_types

module mo_prepadv_types
  implicit none
  type :: t_prepare_adv
    real(kind=8), pointer, contiguous :: mass_flx_me(:,:,:), mass_flx_ic(:,:,:), vol_flx_ic(:,:,:), vn_traj(:,:,:)
  end type t_prepare_adv
end module mo_prepadv_types
"""

_DRIVER = """\
program test_diff
  use mo_nonhydro_types
  use mo_solve_nh_diff
  implicit none
  type(t_nh_state) :: st, clone
  integer :: nnew, ndiff, i, j, k
  real(kind=8) :: v

  nnew = 2
  allocate(st%prog(2))
  do i = 1, 2
    allocate(st%prog(i)%vn(3,4,5), st%prog(i)%w(3,4,5), st%prog(i)%rho(3,4,5), &
             st%prog(i)%exner(3,4,5), st%prog(i)%theta_v(3,4,5))
    do k = 1, 5
      do j = 1, 4
        v = real(100*i + 10*j + k, kind=8) * 1.5d0
        st%prog(i)%vn(:,j,k)      = v
        st%prog(i)%w(:,j,k)       = v + 0.25d0
        st%prog(i)%rho(:,j,k)     = v + 0.5d0
        st%prog(i)%exner(:,j,k)   = v + 0.75d0
        st%prog(i)%theta_v(:,j,k) = v + 0.125d0
      end do
    end do
  end do
  ! Allocate a rank-3 (vt) and a rank-4 (ddt_vn_apc_pc) diag field so the deep
  ! copy exercises clone_ptr3 AND clone_ptr4; the rest stay unassociated (the
  ! => NULL() default), which the clone / compare helpers no-op on.
  allocate(st%diag%vt(3,4,5))
  allocate(st%diag%ddt_vn_apc_pc(3,4,5,3))
  st%diag%vt = 7.0d0
  st%diag%ddt_vn_apc_pc = -3.0d0

  call clone_state_indep_prog(st, clone)

  ! (1) a fresh clone is bit-exact on both prog(nnew) and the full diag.
  call compare_prog_nnew(st, clone, nnew, 'init', ndiff)
  if (ndiff /= 0) then
    print *, 'FAIL: fresh clone prog is not bit-exact, ndiff=', ndiff
    stop 1
  end if
  call compare_diag(st%diag, clone%diag, 'init', ndiff)
  if (ndiff /= 0) then
    print *, 'FAIL: fresh clone diag is not bit-exact, ndiff=', ndiff
    stop 1
  end if

  ! (2) the clone owns independent PROG storage: mutate the original, the clone
  !     must not move, and the compare must SEE exactly the perturbation.
  st%prog(nnew)%vn(1,1,1) = st%prog(nnew)%vn(1,1,1) + 1.0d0
  if (clone%prog(nnew)%vn(1,1,1) == st%prog(nnew)%vn(1,1,1)) then
    print *, 'FAIL: clone aliases the original prog (shallow pointer copy)'
    stop 1
  end if
  call compare_prog_nnew(st, clone, nnew, 'mutated', ndiff)
  if (ndiff /= 1) then
    print *, 'FAIL: compare_prog_nnew should report exactly 1 differing element, ndiff=', ndiff
    stop 1
  end if

  ! (3) the clone owns independent DIAG storage too: mutate a rank-3 and a
  !     rank-4 diag field on the original; the clone must not move and
  !     compare_diag must report exactly the two perturbations.
  st%diag%vt(2,2,2) = st%diag%vt(2,2,2) + 1.0d0
  st%diag%ddt_vn_apc_pc(1,1,1,1) = st%diag%ddt_vn_apc_pc(1,1,1,1) - 1.0d0
  if (clone%diag%vt(2,2,2) == st%diag%vt(2,2,2)) then
    print *, 'FAIL: clone aliases the original diag (shallow pointer copy)'
    stop 1
  end if
  call compare_diag(st%diag, clone%diag, 'mutated', ndiff)
  if (ndiff /= 2) then
    print *, 'FAIL: compare_diag should report exactly 2 differing elements, ndiff=', ndiff
    stop 1
  end if

  call free_state_clone(clone)
  print *, 'PASS'
end program test_diff
"""


def test_solve_nh_diff_deepcopy_and_compare(tmp_path: Path):
    """clone_state_indep_prog deep-copies (independent) and compare_prog_nnew is bit-exact."""
    (tmp_path / "min_types.f90").write_text(_MIN_TYPES)
    (tmp_path / "driver.f90").write_text(_DRIVER)
    shutil.copy(_DIFF_F90, tmp_path / "mo_solve_nh_diff.f90")

    exe = tmp_path / "test_diff"
    compile = subprocess.run([
        "gfortran", "-ffree-line-length-none", "-fcheck=all", "-g", "min_types.f90", "mo_solve_nh_diff.f90",
        "driver.f90", "-o",
        str(exe)
    ],
                             cwd=str(tmp_path),
                             capture_output=True,
                             text=True)
    assert compile.returncode == 0, f"mo_solve_nh_diff.f90 did not compile:\n{compile.stderr}"

    run = subprocess.run([str(exe)], cwd=str(tmp_path), capture_output=True, text=True)
    assert run.returncode == 0 and "PASS" in run.stdout, \
        f"differential helper driver failed:\nstdout={run.stdout}\nstderr={run.stderr}"
