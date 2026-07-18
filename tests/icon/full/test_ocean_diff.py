"""Unit test for the mo_ocean_diff differential helpers used by the ocean binding-swap integration
(runs stock solve_free_sfc_ab_mimetic + SDFG solve_free_sfc_dace_icon on the same input, compares
state BIT-FOR-BIT). Pins three properties against gfortran with a small driver:

  * deep copy is independent BOTH ways -- clone_ocean_state_indep must allocate FRESH targets (every
    field is POINTER, and p_prog(:) is itself a POINTER array, so a shallow dst=src would alias prog
    slots too); mutating either side after cloning must not move the other.
  * compare is bit-exact and exhaustive -- compare_ocean_{prog,diag,aux} report 0 for identical state
    and exactly the perturbation count after single-element perturbations.
  * a_veloc_v snapshot/restore round-trips -- clone_field3/restore_field3/compare_field3 implement the
    save/park/restore dance needed because the PP scheme time-smooths a_veloc_v in place through a
    module pointer (_icon_ocean_patch.py).

Real t_hydro_ocean_state types are replaced with minimal same-name modules carrying only the touched
fields, so the test needs just gfortran (no flang/SDFG/ICON build)."""
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")

_HERE = Path(__file__).resolve().parent
_DIFF_F90 = _HERE / "mo_ocean_diff.f90"

# minimal stand-ins for ICON's types -- same module/field names/ranks (cross-checked against the real
# mo_ocean_types.f90/mo_math_types.f90) so mo_ocean_diff.f90 compiles unchanged. Pointers default to
# => null() so partially-allocated test states have a defined association status.
_MIN_TYPES = """\
module mo_math_types
  implicit none
  type :: t_cartesian_coordinates
    real(kind=8) :: x(3)
  end type t_cartesian_coordinates
end module mo_math_types

module mo_ocean_types
  use mo_math_types, only: t_cartesian_coordinates
  implicit none
  type :: t_hydro_ocean_prog
    real(kind=8), pointer :: h(:,:) => null(), eta_c(:,:) => null(), stretch_c(:,:) => null()
    real(kind=8), pointer :: vn(:,:,:) => null()
    real(kind=8), pointer :: tracer(:,:,:,:) => null()
  end type t_hydro_ocean_prog
  ! Same field names + ranks the real ICON t_hydro_ocean_diag carries, so
  ! mo_ocean_diff's clone_diag_indep / compare_ocean_diag compile unchanged.
  type :: t_hydro_ocean_diag
    real(kind=8), pointer :: vort(:,:,:) => null(), grad(:,:,:) => null(), &
      veloc_adv_horz(:,:,:) => null(), veloc_adv_vert(:,:,:) => null(), &
      laplacian_horz(:,:,:) => null(), laplacian_vert(:,:,:) => null(), &
      press_hyd(:,:,:) => null(), press_grad(:,:,:) => null(), &
      vn_pred(:,:,:) => null(), vn_pred_ptp(:,:,:) => null(), &
      vn_time_weighted(:,:,:) => null(), w(:,:,:) => null(), w_old(:,:,:) => null(), &
      mass_flx_e(:,:,:) => null(), div_mass_flx_c(:,:,:) => null(), ptp_vn(:,:,:) => null()
    real(kind=8), pointer :: h_e(:,:) => null()
    ! shared (never cloned) members, present so the shallow dst = src carries
    ! them across and the clone provably leaves them aliased
    real(kind=8), pointer :: thick_e(:,:) => null()
  end type t_hydro_ocean_diag
  type :: t_hydro_ocean_aux
    real(kind=8), pointer :: g_n(:,:,:) => null(), g_nm1(:,:,:) => null(), g_nimd(:,:,:) => null()
    real(kind=8), pointer :: p_rhs_sfc_eq(:,:) => null(), bc_top_vn(:,:) => null(), &
      bc_bot_vn(:,:) => null(), bc_top_u(:,:) => null(), bc_top_v(:,:) => null(), &
      bc_top_WindStress(:,:) => null()
    type(t_cartesian_coordinates), pointer :: bc_top_veloc_cc(:,:) => null()
  end type t_hydro_ocean_aux
  type :: t_hydro_ocean_state
    type(t_hydro_ocean_prog), pointer :: p_prog(:) => null()
    type(t_hydro_ocean_diag) :: p_diag
    type(t_hydro_ocean_aux) :: p_aux
  end type t_hydro_ocean_state
end module mo_ocean_types
"""

_DRIVER = """\
program test_ocean_diff
  use mo_ocean_types
  use mo_ocean_diff
  implicit none
  type(t_hydro_ocean_state) :: st, clone
  real(kind=8), pointer :: a_veloc_v(:,:,:), aveloc_pre(:,:,:), aveloc_dut(:,:,:)
  integer :: nnew, ndiff, i, j, k
  real(kind=8) :: v

  nnew = 2
  ! p_prog is a POINTER array in ICON (not allocatable): 2 time levels.
  allocate(st%p_prog(2))
  do i = 1, 2
    ! non-default lbounds on h pin exact bound preservation in the clone
    allocate(st%p_prog(i)%h(0:3,0:4), st%p_prog(i)%vn(3,4,5), st%p_prog(i)%tracer(3,4,5,2))
    do j = 0, 4
      v = real(100*i + j, kind=8) * 1.5d0
      st%p_prog(i)%h(:,j) = v
    end do
    do k = 1, 5
      do j = 1, 4
        v = real(100*i + 10*j + k, kind=8) * 0.5d0
        st%p_prog(i)%vn(:,j,k) = v
        st%p_prog(i)%tracer(:,j,k,1) = v + 0.25d0
        st%p_prog(i)%tracer(:,j,k,2) = v - 0.25d0
      end do
    end do
    ! eta_c / stretch_c stay unassociated (non-z* config): the clone /
    ! compare helpers must no-op on them.
  end do
  ! a written rank-3 diag field, an adjacent-phase rank-3 field, and a rank-2
  ! field; vn_pred_ptp / laplacian_vert etc. stay unassociated
  allocate(st%p_diag%vn_pred(3,4,5), st%p_diag%w(3,5,4), st%p_diag%h_e(3,4))
  st%p_diag%vn_pred = 7.0d0
  st%p_diag%w = -3.0d0
  st%p_diag%h_e = 2.0d0
  ! a shared (never cloned) member: the clone must keep ALIASING it
  allocate(st%p_diag%thick_e(3,4))
  st%p_diag%thick_e = 11.0d0
  ! aux: AB term stack + surface RHS + BCs incl. the cartesian wind-stress BC
  allocate(st%p_aux%g_n(3,4,5), st%p_aux%g_nimd(3,4,5), st%p_aux%p_rhs_sfc_eq(3,4))
  allocate(st%p_aux%bc_top_vn(3,4), st%p_aux%bc_top_veloc_cc(3,4))
  st%p_aux%g_n = 1.0d0
  st%p_aux%g_nimd = 2.0d0
  st%p_aux%p_rhs_sfc_eq = 3.0d0
  st%p_aux%bc_top_vn = 4.0d0
  do j = 1, 4
    do i = 1, 3
      st%p_aux%bc_top_veloc_cc(i,j)%x = [1.0d0, 2.0d0, 3.0d0]
    end do
  end do

  call clone_ocean_state_indep(st, clone)

  ! (1) a fresh clone is bit-exact on prog(nnew), diag and aux; bounds preserved.
  call compare_ocean_prog(st, clone, nnew, 'init', ndiff)
  if (ndiff /= 0) then
    print *, 'FAIL: fresh clone prog is not bit-exact, ndiff=', ndiff
    stop 1
  end if
  call compare_ocean_diag(st%p_diag, clone%p_diag, 'init', ndiff)
  if (ndiff /= 0) then
    print *, 'FAIL: fresh clone diag is not bit-exact, ndiff=', ndiff
    stop 1
  end if
  call compare_ocean_aux(st%p_aux, clone%p_aux, 'init', ndiff)
  if (ndiff /= 0) then
    print *, 'FAIL: fresh clone aux is not bit-exact, ndiff=', ndiff
    stop 1
  end if
  if (lbound(clone%p_prog(1)%h, 1) /= 0 .or. ubound(clone%p_prog(1)%h, 2) /= 4) then
    print *, 'FAIL: clone dropped the source lbound/ubound of prog h'
    stop 1
  end if
  if (associated(clone%p_prog, st%p_prog)) then
    print *, 'FAIL: clone p_prog array aliases the original (POINTER array not re-allocated)'
    stop 1
  end if
  if (.not. associated(clone%p_diag%thick_e, st%p_diag%thick_e)) then
    print *, 'FAIL: shared (non-cloned) diag member no longer aliases the original'
    stop 1
  end if
  if (associated(clone%p_prog(nnew)%eta_c)) then
    print *, 'FAIL: clone fabricated an eta_c target for an unassociated source'
    stop 1
  end if

  ! (2) original -> clone independence: mutate the original; the clone must
  !     not move, and each compare must SEE exactly its perturbations.
  st%p_prog(nnew)%h(0,0) = st%p_prog(nnew)%h(0,0) + 1.0d0
  st%p_prog(1)%h(1,1) = st%p_prog(1)%h(1,1) + 1.0d0   ! nold slot: prog compare is nnew-only
  if (clone%p_prog(nnew)%h(0,0) == st%p_prog(nnew)%h(0,0) .or. &
    & clone%p_prog(1)%h(1,1) == st%p_prog(1)%h(1,1)) then
    print *, 'FAIL: clone aliases the original prog (shallow pointer copy)'
    stop 1
  end if
  call compare_ocean_prog(st, clone, nnew, 'mutated', ndiff)
  if (ndiff /= 1) then
    print *, 'FAIL: compare_ocean_prog should report exactly 1 differing element, ndiff=', ndiff
    stop 1
  end if

  st%p_diag%vn_pred(2,2,2) = st%p_diag%vn_pred(2,2,2) + 1.0d0
  st%p_diag%h_e(1,1) = st%p_diag%h_e(1,1) - 1.0d0
  if (clone%p_diag%vn_pred(2,2,2) == st%p_diag%vn_pred(2,2,2)) then
    print *, 'FAIL: clone aliases the original diag (shallow pointer copy)'
    stop 1
  end if
  call compare_ocean_diag(st%p_diag, clone%p_diag, 'mutated', ndiff)
  if (ndiff /= 2) then
    print *, 'FAIL: compare_ocean_diag should report exactly 2 differing elements, ndiff=', ndiff
    stop 1
  end if

  st%p_aux%g_nimd(1,2,1) = st%p_aux%g_nimd(1,2,1) + 1.0d0
  st%p_aux%bc_top_veloc_cc(1,1)%x(2) = st%p_aux%bc_top_veloc_cc(1,1)%x(2) + 1.0d0
  if (clone%p_aux%bc_top_veloc_cc(1,1)%x(2) == st%p_aux%bc_top_veloc_cc(1,1)%x(2)) then
    print *, 'FAIL: clone aliases the original aux cartesian BC (shallow pointer copy)'
    stop 1
  end if
  call compare_ocean_aux(st%p_aux, clone%p_aux, 'mutated', ndiff)
  if (ndiff /= 2) then
    print *, 'FAIL: compare_ocean_aux should report exactly 2 differing elements, ndiff=', ndiff
    stop 1
  end if

  ! (3) clone -> original independence (the REF run writes into the clone).
  v = st%p_diag%w(1,1,1)
  clone%p_diag%w(1,1,1) = clone%p_diag%w(1,1,1) + 5.0d0
  clone%p_prog(nnew)%vn(1,1,1) = clone%p_prog(nnew)%vn(1,1,1) + 5.0d0
  clone%p_prog(nnew)%tracer(1,1,1,2) = clone%p_prog(nnew)%tracer(1,1,1,2) + 5.0d0
  if (st%p_diag%w(1,1,1) /= v) then
    print *, 'FAIL: mutating the clone moved the original diag (shared storage)'
    stop 1
  end if
  call compare_ocean_prog(st, clone, nnew, 'ref-run', ndiff)
  if (ndiff /= 3) then
    print *, 'FAIL: compare_ocean_prog should report exactly 3 differing elements, ndiff=', ndiff
    stop 1
  end if

  ! (4) the a_veloc_v snapshot/park/restore dance the driver performs.
  allocate(a_veloc_v(3,4,5))
  a_veloc_v = 0.5d0
  call clone_field3(aveloc_pre, a_veloc_v)          ! snapshot pre-call
  a_veloc_v(3,3,3) = 9.0d0                          ! "DUT" mutates in place
  call clone_field3(aveloc_dut, a_veloc_v)          ! park the DUT result
  call restore_field3(a_veloc_v, aveloc_pre)        ! hand the "REF" pre-call values
  if (a_veloc_v(3,3,3) /= 0.5d0) then
    print *, 'FAIL: restore_field3 did not reinstate the snapshot values'
    stop 1
  end if
  a_veloc_v(3,3,3) = 9.0d0                          ! "REF" recomputes the same value
  call compare_field3(aveloc_dut, a_veloc_v, 'a_veloc_v', ndiff)
  if (ndiff /= 0) then
    print *, 'FAIL: identical DUT/REF a_veloc_v flagged as different, ndiff=', ndiff
    stop 1
  end if
  a_veloc_v(1,1,1) = -1.0d0                         ! a real divergence
  call compare_field3(aveloc_dut, a_veloc_v, 'a_veloc_v', ndiff)
  if (ndiff /= 1) then
    print *, 'FAIL: compare_field3 should report exactly 1 differing element, ndiff=', ndiff
    stop 1
  end if
  call free_field3(aveloc_pre)
  call free_field3(aveloc_dut)

  ! (5) clean free: releases the clone's targets + its p_prog array, leaves
  !     the original untouched.
  call free_ocean_state_clone(clone)
  if (associated(clone%p_prog)) then
    print *, 'FAIL: free_ocean_state_clone left p_prog associated'
    stop 1
  end if
  if (.not. associated(st%p_prog(nnew)%h) .or. .not. associated(st%p_diag%vn_pred)) then
    print *, 'FAIL: free_ocean_state_clone touched the ORIGINAL state'
    stop 1
  end if
  print *, 'PASS'
end program test_ocean_diff
"""


def test_ocean_diff_deepcopy_and_compare(tmp_path: Path):
    """clone_ocean_state_indep deep-copies independent both ways; compares are bit-exact + exhaustive;
    a_veloc_v snapshot dance round-trips."""
    (tmp_path / "min_types.f90").write_text(_MIN_TYPES)
    (tmp_path / "driver.f90").write_text(_DRIVER)
    shutil.copy(_DIFF_F90, tmp_path / "mo_ocean_diff.f90")

    exe = tmp_path / "test_ocean_diff"
    compile = subprocess.run([
        "gfortran", "-ffree-line-length-none", "-fcheck=all", "-g", "min_types.f90", "mo_ocean_diff.f90", "driver.f90",
        "-o",
        str(exe)
    ],
                             cwd=str(tmp_path),
                             capture_output=True,
                             text=True)
    assert compile.returncode == 0, f"mo_ocean_diff.f90 did not compile:\n{compile.stderr}"

    run = subprocess.run([str(exe)], cwd=str(tmp_path), capture_output=True, text=True)
    assert run.returncode == 0 and "PASS" in run.stdout, \
        f"differential helper driver failed:\nstdout={run.stdout}\nstderr={run.stderr}"
