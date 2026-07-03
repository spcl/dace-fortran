"""The differential ``solve_nh`` patch produces valid Fortran.

:func:`apply_solve_nh_patch` rewrites ``mo_solve_nonhydro`` so ``solve_nh``
becomes a differential driver (SDFG DUT + stock REF, deep-copy + bit-exact
compare) with the original body preserved as ``solve_nh_ref``.  This pins the
transform structurally AND checks the emitted module compiles (``gfortran
-fsyntax-only``) against minimal stand-in types + ``mo_solve_nh_diff`` -- so a
Fortran error in the driver (a bad USE, a mismatched call, a stray name) is
caught here without a full ICON build.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from icon.full._icon_solve_nh_patch import apply_solve_nh_patch

pytestmark = pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")

_HERE = Path(__file__).resolve().parent
_DIFF_F90 = _HERE / "mo_solve_nh_diff.f90"

# Minimal stand-ins for ICON's types (same module + field names).
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

module mo_model_domain
  implicit none
  type :: t_patch
    integer :: id
  end type t_patch
end module mo_model_domain

module mo_intp_data_strc
  implicit none
  type :: t_int_state
    integer :: id
  end type t_int_state
end module mo_intp_data_strc
"""

# A minimal ``mo_solve_nonhydro`` with the real 15-dummy ``solve_nh`` surface and
# a trivial reference body.  The patch must rename this to ``solve_nh_ref`` and
# wrap it in the differential driver.
_MIN_SOLVE_NH = """\
module mo_solve_nonhydro
  use mo_nonhydro_types, only: t_nh_state
  use mo_model_domain,   only: t_patch
  use mo_intp_data_strc, only: t_int_state
  use mo_prepadv_types,  only: t_prepare_adv
  implicit none
contains
  SUBROUTINE solve_nh(p_nh, p_patch, p_int, prep_adv, nnow, nnew, &
                      l_init, l_recompute, lsave_mflx, lprep_adv, lclean_mflx, &
                      idyn_timestep, jstep, dtime, lacc)
    TYPE(t_nh_state),    TARGET, INTENT(INOUT) :: p_nh
    TYPE(t_patch),       TARGET, INTENT(INOUT) :: p_patch
    TYPE(t_int_state),   TARGET, INTENT(IN)    :: p_int
    TYPE(t_prepare_adv), TARGET, INTENT(INOUT) :: prep_adv
    INTEGER, INTENT(IN) :: nnow, nnew
    LOGICAL, INTENT(IN) :: l_init, l_recompute, lsave_mflx, lprep_adv, lclean_mflx
    INTEGER, INTENT(IN) :: idyn_timestep, jstep
    REAL(kind=8), INTENT(IN) :: dtime
    LOGICAL, INTENT(IN) :: lacc
    ! trivial reference dycore body
    p_nh%prog(nnew)%vn = p_nh%prog(nnow)%vn + dtime
  END SUBROUTINE solve_nh
end module mo_solve_nonhydro
"""


def test_solve_nh_patch_structure_and_compiles(tmp_path: Path):
    """apply_solve_nh_patch renames the original to solve_nh_ref, inserts the
    differential driver, and the result compiles."""
    patched = apply_solve_nh_patch(_MIN_SOLVE_NH)

    # Structure: the original survives as solve_nh_ref; the new solve_nh drives
    # the clone / DUT / REF / compare / free.
    assert "SUBROUTINE solve_nh_ref(" in patched
    assert "END SUBROUTINE solve_nh_ref" in patched
    assert "USE mo_solve_nh_diff" in patched
    assert "CALL clone_state_indep_prog(p_nh, nh_ref__dace)" in patched
    assert "CALL solve_nh_dace_icon(" in patched
    assert "CALL solve_nh_ref(nh_ref__dace," in patched
    assert "CALL compare_prog_nnew(p_nh, nh_ref__dace, nnew," in patched
    # the driver must precede the reference (ICON calls ``solve_nh``).
    assert patched.index("SUBROUTINE solve_nh(") < patched.index("SUBROUTINE solve_nh_ref(")

    (tmp_path / "types.f90").write_text(_MIN_TYPES)
    (tmp_path / "patched.f90").write_text(patched)
    shutil.copy(_DIFF_F90, tmp_path / "mo_solve_nh_diff.f90")

    # -fsyntax-only: modules must be compiled in dependency order (types ->
    # mo_solve_nh_diff -> the patched module which USEs it).
    r = subprocess.run(
        ["gfortran", "-fsyntax-only", "-ffree-line-length-none", "types.f90", "mo_solve_nh_diff.f90", "patched.f90"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True)
    assert r.returncode == 0, f"patched solve_nh did not compile:\n{r.stderr}"
