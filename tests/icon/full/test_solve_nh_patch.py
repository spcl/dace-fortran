"""The differential ``solve_nh`` patch produces valid Fortran.

:func:`apply_solve_nh_patch` rewrites ``mo_solve_nonhydro`` so ``solve_nh`` becomes a
differential driver (SDFG DUT + stock REF, deep-copy + bit-exact compare) with the
original body preserved as ``solve_nh_ref``.  Pins the transform structurally AND
checks the emitted module compiles (``gfortran -fsyntax-only``) against minimal
stand-in types -- catches a Fortran error in the driver without a full ICON build.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from icon.full._icon_solve_nh_patch import apply_solve_nh_patch
from icon.full._solve_nh_min_types import MIN_GEOMETRY_TYPES_F90, MIN_STATE_TYPES_F90

pytestmark = pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")

_HERE = Path(__file__).resolve().parent
_DIFF_F90 = _HERE / "mo_solve_nh_diff.f90"

# shared with test_solve_nh_diff.py so the member lists cannot drift apart.
_MIN_TYPES = MIN_STATE_TYPES_F90 + "\n" + MIN_GEOMETRY_TYPES_F90

# minimal mo_solve_nonhydro with the real 15-dummy solve_nh surface; the patch must
# rename this to solve_nh_ref and wrap it in the differential driver.
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

    # original survives as solve_nh_ref; new solve_nh drives clone / DUT / REF / compare / free.
    assert "SUBROUTINE solve_nh_ref(" in patched
    assert "END SUBROUTINE solve_nh_ref" in patched
    assert "USE mo_solve_nh_diff" in patched
    assert "CALL clone_state_indep_prog(p_nh, nh_ref__dace)" in patched
    assert "CALL clone_prepadv_indep(prep_adv, prep_ref__dace)" in patched
    assert "CALL solve_nh_dace_icon(" in patched
    assert "CALL solve_nh_ref(nh_ref__dace," in patched
    # compare covers both prognostic time levels (nnow: DUT-stomp guard) + prep_adv +
    # the full diag, closed by the greppable per-call TOTAL line.
    assert "CALL compare_prog_nnew(p_nh, nh_ref__dace, nnew," in patched
    assert "CALL compare_prog_nnew(p_nh, nh_ref__dace, nnow," in patched
    assert "CALL compare_prepadv(prep_adv, prep_ref__dace," in patched
    assert "CALL compare_diag(p_nh % diag, nh_ref__dace % diag," in patched
    assert "[diff] solve_nh TOTAL: " in patched
    # driver must precede the reference (ICON calls solve_nh).
    assert patched.index("SUBROUTINE solve_nh(") < patched.index("SUBROUTINE solve_nh_ref(")

    (tmp_path / "types.f90").write_text(_MIN_TYPES)
    (tmp_path / "patched.f90").write_text(patched)
    shutil.copy(_DIFF_F90, tmp_path / "mo_solve_nh_diff.f90")

    # -fsyntax-only: modules compiled in dependency order (types -> mo_solve_nh_diff ->
    # the patched module which USEs it).
    r = subprocess.run(
        ["gfortran", "-fsyntax-only", "-ffree-line-length-none", "types.f90", "mo_solve_nh_diff.f90", "patched.f90"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True)
    assert r.returncode == 0, f"patched solve_nh did not compile:\n{r.stderr}"
