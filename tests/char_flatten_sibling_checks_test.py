"""Reproducer for bug 2 (dace-fortran-fixes-needed-6c99810.md):
StripCharacterRuntime flattens sibling-arm input checks.

``hlfir-strip-character-runtime`` folds every CHARACTER compare to TRUE, so a
QE-style ``flag=='c'/'r'/'i'`` dispatch bakes its FIRST compute arm (by
design).  But the SEPARATE input-validation IF statements reference the same
folded compares, so validation branches belonging to the DEAD arms stay live:
the SDFG keeps consulting (and marshalling) the dead arms' optionals.

Fixed by folding each compare against ONE literal -- the first the entity is
tested against -- instead of folding every compare to "equal": the baked arm's
flag is TRUE, every sibling arm's flag FALSE, so the sibling clauses of the
validation IF are dead.  ``test_char_compare_folds_per_literal`` pins that
structurally; ``test_dead_arm_validation_cannot_gate_the_live_arm`` pins the
call contract it buys.  The other tests are CONTROLS documenting the
first-arm-baked contract.

Companion doc: bug2_char_flatten_sibling_checks.md
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from dace_fortran.bindings import FlattenPlan, emit_bindings
from dace_fortran.bindings.fortran_interface import build_auto_interface

pytestmark = [
    pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]

# Minimal QE ``addusxx_g`` shape: character dispatch computed into logicals,
# one COMBINED input-validation IF across all arms (guarded by ``errore``,
# which hlfir-strip-error-helpers erases), then the compute IF/ELSEIF chain.
_KERNEL = """
module disp_mod
  implicit none
contains
subroutine disp(flag, n, y, a_c, a_r)
  implicit none
  character(len=1), intent(in) :: flag
  integer, intent(in) :: n
  real(8), intent(inout) :: y(n)
  real(8), intent(in), optional :: a_c(n)
  real(8), intent(in), optional :: a_r(n)
  logical :: add_c, add_r
  integer :: i
  add_c = ( flag == 'c' .or. flag == 'C' )
  add_r = ( flag == 'r' .or. flag == 'R' )
  if ( .not. (add_c .or. add_r) ) &
     call errore('disp', 'called with incorrect flag: '//flag, 1)
  if ( (add_c .and. .not.present(a_c)) .or. &
       (add_r .and. .not.present(a_r)) ) &
     call errore('disp', 'called with incorrect arguments', 2)
  if ( add_c ) then
     do i = 1, n
        y(i) = y(i) + a_c(i)
     end do
  else if ( add_r ) then
     do i = 1, n
        y(i) = y(i) + 2.0d0 * a_r(i)
     end do
  end if
end subroutine disp
end module disp_mod
"""

# Calls the wrapper the way an ICON/QE caller would: the live arm's optional
# supplied, the dead arm's OMITTED (its dummy is not even referenced here).
_DRIVER = """
subroutine run_disp(n, y, a_c) bind(c, name='run_disp')
  use iso_c_binding
  use disp_dace_bindings
  implicit none
  integer(c_int), value :: n
  real(c_double), intent(inout) :: y(n)
  real(c_double), intent(in) :: a_c(n)
  call disp_dace(n, y, a_c)
  call disp_dace_finalize()
end subroutine run_disp
"""


def _build(tmp_path: Path):
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_KERNEL, sdfg_dir, name="disp", entry="disp_mod::disp").build()
    sdfg.name = "disp"
    return sdfg


def _call_kwargs(sdfg, n, y, a_c, a_r, *, a_c_present, a_r_present):
    """Assemble kwargs against whatever call surface the build produced (the
    dead-arm args disappear from it once the fix lands)."""
    args = sdfg.arglist()
    free = {str(s) for s in sdfg.free_symbols}
    kw = {"n": np.int32(n), "y": y}
    if "a_c" in args:
        kw["a_c"] = a_c
    if "a_r" in args:
        kw["a_r"] = a_r
    if "a_c_present" in free:
        kw["a_c_present"] = a_c_present
    if "a_r_present" in free:
        kw["a_r_present"] = a_r_present
    if "flag" in args:
        import dace
        desc = sdfg.arrays["flag"]
        kw["flag"] = (desc.dtype.type(ord("c")) if isinstance(desc, dace.data.Scalar) else np.array([ord("c")],
                                                                                                    dtype=np.int8))
    return kw


def test_first_arm_baked_and_computes(tmp_path: Path):
    """CONTROL (passes at 6c99810): after char folding the 'c' arm is baked;
    with every presence flag raised the kernel adds ``a_c`` (never the dead
    'r' arm's ``2*a_r``)."""
    sdfg = _build(tmp_path)
    n = 4
    y = np.zeros(n)
    a_c = np.array([1.0, 2.0, 3.0, 4.0])
    a_r = np.full(n, 100.0)  # poison: must never be read
    sdfg(**_call_kwargs(sdfg, n, y, a_c, a_r, a_c_present=1, a_r_present=1))
    np.testing.assert_allclose(y, a_c, err_msg="baked 'c' arm did not compute")


def _flag_stores(sdfg) -> dict:
    """``set_<flag>`` tasklet bodies, one per logical the char dispatch computes."""
    import dace
    return {
        nd.label[len("set_"):]: nd.code.as_string.strip()
        for st in sdfg.all_states()
        for nd in st.nodes() if isinstance(nd, dace.nodes.Tasklet) and nd.label.startswith("set_add_")
    }


def test_char_compare_folds_per_literal(tmp_path: Path):
    """The fix itself, structurally: the compares fold against ONE literal (the
    first arm's), so the baked arm's flag is stored TRUE and every sibling arm's
    flag FALSE.  At 6c99810 every compare folded to "equal", storing TRUE in both
    -- which is what kept the dead arms' validation clauses (and their optionals)
    live."""
    stores = _flag_stores(_build(tmp_path))
    assert set(stores) == {"add_c", "add_r"}, f"dispatch flags not found: {stores}"
    assert stores["add_c"].endswith("= (- 1)"), f"baked 'c' arm not folded TRUE: {stores['add_c']}"
    assert stores["add_r"].endswith("= 0"), f"dead 'r' arm not folded FALSE: {stores['add_r']}"


def test_dead_arm_validation_cannot_gate_the_live_arm(tmp_path: Path):
    """FAILS at 6c99810: the flattened sibling-arm input check consulted the dead
    arm's presence, and the wrapper hardwired it absent -- the contradiction that
    forced the hand-patch on addusxx/newdxx (the kernel demanded ``becphi_r`` be
    present while the 'c' arm computed).

    Three ways out, any of which honours the call contract: the dead arm's
    presence symbol is pruned from the SDFG, or the sibling clause is dead because
    its arm folded FALSE (the per-literal fold), or the wrapper reports the flag
    from the caller's actual ``present()`` instead of a hardwired constant."""
    sdfg = _build(tmp_path)
    free = {str(s) for s in sdfg.free_symbols}

    bindings_path = tmp_path / "disp_bindings.f90"
    iface = build_auto_interface(sdfg._fortran_interface_raw, "disp")
    emit_bindings(sdfg._frozen_signature, iface, FlattenPlan(entries=()), str(bindings_path))
    text = bindings_path.read_text()

    dead_arm_pruned = "a_r_present" not in free
    dead_arm_folded_false = _flag_stores(sdfg).get("add_r", "").endswith("= 0")
    presence_forwarded = "a_r_present = int(merge(1, 0, present(a_r)), c_int)" in text
    assert dead_arm_pruned or dead_arm_folded_false or presence_forwarded, (
        "dead-arm optional still gates the kernel: 'a_r_present' is a free "
        f"symbol of the SDFG (free symbols: {sorted(free)}), its arm did not fold "
        "FALSE, and the emitted wrapper hardwires the flag instead of forwarding "
        "present(a_r)")
    assert "a_r_present = 0  !" not in text, \
        "wrapper still hardwires the dead arm's optional to absent"


def test_wrapper_call_omitting_the_dead_arm_computes(tmp_path: Path):
    """The contract that matters once the dead arm's flag survives as a free symbol:
    a caller of the WRAPPER may omit the dead arm's optional entirely.  The wrapper
    must then report it absent, alias its data onto valid storage, and the baked
    'c' arm must still compute.  At 6c99810 this call returned ``y`` unchanged."""
    import ctypes

    from _util import gfortran_compile_so

    sdfg = _build(tmp_path)
    compiled = sdfg.compile()
    iface = build_auto_interface(sdfg._fortran_interface_raw, "disp")
    bindings_path = tmp_path / "disp_bindings.f90"
    emit_bindings(sdfg._frozen_signature, iface, FlattenPlan(entries=()), str(bindings_path))
    driver_path = tmp_path / "disp_driver.f90"
    driver_path.write_text(_DRIVER)

    build_dir = tmp_path / "bind_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    drv_so = build_dir / "disp_drv.so"
    gfortran_compile_so(drv_so,
                        bindings_path,
                        driver_path,
                        mod_dir=build_dir,
                        link_so=Path(compiled._lib._library_filename))

    lib = ctypes.CDLL(str(drv_so))
    dp = ctypes.POINTER(ctypes.c_double)
    y = np.zeros(4)
    a_c = np.array([1.0, 2.0, 3.0, 4.0])
    lib.run_disp.restype = None
    lib.run_disp.argtypes = [ctypes.c_int, dp, dp]
    lib.run_disp(4, y.ctypes.data_as(dp), a_c.ctypes.data_as(dp))
    np.testing.assert_allclose(y, a_c, err_msg="wrapper call without the dead arm's optional did not compute")


def test_dead_arm_absent_still_computes(tmp_path: Path):
    """Semantic form of the same contract: with the dead arm's optional
    ABSENT (presence 0 / not passed), the baked 'c' arm must still run.
    Guards against the validation branch degenerating into a silent
    early-exit (the shape observed on the QE kernels)."""
    sdfg = _build(tmp_path)
    n = 4
    y = np.zeros(n)
    a_c = np.array([1.0, 2.0, 3.0, 4.0])
    a_r = np.zeros(n)  # dead-arm dummy; zero-filled, never legitimately read
    sdfg(**_call_kwargs(sdfg, n, y, a_c, a_r, a_c_present=1, a_r_present=0))
    np.testing.assert_allclose(y,
                               a_c,
                               err_msg="kernel skipped the baked arm when the dead arm's "
                               "optional was absent")
