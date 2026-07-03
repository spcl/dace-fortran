"""Multi-file Fortran build: several ``.f90`` files (a driver plus the
modules it ``USE``s, in any order) + a mangled entry -> one SDFG.

``build_sdfg_from_files`` stages the files into the scratch dir and
``merge_used_modules`` inlines every ``USE``-d module into the root's
translation unit (the root is the file defining the entry's
procedure), so flang sees one self-contained TU.  These tests pass
multiple files and check the merged SDFG runs correctly.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import have_flang

from dace_fortran import build_sdfg, build_sdfg_from_files

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_MOD_ADD = """
module mod_add
  implicit none
contains
  pure real(8) function add2(a, b)
    real(8), intent(in) :: a, b
    add2 = a + b
  end function add2
end module mod_add
"""

_MOD_SCALE = """
module mod_scale
  use mod_add
  implicit none
contains
  pure real(8) function scale_add(a, b, s)
    real(8), intent(in) :: a, b, s
    scale_add = s * add2(a, b)
  end function scale_add
end module mod_scale
"""

_DRIVER = """
subroutine run(x, y, z, n)
  use mod_add
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: x(n), y(n)
  real(8), intent(out) :: z(n)
  integer :: i
  do i = 1, n
    z(i) = add2(x(i), y(i))
  end do
end subroutine run
"""

# Driver USEs mod_scale which itself USEs mod_add -> transitive merge.
_DRIVER_CHAIN = """
subroutine run_chain(x, y, z, n)
  use mod_scale
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: x(n), y(n)
  real(8), intent(out) :: z(n)
  integer :: i
  do i = 1, n
    z(i) = scale_add(x(i), y(i), 2.0d0)
  end do
end subroutine run_chain
"""


def _write(tmp: Path, **named) -> list:
    """Write ``name=source`` pairs to ``<tmp>/<name>.f90``; return paths."""
    tmp.mkdir(parents=True, exist_ok=True)
    out = []
    for nm, src in named.items():
        p = tmp / f"{nm}.f90"
        p.write_text(src)
        out.append(p)
    return out


@pytest.mark.parametrize("merge_engine", ["fparser", "regex"])
def test_two_files_driver_plus_module(tmp_path: Path, merge_engine):
    """Driver + one ``USE``-d module, files given out of order (both engines)."""
    files = _write(tmp_path / "src", driver=_DRIVER, mod_add=_MOD_ADD)
    sdfg = build_sdfg_from_files(list(reversed(files)),
                                 entry="run",
                                 name="run",
                                 out_dir=tmp_path / "b",
                                 merge_engine=merge_engine)
    n = 16
    rng = np.random.default_rng(0)
    x = np.asfortranarray(rng.random(n))
    y = np.asfortranarray(rng.random(n))
    z = np.zeros(n, order="F")
    sdfg(x=x, y=y, z=z, n=n)
    np.testing.assert_allclose(z, x + y, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("merge_engine", ["fparser", "regex"])
def test_three_files_transitive_use(tmp_path: Path, merge_engine):
    """Driver USEs mod_scale which USEs mod_add -> transitive inline (both engines)."""
    files = _write(tmp_path / "src", mod_add=_MOD_ADD, mod_scale=_MOD_SCALE, driver=_DRIVER_CHAIN)
    sdfg = build_sdfg_from_files(files,
                                 entry="run_chain",
                                 name="run_chain",
                                 out_dir=tmp_path / "b",
                                 merge_engine=merge_engine)
    n = 8
    rng = np.random.default_rng(1)
    x = np.asfortranarray(rng.random(n))
    y = np.asfortranarray(rng.random(n))
    z = np.zeros(n, order="F")
    sdfg(x=x, y=y, z=z, n=n)
    np.testing.assert_allclose(z, 2.0 * (x + y), rtol=1e-12, atol=1e-12)


def test_entry_proc_name_accepts_all_three_spellings():
    """Root-file selection reduces every accepted entry spelling to the bare
    procedure name: the friendly ``module::proc`` qualifier, a mangled Flang
    symbol, and an already-plain name.  (Regression: ``module::proc`` used to
    fall through unstripped and never match any ``subroutine``/``function``.)"""
    from dace_fortran.build import _entry_proc_name
    assert _entry_proc_name("mo_solve_nonhydro::solve_nh") == "solve_nh"
    assert _entry_proc_name("m_array_return::kern") == "kern"
    assert _entry_proc_name("_QMmymodPbar") == "bar"
    assert _entry_proc_name("_QPmain") == "main"
    assert _entry_proc_name("solve_nh") == "solve_nh"
    assert _entry_proc_name(None) is None


@pytest.mark.parametrize("merge_engine", ["fparser", "regex"])
def test_module_qualified_entry_selects_root(tmp_path: Path, merge_engine):
    """A ``module::proc`` entry resolves the root file and builds (both engines).
    Mirrors the bare-name path of ``test_two_files_driver_plus_module`` with the
    driver wrapped in a module and addressed as ``drv::run``."""
    driver = """
module drv
  use mod_add
  implicit none
contains
  subroutine run(x, y, z, n)
    integer, intent(in) :: n
    real(8), intent(in) :: x(n), y(n)
    real(8), intent(out) :: z(n)
    integer :: i
    do i = 1, n
      z(i) = add2(x(i), y(i))
    end do
  end subroutine run
end module drv
"""
    files = _write(tmp_path / "src", driver=driver, mod_add=_MOD_ADD)
    sdfg = build_sdfg_from_files(files, entry="drv::run", name="run", out_dir=tmp_path / "b", merge_engine=merge_engine)
    n = 12
    rng = np.random.default_rng(2)
    x = np.asfortranarray(rng.random(n))
    y = np.asfortranarray(rng.random(n))
    z = np.zeros(n, order="F")
    sdfg(x=x, y=y, z=z, n=n)
    np.testing.assert_allclose(z, x + y, rtol=1e-12, atol=1e-12)


def test_entry_not_found_is_rejected(tmp_path: Path):
    """No input file defines the entry's procedure -> clear error."""
    files = _write(tmp_path / "src", mod_add=_MOD_ADD)
    with pytest.raises(ValueError, match="(?i)no input file defines procedure"):
        build_sdfg_from_files(files, entry="missing", name="x", out_dir=tmp_path / "b")


def test_entry_resolution_contract(tmp_path: Path):
    """Entry contract: ``build_sdfg`` auto-resolves a single
    procedure but errors when the source is empty or *ambiguous*
    (an SDFG targets one specific procedure -- no "first of many"
    guessing); ``build_sdfg_from_files`` always requires ``entry=``
    (it selects the root file)."""
    files = _write(tmp_path / "src", driver=_DRIVER, mod_add=_MOD_ADD)
    with pytest.raises(ValueError, match="(?i)requires entry"):
        build_sdfg_from_files(files, name="run", out_dir=tmp_path / "b")

    two_procs = _DRIVER + "\nsubroutine other(p)\n  real(8) :: p\n  p = 1.0d0\nend subroutine other\n"
    with pytest.raises(ValueError, match="(?i)multiple procedures"):
        build_sdfg(two_procs, name="amb", out_dir=tmp_path / "b2")

    with pytest.raises(ValueError, match="(?i)no SUBROUTINE/FUNCTION"):
        build_sdfg("module m\n  integer :: x\nend module m\n", name="np", out_dir=tmp_path / "b3")
