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

from _util import build_sdfg_from_files, have_flang

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


def test_two_files_driver_plus_module(tmp_path: Path):
    """Driver + one ``USE``-d module, files given out of order."""
    files = _write(tmp_path / "src", driver=_DRIVER, mod_add=_MOD_ADD)
    sdfg = build_sdfg_from_files(list(reversed(files)), tmp_path / "b",
                                 name="run", entry="_QPrun").build()
    n = 16
    rng = np.random.default_rng(0)
    x = np.asfortranarray(rng.random(n))
    y = np.asfortranarray(rng.random(n))
    z = np.zeros(n, order="F")
    sdfg(x=x, y=y, z=z, n=n)
    np.testing.assert_allclose(z, x + y, rtol=1e-12, atol=1e-12)


def test_three_files_transitive_use(tmp_path: Path):
    """Driver USEs mod_scale which USEs mod_add -> transitive inline."""
    files = _write(tmp_path / "src", mod_add=_MOD_ADD, mod_scale=_MOD_SCALE,
                   driver=_DRIVER_CHAIN)
    sdfg = build_sdfg_from_files(files, tmp_path / "b",
                                 name="run_chain", entry="_QPrun_chain").build()
    n = 8
    rng = np.random.default_rng(1)
    x = np.asfortranarray(rng.random(n))
    y = np.asfortranarray(rng.random(n))
    z = np.zeros(n, order="F")
    sdfg(x=x, y=y, z=z, n=n)
    np.testing.assert_allclose(z, 2.0 * (x + y), rtol=1e-12, atol=1e-12)


def test_entry_not_found_is_rejected(tmp_path: Path):
    """No input file defines the entry's procedure -> clear error."""
    files = _write(tmp_path / "src", mod_add=_MOD_ADD)
    with pytest.raises(ValueError, match="(?i)no input file defines procedure"):
        build_sdfg_from_files(files, tmp_path / "b", name="x", entry="_QPmissing")


def test_entry_required(tmp_path: Path):
    """``entry=`` selects the root, so it is mandatory."""
    files = _write(tmp_path / "src", driver=_DRIVER, mod_add=_MOD_ADD)
    with pytest.raises(ValueError, match="(?i)requires entry"):
        build_sdfg_from_files(files, tmp_path / "b", name="run")
