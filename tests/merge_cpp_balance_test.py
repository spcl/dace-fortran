"""Single-TU module merge must keep cpp directives balanced.

``merge_used_modules`` extracts each ``module .. end module`` block (with its cpp
preamble) into one TU.  A whole-module ``#ifdef .. module .. #endif`` wrapper
split across a block boundary leaves an orphan ``#if``/``#endif`` that breaks the
cpp pass; ``_balance_cpp`` drops the orphan side (guarded content already selected
by the real build).

This is the shape ICON's ~150-module USE-closure hits; reproduced minimally here.
"""
import re
from pathlib import Path

import pytest

from _util import have_flang
from dace_fortran import build_sdfg_from_files
from dace_fortran.preprocess import merge_used_modules

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# mo_dbl shares a file with a preceding module wrapped in a whole-module #ifdef:
# mo_dbl's extracted block picks up the orphan #endif, the wrapped block an orphan #if.
_HELPERS = """
#ifdef UNUSED_GUARD
module mo_unused
contains
  pure integer function noop(n) result(r)
    integer, intent(in) :: n
    r = n
  end function noop
end module mo_unused
#endif

module mo_dbl
contains
  pure integer function dbl(n) result(r)
    integer, intent(in) :: n
    r = 2 * n
  end function dbl
end module mo_dbl
"""

_CALLER = """
module apply_dbl_mod
  implicit none
contains
  subroutine apply_dbl(k, out)
    use mo_dbl, only: dbl
    implicit none
    integer, intent(in) :: k
    real(8), intent(out) :: out(4)
    integer :: m, i
    m = dbl(k)
    do i = 1, 4
      out(i) = real(m + i, 8)
    end do
  end subroutine apply_dbl
end module apply_dbl_mod
"""


def test_merge_used_modules_balances_cpp(tmp_path: Path):
    """Merged source has matched ``#if*``/``#endif`` counts even when a USE'd module shares a file with a cpp-wrapped sibling."""
    (tmp_path / "helpers.f90").write_text(_HELPERS)
    merged = merge_used_modules(_CALLER, search_dirs=[str(tmp_path)])
    opens = len(re.findall(r"(?im)^\s*#\s*(?:if|ifdef|ifndef)\b", merged))
    closes = len(re.findall(r"(?im)^\s*#\s*endif\b", merged))
    assert opens == closes, f"unbalanced cpp after merge: {opens} #if* vs {closes} #endif"
    assert "function dbl" in merged.lower(), "the USE'd module body was not merged in"


def test_cpp_wrapped_module_merges_and_builds(tmp_path: Path):
    """End to end: the cpp-wrapped-sibling file merges, ``dbl`` inlines, and the SDFG computes ``out(i) = 2*k + i``."""
    import numpy as np

    caller = tmp_path / "apply_dbl.f90"
    caller.write_text(_CALLER)
    helpers = tmp_path / "helpers.f90"
    helpers.write_text(_HELPERS)

    sdfg = build_sdfg_from_files([caller, helpers],
                                 entry="apply_dbl_mod::apply_dbl",
                                 name="apply_dbl",
                                 out_dir=tmp_path / "build")
    k = 5
    out = np.zeros(4, dtype=np.float64)
    sdfg(k=np.int32(k), out=out)
    ref = np.array([2 * k + i for i in range(1, 5)], dtype=np.float64)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)
