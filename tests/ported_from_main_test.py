"""Drop-in HLFIR ports of the simplest tests from ``origin/main``'s ``tests/fortran/``:
same numerical assertions, swapping the import to HLFIR's ``create_sdfg_from_string``.
Only the short ones so far; allocate-based/PROGRAM-wrapper cases wait on matching HLFIR lowerings.
"""

import numpy as np
import pytest

from _util import have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not available")

# ---------------------------------------------------------------------------
# tests/fortran/fortran_loops_test.py  --  simplest nested-loop case.
# ---------------------------------------------------------------------------


def test_fortran_frontend_loop_region_basic_loop():
    from dace_fortran import build_sdfg

    # Legacy wraps the subroutine in a PROGRAM+CALL; HLFIR runs on the subroutine directly
    # (cross-subroutine lowering not yet implemented). Compute body is identical.
    test_string = """
module loop_test_function_mod
contains
subroutine loop_test_function(a, b, c)
  implicit none
  real(8) :: a(10, 10), b(10, 10), c(10, 10)
  integer :: jk, jl
  do jk = 1, 10
    do jl = 1, 10
      c(jk, jl) = a(jk, jl) + b(jk, jl)
    end do
  end do
end subroutine loop_test_function
end module loop_test_function_mod
"""
    sdfg = build_sdfg(test_string, entry="loop_test_function_mod::loop_test_function", name="loop_test")

    a_test = np.full((10, 10), 2.0, dtype=np.float64)
    b_test = np.full((10, 10), 3.0, dtype=np.float64)
    c_test = np.zeros((10, 10), dtype=np.float64)
    sdfg(a=a_test, b=b_test, c=c_test)

    validate = np.full((10, 10), 5.0, dtype=np.float64)
    assert np.allclose(c_test, validate)
