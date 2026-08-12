"""Value-projection SSA: an allocatable extent that is a loop-variant array element, whose source array is mutated inside the same loop.

``allocate(tmp(sz(i)))`` reads ``sz(i)`` at the ALLOCATE, before the in-iteration
write ``sz(i) = sz(i) + 100``. A correct per-site snapshot freezes the pre-write
value *every iteration*; ``sz(i)`` is neither a constant index (``i`` is the loop
variable) nor loop-invariant, so the old entry-time ``value_symbols`` snapshot is
wrong on both counts. The extent symbol is non-free (defined in the loop), so
DaCe Scope-allocates ``tmp`` inside the loop body at the per-iteration size --
no over-allocation, no 1-past-end write.

Regression for the heap corruption fixed in 0fbe410: a wrong (function-scope,
single free symbol) allocation over-runs the buffer; glibc aborts on the corrupt
free even without a sanitizer, and ASan pinpoints the write
(see ``scripts/lint_generated_kernel.py``). Correctness is oracle'd against f2py,
not hand literals. See ``docs/value_projection_ssa.md``.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH")

SRC = """
module vps_mod
contains
subroutine vps(n, sz, a, out)
  implicit none
  integer, intent(in) :: n
  integer, intent(inout) :: sz(n)
  real(8), intent(in) :: a(n)
  real(8), intent(inout) :: out(n)
  real(8), allocatable :: tmp(:)
  integer :: i, k
  do i = 1, n
    allocate(tmp(sz(i)))
    do k = 1, sz(i)
      tmp(k) = a(i) * real(k, 8)
    end do
    out(i) = tmp(sz(i))
    sz(i) = sz(i) + 100
    deallocate(tmp)
  end do
end subroutine vps
end module vps_mod
"""


def test_loop_variant_element_extent_mutated_source(tmp_path: Path):
    """Per-iteration allocatable sized by a mutated array element -- extent snapshots the pre-write value each iteration."""
    sdfg = build_sdfg(SRC, tmp_path / "sdfg", name="vps", entry="vps_mod::vps").build()
    assert sdfg.arrays["tmp"].transient

    n = 4
    sz = np.asfortranarray(np.array([2, 3, 1, 4], dtype=np.int32))
    a = np.asfortranarray(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))

    sz_s, out_s = sz.copy(), np.zeros(n, dtype=np.float64)
    sdfg(n=np.int32(n), sz=sz_s, a=a, out=out_s)  # aborts here on the heap-bug regression

    # out(i) = tmp(sz_orig(i)) = a(i)*sz_orig(i); sz(i) then += 100.
    np.testing.assert_array_equal(out_s, a * sz.astype(np.float64))
    np.testing.assert_array_equal(sz_s, sz + 100)

    # executable oracle: same source through f2py must agree bit-for-bit.
    sz_r, out_r = sz.copy(), np.zeros(n, dtype=np.float64)
    mod = f2py_compile(SRC, tmp_path / "ref", f"vps_ref_{tmp_path.name}")
    mod.vps_mod.vps(sz_r, a, out_r)
    np.testing.assert_array_equal(out_s, out_r)
    np.testing.assert_array_equal(sz_s, sz_r)
