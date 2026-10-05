"""Arrays private to one loop are placed on the stack (``StorageType.Register(dynamic=True)``)."""

import numpy as np
import pytest

from dace import data, dtypes

from tests._util import build_sdfg

_SRC = """
subroutine row_scan(n, m, a, total)
  implicit none
  integer, intent(in) :: n, m
  real(8), intent(in) :: a(n, m)
  real(8), intent(out) :: total(m)
  real(8), allocatable :: row(:)
  integer :: j
  do j = 1, m
    allocate(row(n))
    row = 2.0d0 * a(:, j)
    total(j) = sum(row)
    deallocate(row)
  end do
end subroutine
"""


def test_a_loop_private_array_is_a_dynamic_register(tmp_path):
    """``row`` is allocated, used and freed inside the ``DO`` body: it becomes a stack VLA of runtime size ``n``,
    while the signature arrays keep their storage."""
    sdfg = build_sdfg(_SRC, tmp_path, name="row_scan", entry="row_scan").build()
    loop_private = [
        name
        for name, desc in sdfg.arrays.items()
        if isinstance(desc, data.Array) and desc.transient and dtypes.is_dynamic_register(desc.storage)
    ]
    assert loop_private, f"no loop-private array was placed on the stack: {sdfg.arrays}"
    assert not any(dtypes.StorageType.Register == sdfg.arrays[name].storage for name in ("a", "total"))

    n, m = 5, 3
    a = np.asfortranarray(np.random.default_rng(0).random((n, m)))
    total = np.zeros(m)
    sdfg(n=n, m=m, a=a, total=total)
    np.testing.assert_allclose(total, 2.0 * a.sum(axis=0))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
