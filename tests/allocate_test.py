"""Verbatim port of f2dace/dev:tests/fortran/allocate_test.py."""

import numpy as np

from tests._util import build_sdfg


def test_fortran_frontend_basic_allocate(tmp_path):
    src = """
subroutine main(d)
  double precision, allocatable, intent(out) :: d(:, :)
  allocate (d(4, 5))
  d(2, 1) = 5.5
end subroutine main"""
    sdfg = build_sdfg(src, tmp_path, name="main").build()
    a = np.full([4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert a[0, 0] == 42
    assert a[1, 0] == 5.5
    assert a[2, 0] == 42


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
