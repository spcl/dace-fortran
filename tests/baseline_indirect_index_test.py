"""Baseline HLFIR coverage: indirect/nested array indexing, out(i) = src(idx(i))."""

from pathlib import Path

import numpy as np
import pytest

from tests._util import build_sdfg, f2py_compile


def _build(src: str, tmp: Path, name: str):
    tmp.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, tmp, name=name, pipeline="hlfir-propagate-shapes").build()


def test_nested_array_indirect(tmp_path):
    """An index array feeds another array read  --  classic indirect access."""
    src = """
subroutine nested_idx(out, idx, src, n)
  implicit none
  integer, intent(in)    :: n
  integer, intent(in)    :: idx(n)
  real(8), intent(in)    :: src(n)
  real(8), intent(inout) :: out(n)
  integer :: i
  do i = 1, n
    out(i) = src(idx(i))
  end do
end subroutine nested_idx
"""
    mod = f2py_compile(src, tmp_path / "ref", "nested_idx")
    sdfg = _build(src, tmp_path / "sdfg", name="nested_idx")

    rng = np.random.default_rng(3)
    n = 7
    src_data = rng.standard_normal(n)
    idx = rng.integers(1, n + 1, size=n, dtype=np.int32)

    out_ref = np.zeros(n, order="F")
    mod.nested_idx(out_ref, np.asfortranarray(idx), np.asfortranarray(src_data))

    out_sdfg = np.zeros(n, dtype=np.float64)
    sdfg(idx=np.ascontiguousarray(idx), src=np.ascontiguousarray(src_data), out=out_sdfg, n=n)
    np.testing.assert_allclose(out_sdfg, out_ref, rtol=1e-12, atol=1e-12)


def test_indirect_read_in_a_condition(tmp_path):
    """``IF (ngc(idx(mc)) > 0)`` (QE ``sticks_dist_new``): the staged condition mints a symbol for the inner index
    read instead of nesting ``idx[mc]`` inside the ``ngc`` memlet subset."""
    src = """
subroutine count_positive(n, idx, ngc, total)
  implicit none
  integer, intent(in) :: n
  integer, intent(in) :: idx(n), ngc(n)
  integer, intent(out) :: total
  integer :: mc
  total = 0
  do mc = 1, n
    if (ngc(idx(mc)) > 0) then
      total = total + mc
    end if
  end do
end subroutine count_positive
"""
    sdfg = build_sdfg(src, tmp_path, name="count_positive", entry="count_positive").build()
    idx = np.array([4, 3, 2, 1], dtype=np.int32)
    ngc = np.array([1, 0, 0, 5], dtype=np.int32)
    total = np.zeros(1, dtype=np.int32)
    sdfg(n=4, idx=idx, ngc=ngc, total=total)
    assert total[0] == 1 + 4  # mc = 1 reads ngc(4) = 5, mc = 4 reads ngc(1) = 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
