"""Smoke tests for the HLFIR -> SDFG frontend: inline Fortran source, build an SDFG,
validate against a numpy reference (E2E-numerical rule) plus structural assertions
against silent SDFG-shape regressions."""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not available")


def test_elementwise_loop(tmp_path):
    """One tasklet, one state  --  c(i) = a(i) + b(i)."""
    src = """
subroutine elementwise_add(a, b, c, n)
  implicit none
  integer, intent(in)    :: n
  real(8), intent(in)    :: a(n), b(n)
  real(8), intent(inout) :: c(n)
  integer :: i
  do i = 1, n
    c(i) = a(i) + b(i)
  end do
end subroutine elementwise_add
"""
    b = build_sdfg(src, tmp_path, name="elementwise_add")
    sdfg = b.build()
    sdfg.validate()
    for nm in ("a", "b", "c"):
        assert nm in sdfg.arrays

    rng = np.random.default_rng(0)
    n = 16
    a = np.ascontiguousarray(rng.standard_normal(n, dtype=np.float64))
    b_arr = np.ascontiguousarray(rng.standard_normal(n, dtype=np.float64))
    c = np.zeros(n, dtype=np.float64)
    expected = a + b_arr
    sdfg(a=a, b=b_arr, c=c, n=n)
    np.testing.assert_allclose(c, expected, rtol=1e-12, atol=1e-12)


def test_read_after_write_shares_access_node(tmp_path):
    """RAW within a loop body: exactly one AccessNode for ``tmp`` in the innermost
    state (single-access-node rule); numerical check catches a dropped RAW edge."""
    from dace.sdfg.state import LoopRegion
    from dace.sdfg import nodes as nd

    src = """
subroutine chained(a, out, n)
  implicit none
  integer, intent(in)    :: n
  real(8), intent(in)    :: a(n)
  real(8), intent(inout) :: out(n)
  real(8) :: tmp(n)
  integer :: i
  do i = 1, n
    tmp(i) = a(i) * 2.0d0
    out(i) = tmp(i) + 1.0d0
  end do
end subroutine chained
"""
    b = build_sdfg(src, tmp_path, name="chained")
    sdfg = b.build()
    sdfg.validate()

    def iter_states(region):
        for n in region.nodes():
            if isinstance(n, LoopRegion):
                yield from iter_states(n)
            elif hasattr(n, "nodes"):
                yield n

    body = next(s for s in iter_states(sdfg) if any(isinstance(n, nd.Tasklet) for n in s.nodes()))
    tmp_nodes = [n for n in body.nodes() if isinstance(n, nd.AccessNode) and n.data == "tmp"]
    assert len(tmp_nodes) == 1, (f"expected a single shared access node for tmp in the body state; "
                                 f"got {len(tmp_nodes)}")

    rng = np.random.default_rng(1)
    n = 8
    a = np.ascontiguousarray(rng.standard_normal(n, dtype=np.float64))
    out = np.zeros(n, dtype=np.float64)
    expected = a * 2.0 + 1.0
    sdfg(a=a, out=out, n=n)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)
