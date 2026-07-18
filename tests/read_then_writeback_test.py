"""Phase I -- read-then-writeback to the same scalar/length-1 array must
produce two distinct access nodes in one state, never a cycle.

velocity_tendencies pattern that surfaced this: after
``hlfir-lift-reduction-operands`` (e3cfbcc68) lifts a reduction temp, both
``max_vcfl_dyn = MAX(...)`` and the writeback land in the same state; the
bridge's ``emit_scalar_assign`` was reusing the cached read node for the
write, giving ONE access node both incoming and outgoing edges -- invalid
SDFG topology.

Reproduces at minimal scale (no MAXVAL, no struct, no inlined callee) so a
regression surfaces independently of Phases F/A/B/G.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_read_then_writeback_no_cycle(tmp_path: Path):
    """Every state with both a read and write of ``out`` must have TWO distinct access nodes for ``out``."""
    src = """
subroutine kernel(out, x, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: x(n)
  real(8), intent(inout) :: out
  integer :: i
  do i = 1, n
    out = max(out, x(i))
  end do
end subroutine kernel
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='kernel').build()

    # For every state touching 'out': if it has both an in-edge and an out-edge on 'out', it must have >= 2 access nodes.
    from dace.sdfg import nodes as nd
    bad = []
    for state in sdfg.all_states():
        out_nodes = [n for n in state.nodes() if isinstance(n, nd.AccessNode) and n.data == "out"]
        if not out_nodes:
            continue
        has_in = any(state.in_degree(n) > 0 for n in out_nodes)
        has_out = any(state.out_degree(n) > 0 for n in out_nodes)
        if has_in and has_out and len(out_nodes) < 2:
            bad.append((state.label, len(out_nodes)))
    assert not bad, f"states with read+write on a single 'out' node: {bad}"

    # End-to-end numerical correctness: kernel must compute max(initial, x[0..n-1]).
    rng = np.random.default_rng(0)
    n = 32
    x = np.asfortranarray(rng.standard_normal(n))
    out = np.array([0.5], dtype=np.float64)
    sdfg(out=out, x=x, n=n)
    assert out[0] == max(0.5, x.max())


def test_read_then_writeback_two_assigns_same_state(tmp_path: Path):
    """Exact velocity_tendencies shape: two statements, second writes the target that was the first's RHS read."""
    src = """
subroutine kernel(state, x, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: x(n)
  real(8), intent(inout) :: state
  real(8) :: tmp
  tmp = max(state, maxval(x(1:n)))
  state = tmp
end subroutine kernel
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='kernel').build()

    from dace.sdfg import nodes as nd
    bad = []
    for s in sdfg.all_states():
        st_nodes = [n for n in s.nodes() if isinstance(n, nd.AccessNode) and n.data == "state"]
        if not st_nodes:
            continue
        has_in = any(s.in_degree(n) > 0 for n in st_nodes)
        has_out = any(s.out_degree(n) > 0 for n in st_nodes)
        if has_in and has_out and len(st_nodes) < 2:
            bad.append((s.label, len(st_nodes)))
    assert not bad, f"read-then-writeback on single 'state' node: {bad}"

    rng = np.random.default_rng(1)
    n = 16
    x = np.asfortranarray(rng.standard_normal(n))
    state = np.array([0.25], dtype=np.float64)
    sdfg(state=state, x=x, n=n)
    assert state[0] == max(0.25, x.max())
