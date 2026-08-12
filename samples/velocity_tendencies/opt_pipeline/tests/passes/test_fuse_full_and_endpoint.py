"""Tests for ``fuse_full_and_endpoint``.

Builds a minimal SDFG that reproduces the stage-3 pattern:
``state_A`` writes ``dst[row, lev]`` for ``lev in 0..N-1`` (a copy from
some source array) and ``state_B`` writes ``dst[row, N] = 0``. The two
states are connected by an unconditional interstate edge. After the
pass runs, expect a single state whose top-level Map iterates the union
range, has a NestedSDFG body, and the originals are gone.
"""

import dace
from dace import dtypes, memlet as mm, nodes
from dace.sdfg import SDFG

from utils.passes.fuse_full_and_endpoint import fuse_full_and_endpoint


def _build_pattern(N: int = 4, ROWS: int = 3) -> SDFG:
    sdfg = SDFG("fuse_pattern")
    sdfg.add_array("src", [ROWS, N], dace.float64, transient=False)
    sdfg.add_array("dst", [ROWS, N + 1], dace.float64, transient=False)

    # state A: full copy for lev = 0..N-1
    s_a = sdfg.add_state("state_a", is_start_block=True)
    rd = s_a.add_read("src")
    wr = s_a.add_write("dst")
    me, mx = s_a.add_map("copy", {"lev": f"0:{N}", "row": f"0:{ROWS}"})
    me.add_in_connector("IN_src")
    me.add_out_connector("OUT_src")
    mx.add_in_connector("IN_dst")
    mx.add_out_connector("OUT_dst")
    t = s_a.add_tasklet("cp", {"a"}, {"o"}, "o = a")
    s_a.add_edge(rd, None, me, "IN_src", mm.Memlet.from_array("src", sdfg.arrays["src"]))
    s_a.add_edge(me, "OUT_src", t, "a", mm.Memlet(data="src", subset="row, lev"))
    s_a.add_edge(t, "o", mx, "IN_dst", mm.Memlet(data="dst", subset="row, lev"))
    s_a.add_edge(mx, "OUT_dst", wr, None,
                 mm.Memlet(data="dst", subset=f"0:{ROWS}, 0:{N}"))

    # state B: zero at lev = N
    s_b = sdfg.add_state_after(s_a, "state_b")
    wr_b = s_b.add_write("dst")
    me_b, mx_b = s_b.add_map("zero", {"row": f"0:{ROWS}"})
    mx_b.add_in_connector("IN_dst")
    mx_b.add_out_connector("OUT_dst")
    tb = s_b.add_tasklet("zr", set(), {"o"}, "o = 0.0")
    # Scope-marker edge: MapEntry must reach every tasklet inside its
    # scope, even when the tasklet has no data inputs.
    s_b.add_edge(me_b, None, tb, None, mm.Memlet())
    s_b.add_edge(tb, "o", mx_b, "IN_dst", mm.Memlet(data="dst", subset=f"row, {N}"))
    s_b.add_edge(mx_b, "OUT_dst", wr_b, None,
                 mm.Memlet(data="dst", subset=f"0:{ROWS}, {N}"))

    return sdfg


def test_fuse_collapses_two_states_to_one():
    sdfg = _build_pattern(N=4, ROWS=3)
    n_states_before = sum(1 for _ in sdfg.all_states())
    n = fuse_full_and_endpoint(sdfg)
    assert n == 1, f"expected 1 fusion, got {n}"

    n_states_after = sum(1 for _ in sdfg.all_states())
    # Two states collapse into one; predecessor / successor wiring stays intact.
    assert n_states_after == n_states_before - 1, \
        f"expected one fewer state, got {n_states_after} (was {n_states_before})"


def test_fused_map_iterates_union_range():
    sdfg = _build_pattern(N=4, ROWS=3)
    fuse_full_and_endpoint(sdfg)

    # Find the (now unique) top-level map and verify its lev range is 0..N (inclusive end).
    found = False
    for state in sdfg.all_states():
        for n in state.nodes():
            if isinstance(n, nodes.MapEntry) and state.entry_node(n) is None:
                params = n.map.params
                assert "lev" in params, f"fused map missing lev axis; params={params}"
                lev_idx = params.index("lev")
                begin, end, step = n.map.range[lev_idx]
                assert str(begin) == "0", f"lev begin {begin} != 0"
                # DaCe end is inclusive: original 0:N (end=N-1=3) extended to end=N=4.
                assert str(end) == "4", f"lev end {end} != 4"
                found = True
    assert found, "no top-level Map left in the fused state"


def test_fused_body_contains_conditional_block():
    sdfg = _build_pattern(N=4, ROWS=3)
    fuse_full_and_endpoint(sdfg)

    # The fused map's body should contain a NestedSDFG whose internal
    # CFG has a ConditionalBlock.
    from dace.sdfg.state import ConditionalBlock
    saw_cb = False
    for state in sdfg.all_states():
        for n in state.nodes():
            if isinstance(n, nodes.NestedSDFG):
                for inner in n.sdfg.nodes():
                    if isinstance(inner, ConditionalBlock):
                        saw_cb = True
                        # Two branches: copy + endpoint
                        assert len(inner.branches) == 2, \
                            f"ConditionalBlock should have 2 branches, has {len(inner.branches)}"
    assert saw_cb, "no ConditionalBlock found in the fused body"


def test_no_fusion_when_endpoint_axis_misaligned():
    """If state B writes a level that's NOT one past state A's end,
    the pattern should NOT match."""
    sdfg = _build_pattern(N=4, ROWS=3)
    # Mutate B to write at lev=2 (not at N=4).
    s_b = next(s for s in sdfg.all_states() if s.label == "state_b")
    for e in list(s_b.edges()):
        if e.data is not None and e.data.data == "dst":
            new_subset = str(e.data.subset).replace(", 4", ", 2")
            e.data = mm.Memlet(data="dst", subset=new_subset)
    n = fuse_full_and_endpoint(sdfg)
    assert n == 0, f"expected 0 fusions on misaligned endpoint, got {n}"
