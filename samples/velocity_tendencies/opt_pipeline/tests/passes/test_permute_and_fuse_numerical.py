"""End-to-end numerical correctness test for permute_layout + fuse_full_and_endpoint.

Builds a small synthetic SDFG that mimics the stage-4 / stage-5 contract:

  * non-transient host inputs/outputs (`src`, `dst`)
  * a transient `tmp` populated by an access->access copy from `src` in
    a `_cpu_to_gpu_copy_in`-style state
  * a body state that writes `tmp[row, lev] = f(src, lev)` for `lev in 0..N-1`
  * an "endpoint" state that writes `tmp[row, N] = 0`
  * a `_gpu_to_cpu_copy_out`-style state that copies `tmp` -> `dst`

Two SDFGs are produced from this template:

  * the **baseline** -- compiled and run as-is
  * the **transformed** -- ``permute_layout(tmp: [1, 0])`` followed by
    ``fuse_full_and_endpoint`` then compiled and run

Both are invoked with the same random ``src`` and the resulting ``dst``
arrays are compared bit-for-bit (the transform only re-orders memory;
no FMA reorder, no precision loss). If they match, both passes
preserve semantics across a real host -> transient -> body -> transient
-> host data flow.

This test is skipped when DaCe doesn't ship ``permute_dimensions``
(branches that have the layout pass under refactor).
"""

import copy
import ctypes
import numpy as np
import pytest

# DaCe's ``libdacestub_*.so`` is built without ``-lgomp`` even though
# the emitted host code references ``omp_get_max_threads`` from the
# OpenMP runtime. Pre-load libgomp into the process global namespace
# so ``dlopen`` of the stub resolves the symbol.
try:
    ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass  # not present -- the test will fail with a clearer message

import dace
from dace import dtypes, memlet as mm
from dace.sdfg import SDFG

pytest.importorskip("dace.transformation.layout.permute_dimensions")

from utils.passes.permute_layout import PermuteConfig, permute_layout
from utils.passes.fuse_full_and_endpoint import fuse_full_and_endpoint


def _build_baseline_sdfg(N: int = 5, ROWS: int = 4, name: str = "perm_fuse_baseline") -> SDFG:
    """Synthetic SDFG mirroring stage-4 shape, but on regular CPU
    storage so the test runs without a GPU. The lift in
    ``permute_layout`` triggers on access->access edges to transients
    regardless of storage."""
    sdfg = SDFG(name)
    sdfg.add_array("src", [ROWS, N], dace.float64, transient=False)
    sdfg.add_array("dst", [ROWS, N + 1], dace.float64, transient=False)
    sdfg.add_array("tmp", [ROWS, N + 1], dace.float64, transient=True)

    # State: access->access copy from src (last-level slot stays uninitialised).
    s_in = sdfg.add_state("_cpu_to_gpu_copy_in", is_start_block=True)
    s_in.add_edge(s_in.add_read("src"), None,
                  s_in.add_write("tmp"), None,
                  mm.Memlet(data="src",
                            subset=f"0:{ROWS}, 0:{N}",
                            other_subset=f"0:{ROWS}, 0:{N}"))

    # State: body -- per-row, per-level non-trivial work that depends on src
    # (the `tmp` value just copied in) so a wrong layout would corrupt the
    # output deterministically.
    s_body = sdfg.add_state_after(s_in, "body")
    rd = s_body.add_read("tmp")
    wr = s_body.add_write("tmp")
    me, mx = s_body.add_map("body_map", {"lev": f"0:{N}", "row": f"0:{ROWS}"})
    me.add_in_connector("IN_tmp")
    me.add_out_connector("OUT_tmp")
    mx.add_in_connector("IN_tmp")
    mx.add_out_connector("OUT_tmp")
    t = s_body.add_tasklet("scale", {"a"}, {"o"}, "o = 2.0 * a + 0.5")
    s_body.add_edge(rd, None, me, "IN_tmp",
                    mm.Memlet(data="tmp", subset=f"0:{ROWS}, 0:{N}"))
    s_body.add_edge(me, "OUT_tmp", t, "a", mm.Memlet(data="tmp", subset="row, lev"))
    s_body.add_edge(t, "o", mx, "IN_tmp", mm.Memlet(data="tmp", subset="row, lev"))
    s_body.add_edge(mx, "OUT_tmp", wr, None,
                    mm.Memlet(data="tmp", subset=f"0:{ROWS}, 0:{N}"))

    # State: endpoint zero-write at lev = N.
    s_end = sdfg.add_state_after(s_body, "endpoint_zero")
    wr_e = s_end.add_write("tmp")
    me_e, mx_e = s_end.add_map("zero_map", {"row": f"0:{ROWS}"})
    mx_e.add_in_connector("IN_tmp")
    mx_e.add_out_connector("OUT_tmp")
    tz = s_end.add_tasklet("zero", set(), {"o"}, "o = 0.0")
    s_end.add_edge(me_e, None, tz, None, mm.Memlet())
    s_end.add_edge(tz, "o", mx_e, "IN_tmp", mm.Memlet(data="tmp", subset=f"row, {N}"))
    s_end.add_edge(mx_e, "OUT_tmp", wr_e, None,
                   mm.Memlet(data="tmp", subset=f"0:{ROWS}, {N}"))

    # State: access->access copy from tmp (full N+1 levels) to dst.
    s_out = sdfg.add_state_after(s_end, "_gpu_to_cpu_copy_out")
    s_out.add_edge(s_out.add_read("tmp"), None,
                   s_out.add_write("dst"), None,
                   mm.Memlet(data="tmp",
                             subset=f"0:{ROWS}, 0:{N + 1}",
                             other_subset=f"0:{ROWS}, 0:{N + 1}"))

    sdfg.validate()
    return sdfg


def _expected_output(src_np: np.ndarray, ROWS: int, N: int) -> np.ndarray:
    """Reference computation: dst[row, lev] = 2.0 * src[row, lev] + 0.5
    for lev in 0..N-1, dst[row, N] = 0."""
    out = np.zeros((ROWS, N + 1), dtype=np.float64)
    out[:, :N] = 2.0 * src_np + 0.5
    out[:, N] = 0.0
    return out


def _compile_and_call(sdfg: SDFG, src_np: np.ndarray, ROWS: int, N: int) -> np.ndarray:
    dst_np = np.zeros((ROWS, N + 1), dtype=np.float64)
    program = sdfg.compile()
    program(src=src_np, dst=dst_np)
    return dst_np


def test_baseline_runs_and_matches_reference():
    """Sanity: the baseline SDFG (no transforms) computes the expected
    function. Establishes the reference oracle."""
    N, ROWS = 5, 4
    sdfg = _build_baseline_sdfg(N=N, ROWS=ROWS, name="numerical_baseline")
    np.random.seed(0)
    src_np = np.random.rand(ROWS, N).astype(np.float64)
    got = _compile_and_call(sdfg, src_np.copy(), ROWS, N)
    want = _expected_output(src_np, ROWS, N)
    np.testing.assert_array_equal(got, want)


def test_permute_only_preserves_numerics():
    """Apply just permute_layout (transpose tmp). Output must be
    identical to the unpermuted baseline -- the transform only
    re-orders memory."""
    N, ROWS = 5, 4
    np.random.seed(1)
    src_np = np.random.rand(ROWS, N).astype(np.float64)

    base_sdfg = _build_baseline_sdfg(N=N, ROWS=ROWS, name="numerical_perm_base")
    base_out = _compile_and_call(base_sdfg, src_np.copy(), ROWS, N)

    perm_sdfg = _build_baseline_sdfg(N=N, ROWS=ROWS, name="numerical_perm_only")
    cfg = PermuteConfig(name="t", permute_map={"tmp": [1, 0]})
    permute_layout(perm_sdfg, cfg, shuffle_map=False)
    perm_out = _compile_and_call(perm_sdfg, src_np.copy(), ROWS, N)

    np.testing.assert_array_equal(perm_out, base_out)


def test_fuse_only_preserves_numerics():
    """Apply just fuse_full_and_endpoint (no permutation). Output must
    be identical to the unfused baseline -- the transform only changes
    state structure / dispatch."""
    N, ROWS = 5, 4
    np.random.seed(2)
    src_np = np.random.rand(ROWS, N).astype(np.float64)

    base_sdfg = _build_baseline_sdfg(N=N, ROWS=ROWS, name="numerical_fuse_base")
    base_out = _compile_and_call(base_sdfg, src_np.copy(), ROWS, N)

    fuse_sdfg = _build_baseline_sdfg(N=N, ROWS=ROWS, name="numerical_fuse_only")
    n_fused = fuse_full_and_endpoint(fuse_sdfg)
    assert n_fused == 1, f"expected 1 fusion, got {n_fused}"
    fuse_out = _compile_and_call(fuse_sdfg, src_np.copy(), ROWS, N)

    np.testing.assert_array_equal(fuse_out, base_out)


def test_permute_then_fuse_preserves_numerics():
    """Apply both transforms in pipeline order (permute first, then
    fuse). Output must be identical to the baseline."""
    N, ROWS = 5, 4
    np.random.seed(3)
    src_np = np.random.rand(ROWS, N).astype(np.float64)

    base_sdfg = _build_baseline_sdfg(N=N, ROWS=ROWS, name="numerical_combo_base")
    base_out = _compile_and_call(base_sdfg, src_np.copy(), ROWS, N)

    combo_sdfg = _build_baseline_sdfg(N=N, ROWS=ROWS, name="numerical_combo")
    cfg = PermuteConfig(name="t", permute_map={"tmp": [1, 0]})
    permute_layout(combo_sdfg, cfg, shuffle_map=False)
    n_fused = fuse_full_and_endpoint(combo_sdfg)
    assert n_fused == 1, f"fuse should still fire after permute; got {n_fused}"
    combo_out = _compile_and_call(combo_sdfg, src_np.copy(), ROWS, N)

    np.testing.assert_array_equal(combo_out, base_out)
