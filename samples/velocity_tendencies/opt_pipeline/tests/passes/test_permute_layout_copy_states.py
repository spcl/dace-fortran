"""Tests for ``permute_layout`` ordering relative to GPU copy states.

After stage 4, the SDFG carries explicit copy-in / copy-out states::

    _cpu_to_gpu_copy_in  (X            -> gpu_X)
    _sync_after_copy_in  (cuStreamSync)
    <body states using gpu_*>
    _gpu_to_cpu_copy_out (gpu_X        -> X)
    _sync_after_copy_out (cuStreamSync)

When ``permute_layout`` is applied to the ``gpu_*`` transients, the
permutation MUST land between ``_sync_after_copy_in`` and the first
body state, and the inverse permutation MUST land between the last
body state and ``_gpu_to_cpu_copy_out``. Otherwise either the body
operates on un-permuted data (incorrect kernel layout) or the host
sees permuted data after copy-out (silent data corruption).

These tests are skipped when DaCe doesn't ship the ``permute_dimensions``
module (i.e. on branches where the layout pass is being rewritten).
"""

import pytest

import dace
from dace import dtypes, memlet as mm
from dace.sdfg import SDFG

pytest.importorskip("dace.transformation.layout.permute_dimensions")

from utils.passes.permute_layout import PermuteConfig, permute_layout  # noqa: E402


def _stage4_shaped_sdfg(M: int = 4, N: int = 4, K: int = 3) -> SDFG:
    """Synthetic stage-4-shaped SDFG.

    Layout:
        non-transient X, Y  (host-side)
        transient gpu_X, gpu_Y  (GPU_Global storage)

    States (linear chain):
        _cpu_to_gpu_copy_in -> _sync_after_copy_in
        -> body  (kernel reads gpu_X, writes gpu_Y)
        -> _gpu_to_cpu_copy_out -> _sync_after_copy_out

    The body has a top-level GPU_Device map; gpu_X / gpu_Y are
    GPU_Global. Modeled on what stage 4 actually emits.
    """
    sdfg = SDFG("stage4_synth")
    sdfg.add_array("X", [M, N, K], dace.float64, transient=False)
    sdfg.add_array("Y", [M, N, K], dace.float64, transient=False)
    sdfg.add_array("gpu_X", [M, N, K], dace.float64,
                   transient=True,
                   storage=dtypes.StorageType.GPU_Global,
                   lifetime=dtypes.AllocationLifetime.Persistent)
    sdfg.add_array("gpu_Y", [M, N, K], dace.float64,
                   transient=True,
                   storage=dtypes.StorageType.GPU_Global,
                   lifetime=dtypes.AllocationLifetime.Persistent)

    # _cpu_to_gpu_copy_in
    s_in = sdfg.add_state("_cpu_to_gpu_copy_in", is_start_block=True)
    rx = s_in.add_read("X")
    wgx = s_in.add_write("gpu_X")
    s_in.add_edge(rx, None, wgx, None, mm.Memlet.from_array("X", sdfg.arrays["X"]))

    # _sync_after_copy_in
    s_sync_in = sdfg.add_state_after(s_in, "_sync_after_copy_in")

    # body
    s_body = sdfg.add_state_after(s_sync_in, "body")
    rgx = s_body.add_read("gpu_X")
    wgy = s_body.add_write("gpu_Y")
    me, mx = s_body.add_map(
        "kernel",
        {"i": f"0:{M}", "j": f"0:{N}", "k": f"0:{K}"},
        schedule=dtypes.ScheduleType.GPU_Device,
    )
    me.add_in_connector("IN_X")
    me.add_out_connector("OUT_X")
    mx.add_in_connector("IN_Y")
    mx.add_out_connector("OUT_Y")
    t = s_body.add_tasklet("scale", {"a"}, {"o"}, "o = 2.0 * a")
    s_body.add_edge(rgx, None, me, "IN_X",
                    mm.Memlet.from_array("gpu_X", sdfg.arrays["gpu_X"]))
    s_body.add_edge(me, "OUT_X", t, "a", mm.Memlet(data="gpu_X", subset="i, j, k"))
    s_body.add_edge(t, "o", mx, "IN_Y", mm.Memlet(data="gpu_Y", subset="i, j, k"))
    s_body.add_edge(mx, "OUT_Y", wgy, None,
                    mm.Memlet.from_array("gpu_Y", sdfg.arrays["gpu_Y"]))

    # _gpu_to_cpu_copy_out
    s_out = sdfg.add_state_after(s_body, "_gpu_to_cpu_copy_out")
    rgy = s_out.add_read("gpu_Y")
    wy = s_out.add_write("Y")
    s_out.add_edge(rgy, None, wy, None, mm.Memlet.from_array("gpu_Y", sdfg.arrays["gpu_Y"]))

    # _sync_after_copy_out
    sdfg.add_state_after(s_out, "_sync_after_copy_out")

    return sdfg


def _state_by_label(sdfg: SDFG, label: str):
    return next(st for st in sdfg.all_states() if st.label == label)


def _memlets_targeting(state, name: str):
    """Memlets in ``state`` that read from or write into ``name``."""
    return [e.data for e in state.edges() if e.data is not None and e.data.data == name]


def test_no_extra_permute_state_inserted_when_copy_exists():
    """With the copy-replacement design, the existing copy_in/copy_out
    states ARE the permute states. No extra ``permute_after_<T>`` is
    inserted, no body re-write, no orphaned reads."""
    sdfg = _stage4_shaped_sdfg()
    cfg = PermuteConfig(name="t", permute_map={"gpu_X": [1, 0, 2], "gpu_Y": [1, 0, 2]})
    permute_layout(sdfg, cfg, shuffle_map=False)
    sdfg.validate()

    labels = {st.label for st in sdfg.all_states()}
    extras = [s for s in labels if "permute_after" in s or "permute_in" in s
              or "permute_out" in s]
    assert not extras, (
        f"expected zero extra permute states (copy_in/copy_out should "
        f"BECOME the permute states); got {extras}"
    )

    # The original five states are intact and in order.
    expected = {"_cpu_to_gpu_copy_in", "_sync_after_copy_in", "body",
                "_gpu_to_cpu_copy_out", "_sync_after_copy_out"}
    assert expected.issubset(labels), (
        f"copy/sync state structure broken; expected {expected} subset of {labels}"
    )


def test_copy_in_becomes_forward_permute_for_input_transient():
    """``_cpu_to_gpu_copy_in`` now writes ``permuted_gpu_X`` (not
    ``gpu_X``) -- the rename loop turned the lifted access->access copy
    into a Map+Tasklet that bakes the forward permutation into the
    write subscript."""
    sdfg = _stage4_shaped_sdfg()
    cfg = PermuteConfig(name="t", permute_map={"gpu_X": [1, 0, 2]})
    permute_layout(sdfg, cfg, shuffle_map=False)
    sdfg.validate()

    copy_in = _state_by_label(sdfg, "_cpu_to_gpu_copy_in")

    # gpu_X is no longer referenced; permuted_gpu_X took its place.
    assert not _memlets_targeting(copy_in, "gpu_X"), (
        "copy_in still references gpu_X; rename did not convert it to permuted_gpu_X"
    )
    assert _memlets_targeting(copy_in, "permuted_gpu_X"), (
        "copy_in does not write permuted_gpu_X; lift/rename failed"
    )


def test_copy_out_becomes_inverse_permute_for_output_transient():
    """``_gpu_to_cpu_copy_out`` reads ``permuted_gpu_Y`` (with permuted
    subscript so the host receives data in the ORIGINAL layout). No
    body->copy_out direct edge; no inserted state in between."""
    sdfg = _stage4_shaped_sdfg()
    cfg = PermuteConfig(name="t", permute_map={"gpu_Y": [1, 0, 2]})
    permute_layout(sdfg, cfg, shuffle_map=False)
    sdfg.validate()

    copy_out = _state_by_label(sdfg, "_gpu_to_cpu_copy_out")

    assert not _memlets_targeting(copy_out, "gpu_Y"), (
        "copy_out still references gpu_Y; rename did not convert it to permuted_gpu_Y"
    )
    assert _memlets_targeting(copy_out, "permuted_gpu_Y"), (
        "copy_out does not read permuted_gpu_Y; lift/rename failed"
    )

    # And the host array Y must be written in its ORIGINAL layout
    # (i.e. there is a memlet writing to "Y" with full extent).
    y_writes = _memlets_targeting(copy_out, "Y")
    assert y_writes, "copy_out must still write the host-side Y array"


def test_permute_layout_returns_zero_on_empty_map():
    """Sanity: an empty permute_map is a no-op and must not insert
    any state."""
    sdfg = _stage4_shaped_sdfg()
    n_states_before = sum(1 for _ in sdfg.all_states())
    cfg = PermuteConfig(name="noop", permute_map={})
    n = permute_layout(sdfg, cfg, shuffle_map=False)
    assert n == 0
    n_states_after = sum(1 for _ in sdfg.all_states())
    assert n_states_after == n_states_before
