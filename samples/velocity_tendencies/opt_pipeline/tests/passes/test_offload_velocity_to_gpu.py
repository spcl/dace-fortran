"""Tests for ``OffloadVelocityToGPU``.

Structural tests (purely in-process, no CUDA needed). Each test builds
a small SDFG that models one piece of the stage-3 contract, runs
``OffloadVelocityToGPU``, and asserts the expected post-conditions.
The pass validates its own output, so every test here is also a
validation test.

Numerical tests aren't included -- stage 4's output requires a GPU for
end-to-end validation, and the velocity pipeline uses ``got_and_want``
snapshots for that higher-level check.
"""
import dace
from dace import dtypes, nodes
from dace import memlet as mm
from dace.sdfg import SDFG

from utils.passes.offload_velocity_to_gpu import OffloadVelocityToGPU


def _inner_sdfg(M=4, N=4) -> SDFG:
    """Inner NSDFG: reads A, writes B[i, j] = 2 * A[i, j]."""
    inner = SDFG("inner")
    for name in ("A", "B"):
        inner.add_array(name, [M, N], dace.float64, transient=False)
    st = inner.add_state()
    ar, bw = st.add_read("A"), st.add_write("B")
    me, mx = st.add_map("k", {"i": f"0:{M}", "j": f"0:{N}"})
    t = st.add_tasklet("mul2", {"a"}, {"o"}, "o = 2.0 * a")
    me.add_in_connector("IN_A")
    me.add_out_connector("OUT_A")
    mx.add_in_connector("IN_B")
    mx.add_out_connector("OUT_B")
    st.add_edge(ar, None, me, "IN_A", mm.Memlet.from_array("A", inner.arrays["A"]))
    st.add_edge(me, "OUT_A", t, "a", mm.Memlet(data="A", subset="i, j"))
    st.add_edge(t, "o", mx, "IN_B", mm.Memlet(data="B", subset="i, j"))
    st.add_edge(mx, "OUT_B", bw, None, mm.Memlet.from_array("B", inner.arrays["B"]))
    return inner


def _stage3_shaped_sdfg(M=4, N=4, K=3) -> SDFG:
    """Minimal stage-3-shaped top SDFG: one top-level MapEntry running
    the per-block work in a NestedSDFG. Non-transient A, B at the top;
    the map range carries no block-descriptor provenance, so the map
    itself is a kernel."""
    top = SDFG("top_stage3")
    top.add_array("A", [M, N, K], dace.float64, transient=False)
    top.add_array("B", [M, N, K], dace.float64, transient=False)
    st = top.add_state()
    me, mx = st.add_map("blocks", {"jb": f"0:{K}"})
    me.add_in_connector("IN_A")
    me.add_out_connector("OUT_A")
    mx.add_in_connector("IN_B")
    mx.add_out_connector("OUT_B")
    n = st.add_nested_sdfg(_inner_sdfg(M=M, N=N), {"A"}, {"B"})
    ar = st.add_read("A")
    bw = st.add_write("B")
    st.add_edge(ar, None, me, "IN_A", mm.Memlet.from_array("A", top.arrays["A"]))
    st.add_edge(me, "OUT_A", n, "A", mm.Memlet(data="A", subset=f"0:{M}, 0:{N}, jb"))
    st.add_edge(n, "B", mx, "IN_B", mm.Memlet(data="B", subset=f"0:{M}, 0:{N}, jb"))
    st.add_edge(mx, "OUT_B", bw, None, mm.Memlet.from_array("B", top.arrays["B"]))
    return top


def _block_loop_sdfg(M=4, N=4, K=3) -> SDFG:
    """Stage-3 shape with a real block loop: ``i_startblk`` /
    ``i_endblk`` come off a ``*_start_block`` / ``*_end_block``
    descriptor array on an interstate edge, which is what makes the
    map a block map."""
    top = SDFG("with_block_map")
    top.add_array("A", [M, N, K], dace.float64, transient=False)
    top.add_array("B", [M, N, K], dace.float64, transient=False)
    top.add_array("p_patch_edges_start_block", [4], dace.int32, transient=False)
    top.add_array("p_patch_edges_end_block", [4], dace.int32, transient=False)

    head = top.add_state("head", is_start_block=True)
    body = top.add_state("body")
    top.add_edge(
        head, body,
        dace.InterstateEdge(assignments={
            "i_startblk": "p_patch_edges_start_block[0]",
            "i_endblk": "p_patch_edges_end_block[0]",
        }))

    me, mx = body.add_map("blocks", {"jb": "i_startblk : i_endblk"})
    me.add_in_connector("IN_A")
    me.add_out_connector("OUT_A")
    mx.add_in_connector("IN_B")
    mx.add_out_connector("OUT_B")
    n = body.add_nested_sdfg(_inner_sdfg(M=M, N=N), {"A"}, {"B"})
    ar, bw = body.add_read("A"), body.add_write("B")
    body.add_edge(ar, None, me, "IN_A", mm.Memlet.from_array("A", top.arrays["A"]))
    body.add_edge(me, "OUT_A", n, "A", mm.Memlet(data="A", subset=f"0:{M}, 0:{N}, jb"))
    body.add_edge(n, "B", mx, "IN_B", mm.Memlet(data="B", subset=f"0:{M}, 0:{N}, jb"))
    body.add_edge(mx, "OUT_B", bw, None, mm.Memlet.from_array("B", top.arrays["B"]))
    return top, me


def _inner_maps(sdfg: SDFG):
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.NestedSDFG):
            for st in n.sdfg.all_states():
                for m in st.nodes():
                    if isinstance(m, nodes.MapEntry):
                        yield m


def test_top_level_map_becomes_kernel_inner_maps_sequential():
    top = _stage3_shaped_sdfg()
    OffloadVelocityToGPU(top)
    top_maps = [
        n for st in top.all_states() for n in st.nodes() if isinstance(n, nodes.MapEntry) and st.entry_node(n) is None
    ]
    assert len(top_maps) == 1
    assert top_maps[0].map.schedule == dtypes.ScheduleType.GPU_Device
    # Maps below a kernel are sequentialized, not nested kernels.
    inner = list(_inner_maps(top))
    assert inner
    for m in inner:
        assert m.map.schedule == dtypes.ScheduleType.Sequential


def test_block_map_stays_host_and_launches_inner_kernel():
    """A block map -- range depends on a symbol an interstate edge
    assigns from a ``*_start_block`` descriptor -- stays on the host
    and the map inside its body becomes the kernel."""
    top, block_entry = _block_loop_sdfg()
    OffloadVelocityToGPU(top)

    assert block_entry.map.schedule == dtypes.ScheduleType.Sequential
    inner = list(_inner_maps(top))
    assert inner
    assert all(m.map.schedule == dtypes.ScheduleType.GPU_Device for m in inner)


def test_block_lookalike_name_is_not_a_block_map():
    """Detection is by descriptor provenance, not by the substring
    ``startblk`` in the range: a symbol that merely looks like a block
    bound must not pin its map to the host."""
    top = _stage3_shaped_sdfg()
    top.add_symbol("i_startblk_scale", dace.int32)
    entry = next(n for st in top.all_states() for n in st.nodes() if isinstance(n, nodes.MapEntry))
    entry.map.range = dace.subsets.Range([(0, dace.symbol("i_startblk_scale"), 1)])

    OffloadVelocityToGPU(top)
    assert entry.map.schedule == dtypes.ScheduleType.GPU_Device


def test_nontransient_arrays_get_gpu_clone_and_copy_states():
    top = _stage3_shaped_sdfg()
    OffloadVelocityToGPU(top)

    assert "gpu_A" in top.arrays
    assert "gpu_B" in top.arrays
    assert top.arrays["gpu_A"].storage == dtypes.StorageType.GPU_Global
    assert top.arrays["gpu_A"].transient is True
    assert top.arrays["gpu_B"].transient is True

    # Originals stay in the signature, CPU-side.
    assert top.arrays["A"].transient is False
    assert top.arrays["B"].transient is False

    labels = {s.label for s in top.states()}
    assert "_cpu_to_gpu_copy_in" in labels
    assert "_gpu_to_cpu_copy_out" in labels
    assert "_sync_after_copy_in" in labels
    assert "_sync_after_copy_out" in labels


def test_kernel_touched_transient_moves_to_gpu():
    """A transient the kernel writes is device data."""
    M, N, K = 4, 4, 3
    top = _stage3_shaped_sdfg(M, N, K)
    top.add_array("t", [M, N, K], dace.float64, transient=True, lifetime=dtypes.AllocationLifetime.SDFG)
    st = next(iter(top.states()))
    mx = next(n for n in st.nodes() if isinstance(n, nodes.MapExit))
    tw = st.add_write("t")
    mx.add_out_connector("OUT_t")
    st.add_edge(mx, "OUT_t", tw, None, mm.Memlet(data="t", subset=f"0:{M}, 0:{N}, 0:{K}"))
    n = next(x for x in st.nodes() if isinstance(x, nodes.NestedSDFG))
    n.sdfg.add_array("C", [M, N], dace.float64, transient=False)
    inner_st = next(iter(n.sdfg.states()))
    inner_mx = next(x for x in inner_st.nodes() if isinstance(x, nodes.MapExit))
    inner_t = next(x for x in inner_st.nodes() if isinstance(x, nodes.Tasklet))
    inner_t.add_out_connector("o2")
    inner_mx.add_in_connector("IN_C")
    inner_mx.add_out_connector("OUT_C")
    inner_st.add_edge(inner_t, "o2", inner_mx, "IN_C", mm.Memlet(data="C", subset="i, j"))
    inner_st.add_edge(inner_mx, "OUT_C", inner_st.add_write("C"), None, mm.Memlet.from_array("C", n.sdfg.arrays["C"]))
    inner_t.code = dace.properties.CodeBlock("o = 2.0 * a\no2 = 3.0 * a")
    n.add_out_connector("C")
    st.add_edge(n, "C", mx, "IN_t", mm.Memlet(data="t", subset=f"0:{M}, 0:{N}, jb"))
    mx.add_in_connector("IN_t")

    OffloadVelocityToGPU(top)
    assert top.arrays["t"].storage == dtypes.StorageType.GPU_Global


def test_interstate_read_array_stays_on_cpu():
    """An array only a host interstate edge reads -- the patch index
    descriptor case -- must stay CPU-side. Moving it to device storage
    is what produced the "accessed on host" violations."""
    top, _ = _block_loop_sdfg()
    OffloadVelocityToGPU(top)

    for name in ("p_patch_edges_start_block", "p_patch_edges_end_block"):
        assert "gpu_" + name not in top.arrays
        assert top.arrays[name].storage != dtypes.StorageType.GPU_Global


def test_scalar_only_host_tasklet_stays_on_host():
    """A free tasklet wired to Scalars only is left on the host, so its
    scalars stay CPU-side too."""
    from dace.sdfg.scope import is_devicelevel_gpu

    top = _stage3_shaped_sdfg()
    top.add_scalar("host_scratch", dace.float64, transient=True)
    st = top.add_state_after(next(iter(top.states())), "host_state")
    ac = st.add_access("host_scratch")
    t = st.add_tasklet("w", {}, {"o"}, "o = 3.14")
    st.add_edge(t, "o", ac, None, mm.Memlet(data="host_scratch", subset="0"))

    OffloadVelocityToGPU(top)
    assert top.arrays["host_scratch"].storage != dtypes.StorageType.GPU_Global
    assert not is_devicelevel_gpu(top, st, t)


def test_array_touched_by_free_tasklet_follows_it_to_the_device():
    """A free tasklet wired to an Array is moved into a 1-iteration GPU
    map, so the array it writes has to be device-resident -- leaving it
    on the host is an ``Illegal copy`` at codegen."""
    top = SDFG("host_only")
    top.add_array("arr", [1], dace.float64, transient=False, storage=dtypes.StorageType.CPU_Heap)
    st = top.add_state()
    ac = st.add_access("arr")
    t = st.add_tasklet("w", {}, {"o"}, "o = 3.14")
    st.add_edge(t, "o", ac, None, mm.Memlet(data="arr", subset="0"))

    OffloadVelocityToGPU(top)
    assert "gpu_arr" in top.arrays
    assert top.arrays["gpu_arr"].storage == dtypes.StorageType.GPU_Global
    # The caller-visible descriptor is untouched and gets a copy-out.
    assert top.arrays["arr"].storage == dtypes.StorageType.CPU_Heap
    assert top.arrays["arr"].transient is False


def test_kernel_side_accessnode_retargeted_hostside_not():
    """The same SDFG can hold a host tasklet and a kernel: the kernel's
    array is cloned and its AccessNodes point at the clone, the
    host-only array keeps its name."""
    top = SDFG("mixed")
    top.add_array("kern_arr", [4, 4], dace.float64, transient=False)
    top.add_scalar("host_arr", dace.float64, transient=False)

    host_st = top.add_state("host_state", is_start_block=True)
    host_ac = host_st.add_access("host_arr")
    host_t = host_st.add_tasklet("w", {}, {"o"}, "o = 1.0")
    host_st.add_edge(host_t, "o", host_ac, None, mm.Memlet(data="host_arr", subset="0"))

    kern_st = top.add_state_after(host_st, label="kern_state")
    me, mx = kern_st.add_map("k", {"i": "0:4", "j": "0:4"})
    me.add_in_connector("IN_kern_arr")
    me.add_out_connector("OUT_kern_arr")
    mx.add_in_connector("IN_kern_arr")
    mx.add_out_connector("OUT_kern_arr")
    kern_read = kern_st.add_read("kern_arr")
    kern_write = kern_st.add_write("kern_arr")
    t = kern_st.add_tasklet("id", {"a"}, {"o"}, "o = a")
    kern_st.add_edge(kern_read, None, me, "IN_kern_arr", mm.Memlet.from_array("kern_arr", top.arrays["kern_arr"]))
    kern_st.add_edge(me, "OUT_kern_arr", t, "a", mm.Memlet(data="kern_arr", subset="i, j"))
    kern_st.add_edge(t, "o", mx, "IN_kern_arr", mm.Memlet(data="kern_arr", subset="i, j"))
    kern_st.add_edge(mx, "OUT_kern_arr", kern_write, None, mm.Memlet.from_array("kern_arr", top.arrays["kern_arr"]))

    OffloadVelocityToGPU(top)

    assert "gpu_kern_arr" in top.arrays
    assert "gpu_host_arr" not in top.arrays
    for n in host_st.nodes():
        if isinstance(n, nodes.AccessNode):
            assert n.data == "host_arr"
    kern_access = [n for n in kern_st.nodes() if isinstance(n, nodes.AccessNode)]
    assert kern_access, "expected AccessNodes in the kernel state"
    for n in kern_access:
        assert n.data == "gpu_kern_arr", (f"kernel-side AccessNode should be retargeted, got data={n.data!r}")


def test_exclude_from_offload_keeps_array_cpu_side():
    """A kernel-side array listed in ``exclude_from_offload`` gets no
    ``gpu_`` clone and keeps its signature entry -- the round-trip case
    where the Fortran caller reads the value on the host after the SDFG
    returns."""
    top = _stage3_shaped_sdfg()
    sig_before = sorted(
        (n, type(d).__name__, tuple(str(x) for x in d.shape)) for n, d in top.arrays.items() if not d.transient)

    OffloadVelocityToGPU(top, exclude_from_offload=("A", ))

    assert "gpu_A" not in top.arrays
    assert "gpu_B" in top.arrays
    assert top.arrays["A"].transient is False
    assert top.arrays["A"].storage in (dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap,
                                       dtypes.StorageType.Register)

    sig_after = sorted(
        (n, type(d).__name__, tuple(str(x) for x in d.shape)) for n, d in top.arrays.items() if not d.transient)
    for n_before in sig_before:
        name = n_before[0]
        matches = [n for n in sig_after if n[0] == name]
        assert matches, f"non-transient {name!r} disappeared from signature"
        assert matches[0] == n_before, (f"signature drift for {name!r}: before={n_before} after={matches[0]}")


def test_multiple_end_states_all_reach_copy_out():
    """Branching tails are fine: every sink state feeds the single
    copy-out state."""
    top = SDFG("multi_end")
    top.add_array("A", [1, 4], dace.float64, transient=False)
    top.add_array("B", [1, 4], dace.float64, transient=False)

    s = top.add_state("start", is_start_block=True)
    end1 = top.add_state("end1")
    end2 = top.add_state("end2")
    top.add_edge(s, end1, dace.InterstateEdge(condition="1 == 1"))
    top.add_edge(s, end2, dace.InterstateEdge(condition="1 == 2"))
    me, mx = s.add_map("k", {"i": "0:4"})
    me.add_in_connector("IN_A")
    me.add_out_connector("OUT_A")
    mx.add_in_connector("IN_B")
    mx.add_out_connector("OUT_B")
    ar, bw = s.add_read("A"), s.add_write("B")
    t = s.add_tasklet("id", {"a"}, {"o"}, "o = a")
    s.add_edge(ar, None, me, "IN_A", mm.Memlet.from_array("A", top.arrays["A"]))
    s.add_edge(me, "OUT_A", t, "a", mm.Memlet(data="A", subset="0, i"))
    s.add_edge(t, "o", mx, "IN_B", mm.Memlet(data="B", subset="0, i"))
    s.add_edge(mx, "OUT_B", bw, None, mm.Memlet.from_array("B", top.arrays["B"]))

    OffloadVelocityToGPU(top)

    copy_out = next(st for st in top.states() if st.label == "_gpu_to_cpu_copy_out")
    for end in (end1, end2):
        assert top.out_edges(end), f"{end.label} lost its outgoing edge"
    reachable = {e.dst for end in (end1, end2) for e in top.out_edges(end)}
    assert copy_out in reachable
