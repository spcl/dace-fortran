"""Promote ``maxvcfl`` scalar reduction into a 2D array + deferred reduce.

Turns the per-block scalar read-modify-write shape

    for block:
        maxvcfl[0] = 0
        for level:
            for nproma_i:
                maxvcfl[0] = max(maxvcfl[0], expr)
        vcflmax[block] = maxvcfl[0]

into

    for block:
        maxvcfl[0:nproma, 0:nlev] = 0                           # 2D map-fill
        for level:                                               # now Map-able
            for nproma_i:                                        # Map
                maxvcfl[nproma_i-1, level-1] = max(
                    maxvcfl[nproma_i-1, level-1], expr)
        vcflmax[block] = max-reduce(maxvcfl[0:nproma, 0:nlev])   # 2D reduce

``maxvcfl`` descriptor: ``Scalar(f64)`` -> ``Array(f64, [nproma, nlev])``
with Fortran-major strides ``(1, nproma)``. ``vcflmax`` stays
``Array(f64, [nblks])`` unchanged; only the producing tasklet is
replaced by a ``Reduce(max)`` over both maxvcfl axes.

The 2D layout lets the level loop parallelise: every (nproma_i, level)
iteration writes a distinct maxvcfl slot, so there is no cross-iteration
write conflict on the outer level loop.
"""
import dace
from dace import data, memlet as mm, nodes
from dace.sdfg.state import LoopRegion


def promote_maxvcfl(sdfg: dace.SDFG,
                    scalar_name: str = "maxvcfl",
                    sink_name: str = "vcflmax",
                    nproma_sym: str = "__CG_global_data__m_nproma",
                    nlev_sym: str = "91") -> int:
    if scalar_name not in sdfg.arrays:
        return 0
    desc = sdfg.arrays[scalar_name]
    if not isinstance(desc, data.Scalar):
        return 0

    dtype = desc.dtype
    transient = desc.transient
    sdfg.remove_data(scalar_name, validate=False)
    # Promote to ``(nproma, nlev)`` Fortran-major. Shape mirrors the
    # classic Fortran ``maxvcfl(nproma, nlev)`` declaration; strides
    # ``(1, nproma)`` make the nproma axis contiguous. That lines up
    # with every other velocity transient and means stage 3's
    # LiftTransients takes the Fortran-packed/append branch, giving
    # a 3D ``(nproma, nlev, nblks)`` lifted layout. No permutation
    # pass needed downstream -- threadIdx.x → nproma gets coalesced
    # access in the level/nproma kernels directly.
    from dace import symbolic
    sdfg.add_array(scalar_name,
                   shape=[nproma_sym, nlev_sym],
                   strides=[1, symbolic.pystr_to_symbolic(nproma_sym)],
                   dtype=dtype, transient=transient)

    rewrites = 0
    for state in list(sdfg.all_states()):
        targets = [n for n in state.nodes()
                   if isinstance(n, nodes.AccessNode) and n.data == scalar_name]
        if not targets:
            continue
        inner_vars = _enclosing_loop_vars(state, 2)
        for an in targets:
            in_edges = list(state.in_edges(an))
            out_edges = list(state.out_edges(an))

            if in_edges and not out_edges:
                (edge,) = in_edges
                src = edge.src
                if (isinstance(src, nodes.Tasklet)
                        and src.code.as_string.strip() == f"{scalar_name}_out = 0"):
                    _rewrite_init(state, an, src, scalar_name, nlev_sym, nproma_sym)
                    rewrites += 1
                    continue
                edge.data = _rmw_memlet(scalar_name, inner_vars)
                rewrites += 1

            elif out_edges and not in_edges:
                if _is_consumer_read(state, an, sink_name):
                    _install_final_reduction(state, an, scalar_name, sink_name,
                                             nlev_sym, nproma_sym)
                    rewrites += 1
                    continue
                for e in out_edges:
                    e.data = _rmw_memlet(scalar_name, inner_vars)
                    rewrites += 1

            else:
                for e in in_edges + out_edges:
                    e.data = _rmw_memlet(scalar_name, inner_vars)
                    rewrites += 1
    return rewrites


def _rmw_memlet(name: str, inner_vars):
    # Shape is ``[nproma, nlev]`` (Fortran-major): nproma is axis 0
    # with stride 1, level is axis 1 with stride nproma. inner_vars[0]
    # is the innermost loop var (nproma); [1] is level.
    level_var, nproma_var = inner_vars[1], inner_vars[0]
    return mm.Memlet(f"{name}[{nproma_var} - 1, {level_var} - 1]")


def _enclosing_loop_vars(state, n: int):
    vars_ = []
    g = state.parent_graph
    while g is not None and len(vars_) < n:
        if isinstance(g, LoopRegion):
            vars_.append(g.loop_variable)
        g = getattr(g, "parent_graph", None)
    return vars_


def _rewrite_init(state, access_node, tasklet, name, nlev_sym, nproma_sym):
    for e in list(state.in_edges(access_node)):
        state.remove_edge(e)
    state.remove_node(tasklet)

    # ``_i`` (nproma) is listed first so DaCe maps it to threadIdx.x --
    # the stride-1 axis, matching the Fortran-major layout installed
    # above.
    me, mx = state.add_map("init_maxvcfl",
                           {"_i": f"0:{nproma_sym}", "_j": f"0:{nlev_sym}"})
    mx.add_in_connector("IN__buf")
    mx.add_out_connector("OUT__buf")
    t = state.add_tasklet("zero", {}, {"_out"}, "_out = 0")
    state.add_nedge(me, t, mm.Memlet())
    state.add_edge(t, "_out", mx, "IN__buf", mm.Memlet(f"{name}[_i, _j]"))
    state.add_edge(mx, "OUT__buf", access_node, None,
                   mm.Memlet(f"{name}[0:{nproma_sym}, 0:{nlev_sym}]"))


def _is_consumer_read(state, access_node, sink_name) -> bool:
    for e in state.out_edges(access_node):
        if not isinstance(e.dst, nodes.Tasklet):
            continue
        for oe in state.out_edges(e.dst):
            if isinstance(oe.dst, nodes.AccessNode) and oe.dst.data == sink_name:
                return True
    return False


def _install_final_reduction(state, access_node, scalar_name, sink_name,
                             nlev_sym, nproma_sym):
    tasklet_edges = [e for e in state.out_edges(access_node)
                     if isinstance(e.dst, nodes.Tasklet)]
    if not tasklet_edges:
        return
    tasklet = tasklet_edges[0].dst
    sink_outs = [e for e in state.out_edges(tasklet)
                 if isinstance(e.dst, nodes.AccessNode) and e.dst.data == sink_name]
    if not sink_outs:
        return
    sink_edge = sink_outs[0]
    sink_subset = sink_edge.data.subset
    sink_access = sink_edge.dst

    for e in list(state.in_edges(tasklet)) + list(state.out_edges(tasklet)):
        state.remove_edge(e)
    state.remove_node(tasklet)

    red = state.add_reduce("lambda a, b: max(a, b)", axes=[0, 1], identity=None)
    state.add_edge(access_node, None, red, None,
                   mm.Memlet(f"{scalar_name}[0:{nproma_sym}, 0:{nlev_sym}]"))
    state.add_edge(red, None, sink_access, None,
                   mm.Memlet(data=sink_name, subset=sink_subset))
