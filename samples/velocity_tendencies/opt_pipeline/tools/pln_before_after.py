"""Build the velocity _for_it_35 pattern, run PerfLoopNesting, show before/after."""
from dace import nodes, SDFG
from dace.transformation.dataflow.perf_loop_nesting import PerfLoopNesting


def _build_case():
    """Reproduce the SDFG used by the velocity _for_it_35 test up to
    (but not including) the PerfLoopNesting call."""
    import dace
    from dace import dtypes, memlet as mm
    from dace.sdfg import SDFG as _S

    NPROMA = dace.symbol("NPROMA")
    KLEV = dace.symbol("KLEV")
    NB = dace.symbol("NB")
    DD0 = dace.symbol("DD0")
    DD2 = dace.symbol("DD2")
    LEV_LO = dace.symbol("LEV_LO")
    LEV_HI = dace.symbol("LEV_HI")
    IST = dace.symbol("IST")
    IEN = dace.symbol("IEN")

    inner = _S("loop_body")
    inner.add_array("cfl_clipping", [NPROMA, KLEV], dace.int32)
    inner.add_array("z_w_con_c", [NPROMA, KLEV], dace.float64)
    inner.add_scalar("cfl_w_limit", dace.float64)
    inner.add_scalar("dtime", dace.float64)
    inner.add_array("maxvcfl", [KLEV, NPROMA], dace.float64)
    inner.add_array("levmask", [NB, KLEV - 1], dace.int32)
    inner.add_array("__CG_p_metrics__m_ddqz_z_half", [DD0, KLEV, DD2], dace.float64,
                    storage=dtypes.StorageType.CPU_Heap)
    inner.add_scalar("tmp_call_7", dace.float64, transient=True)

    ist = inner.add_state("single_state_body", is_start_block=True)
    r_zwcon = ist.add_read("z_w_con_c")
    r_cfllim = ist.add_read("cfl_w_limit")
    r_ddqz_1 = ist.add_read("__CG_p_metrics__m_ddqz_z_half")
    w_cflclip = ist.add_access("cfl_clipping")
    r_cflclip = w_cflclip
    r_dtime = ist.add_read("dtime")
    r_maxvcfl = ist.add_read("maxvcfl")
    r_zwcon_2 = ist.add_read("z_w_con_c")
    r_ddqz_2 = ist.add_read("__CG_p_metrics__m_ddqz_z_half")
    w_levmask = ist.add_write("levmask")
    w_maxvcfl = ist.add_write("maxvcfl")
    w_zwcon = ist.add_write("z_w_con_c")

    me36, mx36 = ist.add_map("single_state_body_map", {"_for_it_36": "IST:IEN + 1"})
    for c in ("z_w_con_c", "cfl_w_limit", "ddqz"):
        me36.add_in_connector("IN_" + c); me36.add_out_connector("OUT_" + c)
    mx36.add_in_connector("IN_cfl_clipping"); mx36.add_out_connector("OUT_cfl_clipping")
    ist.add_edge(r_zwcon, None, me36, "IN_z_w_con_c", mm.Memlet("z_w_con_c[0:NPROMA, 0:KLEV]"))
    ist.add_edge(r_cfllim, None, me36, "IN_cfl_w_limit", mm.Memlet("cfl_w_limit[0]"))
    ist.add_edge(r_ddqz_1, None, me36, "IN_ddqz",
                 mm.Memlet("__CG_p_metrics__m_ddqz_z_half[0:DD0, 0:KLEV, 0:DD2]"))
    t36a = ist.add_tasklet("T_abs", {"z_in"}, {"t_out"}, "t_out = abs(z_in)")
    tmp7 = ist.add_access("tmp_call_7")
    t36b = ist.add_tasklet("T_cmp", {"t_in", "lim_in", "ddqz_in"}, {"c_out"},
                           "c_out = 1 if (t_in > (lim_in * ddqz_in)) else 0")
    ist.add_edge(me36, "OUT_z_w_con_c", t36a, "z_in", mm.Memlet("z_w_con_c[0, 0]"))
    ist.add_edge(t36a, "t_out", tmp7, None, mm.Memlet("tmp_call_7[0]"))
    ist.add_edge(tmp7, None, t36b, "t_in", mm.Memlet("tmp_call_7[0]"))
    ist.add_edge(me36, "OUT_cfl_w_limit", t36b, "lim_in", mm.Memlet("cfl_w_limit[0]"))
    ist.add_edge(me36, "OUT_ddqz", t36b, "ddqz_in",
                 mm.Memlet("__CG_p_metrics__m_ddqz_z_half[0, 0, 0]"))
    ist.add_edge(t36b, "c_out", mx36, "IN_cfl_clipping", mm.Memlet("cfl_clipping[0, 0]"))
    ist.add_edge(mx36, "OUT_cfl_clipping", w_cflclip, None,
                 mm.Memlet("cfl_clipping[0:NPROMA, 0:KLEV]"))

    me37, mx37 = ist.add_map("single_state_body_0_map", {"_for_it_37": "IST:IEN + 1"})
    for c in ("cfl_clipping", "dtime", "z_w_con_c", "maxvcfl", "ddqz"):
        me37.add_in_connector("IN_" + c); me37.add_out_connector("OUT_" + c)
    for c in ("levmask", "maxvcfl", "z_w_con_c"):
        mx37.add_in_connector("IN_" + c); mx37.add_out_connector("OUT_" + c)
    ist.add_edge(r_cflclip, None, me37, "IN_cfl_clipping",
                 mm.Memlet("cfl_clipping[0:NPROMA, 0:KLEV]"))
    ist.add_edge(r_dtime, None, me37, "IN_dtime", mm.Memlet("dtime[0]"))
    ist.add_edge(r_zwcon_2, None, me37, "IN_z_w_con_c",
                 mm.Memlet("z_w_con_c[0:NPROMA, 0:KLEV]"))
    ist.add_edge(r_maxvcfl, None, me37, "IN_maxvcfl",
                 mm.Memlet("maxvcfl[0:KLEV, 0:NPROMA]"))
    ist.add_edge(r_ddqz_2, None, me37, "IN_ddqz",
                 mm.Memlet("__CG_p_metrics__m_ddqz_z_half[0:DD0, 0:KLEV, 0:DD2]"))
    t37 = ist.add_tasklet("T_update",
                          {"c_in", "zw_in", "mv_in", "dt_in", "ddqz_in"},
                          {"lv_out", "mv_out", "zw_out"},
                          "lv_out = c_in\nmv_out = mv_in + dt_in\nzw_out = zw_in * ddqz_in")
    ist.add_edge(me37, "OUT_cfl_clipping", t37, "c_in", mm.Memlet("cfl_clipping[0, 0]"))
    ist.add_edge(me37, "OUT_z_w_con_c", t37, "zw_in", mm.Memlet("z_w_con_c[0, 0]"))
    ist.add_edge(me37, "OUT_maxvcfl", t37, "mv_in", mm.Memlet("maxvcfl[0, 0]"))
    ist.add_edge(me37, "OUT_dtime", t37, "dt_in", mm.Memlet("dtime[0]"))
    ist.add_edge(me37, "OUT_ddqz", t37, "ddqz_in",
                 mm.Memlet("__CG_p_metrics__m_ddqz_z_half[0, 0, 0]"))
    ist.add_edge(t37, "lv_out", mx37, "IN_levmask", mm.Memlet("levmask[0, 0]"))
    ist.add_edge(t37, "mv_out", mx37, "IN_maxvcfl", mm.Memlet("maxvcfl[0, 0]"))
    ist.add_edge(t37, "zw_out", mx37, "IN_z_w_con_c", mm.Memlet("z_w_con_c[0, 0]"))
    ist.add_edge(mx37, "OUT_levmask", w_levmask, None, mm.Memlet("levmask[0:NB, 0:KLEV - 1]"))
    ist.add_edge(mx37, "OUT_maxvcfl", w_maxvcfl, None, mm.Memlet("maxvcfl[0:KLEV, 0:NPROMA]"))
    ist.add_edge(mx37, "OUT_z_w_con_c", w_zwcon, None, mm.Memlet("z_w_con_c[0:NPROMA, 0:KLEV]"))

    outer = _S("velocity_for_it_35")
    outer.add_array("cfl_clipping", [NPROMA, KLEV], dace.int32)
    outer.add_array("z_w_con_c", [NPROMA, KLEV], dace.float64)
    outer.add_scalar("cfl_w_limit", dace.float64)
    outer.add_scalar("dtime", dace.float64)
    outer.add_array("maxvcfl", [KLEV, NPROMA], dace.float64)
    outer.add_array("levmask", [NB, KLEV - 1], dace.int32)
    outer.add_array("__CG_p_metrics__m_ddqz_z_half", [DD0, KLEV, DD2], dace.float64,
                    storage=dtypes.StorageType.CPU_Heap)
    ostate = outer.add_state("outer_state", is_start_block=True)

    pe, px = ostate.add_map("single_state_body_4_map", {"_for_it_35": "LEV_LO:LEV_HI"})
    for c in ("cfl_clipping", "z_w_con_c", "cfl_w_limit", "dtime", "maxvcfl", "ddqz"):
        pe.add_in_connector("IN_" + c); pe.add_out_connector("OUT_" + c)
    for c in ("levmask", "maxvcfl", "z_w_con_c", "cfl_clipping"):
        px.add_in_connector("IN_" + c); px.add_out_connector("OUT_" + c)

    nsdfg = ostate.add_nested_sdfg(
        inner,
        inputs={"cfl_clipping", "z_w_con_c", "cfl_w_limit", "dtime", "maxvcfl",
                "__CG_p_metrics__m_ddqz_z_half"},
        outputs={"levmask", "maxvcfl", "z_w_con_c", "cfl_clipping"},
        symbol_mapping={"NPROMA": NPROMA, "KLEV": KLEV, "NB": NB, "DD0": DD0, "DD2": DD2,
                        "IST": 0, "IEN": NPROMA - 1, "_for_it_35": "_for_it_35"},
    )
    r_cflclip_o = ostate.add_read("cfl_clipping")
    r_zwcon_o = ostate.add_read("z_w_con_c")
    r_cfllim_o = ostate.add_read("cfl_w_limit")
    r_dtime_o = ostate.add_read("dtime")
    r_maxvcfl_o = ostate.add_read("maxvcfl")
    r_ddqz_o = ostate.add_read("__CG_p_metrics__m_ddqz_z_half")
    w_levmask_o = ostate.add_write("levmask")
    w_maxvcfl_o = ostate.add_write("maxvcfl")
    w_zwcon_o = ostate.add_write("z_w_con_c")
    w_cflclip_o = ostate.add_write("cfl_clipping")

    ostate.add_edge(r_cflclip_o, None, pe, "IN_cfl_clipping", mm.Memlet("cfl_clipping[0:NPROMA, 0:KLEV]"))
    ostate.add_edge(r_zwcon_o, None, pe, "IN_z_w_con_c", mm.Memlet("z_w_con_c[0:NPROMA, 0:KLEV]"))
    ostate.add_edge(r_cfllim_o, None, pe, "IN_cfl_w_limit", mm.Memlet("cfl_w_limit[0]"))
    ostate.add_edge(r_dtime_o, None, pe, "IN_dtime", mm.Memlet("dtime[0]"))
    ostate.add_edge(r_maxvcfl_o, None, pe, "IN_maxvcfl", mm.Memlet("maxvcfl[0:KLEV, 0:NPROMA]"))
    ostate.add_edge(r_ddqz_o, None, pe, "IN_ddqz",
                    mm.Memlet("__CG_p_metrics__m_ddqz_z_half[0:DD0, 0:KLEV, 0:DD2]"))

    ostate.add_edge(pe, "OUT_cfl_clipping", nsdfg, "cfl_clipping",
                    mm.Memlet("cfl_clipping[0:NPROMA, 0:KLEV]"))
    ostate.add_edge(pe, "OUT_z_w_con_c", nsdfg, "z_w_con_c",
                    mm.Memlet("z_w_con_c[0:NPROMA, 0:KLEV]"))
    ostate.add_edge(pe, "OUT_cfl_w_limit", nsdfg, "cfl_w_limit", mm.Memlet("cfl_w_limit[0]"))
    ostate.add_edge(pe, "OUT_dtime", nsdfg, "dtime", mm.Memlet("dtime[0]"))
    ostate.add_edge(pe, "OUT_maxvcfl", nsdfg, "maxvcfl", mm.Memlet("maxvcfl[0:KLEV, 0:NPROMA]"))
    ostate.add_edge(pe, "OUT_ddqz", nsdfg, "__CG_p_metrics__m_ddqz_z_half",
                    mm.Memlet("__CG_p_metrics__m_ddqz_z_half[0:DD0, 0:KLEV, 0:DD2]"))

    ostate.add_edge(nsdfg, "levmask", px, "IN_levmask", mm.Memlet("levmask[0:NB, 0:KLEV - 1]"))
    ostate.add_edge(nsdfg, "maxvcfl", px, "IN_maxvcfl", mm.Memlet("maxvcfl[0:KLEV, 0:NPROMA]"))
    ostate.add_edge(nsdfg, "z_w_con_c", px, "IN_z_w_con_c", mm.Memlet("z_w_con_c[0:NPROMA, 0:KLEV]"))
    ostate.add_edge(nsdfg, "cfl_clipping", px, "IN_cfl_clipping", mm.Memlet("cfl_clipping[0:NPROMA, 0:KLEV]"))
    ostate.add_edge(px, "OUT_levmask", w_levmask_o, None, mm.Memlet("levmask[0:NB, 0:KLEV - 1]"))
    ostate.add_edge(px, "OUT_maxvcfl", w_maxvcfl_o, None, mm.Memlet("maxvcfl[0:KLEV, 0:NPROMA]"))
    ostate.add_edge(px, "OUT_z_w_con_c", w_zwcon_o, None, mm.Memlet("z_w_con_c[0:NPROMA, 0:KLEV]"))
    ostate.add_edge(px, "OUT_cfl_clipping", w_cflclip_o, None, mm.Memlet("cfl_clipping[0:NPROMA, 0:KLEV]"))

    outer.validate()
    return outer, ostate


def render(state, sdfg, label):
    print(f"\n{'=' * 66}\n{label}\n{'=' * 66}")
    print(f"outer state '{state.label}' of SDFG '{sdfg.name}'")
    print(f"  nodes: {len(state.nodes())}  edges: {len(state.edges())}")

    top_maps = [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and state.entry_node(n) is None]
    print(f"  top-level MapEntries: {len(top_maps)}")

    for i, pe in enumerate(top_maps, 1):
        px = state.exit_node(pe)
        print(f"\n  [Parent #{i}] '{pe.map.label}' params={pe.map.params}")
        print(f"    in_conns  = {sorted(pe.in_connectors)}")
        print(f"    out_conns = {sorted(px.out_connectors)}")
        body = [x for x in state.nodes() if state.entry_node(x) is pe]
        nsdfgs = [x for x in body if isinstance(x, nodes.NestedSDFG)]
        print(f"    body: {len(body)} nodes ({sum(1 for x in body if isinstance(x, nodes.NestedSDFG))} NSDFG, "
              f"{sum(1 for x in body if isinstance(x, nodes.MapExit))} MapExit)")
        for ns in nsdfgs:
            sub = ns.sdfg
            st = list(sub.states())[0]
            kids = [x for x in st.nodes() if st.entry_node(x) is None and isinstance(x, (nodes.MapEntry, nodes.Tasklet))]
            kid_labels = [f"{type(x).__name__}({(x.map.label if isinstance(x, nodes.MapEntry) else x.label)})" for x in kids]
            arrs = sorted(sub.arrays.keys())
            print(f"    nsdfg '{ns.label}':")
            print(f"      in_conns  = {sorted(ns.in_connectors)}")
            print(f"      out_conns = {sorted(ns.out_connectors)}")
            print(f"      arrays    = {arrs}")
            print(f"      inner state '{st.label}' top-level maps/tasklets: {kid_labels}")

    outer_in = sorted({(type(e.src).__name__, getattr(e.src, "data", getattr(e.src, "label", ""))) for e in state.edges()
                       if isinstance(e.dst, nodes.MapEntry) and state.entry_node(e.dst) is None})
    outer_out = sorted({(type(e.dst).__name__, getattr(e.dst, "data", getattr(e.dst, "label", ""))) for e in state.edges()
                        if isinstance(e.src, nodes.MapExit) and state.entry_node(e.src) is None})
    print(f"\n  outer reads : {outer_in}")
    print(f"  outer writes: {outer_out}")


def main():
    outer, ostate = _build_case()
    render(ostate, outer, "BEFORE PerfLoopNesting")

    applied = outer.apply_transformations_repeated(PerfLoopNesting)
    outer.validate()
    print(f"\n>>> PerfLoopNesting applied {applied} time(s); SDFG still validates.")

    render(ostate, outer, "AFTER PerfLoopNesting")


if __name__ == "__main__":
    main()
