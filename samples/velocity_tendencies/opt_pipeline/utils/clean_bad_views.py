import dace

BAD_VIEW_NAMES = [
    ("ptr_patch_var_128_0", "v_ptr_patch_var_128_verts"),
    ("ptr_patch_var_130_0", "v_ptr_patch_var_130_verts"),
    ("p_patch", "v_p_patch_edges"),
    ("p_patch", "v_p_patch_cells"),
]


def clean_bad_views(sdfg: dace.SDFG):
    for s, parent in sdfg.all_nodes_recursive():
        if isinstance(s, dace.SDFGState):
            for e in s.edges():
                if (
                    isinstance(e.src, dace.nodes.AccessNode)
                    and isinstance(e.dst, dace.nodes.AccessNode)
                    and s.out_degree(e.dst) == 0
                ):
                    src = e.src.data
                    dst = e.dst.data
                    if (src, dst) in BAD_VIEW_NAMES:
                        s.remove_edge(e)
                        s.remove_node(e.dst)
                        if s.in_degree(e.src) == 0 and s.out_degree(e.src) == 0:
                            s.remove_node(e.src)
