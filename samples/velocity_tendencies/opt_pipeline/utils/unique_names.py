import dace
import typing


def unique_names(sdfgs: typing.List[dace.SDFG]):
    for i, sdfg in enumerate(sdfgs):
        sdfg.function_suffix = "_" + str(i)
        visited = set()
        for n, parent in sdfg.all_nodes_recursive():
            if n in visited:
                continue
            # Add unique names for anything that can become CUDA functions to avoid name conflicts
            if isinstance(n, dace.nodes.MapEntry):
                n.map.label = f"{n.map.label}{sdfg.function_suffix}"
                n.label = f"{n.label}{sdfg.function_suffix}"
                visited.add(n.map)
                visited.add(n)
            # if isinstance(n, dace.nodes.MapExit):
            #    n.label = f"{n.label}{sdfg.function_suffix}"
            #    visited.add(n)
            if isinstance(n, dace.nodes.NestedSDFG):
                n.sdfg.function_suffix = "_" + str(i)
                visited.add(n)
