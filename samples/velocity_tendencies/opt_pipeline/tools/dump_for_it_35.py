"""Dump the exact wiring of the `_for_it_35` MapEntry + its NSDFG."""
import argparse
import dace
from dace import nodes

from tools.diagnose_pln import replay_up_to_pln


def dump_memlet(m):
    if m is None:
        return "None"
    return f"Memlet(data={m.data!r}, subset={m.subset}, volume={m.volume}, dynamic={m.dynamic})"


def dump(sdfg):
    for state in sdfg.states():
        for n in state.nodes():
            if isinstance(n, nodes.MapEntry) and "_for_it_35" in n.map.params:
                print(f"=== OUTER MapEntry '{n.map.label}' ===")
                print(f"  map.params={n.map.params}  map.range={n.map.range}  schedule={n.map.schedule}")
                print(f"  in_conns={sorted(n.in_connectors)}  out_conns={sorted(n.out_connectors)}")
                px = state.exit_node(n)
                print(f"  MapExit in_conns={sorted(px.in_connectors)}  out_conns={sorted(px.out_connectors)}")

                print("\n  -- outer in-edges (into MapEntry) --")
                for e in state.in_edges(n):
                    print(f"    {type(e.src).__name__}({getattr(e.src, 'data', getattr(e.src, 'label', ''))!r}).{e.src_conn} -> MapEntry.{e.dst_conn}  {dump_memlet(e.data)}")
                print("  -- outer out-edges (out of MapExit) --")
                for e in state.out_edges(px):
                    print(f"    MapExit.{e.src_conn} -> {type(e.dst).__name__}({getattr(e.dst, 'data', getattr(e.dst, 'label', ''))!r}).{e.dst_conn}  {dump_memlet(e.data)}")

                body = [x for x in state.nodes() if state.entry_node(x) is n]
                nsdfg_node = next(x for x in body if isinstance(x, nodes.NestedSDFG))
                print(f"\n  -- outer->NSDFG edges (via MapEntry->NSDFG) --")
                for e in state.out_edges(n):
                    if e.dst is nsdfg_node:
                        print(f"    MapEntry.{e.src_conn} -> NSDFG.{e.dst_conn}  {dump_memlet(e.data)}")
                print(f"  -- NSDFG->outer edges (NSDFG->MapExit) --")
                for e in state.in_edges(px):
                    if e.src is nsdfg_node:
                        print(f"    NSDFG.{e.src_conn} -> MapExit.{e.dst_conn}  {dump_memlet(e.data)}")

                inner = nsdfg_node.sdfg
                print(f"\n=== NSDFG '{nsdfg_node.label}' sdfg.name='{inner.name}' ===")
                print(f"  NSDFG.in_connectors = {sorted(nsdfg_node.in_connectors)}")
                print(f"  NSDFG.out_connectors = {sorted(nsdfg_node.out_connectors)}")
                print(f"  NSDFG.symbol_mapping = {dict(nsdfg_node.symbol_mapping)}")
                print(f"  NSDFG.sdfg.symbols = {dict(inner.symbols)}")
                print(f"  NSDFG.sdfg.arrays (count={len(inner.arrays)}):")
                for name, desc in inner.arrays.items():
                    print(f"    {name}: {type(desc).__name__} shape={desc.shape} dtype={desc.dtype} transient={desc.transient} storage={desc.storage}")

                ist = list(inner.states())[0]
                print(f"\n  inner state '{ist.label}' nodes:")
                for x in ist.nodes():
                    if isinstance(x, nodes.AccessNode):
                        print(f"    Access '{x.data}'  in={ist.in_degree(x)} out={ist.out_degree(x)}")
                    elif isinstance(x, nodes.MapEntry):
                        print(f"    MapEntry '{x.map.label}'  params={x.map.params} range={x.map.range}")
                        print(f"      in_conns={sorted(x.in_connectors)} out_conns={sorted(x.out_connectors)}")
                    elif isinstance(x, nodes.MapExit):
                        print(f"    MapExit  in_conns={sorted(x.in_connectors)} out_conns={sorted(x.out_connectors)}")
                    elif isinstance(x, nodes.Tasklet):
                        print(f"    Tasklet '{x.label}'  code={x.code.as_string[:80]!r}")
                    elif isinstance(x, nodes.NestedSDFG):
                        print(f"    NSDFG    label='{x.label}'  in={sorted(x.in_connectors)} out={sorted(x.out_connectors)}")
                    else:
                        print(f"    {type(x).__name__}: {x}")

                print(f"\n  inner state edges (top-level only, scoped children summarised):")
                for e in ist.edges():
                    # only show edges where both endpoints are top-level or we care about the boundary
                    if ist.entry_node(e.src) is None and ist.entry_node(e.dst) is None:
                        sl = f"{type(e.src).__name__}({getattr(e.src,'data',getattr(e.src,'label',''))!r}).{e.src_conn}"
                        dl = f"{type(e.dst).__name__}({getattr(e.dst,'data',getattr(e.dst,'label',''))!r}).{e.dst_conn}"
                        print(f"    {sl}  ->  {dl}   {dump_memlet(e.data)}")

                for inner_me in [x for x in ist.nodes() if isinstance(x, nodes.MapEntry)]:
                    inner_mx = ist.exit_node(inner_me)
                    print(f"\n  -- inner map '{inner_me.map.label}' ({inner_me.map.params}={inner_me.map.range}) ---")
                    print(f"     in-edges to MapEntry:")
                    for e in ist.in_edges(inner_me):
                        print(f"       {type(e.src).__name__}({getattr(e.src,'data','')!r}).{e.src_conn} -> {e.dst_conn}  {dump_memlet(e.data)}")
                    print(f"     body (scoped children):")
                    for bn in ist.nodes():
                        if ist.entry_node(bn) is inner_me:
                            lbl = getattr(bn, 'label', getattr(bn, 'data', type(bn).__name__))
                            extra = ""
                            if isinstance(bn, nodes.Tasklet):
                                extra = f" code={bn.code.as_string[:120]!r}"
                            print(f"       {type(bn).__name__} '{lbl}'{extra}")
                    print(f"     out-edges from MapExit:")
                    for e in ist.out_edges(inner_mx):
                        print(f"       {e.src_conn} -> {type(e.dst).__name__}({getattr(e.dst,'data','')!r}).{e.dst_conn}  {dump_memlet(e.data)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="velocity_no_nproma_if_prop_lvn_only_0_istep_1")
    args = p.parse_args()

    sdfg = dace.SDFG.from_file(f"baseline/{args.variant}.sdfgz")
    sdfg.name = args.variant
    sdfg = replay_up_to_pln(sdfg)

    stack = [sdfg]
    seen = {id(sdfg)}
    while stack:
        s = stack.pop()
        dump(s)
        for state in s.states():
            for n in state.nodes():
                if isinstance(n, nodes.NestedSDFG) and id(n.sdfg) not in seen:
                    seen.add(id(n.sdfg))
                    stack.append(n.sdfg)


if __name__ == "__main__":
    main()
