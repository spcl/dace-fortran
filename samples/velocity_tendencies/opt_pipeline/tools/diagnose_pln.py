"""Replay stage1 up to the PLN step and dump every ``_for_it_35`` MapEntry."""
import argparse
import dace
from dace import nodes
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.transformation.dataflow.perf_loop_nesting import PerfLoopNesting
from dace.transformation.interstate import LoopToMap
from dace.transformation.interstate.move_if_into_map import MoveIfIntoMap
from dace.transformation.passes.loop_to_reduce import LoopToReduce
from dace.transformation.passes.simplification.continue_to_condition import ContinueToCondition

from utils.passes.baseline_fix import fix_baseline
from utils.passes.promote_maxvcfl import promote_maxvcfl
from utils.passes.remove_clip_count import remove_clip_count
from utils.passes.uniquify_difcoef import uniquify_difcoef


def replay_up_to_pln(sdfg: dace.SDFG) -> dace.SDFG:
    fix_baseline(sdfg)
    remove_clip_count(sdfg)
    sdfg.validate()
    LoopToReduce().apply_pass(sdfg, {})
    sdfg.validate()
    promote_maxvcfl(sdfg)
    sdfg.validate()
    ContinueToCondition().apply_pass(sdfg, {})
    sdfg.validate()
    uniquify_difcoef(sdfg)
    sdfg.validate()
    sdfg.simplify(skip=["ArrayElimination"])
    sdfg.validate()
    propagate_memlets_sdfg(sdfg)
    sdfg.validate()
    sdfg.apply_transformations_repeated(LoopToMap, permissive=True)
    sdfg.simplify(skip=["ArrayElimination"])
    try:
        sdfg.apply_transformations_repeated(MoveIfIntoMap, validate=False, validate_all=False)
    except Exception as exc:
        print(f"[warn] MoveIfIntoMap step raised: {exc}")
    return sdfg


def walk_sdfgs(sdfg, prefix=""):
    yield prefix or sdfg.name, sdfg
    for nsdfg_node, parent_state in sdfg.all_nodes_recursive():
        if isinstance(nsdfg_node, nodes.NestedSDFG):
            # walk once — all_nodes_recursive already descends but we want labels.
            pass


def describe(n, state, sdfg, indent="  "):
    pe = n
    scope = "top-level" if state.entry_node(pe) is None else f"nested under {state.entry_node(pe).map.label}"
    print(f"{indent}MapEntry '{pe.map.label}' params={pe.map.params} range={pe.map.range} @ state '{state.label}' ({scope})")
    if state.entry_node(pe) is not None:
        print(f"{indent}  -> not top-level; PLN gate 1 fails")
        return
    body = [x for x in state.nodes() if state.entry_node(x) is pe]
    kinds = {}
    for x in body:
        kinds.setdefault(type(x).__name__, []).append(x)
    print(f"{indent}  body kinds: " + ", ".join(f"{k}x{len(v)}" for k, v in kinds.items()))
    nsdfgs = [x for x in body if isinstance(x, nodes.NestedSDFG)]
    if len(nsdfgs) != 1:
        print(f"{indent}  -> #NSDFGs={len(nsdfgs)}; PLN gate 2 fails")
        if nsdfgs:
            for ns in nsdfgs:
                print(f"{indent}    NSDFG '{ns.label}' states={len(list(ns.sdfg.states()))}")
        return
    inner = nsdfgs[0].sdfg
    n_states = len(list(inner.states()))
    print(f"{indent}  inner SDFG '{inner.name}' states={n_states}")
    if n_states != 1:
        for st in inner.states():
            print(f"{indent}    state '{st.label}' nodes={len(st.nodes())}")
        print(f"{indent}  -> PLN gate 3 fails")
        return
    ist = list(inner.states())[0]
    top_nodes = [x for x in ist.nodes() if ist.entry_node(x) is None]
    top_kinds = {}
    for x in top_nodes:
        top_kinds.setdefault(type(x).__name__, []).append(x)
    print(f"{indent}  inner top-level nodes: " + ", ".join(f"{k}x{len(v)}" for k, v in top_kinds.items()))
    for x in top_nodes:
        if isinstance(x, nodes.MapEntry):
            print(f"{indent}    [MapEntry] '{x.map.label}' params={x.map.params}")
        elif isinstance(x, nodes.Tasklet):
            print(f"{indent}    [Tasklet]  '{x.label}'")
        elif isinstance(x, nodes.AccessNode):
            print(f"{indent}    [Access ]  '{x.data}' in={ist.in_degree(x)} out={ist.out_degree(x)}")
        else:
            print(f"{indent}    [{type(x).__name__}] {x}")
    kids = [x for x in top_nodes if isinstance(x, (nodes.MapEntry, nodes.Tasklet))]
    print(f"{indent}  K={len(kids)} (MapEntry|Tasklet at top level)")
    if len(kids) < 2:
        print(f"{indent}  -> PLN gate 4/5 fails (need K>=2)")
    else:
        matches = PerfLoopNesting().can_be_applied_to(sdfg, parent_entry=pe)
        print(f"{indent}  -> can_be_applied_to = {matches}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="velocity_no_nproma_if_prop_lvn_only_0_istep_1")
    args = p.parse_args()

    path = f"baseline/{args.variant}.sdfgz"
    print(f"Loading {path}")
    sdfg = dace.SDFG.from_file(path)
    sdfg.name = args.variant
    sdfg = replay_up_to_pln(sdfg)

    hits = 0
    # Walk every SDFG reachable from top-level
    seen_sdfgs = {id(sdfg): sdfg}
    stack = [sdfg]
    while stack:
        s = stack.pop()
        for st in s.states():
            for n in st.nodes():
                if isinstance(n, nodes.MapEntry) and "_for_it_35" in n.map.params:
                    hits += 1
                    print(f"\n== Hit #{hits} in SDFG '{s.name}' ==")
                    describe(n, st, s)
                if isinstance(n, nodes.NestedSDFG) and id(n.sdfg) not in seen_sdfgs:
                    seen_sdfgs[id(n.sdfg)] = n.sdfg
                    stack.append(n.sdfg)

    print(f"\nTotal _for_it_35 MapEntries found: {hits}")


if __name__ == "__main__":
    main()
