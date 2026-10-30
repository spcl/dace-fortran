# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Demote interstate-assigned symbols that shape no iteration domain or data access to scalars.

A counter such as ICON's ``clip_count = clip_count + 1`` is data spelled as a symbol: a loop updating it reads as
carrying a symbol and stays sequential, while as a scalar it is a write the reduction passes handle.
"""

import ast
from collections.abc import Container
from typing import Any

import dace
from dace import SDFG, dtypes, symbolic
from dace.sdfg import nodes
from dace.sdfg.replace import replace_in_codeblock, tokenize_cpp
from dace.sdfg.state import ConditionalBlock, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from ordered_set import OrderedSet

from dace_fortran.dace_types import explicit_cf_compatible

#: The state ScalarToSymbolPromotion puts first to initialize a promoted transient a read may reach unassigned.
PROMOTED_INIT_STATE = "init_promoted_scalars"


def subscript_index_symbols(code: object, containers: Container[str]) -> set[str]:
    """Names an index into one of ``containers`` is written in: ``k`` in ``A[k, i] > 0``."""
    names: set[str] = set()
    for tree in code if isinstance(code, list) else [code]:
        if not isinstance(tree, ast.AST):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in containers:
                names.update(n.id for n in ast.walk(node.slice) if isinstance(n, ast.Name))
    return names


def structural_symbols(sdfg: SDFG) -> set[str]:
    """Symbols of ``sdfg``'s own blocks that a shape, subset, map range, loop domain or nested mapping reads."""
    found: set[str] = set()
    for desc in sdfg.arrays.values():
        found.update(str(s) for s in desc.free_symbols)
    for block in sdfg.all_control_flow_blocks():
        if isinstance(block, LoopRegion):
            if block.loop_variable:
                found.add(block.loop_variable)
            for code in (block.init_statement, block.loop_condition, block.update_statement):
                if code is not None:
                    found.update(str(s) for s in code.get_free_symbols())
        if isinstance(block, (LoopRegion, ConditionalBlock)):
            for code in block.get_meta_codeblocks():
                found.update(subscript_index_symbols(code.code, sdfg.arrays))
    for edge in sdfg.all_interstate_edges():
        found.update(subscript_index_symbols(edge.data.condition.code, sdfg.arrays))
        for value in edge.data.assignments.values():
            found.update(subscript_index_symbols(ast.parse(value), sdfg.arrays))
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry):
                found.update(str(s) for s in node.map.range.free_symbols)
            elif isinstance(node, nodes.NestedSDFG):
                for value in node.symbol_mapping.values():
                    found.update(str(s) for s in symbolic.pystr_to_symbolic(value).free_symbols)
            elif isinstance(node, nodes.Tasklet) and node.code.language is dtypes.Language.Python:
                found.update(
                    subscript_index_symbols(node.code.code, node.in_connectors.keys() | node.out_connectors.keys())
                )
        for dataflow_edge in state.edges():
            found.update(str(s) for s in dataflow_edge.data.free_symbols)
    return found


def tasklet_reads(tasklet: nodes.Tasklet, name: str) -> bool:
    if tasklet.code.language is dtypes.Language.Python:
        return name in tasklet.free_symbols
    return name in tokenize_cpp.findall(tasklet.code.as_string)


def assignment_tasklet(state: SDFGState, lhs: str, rhs: str) -> None:
    """Write the interstate assignment ``lhs = rhs`` into ``state``; every container ``rhs`` reads gets a connector.

    The expression stays as written (no SymPy round trip, which would change its rounding).
    """
    arrays = state.sdfg.arrays
    reads: dict[str, tuple[str, str]] = {}

    def connector(data: str, expr: str) -> ast.Name:
        return ast.Name(id=reads.setdefault(expr, (f"_in_{data}_{len(reads)}", data))[0], ctx=ast.Load())

    class ReadsToConnectors(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            if isinstance(node.value, ast.Name) and node.value.id in arrays:
                return connector(node.value.id, ast.unparse(node))
            return self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> ast.AST:
            return connector(node.id, f"{node.id}[0]") if node.id in arrays else node

    body = ReadsToConnectors().visit(ast.parse(rhs.strip(), mode="eval")).body
    out = f"_out_{lhs}"
    tasklet = state.add_tasklet(
        f"assign_{lhs}", {conn for conn, _ in reads.values()}, {out}, f"{out} = {ast.unparse(body)}"
    )
    accesses: dict[str, nodes.AccessNode] = {}
    for expr, (conn, data) in reads.items():
        if data not in accesses:
            accesses[data] = state.add_access(data)
        state.add_edge(accesses[data], None, tasklet, conn, dace.Memlet(expr))
    state.add_edge(tasklet, out, state.add_access(lhs), None, dace.Memlet(f"{lhs}[0]"))


def demote_symbol(sdfg: SDFG, name: str) -> None:
    """Turn the symbol ``name`` of ``sdfg`` into a transient scalar of the same dtype."""
    dtype = sdfg.symbols[name]
    sdfg.remove_symbol(name)
    sdfg.add_scalar(name, dtype, transient=True)
    conn = f"_in_{name}"
    for state in sdfg.states():
        scope = None
        for node in list(state.nodes()):
            if not isinstance(node, nodes.Tasklet) or not tasklet_reads(node, name):
                continue
            replace_in_codeblock(node.code, {name: conn}, node, sdfg)
            node.add_in_connector(conn)
            access = state.add_access(name)
            state.add_edge(access, None, node, conn, dace.Memlet(f"{name}[0]"))
            scope = scope or state.scope_dict()
            if scope[node] is not None:
                state.add_edge(scope[node], None, access, None, dace.Memlet())
    # One state per assigning edge: ``add_state_before`` would move every in-edge of a join onto one write.
    for edge in list(sdfg.all_interstate_edges()):
        if name not in edge.data.assignments:
            continue
        rhs = edge.data.assignments.pop(name)
        cfg = edge.dst.parent_graph
        state = cfg.add_state(f"_{edge.dst}_sym_assign")
        cfg.remove_edge(edge)
        cfg.add_edge(edge.src, state, edge.data)
        cfg.add_edge(state, edge.dst, dace.InterstateEdge())
        assignment_tasklet(state, name, rhs)


def drop_undefined_init(sdfg: SDFG, name: str) -> None:
    """Forget the entry value ScalarToSymbolPromotion gave ``name`` to keep it out of the signature.

    A read reaching that value sees Fortran's undefined variable, so the value means nothing. As a scalar write it
    would be an access outside the loop that owns ``name``, and LoopToMap would refuse to privatize it per iteration.
    """
    start = sdfg.start_block
    if start.label == PROMOTED_INIT_STATE:
        for edge in sdfg.out_edges(start):
            edge.data.assignments.pop(name, None)


@explicit_cf_compatible
class DemoteNonStructuralSymbols(ppl.Pass):
    """Demote every interstate-assigned symbol that no iteration domain or data access reads to a scalar."""

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        demoted = 0
        for sd in sdfg.all_sdfgs_recursive():
            assigned = OrderedSet(name for edge in sd.all_interstate_edges() for name in edge.data.assignments)
            # A symbol entering from outside has no definition here to turn into a scalar write.
            outside = sd.free_symbols if sd.parent_nsdfg_node is None else sd.parent_nsdfg_node.symbol_mapping.keys()
            structural = None
            for name in assigned:
                if name not in sd.symbols or name in outside:
                    continue
                structural = structural if structural is not None else structural_symbols(sd)
                if name in structural:
                    continue
                drop_undefined_init(sd, name)
                demote_symbol(sd, name)
                demoted += 1
                # A demotion adds tasklets and accesses whose subscripts can name other candidates.
                structural = None
        return demoted or None
