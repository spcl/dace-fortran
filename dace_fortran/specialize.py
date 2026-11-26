# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Specialize an SDFG for known scalar and symbol values: the Fortran frontend's namelist and configuration values.

Moved from the FaCe branch of DaCe, so that DaCe keeps main's ``specialize_scalar``: only dace-fortran uses the batched
form, which folds every value in one pass and also reaches the tasklets the values feed.
"""

import copy
import numbers
from typing import cast

import dace
from dace.properties import CodeBlock
from dace.sdfg import nodes as nd
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.nodes import NestedSDFG
from dace.sdfg.state import ConditionalBlock, LoopRegion

from dace_fortran.dace_types import connectors


def _specialize_scalar_impl(root: "dace.SDFG", sdfg: "dace.SDFG", scalars: dict[str, float | int | str]):
    # This function replaces the scalars named by <scalars> with their constant values
    # A scalar can appear on:
    # 1. Interstate Edge
    # -> For 1: Replace occurence on the interstate edge with scalar_name
    # 2. Dynamic Input to a Map
    # -> For 2: Rm. dynamic in connector, remove the edge and the node if the degree is None
    # 3. Access Node
    # -> If access node is used then e.g. [scalar] -> [tasklet]
    # -> then create a [tasklet] that uses the scalar_val as a constant value inside
    import ast
    import re

    import numpy as np
    from dace.frontend.python import astutils

    def _token_replace(code: str, src: str, dst: str) -> str:
        # Whole identifiers only (`o = _in*3`), not attributes (`x._in`)
        return re.sub(r"(?<![A-Za-z0-9_.])" + re.escape(src) + r"(?![A-Za-z0-9_])", dst, code).strip()

    def _scalar_literal(value: float | str, dtype) -> str:
        # An integer replacing a floating-point scalar stays a float literal: `_in / 2` must not become `5 / 2`
        floating = dtype is not None and np.issubdtype(dtype.as_numpy_dtype(), np.floating)
        if isinstance(value, str):
            if not floating:
                return value
            try:
                value = float(value)
            except ValueError:
                return str(value)
        if floating:
            # At the scalar's width: 0.1 stored in a float32 is not the double 0.1
            return repr(float(dtype.type(value)))
        return str(value)

    def rewrite_code(code: CodeBlock | None, repl: dict[str, str]) -> CodeBlock | None:
        if code is None:  # A while loop has no init or update statement
            return None
        if code.language == dace.dtypes.Language.Python:
            # On the AST: precedence holds, and a sympy round-trip loses float64 digits and turns 1/3 into int division
            replacer = astutils.ASTFindReplace(dict(repl))
            body = cast(list[ast.AST], CodeBlock(code.as_string, dace.dtypes.Language.Python).code)
            new_body = [ast.fix_missing_locations(replacer.visit(stmt)) for stmt in body]
            return CodeBlock(code=new_body, language=dace.dtypes.Language.Python)
        text = code.as_string
        for src, dst in repl.items():
            # A negative or compound value keeps its own parentheses: `y-x` must not become `y--1.5`
            text = _token_replace(text, src, dst if re.fullmatch(r"[\w.]+", dst) else f"({dst})")
        return CodeBlock(code=text, language=code.language)

    def carried_scalar(edge: MultiConnectorEdge[dace.Memlet]) -> str | None:
        # A copy's memlet may name either endpoint
        for node in (edge.src, edge.dst):
            if isinstance(node, nd.AccessNode) and node.data in scalars:
                return node.data
        return edge.data.data if edge.data is not None and edge.data.data in scalars else None

    def removable(node: nd.Node) -> bool:
        return not isinstance(node, nd.CodeNode) or not node.has_side_effects(sdfg)

    # Captured before the descriptors are removed below (see _scalar_literal).
    scalar_dtypes = {name: (sdfg.arrays[name].dtype if name in sdfg.arrays else None) for name in scalars}
    literals = {name: _scalar_literal(val, scalar_dtypes[name]) for name, val in scalars.items()}

    # Before replacing anything collect all nested SDFGs and their in-out edges for recursion
    nsdfgs = [
        (node, state, state.in_edges(node), state.out_edges(node))
        for state in sdfg.states()
        for node in state.nodes()
        if isinstance(node, nd.NestedSDFG)
    ]

    # If we are the root SDFG then we need can't remove non-transient scalar (but will just not use it)
    # For nestedSDFGs we will remove
    if root != sdfg:
        for name in scalars:
            if name in sdfg.arrays:
                sdfg.remove_data(name, validate=False)

            if name in sdfg.symbols:
                sdfg.remove_symbol(name)

    for state in sdfg.states():
        # Check dynamic inputs
        for e in cast(list[MultiConnectorEdge[dace.Memlet]], state.edges()):
            if e not in state.edges():
                continue
            scalar_name = carried_scalar(e)
            if scalar_name is None:
                continue
            literal = literals[scalar_name]

            src = e.src
            dst = e.dst
            # What a map entry keeps an edge to once the scalar's path through it is gone
            scope_target = dst

            if isinstance(dst, nd.Tasklet) and e.dst_conn is not None:
                dst.code = rewrite_code(dst.code, {e.dst_conn: literal})
                dst.remove_in_connector(e.dst_conn)
            elif isinstance(dst, nd.AccessNode) and dst.data not in scalars and not e.data.is_empty():
                # A copy out of the scalar: its target still needs the value
                scope_target = state.add_tasklet(
                    f"{scalar_name}_value", {}, connectors(["__out"]), f"__out = {literal}"
                )
                subset = e.data.get_dst_subset(e, state) or dace.subsets.Range.from_array(sdfg.arrays[dst.data])
                dst_memlet = dace.Memlet(data=dst.data, subset=copy.deepcopy(subset), wcr=e.data.wcr)
                state.add_edge(scope_target, "__out", dst, e.dst_conn, dst_memlet)
            elif isinstance(dst, nd.LibraryNode) and e.dst_conn is not None:
                # The expansion reads the connector, so it keeps it
                scope_target = state.add_tasklet(
                    f"{scalar_name}_value", {}, connectors(["__out"]), f"__out = {literal}"
                )
                name, desc = sdfg.add_scalar(
                    f"__{scalar_name}_value", scalar_dtypes[scalar_name], transient=True, find_new_name=True
                )
                access = state.add_access(name)
                state.add_edge(scope_target, "__out", access, None, dace.Memlet.from_array(name, desc))
                state.add_edge(access, None, dst, e.dst_conn, dace.Memlet.from_array(name, desc))
            elif e.dst_conn is not None:
                dst.remove_in_connector(e.dst_conn)

            state.remove_edge(e)
            if e.src_conn is not None:
                src.remove_out_connector(e.src_conn)

            if isinstance(src, nd.MapEntry) and (state.out_degree(src) == 0 or state.in_degree(scope_target) == 0):
                # Add a dep edge, to not invalidate the map
                state.add_edge(src, None, scope_target, None, dace.memlet.Memlet())
            elif state.degree(src) == 0 and removable(src):
                state.remove_node(src)
            if state.in_degree(dst) == 0:
                if isinstance(dst, nd.MapExit):
                    state.add_edge(src, None, dst, None, dace.memlet.Memlet())
                elif state.degree(dst) == 0 and removable(dst):
                    state.remove_node(dst)

        # Uses as a free name: map ranges, memlet subsets, tasklet code, nested symbol mappings
        state.replace_dict(literals)

    # Replace on for CFGs as
    for cfg in sdfg.all_control_flow_regions():
        if isinstance(cfg, LoopRegion):
            cfg.loop_condition = rewrite_code(cfg.loop_condition, literals)
            cfg.init_statement = rewrite_code(cfg.init_statement, literals)
            cfg.update_statement = rewrite_code(cfg.update_statement, literals)
            assert cfg.loop_variable not in scalars, (
                f"Loop variable {cfg.loop_variable} cannot be the same as a specialized scalar"
            )
        if isinstance(cfg, ConditionalBlock):
            for i, (n_cond, n_body) in enumerate(cfg.branches):
                if n_cond is not None:
                    cfg.branches[i] = (rewrite_code(n_cond, literals), n_body)

    # Not recursive: a nested SDFG binds names of its own and is reached through its connectors below
    for edge in sdfg.all_interstate_edges():
        edge.data.replace_dict(literals, replace_keys=False)

    if root != sdfg:
        for scalar_name in scalars:
            sdfg.parent_nsdfg_node.symbol_mapping.pop(scalar_name, None)

    for nsdfg_node, state, in_edges, out_edges in nsdfgs:
        # Captured before the loop above, which may have deleted the nest
        if nsdfg_node not in state.nodes():
            continue
        assert not any(oe.data.data in scalars for oe in out_edges)
        # By connector: one scalar can feed several
        inner = {ie.dst_conn: scalars[ie.data.data] for ie in in_edges if ie.data.data in scalars}
        if inner:
            # Avoid import loop
            from dace.transformation.passes.length_one_array_scalar_conversion import rewrite_code_slots, rewrite_refs

            # An inner length-1 array is read as ``y[0]``: only the name takes the literal
            subscripted = {name: name for name in inner}
            rewrite_code_slots(nsdfg_node.sdfg, lambda text: rewrite_refs(text, subscripted))
            _specialize_scalar_impl(root, nsdfg_node.sdfg, inner)


def _interacting(values: dict[str, float | int | str]) -> bool:
    """Whether a value names another entry (``{a: 'b+1', b: '2'}``), where one batched substitution differs from
    substituting one at a time."""
    import re

    names = values.keys()
    return any(
        isinstance(val, str) and ((names - {name}) & set(re.findall(r"[A-Za-z_]\w*", val)))
        for name, val in values.items()
    )


def specialize_scalar(sdfg: "dace.SDFG", scalar_name: str, scalar_val: float | str):
    """Bake a scalar data container to a constant. Single-entry case of :func:`specialize_scalars`."""
    assert isinstance(scalar_name, str), f"Expected scalar name to be str got {type(scalar_name)}"
    specialize_scalars(sdfg, {scalar_name: scalar_val})


def specialize_scalars(sdfg: "dace.SDFG", values: dict[str, float | int | str]):
    """Bake scalar data containers to constant values, recursively through nested SDFGs, in one walk: reads fold
    into the tasklets, edges and connectors go, and loop/branch conditions and interstate edges are rewritten.

    :param sdfg: The SDFG to specialize.
    :param values: Map of scalar name to the constant value to substitute in.
    """
    if not values:
        return

    scalars: dict[str, float | int | str] = {}
    for name, val in values.items():
        # sympy registers its numbers with the ``numbers`` ABCs
        assert isinstance(val, (str, numbers.Real)), f"Expected float, int, str or sympy.Number, got {val!r}"
        scalars[name] = val if isinstance(val, str) else int(val) if isinstance(val, numbers.Integral) else float(val)

    if _interacting(scalars):
        # Cannot fold into one pass without changing the result -- keep the sequential meaning.
        for name, val in scalars.items():
            _specialize_scalar_impl(sdfg, sdfg, {name: val})
        return

    _specialize_scalar_impl(sdfg, sdfg, scalars)


def specialize_symbols(sdfg: "dace.SDFG", values: dict[str, float | int | str]) -> None:
    """Bake free symbols to constant values in one walk over every nested SDFG, dropping them from each
    ``symbol_mapping``. For a scalar data container holding the value, use :func:`specialize_scalars`.

    :param sdfg: The SDFG to specialize.
    :param values: Map of symbol name to the constant value to substitute in.
    """
    if not values:
        return
    if _interacting(values):
        # Cannot fold into one pass without changing the result -- keep the sequential meaning.
        for name, value in values.items():
            specialize_symbols(sdfg, {name: value})
        return
    bake_symbols(sdfg, {name: str(value) for name, value in values.items()})


def scope_iterators(sdfg: "dace.SDFG") -> set[str]:
    """Iterators of the loops and maps of ``sdfg``, not of its nested SDFGs."""
    loops = {r.loop_variable for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)}
    return loops | {p for s in sdfg.states() for n in s.nodes() if isinstance(n, nd.MapEntry) for p in n.map.params}


def nested_bindings(sdfg: "dace.SDFG") -> set[str]:
    """Names ``sdfg`` binds itself, which a value mapped in from outside cannot replace."""
    return {name for edge in sdfg.all_interstate_edges() for name in edge.data.assignments} | scope_iterators(sdfg)


def bake_symbols(sdfg: "dace.SDFG", vals: dict[str, str]) -> None:
    iterators = scope_iterators(sdfg)
    vals = {name: val for name, val in vals.items() if name not in iterators}
    # Drop their interstate definitions first: replace_dict would turn the key into a literal (`5 = ...`)
    for e in sdfg.all_interstate_edges():
        for name in vals:
            e.data.assignments.pop(name, None)
    free = {str(s) for s in sdfg.free_symbols}
    present = {name: val for name, val in vals.items() if name in sdfg.symbols or name in free}
    if present:
        sdfg.replace_dict(present)
    for name in vals:
        if name in sdfg.symbols:
            sdfg.remove_symbol(name)
    # Inside a nest a name means its symbol_mapping value: `foo(n + 1)` bakes the nested `n` to the constant it became
    for state in sdfg.states():
        for node in state.nodes():
            if not isinstance(node, NestedSDFG):
                continue
            bound = nested_bindings(node.sdfg)
            mapped = {name: dace.symbolic.pystr_to_symbolic(str(val)) for name, val in node.symbol_mapping.items()}
            inner = {name: str(val) for name, val in mapped.items() if not val.free_symbols and name not in bound}
            for name in inner:
                del node.symbol_mapping[name]
            if inner:
                bake_symbols(node.sdfg, inner)
