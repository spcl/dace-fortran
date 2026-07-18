# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Optional, List, Iterable, Set, Tuple, Dict

import networkx as nx
import numpy as np
import fparser.two.Fortran2003 as f03
from fparser.two.utils import Base, walk

from . import utils
from . import types
from . import analysis
from .. import ast_utils


def _nearest_scope_with_spec_part(node: Base) -> Optional[Base]:
    """The nearest enclosing scope (module / main program / subprogram) of
    ``node`` that has a ``Specification_Part``.  A consolidated ``USE`` can only
    live in such a scope; a contained subprogram whose body is only
    host-associated references has none, so its imports bubble up to here."""
    p: Optional[Base] = node
    while p is not None:
        if isinstance(p, (f03.Module, f03.Main_Program, f03.Function_Subprogram, f03.Subroutine_Subprogram)):
            if any(isinstance(c, f03.Specification_Part) for c in p.children):
                return p
        p = p.parent
    return None


def consolidate_uses(ast: f03.Program, alias_map: Optional[types.SPEC_TABLE] = None) -> f03.Program:
    """Rewrites every `USE` statement to `USE ..., ONLY: ...`, listing only symbols actually used in that scope."""
    alias_map = alias_map or analysis.alias_specs(ast)
    for sp in reversed(walk(ast, f03.Specification_Part)):
        # Fix the kind parameter of literals referring to a variable: fparser leaves them as plain strings, not Names.
        for lit in walk(sp.parent,
                        (f03.Real_Literal_Constant, f03.Signed_Real_Literal_Constant, f03.Int_Literal_Constant,
                         f03.Signed_Int_Literal_Constant, f03.Logical_Literal_Constant)):
            val, kind = lit.children
            if not isinstance(kind, str):
                continue
            # Only a *named* kind (``0.0_wp``) needs wrapping into a ``Name`` so
            # later passes can resolve it.  A *numeric* kind suffix (``0.0_8``,
            # ``1_4`` -- ubiquitous in real Fortran) is left as the plain string
            # fparser produced: ``f03.Name('8')`` is not a valid identifier and
            # would raise ``NoMatchError``.
            if kind.isidentifier():
                utils.set_children(lit, (val, f03.Name(kind)))

        use_map: Dict[str, Set[str]] = {}
        for nm in walk(sp.parent, f03.Name):
            if isinstance(nm.parent, (f03.Use_Stmt, f03.Only_List, f03.Rename)):
                continue
            # Find the module `nm` was really imported from.
            sc_spec = analysis.search_scope_spec(nm)
            if not sc_spec:
                continue
            box = alias_map[sc_spec].parent
            if box is not sp.parent and isinstance(
                    box, (f03.Function_Subprogram, f03.Subroutine_Subprogram, f03.Main_Program)):
                # `nm` is used in a deeper subprogram: consolidate it there, when
                # that subprogram's own Specification_Part is processed -- UNLESS
                # it has none (a body of only host-associated references), in
                # which case the import must be retained at the nearest enclosing
                # scope that does (here), or it would be dropped and dangle.
                if _nearest_scope_with_spec_part(box) is not sp.parent:
                    continue
            spec = analysis.search_real_ident_spec(nm.string, sc_spec, alias_map)
            if not spec or spec not in alias_map:
                continue
            if alias_map[spec].parent is sp.parent:
                # If `nm` is just referring to the subprogram that `sp` is a part of, then just leave it be.
                continue
            if len(spec) == 2:
                mod_spec = spec[:-1]
            elif len(spec) == 3 and spec[-2] == analysis.INTERFACE_NAMESPACE:
                mod_spec = spec[:-2]
            else:
                continue
            if not isinstance(alias_map[mod_spec], f03.Module_Stmt):
                # Objects defined inside a free function cannot be imported; so we must already be in that function.
                continue
            nm_mod = mod_spec[0]
            sp_mod = sp
            while sp_mod and not isinstance(sp_mod, (f03.Module, f03.Main_Program)):
                sp_mod = sp_mod.parent
            if sp_mod and nm_mod == utils.find_name_of_node(sp_mod):
                continue
            if nm.string == spec[-1]:
                u = nm.string
            else:
                u = f"{nm.string} => {spec[-1]}"
            if nm_mod not in use_map:
                use_map[nm_mod] = set()
            use_map[nm_mod].add(u)
        nuses: List[f03.Use_Stmt] = [
            f03.Use_Stmt(f"use {k}, only: {', '.join(sorted(use_map[k]))}") for k in use_map.keys()
        ]
        utils.set_children(sp, nuses + [c for c in sp.children if not isinstance(c, f03.Use_Stmt)])
    return ast


def keep_sorted_used_modules(ast: f03.Program, entry_points: Optional[Iterable[types.SPEC]] = None) -> f03.Program:
    """Drops modules not transitively reachable (via `USE`) from `entry_points` (all modules
    if None), and topologically sorts the survivors so each is defined before it is used."""
    TOPLEVEL = '__toplevel__'

    def _get_module(n: Base) -> str:
        p = n
        while p and not isinstance(p, (f03.Module, f03.Main_Program)):
            p = p.parent
        if not p:
            return TOPLEVEL
        else:
            p_stmt = ast_utils.singular(ast_utils.children_of_type(p, (f03.Module_Stmt, f03.Program_Stmt)))
            return utils.find_name_of_stmt(p_stmt).lower()

    g = nx.DiGraph()  # edge u->v: u must come before v (v depends on u).
    for c in ast.children:
        g.add_node(_get_module(c))
    g.add_node(TOPLEVEL)

    for u in walk(ast, f03.Use_Stmt):
        u_name = ast_utils.singular(ast_utils.children_of_type(u, f03.Name)).string.lower()
        v_name = _get_module(u)
        g.add_edge(u_name, v_name)

    entry_modules: Set[str]
    if entry_points is None:
        entry_modules = set(g.nodes) | {TOPLEVEL}
    else:
        entry_modules = {ep[0] for ep in entry_points if ep[0] in g.nodes} | {TOPLEVEL}

    assert all(g.has_node(em) for em in entry_modules)
    used_modules: Set[str] = {anc for em in entry_modules for anc in nx.ancestors(g, em)} | entry_modules
    h = g.subgraph(used_modules).to_directed()

    top_ord = {n: i for i, n in enumerate(nx.lexicographical_topological_sort(h))}
    top_ord[TOPLEVEL] = g.number_of_nodes() + 1

    utils.set_children(ast, [n for n in ast.children if _get_module(n) in used_modules])
    assert all(_get_module(n) in top_ord for n in ast.children)
    utils.set_children(ast, sorted(ast.children, key=lambda x: top_ord[_get_module(x)]))

    return ast


def prune_coarsely(ast: f03.Program, keepers: Iterable[types.SPEC]) -> f03.Program:
    """Iteratively removes functions/types/interfaces/variables not reachable from `keepers`,
    to a fixed point."""
    removed_something = None
    while removed_something is None or removed_something:
        removed_something = False
        ast = consolidate_uses(ast)
        ast = keep_sorted_used_modules(ast, keepers)
        ident_map = analysis.identifier_specs(ast)
        alias_map = analysis.alias_specs(ast)
        iface_map = analysis.interface_specs(ast, alias_map)

        used_fns: Set[types.SPEC] = set(keepers)
        for k, v in ident_map.items():
            if len(k) < 2 or not isinstance(v, (f03.Function_Stmt, f03.Subroutine_Stmt)):
                continue
            vname = utils.find_name_of_stmt(v)
            box = alias_map[k[:-2] if k[-2] == analysis.INTERFACE_NAMESPACE else k[:-1]].parent
            for nm in walk(box, f03.Name):
                if (nm.string != vname or isinstance(nm.parent, (f03.Rename, f03.Use_Stmt)) or isinstance(
                        nm.parent,
                    (f03.Function_Stmt, f03.End_Function_Stmt, f03.Subroutine_Stmt, f03.End_Subroutine_Stmt))):
                    continue
                scope_spec = analysis.search_scope_spec(nm)
                if scope_spec == k:
                    continue
                used_fns.add(k)
                break
        for k, v in alias_map.items():
            if not isinstance(v, (f03.Function_Stmt, f03.Subroutine_Stmt)):
                continue
            if k not in ident_map:
                used_fns.add(analysis.ident_spec(v))
        for fref in walk(ast, (f03.Function_Reference, f03.Call_Stmt)):
            scope_spec = analysis.find_scope_spec(fref)
            name, _ = fref.children
            if isinstance(name, f03.Intrinsic_Name):
                continue
            fref_spec = analysis.search_real_ident_spec(name.string, scope_spec, alias_map)
            if fref_spec and len(fref_spec) == 1:
                used_fns.add(fref_spec)
        for k, vs in iface_map.items():
            for v in vs:
                used_fns.add(v)
        for k, v in ident_map.items():
            if not isinstance(v, (f03.Function_Stmt, f03.Subroutine_Stmt)):
                continue
            if k not in used_fns:
                utils.remove_self(v.parent)
                removed_something = True

        used_types: Set[types.SPEC] = set()
        for k, v in ident_map.items():
            if not isinstance(v, f03.Derived_Type_Stmt):
                continue
            vname = utils.find_name_of_stmt(v)
            box = alias_map[k[:-1]].parent
            for nm in walk(box, f03.Name):
                if nm.string != vname or isinstance(nm.parent, (f03.Rename, f03.Use_Stmt)):
                    continue
                if isinstance(nm.parent, (f03.Derived_Type_Stmt, f03.End_Type_Stmt)) and nm.parent.parent is v.parent:
                    continue
                scope_spec = analysis.search_scope_spec(nm)
                if scope_spec == k:
                    continue
                used_types.add(k)
                break
        for k, v in alias_map.items():
            if not isinstance(v, f03.Derived_Type_Stmt):
                continue
            if k not in ident_map:
                used_types.add(analysis.ident_spec(v))
        for k, v in ident_map.items():
            if not isinstance(v, f03.Derived_Type_Stmt):
                continue
            if k not in used_types:
                utils.remove_self(v.parent)
                removed_something = True

        used_ifaces: Set[types.SPEC] = set()
        for k, v in ident_map.items():
            if len(k) < 2 or k[-2] != analysis.INTERFACE_NAMESPACE:
                continue
            vname = utils.find_name_of_stmt(v)
            box = alias_map[k[:-2]].parent
            for nm in walk(box, f03.Name):
                if nm.string != vname or isinstance(nm.parent, (f03.Rename, f03.Use_Stmt)):
                    continue
                if isinstance(nm.parent, (f03.Interface_Stmt, f03.End_Interface_Stmt)) and nm.parent.parent is v.parent:
                    continue
                scope_spec = analysis.search_scope_spec(nm)
                if scope_spec == k or scope_spec == k[:-2] + k[-1:]:
                    continue
                used_ifaces.add(k)
                break
        for k, v in alias_map.items():
            vspec = analysis.ident_spec(v)
            if len(vspec) < 2 or vspec[-2] != analysis.INTERFACE_NAMESPACE:
                continue
            if k not in ident_map:
                used_ifaces.add(vspec)
        for k, v in ident_map.items():
            if len(k) < 2 or k[-2] != analysis.INTERFACE_NAMESPACE:
                continue
            if k not in used_ifaces:
                utils.remove_self(v.parent)
                removed_something = True

        used_vars: Set[types.SPEC] = set()
        for k, v in ident_map.items():
            if not isinstance(v, (f03.Entity_Decl, f03.Proc_Decl)):
                continue
            vname = utils.find_name_of_stmt(v)
            box = alias_map[k[:-1]].parent
            for nm in walk(box, f03.Name):
                if nm.string != vname or isinstance(nm.parent, (f03.Rename, f03.Use_Stmt)) or nm.parent is v:
                    continue
                scope_spec = analysis.search_scope_spec(nm)
                if scope_spec == k:
                    continue
                used_vars.add(k)
                break
        for k, v in alias_map.items():
            if not isinstance(v, (f03.Entity_Decl, f03.Proc_Decl)):
                continue
            if k not in ident_map:
                used_vars.add(analysis.ident_spec(v))
        for k, v in ident_map.items():
            if not isinstance(v, (f03.Entity_Decl, f03.Proc_Decl)):
                continue
            if k not in used_vars:
                elist = v.parent
                utils.remove_self(v)
                elist_tdecl = elist.parent
                assert isinstance(
                    elist_tdecl,
                    (f03.Type_Declaration_Stmt, f03.Procedure_Declaration_Stmt, f03.Proc_Component_Def_Stmt))
                if not elist.children:
                    utils.remove_self(elist_tdecl)
                removed_something = True

    for iface in walk(ast, f03.Interface_Stmt):
        name, = iface.children
        if name and name != 'ABSTRACT':
            continue
        idef = iface.parent
        if not idef.children[1:-1]:
            utils.remove_self(idef)

    ast = keep_sorted_used_modules(ast, keepers)
    return ast


def prune_dangling_interface_bodies(ast: f03.Program) -> f03.Program:
    """Removes interface bodies left dangling by pruning.

    An interface body's ``IMPORT`` can name a type/kind pruning drops as unused
    external baggage (ICON's halo ``interface_exchange_data_*`` importing
    ``t_comm_pattern_collection`` etc.) -- it then imports a name that no longer
    exists and cannot compile. Such a body is also unreferenced (its DEFERRED
    binding was pruned with its type), so drop it; then remove any interface
    block emptied as a result.

    Gated by the caller on :data:`analysis.TOLERATE_EXTERNAL_USES`: with full
    resolution every ``IMPORT`` resolves, so this is a no-op."""
    alias_map = analysis.alias_specs(ast)
    for imp in list(walk(ast, f03.Import_Stmt)):
        body = imp.parent.parent  # Specification_Part -> (Subroutine|Function)_Body
        if not isinstance(body.parent, f03.Interface_Block):
            continue
        host_scope = analysis.find_scope_spec(body.parent)
        if any(analysis.search_real_ident_spec(nm.string, host_scope, alias_map) is None for nm in walk(imp, f03.Name)):
            utils.remove_self(body)
    # Drop generic-interface MODULE PROCEDURE members whose target subprogram was pruned:
    # reachability prunes unused specifics but their names linger in the generic's member
    # list -- gfortran: "Procedure 'x' in generic interface 'y' is neither function nor subroutine".
    defined: Set[str] = set()
    for sp in walk(ast, (f03.Subroutine_Subprogram, f03.Function_Subprogram)):
        stmt = ast_utils.atmost_one(ast_utils.children_of_type(sp, (f03.Subroutine_Stmt, f03.Function_Stmt)))
        nm = utils.find_name_of_stmt(stmt) if stmt is not None else None
        if nm:
            defined.add(nm.lower())
    for proc_stmt in list(walk(ast, f03.Procedure_Stmt)):
        if not isinstance(proc_stmt.parent, f03.Interface_Block):
            continue
        name_list = proc_stmt.children[0]
        names = walk(name_list, f03.Name)
        kept = [nm for nm in names if nm.string.lower() in defined]
        if len(kept) == len(names):
            continue
        if not kept:
            utils.remove_self(proc_stmt)
        else:
            name_list.items = tuple(kept)
    for iface in walk(ast, f03.Interface_Stmt):
        name, = iface.children
        if name and name != 'ABSTRACT':
            continue
        idef = iface.parent
        if not idef.children[1:-1]:
            utils.remove_self(idef)
    return ast


def prune_unused_objects(ast: f03.Program, keepers: List[types.SPEC]) -> f03.Program:
    """Fine-grained pruning: removes any object not reachable (by usage) from `keepers`.

    Precondition: indirections (e.g. interface calls) must already be resolved."""
    PRUNABLE_OBJECT_CLASSES = (f03.Program_Stmt, f03.Subroutine_Stmt, f03.Function_Stmt, f03.Derived_Type_Stmt,
                               f03.Entity_Decl, f03.Component_Decl)

    ident_map = analysis.identifier_specs(ast)
    alias_map = analysis.alias_specs(ast)
    survivors: Set[types.SPEC] = set(keepers)
    keeper_nodes = [alias_map[k] for k in keepers]
    assert all(isinstance(k, PRUNABLE_OBJECT_CLASSES) for k in keeper_nodes)

    def _keep_from(node: Base):
        for nm in walk(node, f03.Name):
            loc = analysis.search_real_local_alias_spec(nm, alias_map)
            scope_spec = analysis.search_scope_spec(nm.parent)
            if not loc or not scope_spec: continue
            nm_spec = analysis.ident_spec(alias_map[loc])
            if isinstance(nm.parent, f03.Entity_Decl) and nm is nm.parent.children[0]:
                fnargs = ast_utils.atmost_one(ast_utils.children_of_type(alias_map[scope_spec], f03.Dummy_Arg_List))
                fnargs = fnargs.children if fnargs else tuple()
                if any(a.string == nm.string for a in fnargs):
                    survivors.add(nm_spec)
                continue
            if isinstance(nm.parent, f03.Component_Decl) and nm is nm.parent.children[0]: continue
            if isinstance(nm.parent, f03.Pointer_Assignment_Stmt) and nm is nm.parent.children[0]: continue

            for j in reversed(range(len(scope_spec))):
                anc_spec = scope_spec[:j + 1]
                if anc_spec in survivors: continue
                # INTERFACE_NAMESPACE is a synthetic scope-spec segment, not an object, so it's
                # absent from alias_map -- skip an ancestor that doesn't resolve rather than index it blindly.
                anc_node = alias_map.get(anc_spec)
                if anc_node is None: continue
                survivors.add(anc_spec)
                if isinstance(anc_node, PRUNABLE_OBJECT_CLASSES):
                    _keep_from(anc_node.parent)

            if not nm_spec or nm_spec not in alias_map or nm_spec in survivors: continue
            survivors.add(nm_spec)
            keep_node = alias_map[nm_spec]
            if isinstance(keep_node, PRUNABLE_OBJECT_CLASSES):
                _keep_from(keep_node.parent)
            elif isinstance(keep_node, f03.Interface_Stmt):
                # An UNRESOLVED generic INTERFACE reference (deconstruct_interface_calls couldn't
                # pick a specific, e.g. a keyword-arg call to ICON's smooth_oncells) keeps the whole
                # block: its MODULE PROCEDURE names recurse to candidates, so it stays bindable
                # rather than being emptied to a dangling generic. A resolved generic never hits this.
                _keep_from(keep_node.parent)
        # Component accesses keep the components they touch. A pointer-component WRITE
        # (``this % comp => x``) is a Data_Pointer_Object, not a Data_Ref, so it's invisible to
        # walk(Data_Ref) -- without this, an only-ever-assigned pointer component is wrongly pruned.
        comp_refs: List[Base] = list(walk(node, f03.Data_Ref))
        comp_refs += [dpo for dpo in walk(node, f03.Data_Pointer_Object) if '%' in dpo.tofortran()]
        for dr in comp_refs:
            root, rest = analysis._lookup_dataref(dr, alias_map)
            if rest and isinstance(rest[0], f03.Section_Subscript_List):
                root, rest = f03.Part_Ref(f"{root.tofortran()}({rest[0].tofortran()})"), rest[1:]
            scope_spec = analysis.find_scope_spec(dr)
            for upto in range(1, len(rest) + 1):
                anc_nodes: Tuple[f03.Name, ...] = (root, ) + rest[:upto]
                ancref = f03.Data_Ref('%'.join([c.tofortran() for c in anc_nodes]))
                ancspec = analysis.find_dataref_component_spec(ancref, scope_spec, alias_map)
                survivors.add(ancspec)

    for k in keeper_nodes:
        _keep_from(k.parent)

    killed: Set[types.SPEC] = set()
    for ns in sorted(set(ident_map.keys()) - survivors):
        ns_node = ident_map[ns]
        if not isinstance(ns_node, PRUNABLE_OBJECT_CLASSES): continue
        is_killed = False
        for i in range(len(ns) - 1):
            anc_spec = ns[:i + 1]
            if anc_spec in killed:
                killed.add(ns)
                is_killed = True
                break
        if is_killed: continue
        ns_typ = analysis.find_type_of_entity(ns_node, alias_map)
        if isinstance(ns_node, f03.Entity_Decl) and ns_typ.pointer:
            for pa in walk(ast, f03.Pointer_Assignment_Stmt):
                dst = pa.children[0]
                if not isinstance(dst, f03.Name): continue
                dst_spec = analysis.search_real_local_alias_spec(dst, alias_map)
                if dst_spec and alias_map[dst_spec] is ns_node:
                    utils.remove_self(pa)
        if isinstance(ns_node, f03.Entity_Decl):
            elist = ns_node.parent
            utils.remove_self(ns_node)
            elist_tdecl = elist.parent
            assert isinstance(elist_tdecl, f03.Type_Declaration_Stmt)
            if not elist.children:
                utils.remove_self(elist_tdecl)
            elist_spart = elist_tdecl.parent
            assert isinstance(elist_spart, f03.Specification_Part)
            for c in elist_spart.children:
                if not isinstance(c, f03.Equivalence_Stmt): continue
                _, eqvs = c.children
                eqvs = eqvs.children if eqvs else tuple()
                for eqv in eqvs:
                    eqa, eqbs = eqv.children
                    eqbs = eqbs.children if eqbs else tuple()
                    eqz = (eqa, ) + eqbs
                    assert all(isinstance(z, f03.Part_Ref) for z in eqz) and len(eqz) == 2
                    eqz = tuple(z for z in eqz if analysis.search_real_local_alias_spec(z.children[0], alias_map) != ns)
                    if len(eqz) < 2:
                        utils.remove_self(eqv)
                _, eqvs = c.children
                if not (eqvs.children if eqvs else tuple()):
                    utils.remove_self(c)
            if not elist_spart.children:
                utils.remove_self(elist_spart)
        elif isinstance(ns_node, f03.Component_Decl):
            clist = ns_node.parent
            utils.remove_self(ns_node)
            tdef = clist.parent
            assert isinstance(tdef, f03.Data_Component_Def_Stmt)
            if not clist.children:
                utils.remove_self(tdef)
        else:
            utils.remove_self(ns_node.parent)
        killed.add(ns)

    # A type whose every component was pruned (CLOUDSC's stubbed PERFORMANCE_TIMER) must not
    # stay memberless: variables of it still exist, and numpy f2py builds a NULL module wrapper
    # for one, segfaulting on import (PyInit_*). Keep a placeholder component instead.
    for tdef in walk(ast, f03.Derived_Type_Def):
        if not walk(tdef, f03.Component_Decl):
            tstmt = ast_utils.singular(ast_utils.children_of_type(tdef, f03.Derived_Type_Stmt))
            utils.replace_node(tstmt, (tstmt, f03.Data_Component_Def_Stmt("INTEGER :: pruned_type_placeholder")))

    for m in walk(ast, f03.Module):
        _, sp, ex, subp = utils._get_module_or_program_parts(m)
        empty_spec = not sp or all(isinstance(c, (f03.Save_Stmt, f03.Implicit_Part)) for c in sp.children)
        empty_exec = not ex or not ex.children
        empty_subp = not subp or all(isinstance(c, f03.Contains_Stmt) for c in subp.children)
        if empty_spec and empty_exec and empty_subp:
            utils.remove_self(m)

    consolidate_uses(ast, alias_map)
    return ast


def prune_branches(ast: f03.Program, alias_map: Optional[types.SPEC_TABLE] = None) -> f03.Program:
    """Prunes dead branches from `If_Construct`/`If_Stmt` by evaluating their conditions
    at compile time.

    `alias_map`: None builds it from `ast` (whole-program case). A caller holding a
    substituted FRAGMENT (whose USEd modules aren't present, so alias_specs can't run)
    passes an empty map: only already-folded .TRUE./.FALSE. conditions then fold --
    exactly what's needed to prune PRESENT(absent)-dead branches."""
    if alias_map is None:
        alias_map = analysis.alias_specs(ast)
    for ib in walk(ast, f03.If_Construct):
        _prune_branches_in_ifblock(ib, alias_map)
    for ib in walk(ast, f03.If_Stmt):
        _prune_branches_in_ifstmt(ib, alias_map)
    return ast


def _prune_branches_in_ifblock(ib: f03.If_Construct, alias_map: types.SPEC_TABLE):
    """Helper to prune an `If_Construct` (a multi-line IF block)."""
    ifthen = ib.children[0]
    assert isinstance(ifthen, f03.If_Then_Stmt)
    cond, = ifthen.children
    cval = analysis._const_eval_basic_type(cond, alias_map)
    if cval is None:
        return
    assert isinstance(cval, np.bool_)

    elifat = [idx for idx, c in enumerate(ib.children) if isinstance(c, (f03.Else_If_Stmt, f03.Else_Stmt))]
    if cval:
        cut = elifat[0] if elifat else -1
        actions = ib.children[1:cut]
        utils.replace_node(ib, actions)
        return
    elif not elifat:
        utils.remove_self(ib)
        return

    cut = elifat[0]
    cut_cond_node = ib.children[cut]
    if isinstance(cut_cond_node, f03.Else_Stmt):
        actions = ib.children[cut + 1:-1]
        utils.replace_node(ib, actions)
        return

    assert isinstance(cut_cond_node, f03.Else_If_Stmt)
    cut_cond, _ = cut_cond_node.children
    utils.remove_children(ib, ib.children[1:(cut + 1)])
    utils.set_children(ifthen, (cut_cond, ))
    _prune_branches_in_ifblock(ib, alias_map)


def _prune_branches_in_ifstmt(ib: f03.If_Stmt, alias_map: types.SPEC_TABLE):
    """Helper to prune an `If_Stmt` (a single-line IF statement)."""
    cond, actions = ib.children
    cval = analysis._const_eval_basic_type(cond, alias_map)
    if cval is None:
        return
    assert isinstance(cval, np.bool_)
    if cval:
        utils.replace_node(ib, actions)
    else:
        utils.remove_self(ib)
    expart = ib.parent
    if isinstance(expart, f03.Execution_Part) and not expart.children:
        utils.remove_self(expart)
