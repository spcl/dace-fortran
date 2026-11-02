# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""AST cleanup and canonicalization passes for the Fortran frontend, run after
parsing and desugaring: disambiguates language constructs, standardizes
identifier names, drops constructs irrelevant to dataflow analysis, and
restructures global data.
"""

from __future__ import annotations

from typing import Union

import fparser.two.Fortran2003 as f03
from fparser.api import get_reader
from fparser.two.utils import NumberBase, walk

from .. import ast_utils
from . import analysis, pruning, types, utils

#: GNU/g77-extension procedures gfortran provides but that are neither Fortran
#: intrinsics nor defined in any module -- bare external library calls (flang
#: doesn't implement them). Extend as other g77 externals are encountered.
EXTERNAL_BUILTIN_PROCEDURES = frozenset({"rand", "irand", "srand"})


def correct_for_function_calls(ast: f03.Program) -> f03.Program:
    """Disambiguates array accesses from function calls: `A(i)` and `F(x)` parse
    identically, and fparser defaults to `Part_Ref` (array access), so this uses
    type analysis to reclassify unresolved `Part_Ref`/`Structure_Constructor`
    nodes as `Function_Reference` (and intrinsics as `Intrinsic_Function_Reference`)."""
    alias_map = analysis.alias_specs(ast)

    # Statement functions parse as assignments to array-like expressions -- find them first.
    for asgn in walk(ast, f03.Assignment_Stmt):
        lv, _, _ = asgn.children
        if not isinstance(lv, (f03.Part_Ref, f03.Structure_Constructor, f03.Function_Reference)):
            continue
        if walk(lv, f03.Subscript_Triplet):
            # LHS with a subscript triplet is an array section, not a statement function.
            continue
        lv, _ = lv.children
        lvloc = analysis.search_real_local_alias_spec(lv, alias_map)
        if not lvloc:
            continue
        lv = alias_map[lvloc]
        if not isinstance(lv, f03.Entity_Decl):
            continue
        lv_type = analysis.find_type_of_entity(lv, alias_map)
        if not lv_type or lv_type.shape:
            # If it has a shape, it's an array, not a statement function.
            continue

        stmt_fn = f03.Stmt_Function_Stmt(asgn.tofortran())
        ex = asgn.parent
        # Walk up to the enclosing execution part.
        while not isinstance(ex, f03.Execution_Part):
            ex = ex.parent
        sp = ast_utils.atmost_one(ast_utils.children_of_type(ex.parent, f03.Specification_Part))
        assert sp
        utils.append_children(sp, stmt_fn)
        utils.remove_self(asgn)

    alias_map = analysis.alias_specs(ast)

    # Iteratively disambiguate Part_Ref nodes until fixed-point (fixing one may reveal another).
    # TODO: `Function_Reference(...)` sometimes generates inner `Part_Ref`s needing another pass -- avoid the reloop.
    changed = None
    while changed is None or changed:
        changed = False
        for pr in walk(ast, f03.Part_Ref):
            if isinstance(pr.parent, f03.Data_Ref):
                dref = pr.parent
                scope_spec = analysis.find_scope_spec(dref)
                comp_spec = analysis.find_dataref_component_spec(dref, scope_spec, alias_map)
                if comp_spec not in alias_map:
                    # Component of an externalised/unresolvable (MATCH_ALL) type -- genuine
                    # member access on an opaque type, not a disguised function call.
                    continue
                comp_type_spec = analysis.find_type_of_entity(alias_map[comp_spec], alias_map)
                if not comp_type_spec:
                    # If no type can be found for the component, it's likely a function call.
                    utils.replace_node(dref, f03.Function_Reference(dref.tofortran()))
                    changed = True
            else:
                pr_name, _ = pr.children
                if isinstance(pr_name, f03.Name):
                    pr_spec = analysis.search_real_local_alias_spec(pr_name, alias_map)
                    if pr_spec in alias_map and isinstance(alias_map[pr_spec], (f03.Function_Stmt, f03.Interface_Stmt)):
                        # If the name resolves to a function/interface, it's a function reference.
                        utils.replace_node(pr, f03.Function_Reference(pr.tofortran()))
                        changed = True
                elif isinstance(pr_name, f03.Data_Ref):
                    scope_spec = analysis.find_scope_spec(pr_name)
                    pr_type_spec = analysis.find_type_dataref(pr_name, scope_spec, alias_map)
                    if not pr_type_spec:
                        # If no type can be found for the data reference, it's likely a function call.
                        utils.replace_node(pr, f03.Function_Reference(pr.tofortran()))
                        changed = True

    # Disambiguate Structure_Constructor nodes that might actually be function calls.
    for sc in walk(ast, f03.Structure_Constructor):
        scope_spec = analysis.find_scope_spec(sc)

        # TODO: Add ref.
        sc_type, _ = sc.children
        sc_type_spec = analysis.search_real_ident_spec(sc_type.string, scope_spec, alias_map)
        if not sc_type_spec:
            if sc_type.string.lower() in EXTERNAL_BUILTIN_PROCEDURES:
                # Bare g77-extension call (e.g. `rand()`) mis-parsed as a structure
                # constructor, with no Fortran definition -- reclassify as a function reference.
                utils.replace_node(sc, f03.Function_Reference(sc.tofortran()))
                continue
            # Unresolved constructor/call name (e.g. ICON's external `p_mpi_wtime`) -- under
            # tolerance leave it for pruning to drop; strict mode asserts as before.
            if not analysis.OPTIONS.tolerate_external_uses:
                raise AssertionError(f"cannot find {sc_type.string} / {scope_spec}")
            continue
        sc_decl = alias_map[sc_type_spec]
        if isinstance(sc_decl, (f03.Function_Stmt, f03.Interface_Stmt, f03.Stmt_Function_Stmt)):
            # If the constructor's name resolves to a function, it's a function reference.
            utils.replace_node(sc, f03.Function_Reference(sc.tofortran()))

    # Convert generic Function_Reference/Call_Stmt to Intrinsic_Function_Reference where applicable.
    for fref in walk(ast, (f03.Function_Reference, f03.Call_Stmt)):
        scope_spec = analysis.find_scope_spec(fref)

        name, args = fref.children
        name = name.string
        if not f03.Intrinsic_Name.match(name):
            continue
        fref_spec = (*scope_spec, name)
        if fref_spec in alias_map:
            # Shadowed by a user-defined entity -- not an intrinsic call.
            continue
        if isinstance(fref, f03.Function_Reference):
            # Preserve args when swapping in the specific intrinsic node.
            repl = f03.Intrinsic_Function_Reference(fref.tofortran())
            utils.set_children(repl, (f03.Intrinsic_Name(name), args))
            utils.replace_node(fref, repl)
        else:
            utils.set_children(fref, (f03.Intrinsic_Name(name), args))

    return ast


def remove_access_and_bind_statements(ast: f03.Program) -> f03.Program:
    """Removes access-control (`PUBLIC`/`PRIVATE`/`Private_Components_Stmt`) and
    `BIND(C, ...)` statements -- irrelevant to dataflow analysis and codegen."""
    # TODO: This can get us into ambiguity and unintended shadowing.

    # Also removes access statements on interfaces.
    for acc in walk(ast, f03.Access_Stmt):
        # TODO: Add ref.
        kind, _alist = acc.children
        assert kind.upper() in {"PUBLIC", "PRIVATE"}
        spec = acc.parent
        utils.remove_self(acc)
        if not spec.children:
            # Remove the parent too if it's now empty.
            utils.remove_self(spec)

    for acc in walk(ast, f03.Private_Components_Stmt):
        utils.remove_self(acc)

    for bind in walk(ast, f03.Language_Binding_Spec):
        if isinstance(bind.parent, (f03.Suffix, f03.Subroutine_Stmt, f03.Function_Stmt)):
            # Part of a tuple -- replace with None to keep tuple structure.
            utils.replace_node(bind, None)
        else:
            par = bind.parent
            utils.remove_self(bind)
            if not par.children:
                # Replace empty parent with None.
                utils.replace_node(par, None)
    for bind in walk(ast, f03.Type_Attr_Spec):
        b, _c = bind.children
        if b == "BIND":
            par = bind.parent
            utils.remove_self(bind)
            if not par.children:
                # Replace empty parent with None.
                utils.replace_node(par, None)

    return ast


def assign_globally_unique_subprogram_names(ast: f03.Program, keepers: set[types.SPEC]) -> f03.Program:
    """Renames subprograms to be globally unique, avoiding cross-module name
    collisions and keyword clashes. `keepers` (entry points) retain their names."""
    SUFFIX, COUNTER = "fn", 0

    ident_map = analysis.identifier_specs(ast)
    alias_map = analysis.alias_specs(ast)

    # Find names that collide or clash with reserved keywords.
    known_names: set[str] = {k[-1] for k in ident_map}
    name_counts: dict[str, int] = dict.fromkeys(known_names, 0)
    for k in ident_map:
        name_counts[k[-1]] += 1
    name_collisions: set[str] = {k for k, v in name_counts.items() if v > 1 or k.lower() in KEYWORDS_TO_AVOID}

    # Assign fresh names to colliding/reserved subprograms.
    uident_map: dict[types.SPEC, str] = {}
    for k in ident_map:
        if k in keepers:
            continue
        if k[-1] in name_collisions:
            # Suffix + counter, bumping past any existing name.
            uname, COUNTER = f"{k[-1]}_{SUFFIX}_{COUNTER}", COUNTER + 1
            while uname in known_names:
                uname, COUNTER = f"{k[-1]}_{SUFFIX}_{COUNTER}", COUNTER + 1
        else:
            uname = k[-1]
        uident_map[k] = uname
    uident_map.update({k: k[-1] for k in keepers})

    # PHASE 1.a: remove imports of to-be-renamed functions (re-imported under the new name later).
    for use in walk(ast, f03.Use_Stmt):
        mod_name = ast_utils.singular(ast_utils.children_of_type(use, f03.Name)).string
        mod_spec = (mod_name,)
        olist = ast_utils.atmost_one(ast_utils.children_of_type(use, f03.Only_List))
        if not olist:
            continue
        survivors = []
        for c in olist.children:
            if isinstance(c, f03.Rename):
                # Renamed uses don't survive -- replaced by direct uses.
                continue
            assert isinstance(c, f03.Name)
            tgt_spec = analysis.find_real_ident_spec(c.string, mod_spec, alias_map)
            assert tgt_spec in ident_map and tgt_spec in uident_map
            if not isinstance(ident_map[tgt_spec], (f03.Function_Stmt, f03.Subroutine_Stmt)):
                # Leave non-function uses alone.
                survivors.append(c)
        if survivors:
            utils.set_children(olist, survivors)
        else:
            utils.remove_self(use)

    # PHASE 1.b: replace all the function callsites.
    for fref in walk(ast, (f03.Function_Reference, f03.Call_Stmt)):
        scope_spec = analysis.find_scope_spec(fref)

        # TODO: Add ref.
        name, _ = fref.children
        if not isinstance(name, f03.Name):
            # Intrinsics aren't renamed.
            assert isinstance(name, f03.Intrinsic_Name), f"{fref}"
            continue
        fspec = analysis.find_real_ident_spec(name.string, scope_spec, alias_map)
        assert fspec in ident_map and fspec in uident_map
        assert isinstance(ident_map[fspec], (f03.Function_Stmt, f03.Subroutine_Stmt))
        uname = uident_map[fspec]
        ufspec = (*fspec[:-1], uname)
        name.string = uname

        # Find the enclosing execution + specification parts.
        execution_part = fref.parent
        while not isinstance(execution_part, f03.Execution_Part):
            execution_part = execution_part.parent
        subprog = execution_part.parent
        specification_part = ast_utils.atmost_one(ast_utils.children_of_type(subprog, f03.Specification_Part))

        # Find the current module/program name (to decide if a USE is needed).
        cmod = fref.parent
        while cmod and not isinstance(cmod, (f03.Module, f03.Main_Program)):
            cmod = cmod.parent
        if cmod:
            stmt, _, _, _ = utils.get_module_or_program_parts(cmod)
            cmod = ast_utils.singular(ast_utils.children_of_type(stmt, f03.Name)).string.lower()
        else:
            # If not in a module/main program, it must be a nested subprogram.
            subp = list(ast_utils.children_of_type(ast, f03.Subroutine_Subprogram))
            assert subp
            stmt = ast_utils.singular(ast_utils.children_of_type(subp[0], f03.Subroutine_Stmt))
            cmod = ast_utils.singular(ast_utils.children_of_type(stmt, f03.Name)).string.lower()

        assert len(ufspec) >= 1
        if len(ufspec) == 1:
            # Toplevel subprograms are globally visible, no USE statement needed.
            continue
        mod = ufspec[0]
        if mod == cmod:
            # If the function is defined in the current module, no USE statement needed.
            continue

        # Add a USE for the renamed function (consolidated later).
        if not specification_part:
            utils.append_children(subprog, f03.Specification_Part(get_reader(f"use {mod}, only: {uname}")))
        else:
            utils.prepend_children(specification_part, f03.Use_Stmt(f"use {mod}, only: {uname}"))

    # PHASE 1.d: replaces actual function names in their definitions.
    for k, v in ident_map.items():
        if not isinstance(v, (f03.Function_Stmt, f03.Subroutine_Stmt)):
            continue
        assert k in uident_map
        if uident_map[k] == k[-1]:
            continue
        oname, uname = k[-1], uident_map[k]
        ast_utils.singular(ast_utils.children_of_type(v, f03.Name)).string = uname
        fdef = v.parent
        end_stmt = ast_utils.singular(
            ast_utils.children_of_type(fdef, (f03.End_Function_Stmt, f03.End_Subroutine_Stmt))
        )
        kw, _end_name = end_stmt.children
        utils.set_children(end_stmt, (kw, f03.Name(uname)))
        # Function names also work as an in-scope variable (the return value).
        if isinstance(v, f03.Function_Stmt):
            for nm in walk(
                tuple(ast_utils.children_of_type(fdef, (f03.Specification_Part, f03.Execution_Part))), f03.Name
            ):
                if nm.string != oname:
                    continue
                nm_spec = analysis.search_local_alias_spec(nm)
                assert nm_spec is not None, f"`{nm}` has no local alias spec"
                # Adjust local spec to the function's original spec (handles the name-as-return-var case).
                local_spec = nm_spec[:-2] + nm_spec[-1:]
                assert local_spec in ident_map, f"`{local_spec}` is not a valid identifier"
                assert ident_map[local_spec] is v, f"`{local_spec}` does not refer to `{v}`"
                nm.string = uname

    return ast


def add_use_to_specification(scdef: utils.SCOPE_OBJECT_TYPES, clause: str) -> None:
    """Prepends a `USE` statement (`clause`) to `scdef`'s specification part, creating one if absent."""
    specification_part = ast_utils.atmost_one(ast_utils.children_of_type(scdef, f03.Specification_Part))
    if not specification_part:
        utils.append_children(scdef, f03.Specification_Part(get_reader(clause)))
    else:
        utils.prepend_children(specification_part, f03.Use_Stmt(clause))


KEYWORDS_TO_AVOID = {k.lower() for k in ("for", "in", "beta", "input", "this")}


def assign_globally_unique_variable_names(ast: f03.Program, keepers: set[Union[str, types.SPEC]]) -> f03.Program:
    """Renames variables to be globally unique, avoiding cross-module collisions
    and keyword clashes. `keepers` (specs or bare names) retain their names,
    including entry-point arguments unless a kept name is itself a keyword."""
    SUFFIX, COUNTER = "var", 0

    ident_map = analysis.identifier_specs(ast)
    alias_map = analysis.alias_specs(ast)

    # Find names that collide or clash with reserved keywords.
    known_names: set[str] = {k[-1].lower() for k in ident_map}
    name_counts: dict[str, int] = dict.fromkeys(known_names, 0)
    for k in ident_map:
        name_counts[k[-1].lower()] += 1
    name_collisions: set[str] = {k for k, v in name_counts.items() if v > 1 or k in KEYWORDS_TO_AVOID}

    # Entry-point args may keep their original names.
    entry_point_args: set[types.SPEC] = set()
    for keeper in keepers:
        if not isinstance(keeper, tuple) or keeper not in ident_map:
            continue
        fn = ident_map[keeper]
        if not isinstance(fn, (f03.Subroutine_Stmt, f03.Function_Stmt)):
            continue
        args = ast_utils.atmost_one(ast_utils.children_of_type(fn, f03.Dummy_Arg_List))
        args = args.children if args else tuple()
        for a in args:
            entry_point_args.add((*keeper, a.string))

    # Assign fresh names to colliding/reserved/not-kept variables.
    uident_map: dict[types.SPEC, str] = {}
    for k in ident_map:
        if k[-1].lower() not in KEYWORDS_TO_AVOID and k in entry_point_args:
            # Keep entry-point args unless they clash with a keyword.
            continue
        if k in keepers:
            continue
        if k[-1] in keepers:
            # Bare variable name (anywhere) requested to keep.
            continue
        if k[-1].lower() in name_collisions:
            # Suffix + counter, bumping past any existing name.
            uname, COUNTER = f"{k[-1]}_{SUFFIX}_{COUNTER}", COUNTER + 1
            while uname in known_names:
                uname, COUNTER = f"{k[-1]}_{SUFFIX}_{COUNTER}", COUNTER + 1
        else:
            uname = k[-1]
        uident_map[k] = uname
    uident_map.update({k: k[-1] for k in keepers if isinstance(k, tuple)})

    # PHASE 1.a: remove imports of to-be-renamed variables (re-imported under the new name later).
    for use in walk(ast, f03.Use_Stmt):
        mod_name = ast_utils.singular(ast_utils.children_of_type(use, f03.Name)).string
        mod_spec = (mod_name,)
        olist = ast_utils.atmost_one(ast_utils.children_of_type(use, f03.Only_List))
        if not olist:
            continue
        survivors = []
        for c in olist.children:
            if isinstance(c, f03.Rename):
                # Renamed uses don't survive -- replaced by direct uses.
                continue
            assert isinstance(c, f03.Name)
            tgt_spec = analysis.find_real_ident_spec(c.string, mod_spec, alias_map)
            assert tgt_spec in ident_map and tgt_spec in uident_map
            if not isinstance(ident_map[tgt_spec], f03.Entity_Decl):
                # Leave non-variable uses alone.
                survivors.append(c)
        if survivors:
            utils.set_children(olist, survivors)
        else:
            utils.remove_self(use)

    # PHASE 1.b: replace variable names used as call keywords -- must run early to avoid ambiguity (e.g. `fn(kw=kw)`).
    for kv in walk(ast, f03.Actual_Arg_Spec):
        fref = kv.parent.parent
        if not isinstance(fref, (f03.Function_Reference, f03.Call_Stmt)):
            continue
        callee, _ = fref.children
        if isinstance(callee, f03.Intrinsic_Name):
            # Intrinsics don't have their internal variables renamed.
            continue
        callee_spec = analysis.search_real_local_alias_spec(callee, alias_map)
        assert callee_spec is not None, f"cannot resolve callee `{callee}`"
        cspec = analysis.ident_spec(alias_map[callee_spec])
        assert cspec
        k, _ = kv.children
        assert isinstance(k, f03.Name)
        kspec = analysis.find_real_ident_spec(k.string, cspec, alias_map)
        assert kspec in ident_map and kspec in uident_map
        assert isinstance(ident_map[kspec], f03.Entity_Decl)
        k.string = uident_map[kspec]

    # PHASE 1.c: replace all direct references to variables.
    for vref in walk(ast, f03.Name):
        if isinstance(vref.parent, f03.Entity_Decl):
            # Don't rename the declarations yet, only usages.
            continue
        vspec = analysis.search_real_local_alias_spec(vref, alias_map)
        if not vspec:
            # Not a valid alias (e.g., a structure component that is not a variable).
            continue
        if not isinstance(alias_map[vspec], f03.Entity_Decl):
            # Does not refer to a variable (e.g., a function name).
            continue
        edcl = alias_map[vspec]
        fdef = utils.find_scope_ancestor(edcl)
        if isinstance(fdef, f03.Function_Subprogram) and utils.find_name_of_node(fdef) == utils.find_name_of_node(edcl):
            # Function return variables must retain their names.
            continue

        scope_spec = analysis.find_scope_spec(vref)
        vspec = analysis.find_real_ident_spec(vspec[-1], scope_spec, alias_map)
        assert vspec in ident_map
        if vspec not in uident_map:
            # TODO: `vspec` **should** be in `uident_map` if it is a variable (whether we rename it or not).
            continue
        uname = uident_map[vspec]
        vref.string = uname

        # Global (module-level) var -- add a USE statement if needed.
        if len(vspec) > analysis.MODULE_MEMBER_DEPTH:
            # Not directly in a toplevel module -- nothing to import.
            continue
        mod, _ = vspec
        if not isinstance(alias_map[(mod,)], f03.Module_Stmt):
            # We can only import modules.
            continue

        # Find the nearest specification part (or lack thereof).
        scdef = alias_map[scope_spec].parent
        # Find out the current module name.
        cmod = scdef
        while not isinstance(cmod.parent, f03.Program):
            cmod = cmod.parent
        if utils.find_name_of_node(cmod) == mod:
            # If the variable is already defined in the current module, no import needed.
            continue
        add_use_to_specification(scdef, f"use {mod}, only: {uname}")

    # PHASE 1.d: replace variable names used as "kind" specifiers in literal constants.
    for lit in walk(ast, f03.Real_Literal_Constant):
        val, kind = lit.children
        if not kind:
            continue
        # Fparser sometimes parses kind as a plain `str` instead of a `Name`.
        assert isinstance(kind, str)
        scope_spec = analysis.find_scope_spec(lit)
        kind_spec = analysis.search_real_ident_spec(kind, scope_spec, alias_map)
        if not kind_spec or kind_spec not in uident_map:
            continue
        uname = uident_map[kind_spec]
        utils.set_children(lit, (val, uname))

        # Global kind var -- add a USE statement if needed.
        if len(kind_spec) > analysis.MODULE_MEMBER_DEPTH:
            # Not directly in a toplevel module -- nothing to import.
            continue
        mod, _ = kind_spec
        if not isinstance(alias_map[(mod,)], f03.Module_Stmt):
            # We can only import modules.
            continue

        # Find the nearest specification part (or lack thereof).
        scdef = alias_map[scope_spec].parent
        # Find out the current module name.
        cmod = scdef
        while not isinstance(cmod.parent, f03.Program):
            cmod = cmod.parent
        if utils.find_name_of_node(cmod) == mod:
            # If the variable is already defined in the current module, no import needed.
            continue
        add_use_to_specification(scdef, f"use {mod}, only: {uname}")

    # PHASE 1.e: replace variable names in their declarations.
    for k, v in ident_map.items():
        if not isinstance(v, f03.Entity_Decl):
            continue
        if k not in uident_map or uident_map[k] == k[-1]:
            # TODO: `k` **should** be in `uident_map` if it is a variable (whether we rename it or not).
            continue
        oname, uname = k[-1], uident_map[k]
        fdef = utils.find_scope_ancestor(v)
        if isinstance(fdef, f03.Function_Subprogram) and utils.find_name_of_node(fdef) == oname:
            # Function return variables must retain their names.
            continue
        ast_utils.singular(ast_utils.children_of_type(v, f03.Name)).string = uname

    return ast


def lower_identifier_names(ast: f03.Program) -> f03.Program:
    """Lower-cases all `Name` nodes and numeric-literal `KIND` specifiers (Fortran is case-insensitive)."""
    for nm in walk(ast, f03.Name):
        nm.string = nm.string.lower()
    # Kind specifiers in numeric literals too (e.g. 1.0_REAL8).
    for num in walk(ast, NumberBase):
        val, kind = num.children
        if isinstance(kind, str):
            utils.set_children(num, (val, kind.lower()))
    return ast


GLOBAL_DATA_OBJ_NAME = "global_data"
GLOBAL_DATA_TYPE_NAME = "global_data_type"


def consolidate_global_data_into_arg(ast: f03.Program, always_add_global_data_arg: bool = False) -> f03.Program:
    """Consolidates module-level global variables into one `global_data_type` derived
    type, passed as a `global_data` argument to every subprogram (and threaded through
    call sites), replacing direct global accesses with `global_data % var`. Makes
    global data dependencies explicit for dataflow analysis."""
    alias_map = analysis.alias_specs(ast)
    GLOBAL_DATA_MOD_NAME = "global_mod"
    if (GLOBAL_DATA_MOD_NAME,) in alias_map:
        # global_mod already exists -- nothing to do.
        return ast

    derived_type_lines: list[str] = []
    global_var_lines: list[str] = []
    # Collect derived types for the global module.
    for dt in walk(ast, f03.Derived_Type_Def):
        dtspec = analysis.ident_spec(ast_utils.singular(ast_utils.children_of_type(dt, f03.Derived_Type_Stmt)))
        mod, dtname = dtspec
        derived_type_lines.append(f"use {mod}, only : {dtname}")
    # Collect global vars for the global data structure.
    for m in walk(ast, f03.Module):
        spart = ast_utils.atmost_one(ast_utils.children_of_type(m, f03.Specification_Part))
        if not spart:
            continue
        for tdecl in ast_utils.children_of_type(spart, f03.Type_Declaration_Stmt):
            _typ, attr, _ = tdecl.children
            if "PARAMETER" in f"{attr}":
                # PARAMETERs are not global data: they and their references stay in place.
                continue
            global_var_lines.append(tdecl.tofortran())
    all_derived_types = "\n".join(derived_type_lines)
    all_global_vars = "\n".join(global_var_lines)

    # Replace references to global variables with data-refs.
    for nm in walk(ast, f03.Name):
        par = nm.parent
        if isinstance(par, (f03.Entity_Decl, f03.Use_Stmt, f03.Rename, f03.Only_List)):
            continue
        if isinstance(par, (f03.Part_Ref, f03.Data_Ref)):
            while par and isinstance(par.parent, (f03.Part_Ref, f03.Data_Ref)):
                par = par.parent
            scope_spec = analysis.search_scope_spec(par)
            assert scope_spec is not None, f"`{par}` has no scope spec"
            root, _, _ = analysis.dataref_root(par, scope_spec, alias_map)
            if root is not nm:
                continue
        local_spec = analysis.search_real_local_alias_spec(nm, alias_map)
        if not local_spec:
            continue
        assert local_spec in alias_map
        if not isinstance(alias_map[local_spec], f03.Entity_Decl):
            continue
        edecl_spec = analysis.ident_spec(alias_map[local_spec])
        assert len(edecl_spec) >= analysis.MODULE_MEMBER_DEPTH, (
            f"Fortran cannot possibly have a top-level global variable, outside any module; got {edecl_spec}"
        )
        if len(edecl_spec) != analysis.MODULE_MEMBER_DEPTH:
            # We cannot possibly have a module level variable declaration.
            continue
        mod, var = edecl_spec
        assert (mod,) in alias_map
        if not isinstance(alias_map[(mod,)], f03.Module_Stmt):
            continue
        if (typ := analysis.find_type_of_entity(alias_map[local_spec], alias_map)) and typ.const:
            continue
        if isinstance(nm.parent, f03.Part_Ref):
            _, subsc = nm.parent.children
            utils.replace_node(nm.parent, f03.Data_Ref(f"{GLOBAL_DATA_OBJ_NAME} % {var}({subsc})"))
        else:
            utils.replace_node(nm, f03.Data_Ref(f"{GLOBAL_DATA_OBJ_NAME} % {var}"))

    if all_global_vars or always_add_global_data_arg:
        # Make `global_data` an argument to every defined function.
        for fn in walk(ast, (f03.Function_Subprogram, f03.Subroutine_Subprogram)):
            stmt = ast_utils.singular(ast_utils.children_of_type(fn, utils.NAMED_STMTS_OF_INTEREST_CLASSES))
            assert isinstance(stmt, (f03.Function_Stmt, f03.Subroutine_Stmt))
            prefix, name, dummy_args, whatever = stmt.children
            if dummy_args:
                utils.prepend_children(dummy_args, f03.Name(GLOBAL_DATA_OBJ_NAME))
            else:
                utils.set_children(stmt, (prefix, name, f03.Dummy_Arg_Name_List(GLOBAL_DATA_OBJ_NAME), whatever))
            spart = ast_utils.atmost_one(ast_utils.children_of_type(fn, f03.Specification_Part))
            use_stmt = f"use {GLOBAL_DATA_MOD_NAME}, only : {GLOBAL_DATA_TYPE_NAME}"
            if spart:
                utils.prepend_children(spart, f03.Use_Stmt(use_stmt))
            else:
                utils.set_children(
                    fn, [*fn.children[:1], f03.Specification_Part(get_reader(use_stmt)), *fn.children[1:]]
                )
            spart = ast_utils.singular(ast_utils.children_of_type(fn, f03.Specification_Part))
            decl_idxs = [idx for idx, v in enumerate(spart.children) if isinstance(v, f03.Type_Declaration_Stmt)]
            decl_idx = decl_idxs[0] if decl_idxs else len(spart.children)
            utils.set_children(
                spart,
                [
                    *spart.children[:decl_idx],
                    f03.Type_Declaration_Stmt(f"type({GLOBAL_DATA_TYPE_NAME}) :: {GLOBAL_DATA_OBJ_NAME}"),
                    *spart.children[decl_idx:],
                ],
            )
        for fcall in walk(ast, (f03.Function_Reference, f03.Call_Stmt)):
            fn, args = fcall.children
            fnspec = analysis.search_real_local_alias_spec(fn, alias_map)
            if not fnspec:
                continue
            fnstmt = alias_map[fnspec]
            assert isinstance(fnstmt, (f03.Function_Stmt, f03.Subroutine_Stmt))
            if args:
                utils.prepend_children(args, f03.Name(GLOBAL_DATA_OBJ_NAME))
            else:
                utils.set_children(fcall, (fn, f03.Actual_Arg_Spec_List(GLOBAL_DATA_OBJ_NAME)))
        # NOTE: variables aren't removed here -- later pruning drops them.

    global_mod = f03.Module(
        get_reader(f"""
module {GLOBAL_DATA_MOD_NAME}
  {all_derived_types}

  type {GLOBAL_DATA_TYPE_NAME}
    {all_global_vars}
  end type {GLOBAL_DATA_TYPE_NAME}
end module {GLOBAL_DATA_MOD_NAME}
""")
    )
    utils.prepend_children(ast, global_mod)

    return pruning.consolidate_uses(ast)


def rename_clashing_specifics(ast: f03.Program, renames: dict[str, str]) -> int:
    """Renames a specific procedure that shares its name with its own generic
    interface (ICON's ``mo_mpi``: ``INTERFACE p_wait`` has a specific also named
    ``p_wait``) -- breaks once the inliner renames the generic's specifics, since
    the bare name turns ambiguous and ``USE ... => p_wait`` dangles. ``renames``
    maps ``old -> new``: only the specific's definition (+ ``MODULE PROCEDURE``
    entry) is renamed when ``old`` names both; the generic and call sites stay
    untouched. Non-colliding names (e.g. external ``mpi_*``) are skipped;
    returns the rename count."""
    renamed = 0
    for old, new in renames.items():
        old = old.lower()

        def _generic_name(ib: f03.Interface_Block) -> str | None:
            st = ast_utils.atmost_one(ast_utils.children_of_type(ib, f03.Interface_Stmt))
            head = st.children[0] if st is not None else None
            return str(head).lower() if isinstance(head, f03.Name) else None

        generics = [ib for ib in walk(ast, f03.Interface_Block) if _generic_name(ib) == old]
        specifics = [
            sp
            for sp in walk(ast, (f03.Subroutine_Subprogram, f03.Function_Subprogram))
            if (utils.find_name_of_node(sp) or "").lower() == old
        ]
        if not (generics and specifics):
            continue  # not a generic/specific collision defined in this unit -- leave it

        for sp in specifics:
            for stmt in walk(sp, (f03.Subroutine_Stmt, f03.Function_Stmt)):
                utils.replace_node(stmt.children[1], f03.Name(new))
            for endst in walk(sp, (f03.End_Subroutine_Stmt, f03.End_Function_Stmt)):
                if endst.children[1] is not None:
                    utils.replace_node(endst.children[1], f03.Name(new))
            renamed += 1
        for ib in generics:
            for ps in walk(ib, f03.Procedure_Stmt):
                for nm in walk(ps.children[0], f03.Name):
                    if str(nm).lower() == old:
                        utils.replace_node(nm, f03.Name(new))
    return renamed
