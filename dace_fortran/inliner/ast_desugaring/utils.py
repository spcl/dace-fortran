# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

from copy import deepcopy
from typing import Union, Tuple, Optional, List, Iterable

from fparser.api import get_reader
import fparser.two.Fortran2003 as f03
from fparser.two.utils import Base, BlockBase

from .. import ast_utils

# Type Aliases for common node groupings
# Represents program entry points like the main program, subroutines, and functions.
ENTRY_POINT_OBJECT_TYPES = Union[f03.Main_Program, f03.Subroutine_Subprogram, f03.Function_Subprogram]
ENTRY_POINT_OBJECT_CLASSES = (f03.Main_Program, f03.Subroutine_Subprogram, f03.Function_Subprogram)
# Represents nodes that define a new scope (e.g., modules, functions, derived types).
SCOPE_OBJECT_TYPES = Union[f03.Main_Program, f03.Module, f03.Function_Subprogram, f03.Subroutine_Subprogram,
                           f03.Derived_Type_Def, f03.Interface_Block, f03.Subroutine_Body, f03.Function_Body,
                           f03.Stmt_Function_Stmt]
SCOPE_OBJECT_CLASSES = (f03.Main_Program, f03.Module, f03.Function_Subprogram, f03.Subroutine_Subprogram,
                        f03.Derived_Type_Def, f03.Interface_Block, f03.Subroutine_Body, f03.Function_Body,
                        f03.Stmt_Function_Stmt)
# Represents statements that have a name and are of interest for analysis.
NAMED_STMTS_OF_INTEREST_TYPES = Union[f03.Program_Stmt, f03.Module_Stmt, f03.Function_Stmt, f03.Subroutine_Stmt,
                                      f03.Derived_Type_Stmt, f03.Component_Decl, f03.Entity_Decl, f03.Specific_Binding,
                                      f03.Generic_Binding, f03.Interface_Stmt, f03.Stmt_Function_Stmt,
                                      f03.Proc_Component_Def_Stmt, f03.Proc_Decl]
NAMED_STMTS_OF_INTEREST_CLASSES = (f03.Program_Stmt, f03.Module_Stmt, f03.Function_Stmt, f03.Subroutine_Stmt,
                                   f03.Derived_Type_Stmt, f03.Component_Decl, f03.Entity_Decl, f03.Specific_Binding,
                                   f03.Generic_Binding, f03.Interface_Stmt, f03.Stmt_Function_Stmt,
                                   f03.Proc_Component_Def_Stmt, f03.Proc_Decl)


def find_name_of_stmt(node: NAMED_STMTS_OF_INTEREST_TYPES) -> Optional[str]:
    """Name of a statement node, or None for anonymous blocks."""
    if isinstance(node, f03.Specific_Binding):
        # Ref: https://github.com/stfc/fparser/blob/8c870f84edbf1a24dfbc886e2f7226d1b158d50b/src/fparser/two/Fortran2003.py#L2504
        _, _, _, bname, _ = node.children
        name = bname
    elif isinstance(node, f03.Generic_Binding):
        _, bname, _ = node.children
        name = bname
    elif isinstance(node, f03.Interface_Stmt):
        name, = node.children
        if name == 'ABSTRACT':
            return None
    elif isinstance(node, f03.Proc_Component_Def_Stmt):
        tgt, attrs, plist = node.children
        assert len(plist.children) == 1, \
            f"Only one procedure per statement is accepted due to Fparser bug. Break down the line: {node}"
        # Name comes from the proc-decl list, not ``tgt``: for ``procedure(fun), pointer :: nofun``,
        # ``tgt`` is ``fun``, which would mis-name the ``nofun`` component.
        decl = plist.children[0]
        name = decl if isinstance(decl, f03.Name) else ast_utils.singular(ast_utils.children_of_type(decl, f03.Name))
    else:
        # TODO: Test out other type specific ways of finding names.
        name = ast_utils.singular(ast_utils.children_of_type(node, f03.Name))
    if name:
        name = f"{name}"
    return name


def find_name_of_node(node: Base) -> Optional[str]:
    """Name of a node's contained named statement, or None."""
    if isinstance(node, NAMED_STMTS_OF_INTEREST_CLASSES):
        return find_name_of_stmt(node)
    stmt = ast_utils.atmost_one(ast_utils.children_of_type(node, NAMED_STMTS_OF_INTEREST_CLASSES))
    if not stmt:
        return None
    return find_name_of_stmt(stmt)


def find_scope_ancestor(node: Base) -> Optional[SCOPE_OBJECT_TYPES]:
    """Nearest ancestor node that defines a scope, or None."""
    anc = node.parent
    while anc and not isinstance(anc, SCOPE_OBJECT_CLASSES):
        anc = anc.parent
    return anc


def find_named_ancestor(node: Base) -> Optional[NAMED_STMTS_OF_INTEREST_TYPES]:
    """Nearest named-statement-of-interest ancestor, or None."""
    anc = find_scope_ancestor(node)
    if not anc:
        return None
    return ast_utils.atmost_one(ast_utils.children_of_type(anc, NAMED_STMTS_OF_INTEREST_CLASSES))


def lineage(anc: Base, des: Base) -> Optional[Tuple[Base, ...]]:
    """Path from anc to des, or None if des is not a descendant of anc."""
    if anc is des:
        return (anc, )
    if not des.parent:
        return None
    lin = lineage(anc, des.parent)
    if not lin:
        return None
    return lin + (des, )


def _reparent_children(node: Base):
    """Fixes up `parent` pointers on all children to point back to `node`."""
    for c in node.children:
        if isinstance(c, Base):
            c.parent = node


def set_children(par: Base, children: Iterable[Union[Base, str]]):
    """Replaces `par`'s children, handling both `.items`- and `.content`-based nodes."""
    assert hasattr(par, 'content') != hasattr(par, 'items')
    if hasattr(par, 'items'):
        par.items = tuple(children)
    elif hasattr(par, 'content'):
        if not children:
            remove_self(par)
        else:
            par.content = list(children)
    if children:
        _reparent_children(par)


def remove_self(nodes: Union[Base, List[Base]]):
    """Removes one or more nodes from their parent's children."""
    if isinstance(nodes, Base):
        nodes = [nodes]
    for n in nodes:
        remove_children(n.parent, n)


def replace_node(node: Base, subst: Union[None, Base, Iterable[Base]]):
    """Replaces `node` with `subst` (None deletes it; can be a single node or iterable)."""
    # Ensure substituted nodes aren't the same object reused at multiple sites.
    par = node.parent
    repls = []
    for c in par.children:
        if c is not node:
            repls.append(c)
            continue
        if subst is None or isinstance(subst, Base):
            subst = [subst]
        repls.extend(subst)
    if isinstance(par, f03.Loop_Control) and isinstance(subst, Base):
        _, cntexpr, _, _ = par.children
        if cntexpr:
            loopvar, looprange = cntexpr
            for i in range(len(looprange)):
                if looprange[i] is node:
                    looprange[i] = subst
                    subst.parent = par
    set_children(par, repls)


def append_children(par: Base, children: Union[Base, List[Base]]):
    """Appends one or more children (a single node or a list) to `par`."""
    if isinstance(children, Base):
        children = [children]
    set_children(par, list(par.children) + children)


def prepend_children(par: Base, children: Union[Base, List[Base]]):
    """Prepends one or more children (a single node or a list) to `par`."""
    if isinstance(children, Base):
        children = [children]
    set_children(par, children + list(par.children))


def remove_children(par: Base, children: Union[Base, List[Base]]):
    """Removes specific children (a single node or a list) from `par`."""
    if isinstance(children, Base):
        children = [children]
    cids = {id(c) for c in children}
    repl = [c for c in par.children if id(c) not in cids]
    set_children(par, repl)


def copy_fparser_node(n: Base) -> Base:
    """Copies a node by re-parsing its Fortran text; falls back to deepcopy on failure."""
    try:
        nstr = n.tofortran()
        if isinstance(n, BlockBase):
            x = Base.__new__(type(n), get_reader(nstr))
        else:
            x = Base.__new__(type(n), nstr)
        assert x is not None
        return x
    except (RuntimeError, AssertionError):
        return deepcopy(n)


def _get_module_or_program_parts(mod: Union[f03.Module, f03.Main_Program]) \
        -> Tuple[
            Union[f03.Module_Stmt, f03.Program_Stmt],
            Optional[f03.Specification_Part],
            Optional[f03.Execution_Part],
            Optional[f03.Module_Subprogram_Part],
        ]:
    """Splits a Module/Main_Program node into (stmt, spec_part, exec_part, subprogram_part)."""
    # A module/program statement must exist.
    stmt = ast_utils.singular(
        ast_utils.children_of_type(mod, f03.Module_Stmt if isinstance(mod, f03.Module) else f03.Program_Stmt))
    spec = list(ast_utils.children_of_type(mod, f03.Specification_Part))
    assert len(spec) <= 1, f"A module/program cannot have more than one specification parts, found {spec} in {mod}"
    spec = spec[0] if spec else None
    expart = list(ast_utils.children_of_type(mod, f03.Execution_Part))
    assert len(expart) <= 1, f"A module/program cannot have more than one execution parts, found {spec} in {mod}"
    expart = expart[0] if expart else None
    subp = list(ast_utils.children_of_type(mod, f03.Module_Subprogram_Part))
    assert len(subp) <= 1, f"A module/program cannot have more than one subprogram parts, found {subp} in {mod}"
    subp = subp[0] if subp else None
    return stmt, spec, expart, subp
