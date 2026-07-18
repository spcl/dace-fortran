# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Supportability analysis for static-vtable monomorphisation of polymorphic TBP
dispatch (ICON-O ocean-solver pattern): recognises programs where a runtime
``this%act%solve`` dispatch (flang lowers to ``fir.dispatch``, rejected by the
bridge) can be soundly replaced by a static ``if`` ladder over the closed set of
concrete subtypes, and produces :class:`MonomorphizationPlan`; else rejects via
:class:`UnsupportedProgram` (fail loudly instead of mis-compiling).

Locked accepted class: single-level inheritance only (no chains/diamonds --
Fortran has no multiple inheritance); no ``CLASS(*)`` (no closed subtype set);
every ``DEFERRED`` binding overridden by every concrete arm.
"""
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import fparser.two.Fortran2003 as f03
from fparser.api import get_reader
from fparser.two.parser import ParserFactory
from fparser.two.utils import walk

from dace_fortran.inliner.ast_desugaring.utils import find_name_of_stmt

EXTENDS_RE = re.compile(r"EXTENDS\s*\(\s*(\w+)\s*\)", re.IGNORECASE)


class UnsupportedProgram(Exception):
    """Program/hierarchy outside the class monomorphisation can soundly rewrite.

    ``reason`` is a human-readable explanation of which restriction was violated."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class ConcreteArm:
    """One concrete subtype registered to an abstract base -- one arm of the
    generated static-dispatch ladder."""
    type_name: str
    #: deferred binding name -> the concrete procedure this subtype binds it to
    bindings: Dict[str, str]


@dataclass
class MonomorphizationPlan:
    """The closed set of concrete arms for one abstract base.  The rewrite emits
    every arm (emit-all-always: no collapse), one static call each."""
    abstract_base: str
    deferred: List[str]
    arms: List[ConcreteArm]


@dataclass
class TypeInfo:
    name: str
    abstract: bool
    parent: Optional[str]
    #: deferred binding name -> interface name (target is None for deferred)
    deferred: Dict[str, Optional[str]]
    #: overriding binding name -> concrete procedure name (``proc :: b => p``)
    overrides: Dict[str, str]


def parse_program(source: str) -> f03.Program:
    """Parse one Fortran source string into an fparser2 AST (f2008), matching the
    inliner's own parser configuration."""
    parser = ParserFactory().create(std="f2008")
    return parser(get_reader(source))


def read_type_info(dtd: f03.Derived_Type_Def) -> TypeInfo:
    """Extract the name / abstract flag / parent / deferred + overriding bindings
    of one ``Derived_Type_Def``."""
    stmt = walk(dtd, f03.Derived_Type_Stmt)[0]
    text = str(stmt)
    head, _, tail = text.partition("::")
    name = (tail.strip() or head.strip()).split("(")[0].split()[-1].lower()
    abstract = "ABSTRACT" in head.upper()
    extends = EXTENDS_RE.search(head)
    parent = extends.group(1).lower() if extends else None

    deferred: Dict[str, Optional[str]] = {}
    overrides: Dict[str, str] = {}
    for binding in walk(dtd, f03.Specific_Binding):
        btext = str(binding)
        attrs, _, after = btext.partition("::")
        bname = find_name_of_stmt(binding)
        bname = (str(bname).lower() if bname is not None else after.split("=>")[0].strip().lower())
        if "DEFERRED" in attrs.upper():
            iface = walk(binding, f03.Name)
            deferred[bname] = str(iface[0]).lower() if iface else None
        elif "=>" in after:
            overrides[bname] = after.split("=>", 1)[1].strip().lower()
    return TypeInfo(name, abstract, parent, deferred, overrides)


def reject_unlimited_polymorphic(ast: f03.Program, scopes: Optional[List[f03.Base]] = None) -> None:
    """Reject ``CLASS(*)`` -- it has no closed subtype set to ladder over.

    ``scopes`` restricts the check to those nodes (avoids false positives from
    unrelated ``CLASS(*)`` containers pulled in by a large extraction closure);
    without it the whole program is scanned."""
    roots = scopes if scopes is not None else [ast]
    for root in roots:
        for spec in walk(root, f03.Declaration_Type_Spec):
            if str(spec).replace(" ", "").upper() == "CLASS(*)":
                raise UnsupportedProgram("unlimited polymorphic `CLASS(*)` has no closed set of concrete "
                                         "subtypes to enumerate into dispatch arms")


def analyze(ast: f03.Program, only_bases: Optional[Iterable[str]] = None) -> List[MonomorphizationPlan]:
    """Return one :class:`MonomorphizationPlan` per monomorphisable abstract base
    (``[]`` if none), or raise :class:`UnsupportedProgram` if a hierarchy cannot
    be soundly rewritten.

    ``only_bases`` restricts analysis + validation (incl. ``CLASS(*)`` rejection)
    to those bases -- what :func:`monomorphize` passes, so an unrelated dispatch
    root or ``CLASS(*)`` container pulled in by a large extraction closure isn't
    wrongly rejected. ``None`` validates the whole program."""
    only = {b.lower() for b in only_bases} if only_bases is not None else None
    if only is None:
        reject_unlimited_polymorphic(ast)

    dtds = {read_type_info(d).name: d for d in walk(ast, f03.Derived_Type_Def)}
    types = {name: read_type_info(d) for name, d in dtds.items()}
    children: Dict[str, List[TypeInfo]] = {}
    for ti in types.values():
        if ti.parent is not None:
            children.setdefault(ti.parent, []).append(ti)

    plans: List[MonomorphizationPlan] = []
    for name, base in types.items():
        if not base.deferred:
            continue  # not a dispatch root -- nothing deferred to resolve
        if only is not None and name not in only:
            continue  # a dispatch root the spec does not monomorphise -- leave it

        # single-level guard: the abstract base must be a root.
        if base.parent is not None:
            raise UnsupportedProgram(f"abstract base `{name}` itself extends `{base.parent}`: inheritance "
                                     f"depth > 1 is not supported (single-level hierarchies only)")

        kids = children.get(name, [])
        for kid in kids:
            if children.get(kid.name):
                grand = sorted(g.name for g in children[kid.name])
                raise UnsupportedProgram(f"subtype `{kid.name}` of `{name}` is itself extended by {grand}: "
                                         f"inheritance depth > 1 is not supported (single-level only)")
            if kid.abstract:
                raise UnsupportedProgram(f"subtype `{kid.name}` of `{name}` is itself abstract: only "
                                         f"concrete leaf subtypes can become dispatch arms")

        if not kids:
            raise UnsupportedProgram(f"abstract base `{name}` has no concrete subtype in the translation "
                                     f"unit: there is nothing to dispatch to")

        arms: List[ConcreteArm] = []
        for kid in kids:
            missing = sorted(d for d in base.deferred if d not in kid.overrides)
            if missing:
                raise UnsupportedProgram(f"concrete subtype `{kid.name}` does not override deferred "
                                         f"binding(s) {missing} of `{name}`")
            arms.append(ConcreteArm(kid.name, {d: kid.overrides[d] for d in base.deferred}))

        if only is not None:
            # Unsound CLASS(*) can only live in this base's/arms' defs -- elsewhere is untouched by the rewrite.
            reject_unlimited_polymorphic(ast, scopes=[dtds[name]] + [dtds[kid.name] for kid in kids])

        plans.append(MonomorphizationPlan(name, sorted(base.deferred), arms))

    return plans


def analyze_source(source: str) -> List[MonomorphizationPlan]:
    """Parse + analyse one Fortran source string (convenience for tests/specs)."""
    return analyze(parse_program(source))
