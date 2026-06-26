# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Supportability analysis for static-vtable monomorphisation of polymorphic
type-bound-procedure (TBP) dispatch -- the ICON-O ocean-solver pattern.

The monomorphisation pass replaces a runtime virtual dispatch (``this%act%solve``
on a ``CLASS(abstract)`` entity, which flang lowers to ``fir.dispatch`` and the
dace-fortran bridge rejects) with a static ``if`` ladder over the *closed set* of
concrete subtypes registered to the abstract base -- one arm per subtype, each a
direct call.  That rewrite is only sound for a restricted class of hierarchies.

This module is the front-end that *recognises* that class and produces the
:class:`MonomorphizationPlan` (the closed arm set the rewrite will emit), or
rejects the program with :class:`UnsupportedProgram` carrying a precise reason --
so an out-of-scope program fails loudly and detectably instead of being
mis-compiled.  The accepted class (locked design):

  * single-level inheritance only -- every concrete subtype ``EXTENDS`` the
    abstract base *directly* (no chains, no intermediate abstract layers).
    Fortran has no multiple inheritance, so diamonds are impossible by language.
  * no unlimited polymorphism (``CLASS(*)``) -- there is no closed subtype set.
  * every ``DEFERRED`` binding of the base is overridden by every concrete arm.
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
    """A program (or type hierarchy) outside the class the monomorphisation pass
    can soundly rewrite.

    ``reason`` is a human-readable explanation of which restriction was violated,
    so callers can detect the rejection and surface *why* rather than failing with
    an opaque crash deeper in the pipeline.
    """

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

    With ``scopes`` given, only ``CLASS(*)`` *within those nodes* is rejected
    (the monomorphised axis's base + arm type definitions). A large extraction
    closure routinely pulls in unrelated generic containers -- a hash table /
    key-value store keyed on ``CLASS(*)``, the comm-pattern infra -- that are
    nowhere near the laddered axes and are externalised downstream; rejecting on
    those is a false positive. Without ``scopes``, the whole program is scanned
    (the strict, axis-agnostic check the standalone analysis tests rely on)."""
    roots = scopes if scopes is not None else [ast]
    for root in roots:
        for spec in walk(root, f03.Declaration_Type_Spec):
            if str(spec).replace(" ", "").upper() == "CLASS(*)":
                raise UnsupportedProgram("unlimited polymorphic `CLASS(*)` has no closed set of concrete "
                                         "subtypes to enumerate into dispatch arms")


def analyze(ast: f03.Program, only_bases: Optional[Iterable[str]] = None) -> List[MonomorphizationPlan]:
    """Return one :class:`MonomorphizationPlan` per monomorphisable abstract base,
    or raise :class:`UnsupportedProgram` if the program uses polymorphism in a
    shape the pass cannot soundly rewrite.

    A program with no abstract dispatch yields ``[]`` (nothing to do -- not an
    error).

    ``only_bases`` restricts the analysis to the named abstract bases (the spec's
    ladder axes) -- their plans are built and their hierarchies validated, and
    the ``CLASS(*)`` rejection is scoped to *their* type definitions. This is what
    :func:`monomorphize` passes: a large extraction closure pulls in unrelated
    dispatch roots and generic ``CLASS(*)`` containers that the spec never
    rewrites, and validating those would reject the whole program for a hierarchy
    the pass never touches. With ``only_bases`` ``None`` the whole program is
    validated (the axis-agnostic mode the standalone analysis tests use)."""
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
            # Scoped CLASS(*) check: the ladder structurally depends only on this
            # base and its concrete arm type definitions, so an unsound CLASS(*)
            # would have to live in one of those. CLASS(*) elsewhere in the
            # closure is untouched by the rewrite (and externalised downstream).
            reject_unlimited_polymorphic(ast, scopes=[dtds[name]] + [dtds[kid.name] for kid in kids])

        plans.append(MonomorphizationPlan(name, sorted(base.deferred), arms))

    return plans


def analyze_source(source: str) -> List[MonomorphizationPlan]:
    """Parse + analyse one Fortran source string (convenience for tests/specs)."""
    return analyze(parse_program(source))
