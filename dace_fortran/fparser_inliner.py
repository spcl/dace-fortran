# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""fparser-based single-TU module inliner (opt-in alternative to the
regex ``merge_used_modules`` text-splicer).

This is a faithful port of the upstream DaCe Fortran-frontend tooling
(``create_preprocessed_ast.py`` + the ``ast_desugaring`` package),
restricted to its **source-text** product: it parses a multi-file
Fortran project into one combined fparser AST, resolves ``USE``
statements and inlines the needed modules, prunes everything not
reachable from the requested entry point, runs the desugaring /
optimization pipeline, and serialises the result back to a single
self-contained ``.f90`` file.

Upstream entry-point map (what the windmill tooling produces):

- ``tools/create_preprocessed_ast.py`` -> a preprocessed single
  ``.f90`` written from ``run_fparser_transformations(...).tofortran()``.
- ``tools/create_singular_sdfg_from_ast.py`` -> an SDFG (NOT ported
  here: that is the windmill SDFG-construction path, out of scope for a
  source-text single-TU producer; dace-fortran has its own HLFIR
  builder for the SDFG step).

The heavy ``fortran_parser.py`` SDFG translator and ``fix_utils.py``
(SDFG post-processing) are deliberately *not* vendored -- they are not
on the ``inline_to_single_tu`` path.  Only ``ParseConfig``,
``create_fparser_ast``, ``construct_full_ast``,
``run_fparser_transformations`` (and the small parsing helpers) are
ported, plus the ``ast_desugaring`` package they depend on.

Public API
----------
``inline_to_single_tu(sources, entry, ...) -> Path``  -- the headline
entry point: emit ONE combined ``.f90`` and return its path.  It also
exposes the combined fparser AST via ``inline_to_ast(...)`` for callers
that want to inspect / further-transform the tree before serialisation.
"""
import argparse
import logging
import re
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import fparser.two.Fortran2003 as f03
from fparser.api import get_reader
from fparser.two.C99Preprocessor import CPP_CLASS_NAMES
from fparser.two.parser import ParserFactory
from fparser.two.utils import Base, FortranSyntaxError, walk

from dace_fortran.external_functions import ExternalFunction, dont_inline_names, validate
from dace_fortran.inliner.ast_desugaring import (analysis, cleanup, desugaring, optimizations, pruning,
                                                 specialize_at_source as specialize_at_source_mod, types, utils)
from dace_fortran.inliner.ast_desugaring.monomorphize_rewrite import monomorphize_auto
from dace_fortran.inliner.ast_utils import atmost_one, children_of_type, singular

logger = logging.getLogger(__name__)

#: Stub implementations of the standard intrinsic modules so ``USE
#: iso_c_binding`` / ``USE iso_fortran_env`` resolve during parsing even
#: when the full standard-library sources are not on the search path.
#: Ported verbatim from ``tools/create_preprocessed_ast.py``.
BUILTINS = """
! This file provides stub implementations for standard Fortran intrinsic modules.
! These are included during preprocessing to ensure that `USE` statements for
! these common modules can be resolved by the parser, even if the full
! standard library implementations are not available.

module iso_c_binding
  integer, parameter :: c_int8_t = 1, c_int16_t = 2, c_int32_t = 4, c_int64_t = 8
  integer, parameter :: c_char = c_int8_t, c_signed_char = c_char, c_bool = c_int8_t, c_int = c_int32_t, c_long = c_int, c_size_t = c_int64_t
  integer, parameter :: c_float = 4, c_double = 8

  type c_ptr
  end type c_ptr

  type c_funptr
  end type c_funptr

  type(c_ptr), parameter :: c_null_ptr = c_ptr()
  character(kind=c_char), parameter :: c_null_char = char(0)

  interface c_f_pointer
    module procedure :: cfp_logical_r3
  end interface c_f_pointer

  interface c_f_procpointer
  end interface c_f_procpointer

  interface c_loc
  end interface c_loc

  interface c_associated
    module procedure :: cass_cptr
  end interface c_associated
contains
  subroutine cfp_logical_r3(cptr, fptr, shape, lower)
    type(c_ptr), intent(in) :: cptr
    logical, pointer, intent(out) :: fptr(:, :, :)
    integer, optional :: shape(:)
    integer, optional :: lower(:)
  end subroutine cfp_logical_r3

  logical function cass_cptr(a, b)
    type(c_ptr), intent(in) :: a
    type(c_ptr), optional, intent(in) :: b
  end function cass_cptr
end module iso_c_binding

module iso_fortran_env
  integer, parameter :: real32 = 4
  integer, parameter :: real64 = 8
  integer, parameter :: int32 = 4
  integer, parameter :: int64 = 8
  integer, parameter :: error_unit = 0
  integer, parameter :: output_unit = 6
  character, parameter :: compiler_version = "", compiler_options = ""
end module iso_fortran_env
"""

#: Names of the intrinsic-module stubs ``BUILTINS`` defines.  These are
#: injected only so the inliner's fparser parse resolves ``USE iso_c_binding``
#: / ``USE iso_fortran_env``; the real modules are compiler-provided, so any
#: stub that survives pruning (e.g. when nothing prunes it because no entry
#: point is given) must be dropped before the text is handed to flang -- a
#: stub definition would otherwise collide with the compiler's own.
BUILTIN_STUB_MODULE_NAMES = frozenset({"iso_c_binding", "iso_fortran_env"})

#: Compiler-provided intrinsic modules.  Their ``USE`` statements must survive
#: the merge so the serialised Fortran compiles (flang supplies the real module),
#: but the pipeline strips them as "resolved internally" -- and the symbols they
#: export (``c_double_complex``, ``c_ptr``, kind parameters, ...) are not module
#: procedures, so ``restore_cross_module_uses`` does not bring them back.  We
#: instead capture these ``USE``s before the pipeline and restore them after.
#: They are otherwise irrelevant to the SDFG dace emits (kinds resolve to plain
#: integers downstream), so a verbatim pass-through is exactly right.
INTRINSIC_MODULE_NAMES = frozenset({
    "iso_c_binding", "iso_fortran_env", "ieee_arithmetic", "ieee_exceptions", "ieee_features", "omp_lib",
    "omp_lib_kinds", "openacc", "mpi", "mpi_f08"
})

#: The subset of the above that are EXTERNAL LIBRARIES (MPI / OpenMP / OpenACC)
#: rather than Fortran-standard intrinsic modules.  flang supplies them when it
#: compiles, so they are preserved by default -- but under
#: ``tolerate_external_uses`` (a self-contained kernel extraction) their ``USE``s
#: are DROPPED like any other unresolved external library: the kernel
#: externalises all communication / synchronisation, the directives are
#: pre-stripped (``-U_OPENMP -U_OPENACC``), and no ``.mod`` is on a downstream
#: gfortran's path.  This is exactly the ``netcdf`` / ``mpi`` / ``cdi`` set the
#: tolerate mode is documented to ingest-then-drop.
EXTERNAL_LIBRARY_MODULE_NAMES = frozenset({"omp_lib", "omp_lib_kinds", "openacc", "mpi", "mpi_f08"})


def _preserved_intrinsic_modules(ast: Optional[f03.Program] = None) -> frozenset:
    """The compiler-provided modules whose ``USE`` is captured before the
    pipeline and restored after.  Under ``tolerate_external_uses`` the external
    -library group (:data:`EXTERNAL_LIBRARY_MODULE_NAMES`) is excluded so those
    ``USE``s are dropped, not resurrected -- see that constant's note.

    EXCEPTION: an external-library module that is ACTUALLY PROVIDED as a stub
    ``MODULE`` in the merged closure is NOT external here -- the inlined halo
    mode injects a real ``module mpi`` (see ``tests/icon/_halo_modes._MPI_STUB``),
    so gfortran gets its ``.mod`` from that stub and ``USE mpi`` must SURVIVE:
    ``mo_mpi`` resolves the raw ``mpi_recv`` / ``mpi_irecv`` / ``mpi_send`` /
    ``mpi_isend`` calls against the stub's assumed-type interfaces instead of
    leaving them implicit externals (which forces the unsound
    ``-fallow-argument-mismatch`` on the reference build).  In the external halo
    mode no such stub is provided and ``mo_mpi`` is not inlined, so the group is
    still dropped as before.  Pass ``ast`` to enable the provided-module check."""
    if not analysis.TOLERATE_EXTERNAL_USES:
        return INTRINSIC_MODULE_NAMES
    preserved = INTRINSIC_MODULE_NAMES - EXTERNAL_LIBRARY_MODULE_NAMES
    if ast is not None:
        provided = set()
        for ms in walk(ast, f03.Module_Stmt):
            nm = next(iter(children_of_type(ms, f03.Name)), None)
            if nm is not None:
                provided.add(nm.string.lower())
        preserved = preserved | (EXTERNAL_LIBRARY_MODULE_NAMES & provided)
    return frozenset(preserved)


def strip_builtin_stub_modules(ast: f03.Program) -> f03.Program:
    """Remove the injected intrinsic-module stubs (see ``BUILTINS`` /
    :data:`BUILTIN_STUB_MODULE_NAMES`) from a merged AST, in place.

    The pruning pipeline drops them automatically when an entry point scopes
    the closure, but a whole-project merge (no entry) keeps every top-level
    unit -- including the stubs.  flang supplies the real intrinsic modules,
    so the stub blocks are redundant and would collide; this removes them."""
    kept = []
    for child in ast.children:
        if isinstance(child, f03.Module):
            stmt = atmost_one(children_of_type(child, f03.Module_Stmt))
            name = stmt.children[1].string.lower() if stmt else None
            if name in BUILTIN_STUB_MODULE_NAMES:
                continue
        kept.append(child)
    if len(kept) != len(ast.children):
        ast.init(kept)
    return ast


def find_all_f90_files(root: Path) -> Iterable[Path]:
    """Yield every Fortran source under ``root`` (recursively), or ``root``
    itself when it is a file.  Ported from ``tools/helpers.py``.
    """
    if root.is_file():
        yield root
        return
    for pat in ("*.f90", "*.F90", "*.incf"):
        yield from root.rglob(pat)


class ParseConfig:
    """Configuration for parsing and inlining a Fortran project.

    A faithful port of ``fortran_parser.ParseConfig`` restricted to the
    fields the source-text inliner consumes.  Canonicalises the various
    accepted input forms up front (``sources`` list/dict, single-tuple
    entry points, etc.).
    """

    def __init__(self,
                 sources: Union[None, List[Path], Dict[str, str]] = None,
                 entry_points: Union[None, types.SPEC, List[types.SPEC]] = None,
                 do_not_prune: Union[None, types.SPEC, List[types.SPEC]] = None,
                 do_not_rename: Union[None, types.SPEC, List[types.SPEC]] = None,
                 make_noop: Union[None, types.SPEC, List[types.SPEC]] = None,
                 ast_checkpoint_dir: Union[None, str, Path] = None,
                 consolidate_global_data: bool = False,
                 rename_uniquely: bool = False,
                 do_not_prune_type_components: bool = False,
                 keep_type_components: Optional[Dict[str, Iterable[str]]] = None,
                 monomorphize: bool = True,
                 rename_specifics: Optional[Dict[str, str]] = None,
                 specialize_at_source: Optional[Iterable[str]] = None,
                 f2py_safe_empty_types: bool = False):
        # Make the configs canonical, by processing the various types upfront.
        if not sources:
            sources = {}
        elif isinstance(sources, list):
            sources = {str(p): Path(p).read_text() for p in sources}
        if not entry_points:
            entry_points = []
        elif isinstance(entry_points, tuple):
            entry_points = [entry_points]
        if not do_not_prune:
            do_not_prune = []
        elif isinstance(do_not_prune, tuple):
            do_not_prune = [do_not_prune]
        do_not_prune = list({x for x in entry_points + do_not_prune})
        if not do_not_rename:
            do_not_rename = []
        elif isinstance(do_not_rename, tuple):
            do_not_rename = [do_not_rename]
        do_not_rename = list({x for x in entry_points + do_not_rename})
        if not make_noop:
            make_noop = []
        elif isinstance(make_noop, tuple):
            make_noop = [make_noop]
        if isinstance(ast_checkpoint_dir, str):
            ast_checkpoint_dir = Path(ast_checkpoint_dir)

        self.sources: Dict[str, str] = sources
        self.entry_points: List[types.SPEC] = entry_points
        self.config_injections: list = []
        #: Lower-cased names of stubbed LOGICAL functions whose body is replaced
        #: with ``<result> = .FALSE.`` (a subset of ``make_noop``); populated by
        #: :func:`inline_to_ast` from its ``make_return_false`` argument.
        self.make_return_false: Set[str] = set()
        self.do_not_prune: List[types.SPEC] = do_not_prune
        self.do_not_rename: List[types.SPEC] = do_not_rename
        self.make_noop: List[types.SPEC] = make_noop
        #: Lower-cased names of the EXPLICIT make_noop procedures (snapshotted
        #: before :func:`inline_to_ast` merges the do_not_emit/keep_external
        #: stubs in).  A call to one of these is a semantic no-op and is
        #: dropped outright; keep_external stubs keep their call sites (the
        #: bridge or an external implementation handles them).
        self.drop_noop_calls: Set[str] = {s[-1].lower() for s in make_noop}
        self.ast_checkpoint_dir = ast_checkpoint_dir
        self.consolidate_global_data = consolidate_global_data
        self.rename_uniquely = rename_uniquely
        self.do_not_prune_type_components = do_not_prune_type_components
        #: ``typename -> [component names]`` -- derived-type components to keep
        #: through pruning EVEN when no body reference reaches them.  Unlike
        #: :attr:`do_not_prune_type_components` (all-or-nothing over every type),
        #: this preserves EXACTLY the named members, at their source declaration
        #: positions (pruning removes non-survivors in place, so kept members
        #: keep their relative order).  Used to make one kernel's extracted
        #: single-TU carry the union of struct members a SIBLING kernel also
        #: consumes, so a per-member-SoA callback ABI lines up member-for-member
        #: on both sides (the marshal-expansion leaf order == the binding shim's
        #: slot order, both being source declaration order).  Type / component
        #: names are matched case-insensitively.  Resolved to
        #: ``Component_Decl`` specs by :meth:`keep_named_type_components`.
        self.keep_type_components: Dict[str, List[str]] = {
            t.lower(): [c.lower() for c in comps]
            for t, comps in (keep_type_components or {}).items()
        }
        #: Run the single-level abstract-dispatch monomorphisation pass (default
        #: on, always): collapse a ``CLASS(base)`` virtual dispatch the bridge
        #: cannot lower into a static call.  A precise no-op when the program
        #: has no live abstract dispatch (the common case) and when the dispatch
        #: has been externalised away.
        self.monomorphize = monomorphize
        #: ``old -> new`` renames for a specific module procedure that shares its
        #: name with the generic interface it belongs to (see
        #: :func:`cleanup.rename_clashing_specifics`).  Applied before the
        #: externalisation / interface deconstruction that the collision breaks.
        self.rename_specifics: Dict[str, str] = dict(rename_specifics or {})
        #: Names of subprograms to SPECIALIZE to their call sites by source-level
        #: inlining (per-call-site monomorphization), in addition to the structural
        #: module merge.  Used for ICON's halo ``sync_patch_array`` family, whose
        #: runtime-selected ``p_pat => p_patch%comm_pat_<typ>`` rebind the bridge
        #: cannot lower while ``typ`` is symbolic: inlining the wrapper lets the call
        #: site's compile-time-constant ``typ`` flow in so the constant-fold /
        #: branch-prune collapses the ladder to a single-source rebind BEFORE the
        #: bridge's pointer-rewrite (HLFIR inlining is too late).  See
        #: :mod:`inliner.ast_desugaring.specialize_at_source`.
        self.specialize_at_source: List[str] = [n.lower() for n in (specialize_at_source or [])]
        #: Give emptied derived types a placeholder member so numpy f2py can wrap the TU.
        #: Only the f2py-wrapped path (CLOUDSC) sets this; see :func:`pruning.prune_unused_objects`.
        self.f2py_safe_empty_types = f2py_safe_empty_types

    def set_all_possible_entry_points_from(self, ast: f03.Program):
        """Treat every top-level subprogram / main program as an entry point
        (used when no explicit entry point was supplied)."""
        self.entry_points = [
            analysis.ident_spec(singular(children_of_type(c, utils.NAMED_STMTS_OF_INTEREST_CLASSES)))
            for c in walk(ast, utils.ENTRY_POINT_OBJECT_CLASSES) if isinstance(c, utils.ENTRY_POINT_OBJECT_CLASSES)
        ]
        self.do_not_prune = list({x for x in self.entry_points + self.do_not_prune})

    def avoid_pruning_type_components(self, ast: f03.Program):
        """Mark every derived-type component to be preserved during pruning."""
        ident_map = analysis.identifier_specs(ast)
        comp_specs = [k for k, v in ident_map.items() if isinstance(v, f03.Component_Decl)]
        self.do_not_prune = list({x for x in comp_specs + self.do_not_prune})

    def keep_named_type_components(self, ast: f03.Program):
        """Mark the specific derived-type components named in
        :attr:`keep_type_components` to be preserved during pruning.

        A component spec is ``(module, typename, component)`` (see
        :func:`analysis.identifier_specs`); we match on the last two tuple
        elements case-insensitively so the caller names only ``typename`` +
        ``component`` (the defining module is resolved from wherever the type
        lands after the merge).  Unmatched entries are ignored (a type the
        merge pruned entirely before this runs contributes nothing)."""
        if not self.keep_type_components:
            return
        ident_map = analysis.identifier_specs(ast)
        keep: List[types.SPEC] = []
        for spec, node in ident_map.items():
            if not isinstance(node, f03.Component_Decl) or len(spec) < 2:
                continue
            tname, cname = spec[-2].lower(), spec[-1].lower()
            if cname in self.keep_type_components.get(tname, ()):
                keep.append(spec)
        self.do_not_prune = list({x for x in keep + self.do_not_prune})


def top_level_objects_map(ast: f03.Program, path: str) -> Dict[str, Base]:
    """Map lowercase names of top-level objects (modules, main programs) to
    their fparser nodes.  Warns (and skips) leftover cpp directives."""
    out: Dict[str, Base] = {}
    for top in ast.children:
        if type(top).__name__ in CPP_CLASS_NAMES:
            logger.warning(
                "Resolve the C++ preprocessor statements before starting to do anything with it; got `%s` in %s", top,
                path)
            continue
        name = utils.find_name_of_node(top)
        assert name
        out[name.lower()] = top
    return out


def _get_toplevel_objects(path_f90: Tuple[str, str], parser, sources: Dict[str, str]) -> Dict[str, Base]:
    """Parse one source file, resolve its ``INCLUDE`` statements by text
    substitution from ``sources``, and map its top-level objects."""
    path, f90 = path_f90
    assert isinstance(f90, str)
    try:
        # C++ preprocessor would not resolve the Fortran include statements, so we resolve them ourselves first.
        cast = parser(get_reader(f90))
        inc_map = {}
        for inc in walk(cast, f03.Include_Stmt):
            file, = inc.children
            repls = {k: c for k, c in sources.items() if k.endswith(f"{file}")}
            if not repls:
                logger.warning("Could not find the file to include `%s` in %s; moving on", inc, path)
                continue
            if len(repls) > 1:
                logger.warning("Found multiple candidate files to include `%s` in %s: %s; proceeding arbitrarily", inc,
                               path, sorted(repls.keys()))
            _, content = repls.popitem()
            inc_map[inc.tofortran()] = content
        if inc_map:
            f90_again = cast.tofortran()
            for k, v in inc_map.items():
                f90_again = f90_again.replace(k, v)
            cast = parser(get_reader(f90_again))
        return top_level_objects_map(cast, path)
    except FortranSyntaxError as e:
        logger.warning("Could not parse `%s`; got %s", path, e)
        return {}


def construct_full_ast(sources: Dict[str, str],
                       parser,
                       entry_points: Optional[Iterable[types.SPEC]] = None) -> f03.Program:
    """Combine every source file into one fparser AST, resolving
    ``INCLUDE`` directives and pruning modules unreachable from
    ``entry_points`` (all modules kept when ``entry_points`` is ``None``)."""
    tops: Dict[str, Base] = {}
    for path, f90 in sources.items():
        ctops = _get_toplevel_objects((path, f90), parser=parser, sources=sources)
        if ctops.keys() & tops.keys():
            logger.warning("Found duplicate names for top-level objects: %s", ctops.keys() & tops.keys())
        tops.update(ctops)

    ast = f03.Program(get_reader(''))
    ast.content = []
    for _, v in tops.items():
        utils.append_children(ast, v)

    ast = pruning.keep_sorted_used_modules(ast, entry_points)
    return ast


def _module_name_of_use(use: f03.Use_Stmt) -> Optional[str]:
    """The module name a ``USE`` statement imports from (its first ``Name``
    child -- the ``ONLY:`` list / renames are separate child nodes)."""
    nm = next(iter(children_of_type(use, f03.Name)), None)
    return nm.string.lower() if nm else None


def _scope_visible_names(scope: Base, host_spec: Optional[f03.Specification_Part]):
    """Names already bound in ``scope`` that must NOT be re-imported / shadowed,
    plus the set of modules ``scope`` imports *whole* (``USE x`` with no
    ``ONLY:``, which brings in every public name of ``x``).

    Collects: the procedure's own name + dummy arguments, locally declared
    entities, and every ``USE``-imported name -- from the scope's own
    specification part and, for a module-contained procedure, the host module's
    specification part (host association)."""
    visible: Set[str] = set()
    whole_use_mods: Set[str] = set()
    own_stmt = atmost_one(children_of_type(scope, (f03.Subroutine_Stmt, f03.Function_Stmt, f03.Program_Stmt)))
    if own_stmt is not None:
        for nm in walk(own_stmt, f03.Name):  # proc name + dummy-arg names (+ result)
            visible.add(nm.string.lower())
    own_spec = atmost_one(children_of_type(scope, f03.Specification_Part))
    for spec in (own_spec, host_spec):
        if spec is None:
            continue
        for use in walk(spec, f03.Use_Stmt):
            only = atmost_one(children_of_type(use, f03.Only_List))
            if only is None:
                mnm = _module_name_of_use(use)
                if mnm:
                    whole_use_mods.add(mnm)
            else:
                for nm in walk(only, f03.Name):  # ONLY: a, b => c -> a, b, c all blocked
                    visible.add(nm.string.lower())
        for ent in walk(spec, f03.Entity_Decl):
            nm = next(iter(children_of_type(ent, f03.Name)), None)
            if nm:
                visible.add(nm.string.lower())
    return visible, whole_use_mods


def _prepend_use(scope: Base, clause: str):
    """Add a ``USE`` statement to the front of ``scope``'s specification part,
    creating one (in the correct position, right after the opening statement)
    when the scope has none -- so the ``USE`` lands before ``IMPLICIT`` /
    declarations rather than after the body (which is illegal Fortran)."""
    spec = atmost_one(children_of_type(scope, f03.Specification_Part))
    if spec is not None:
        utils.prepend_children(spec, f03.Use_Stmt(clause))
    else:
        kids = list(scope.children)
        kids.insert(1, f03.Specification_Part(get_reader(clause)))
        utils.set_children(scope, kids)


_SCOPE_CLASSES = (f03.Module, f03.Main_Program, f03.Subroutine_Subprogram, f03.Function_Subprogram)
_SCOPE_STMT_CLASSES = (f03.Module_Stmt, f03.Program_Stmt, f03.Subroutine_Stmt, f03.Function_Stmt)


def _scope_qualname(scope: Base) -> Tuple[str, ...]:
    """The qualified name of ``scope`` -- the lower-cased names of the enclosing
    program units, root-first (``('m', 'foo')`` for subprogram ``foo`` in module
    ``m``).  Stable across the (no-body-inlining) merge, so it keys a scope
    before and after the pipeline."""
    names: List[str] = []
    node: Optional[Base] = scope
    while node is not None:
        if isinstance(node, _SCOPE_CLASSES):
            stmt = atmost_one(children_of_type(node, _SCOPE_STMT_CLASSES))
            nm = utils.find_name_of_stmt(stmt) if stmt is not None else None
            if nm:
                names.append(nm.lower())
        node = node.parent
    return tuple(reversed(names))


def collect_intrinsic_uses(ast: f03.Program) -> Dict[Tuple[str, ...], List[str]]:
    """Record every ``USE`` of a compiler-provided intrinsic module
    (:data:`INTRINSIC_MODULE_NAMES`), keyed by the qualified name of the scope
    that holds it.  Run BEFORE the pipeline, which strips these ``USE``s."""
    preserved = _preserved_intrinsic_modules(ast)
    captured: Dict[Tuple[str, ...], List[str]] = {}
    for use in walk(ast, f03.Use_Stmt):
        if _module_name_of_use(use) not in preserved:
            continue
        scope = use.parent
        while scope is not None and not isinstance(scope, _SCOPE_CLASSES):
            scope = scope.parent
        if scope is None:
            continue
        captured.setdefault(_scope_qualname(scope), []).append(str(use).strip())
    return captured


def restore_intrinsic_uses(ast: f03.Program, captured: Dict[Tuple[str, ...], List[str]]) -> f03.Program:
    """Re-add the intrinsic-module ``USE``s captured by
    :func:`collect_intrinsic_uses` to their original scopes (matched by
    qualified name), so the merged Fortran still compiles -- flang supplies the
    real intrinsic module.  No-op for scopes that already carry the ``USE``."""
    if not captured:
        return ast
    # 1. Drop every intrinsic-module ``USE`` the pipeline left behind.  Those are
    #    redistributed *partial* ``ONLY:`` imports of the (now-stripped) stub's
    #    names (``ONLY: c_int``) -- they neither import what the stub lacked
    #    (``c_double_complex``) nor carry the ``, INTRINSIC`` attribute, so flang
    #    rejects them.  We replace the lot with the captured originals.
    preserved = _preserved_intrinsic_modules(ast)
    for use in list(walk(ast, f03.Use_Stmt)):
        if _module_name_of_use(use) in preserved:
            utils.remove_self(use)
    # 2. Restore the original intrinsic ``USE``s at their scopes (matched by
    #    qualified name); a module-level ``USE`` is host-associated into every
    #    contained procedure, so the per-subprogram partials are not needed.
    #
    #    EXCEPTION: an external-library ``USE`` whose stub ``MODULE`` was folded
    #    away and pruned during the pipeline (the inlined-halo ``module mpi``
    #    constants stub: once ``mpi_status_size`` &c. fold to literals nothing
    #    references it, so pruning drops it).  Restoring ``USE mpi`` then
    #    dangles -- flang has no ``mpi.mod`` -- so skip restoring an
    #    external-library ``USE`` whose module is no longer defined in the AST.
    #    A genuine flang-provided intrinsic (``iso_c_binding`` &c.) has no
    #    in-AST ``MODULE`` either but is NOT external-library, so it is still
    #    restored.
    still_defined = {(stmt.children[1].string.lower() if stmt else None)
                     for m in walk(ast, f03.Module)
                     for stmt in [atmost_one(children_of_type(m, f03.Module_Stmt))]}
    for scope in walk(ast, _SCOPE_CLASSES):
        for clause in captured.get(_scope_qualname(scope), []):
            mod = _module_name_of_use(f03.Use_Stmt(clause))
            if mod in EXTERNAL_LIBRARY_MODULE_NAMES and mod not in still_defined:
                continue  # stub folded + pruned -> restoring would dangle
            _prepend_use(scope, clause)
    return ast


def _defined_proc_names(ast: f03.Program) -> Set[str]:
    """Names of every procedure DEFINED with a body (module / internal
    subprograms).  Interface-body declarations are not subprograms, so they are
    excluded -- which is exactly what lets us tell a declared-external procedure
    from a locally-defined one."""
    names: Set[str] = set()
    for sp in walk(ast, (f03.Subroutine_Subprogram, f03.Function_Subprogram)):
        stmt = atmost_one(children_of_type(sp, (f03.Subroutine_Stmt, f03.Function_Stmt)))
        nm = utils.find_name_of_stmt(stmt) if stmt is not None else None
        if nm:
            names.add(nm.lower())
    return names


def _interface_block_decl_names(ib: f03.Interface_Block) -> Set[str]:
    """The lower-cased names of the procedures an ``INTERFACE`` block declares."""
    return {
        nm.lower()
        for nm in (utils.find_name_of_stmt(s) for s in walk(ib, (f03.Function_Stmt, f03.Subroutine_Stmt))) if nm
    }


def collect_external_interfaces(ast: f03.Program) -> Dict[Tuple[str, ...], List[str]]:
    """Capture ``INTERFACE`` blocks that declare *external* procedures -- an
    explicit interface for a procedure with no definition anywhere in the
    project (e.g. a C-library function via ``BIND(C)``), keyed by the scope's
    qualified name.  The pipeline removes these ("no candidate to resolve to"),
    but they are declarations the serialised Fortran needs to compile (to an
    object -- the external symbol is resolved at link time, which the bridge's
    compile-to-HLFIR step never does)."""
    defined = _defined_proc_names(ast)
    captured: Dict[Tuple[str, ...], List[str]] = {}
    for ib in walk(ast, f03.Interface_Block):
        # An ABSTRACT INTERFACE declares deferred-binding *templates* (referenced
        # by ``PROCEDURE(name), DEFERRED``), never an external link-time
        # procedure that a call binds to -- so it is not the kind of declaration
        # this capture/restore preserves.  Worse, restoring one would resurrect a
        # body whose ``IMPORT``ed types were pruned as dead baggage (ICON's
        # comm-pattern ``interface_exchange_data_*`` import ``t_comm_pattern_collection``,
        # ``t_ptr_3d_dp``, ...), leaving a dangling ``IMPORT`` that cannot compile.
        istmt = ib.children[0]
        if isinstance(istmt, f03.Interface_Stmt) and istmt.children[0] == 'ABSTRACT':
            continue
        decl = _interface_block_decl_names(ib)
        # Skip a generic interface (no procedure bodies -> ``MODULE PROCEDURE``)
        # and any interface a locally-defined procedure backs (the pipeline
        # resolves / inlines that one correctly).
        if not decl or (decl & defined):
            continue
        scope = ib.parent
        while scope is not None and not isinstance(scope, _SCOPE_CLASSES):
            scope = scope.parent
        if scope is None:
            continue
        captured.setdefault(_scope_qualname(scope), []).append(str(ib))
    return captured


def restore_external_interfaces(ast: f03.Program, captured: Dict[Tuple[str, ...], List[str]]) -> f03.Program:
    """Re-insert the external-procedure ``INTERFACE`` blocks captured by
    :func:`collect_external_interfaces` into their scopes' specification parts
    (matched by qualified name), so the calls to those external procedures have
    an explicit interface and the single TU compiles."""
    if not captured:
        return ast
    for scope in walk(ast, _SCOPE_CLASSES):
        blocks = captured.get(_scope_qualname(scope))
        if not blocks:
            continue
        spec = atmost_one(children_of_type(scope, f03.Specification_Part))
        # The pipeline may have left a MANGLED in-place copy of a captured block
        # (``remove_access_and_bind_statements`` strips its ``BIND(C)``, const-eval
        # folds its kinds) which no longer matches the verbatim capture -- so it
        # would not be deduped by text and the restore would add a SECOND
        # declaration of the same external procedure (a compile error).  Drop any
        # existing block that overlaps a captured block's procedure names first;
        # the verbatim capture re-added below is the single source of truth.
        captured_names = {
            n
            for text in blocks
            for n in _interface_block_decl_names(f03.Interface_Block(get_reader(text)))
        }
        if spec is not None:
            for ib in list(walk(spec, f03.Interface_Block)):
                if _interface_block_decl_names(ib) & captured_names:
                    utils.remove_self(ib)
        present = {str(ib) for ib in walk(spec, f03.Interface_Block)} if spec is not None else set()
        for text in blocks:
            if text in present:
                continue
            if spec is None:
                kids = list(scope.children)
                kids.insert(1, f03.Specification_Part(get_reader(text)))
                utils.set_children(scope, kids)
                spec = atmost_one(children_of_type(scope, f03.Specification_Part))
            else:
                utils.append_children(spec, f03.Interface_Block(get_reader(text)))
            present.add(text)
    return ast


def restore_cross_module_uses(ast: f03.Program) -> f03.Program:
    """Re-add the inter-module ``USE`` statements the pipeline strips, so the
    serialised Fortran is a valid, single-file-compilable program (in place).

    The inliner resolves every symbol through its own alias map and drops
    ``USE`` statements as redundant -- correct for its native AST->SDFG path,
    but a real compiler (flang / gfortran) needs the ``USE`` for legal scoping.
    For every reference to a procedure that survives as a *different* module's
    procedure, this restores ``USE <mod>, ONLY: <proc>`` -- at the host module
    level (host-associated to every contained procedure, matching the original
    style), or in the scope itself for a free subprogram / main program.  Names
    that are locally declared / dummy arguments / siblings / already imported,
    and procedure names defined in more than one module (ambiguous), are left
    untouched -- so a local array or variable that merely shares a name with
    some module procedure is never mis-imported."""
    # Map each module-procedure name to its defining module(s).
    proc_mods: Dict[str, Set[str]] = {}
    for mod in walk(ast, f03.Module):
        mname = utils.find_name_of_stmt(atmost_one(children_of_type(mod, f03.Module_Stmt)))
        subpart = atmost_one(children_of_type(mod, f03.Module_Subprogram_Part))
        if not mname or subpart is None:
            continue
        for sp in children_of_type(subpart, (f03.Subroutine_Subprogram, f03.Function_Subprogram)):
            pstmt = atmost_one(children_of_type(sp, (f03.Subroutine_Stmt, f03.Function_Stmt)))
            pname = utils.find_name_of_stmt(pstmt) if pstmt is not None else None
            if pname:
                proc_mods.setdefault(pname.lower(), set()).add(mname.lower())
    # Only unambiguously-defined module procedures can be safely imported by name.
    unique_proc_mod = {p: next(iter(m)) for p, m in proc_mods.items() if len(m) == 1}
    if not unique_proc_mod:
        return ast

    # Plan the additions per target node (a module, or a free scope), so two
    # sibling procedures needing the same import add it once at module level.
    plan: Dict[int, Tuple[Base, Dict[str, Set[str]]]] = {}
    planned: Dict[int, Set[str]] = {}
    for scope in walk(ast, (f03.Subroutine_Subprogram, f03.Function_Subprogram, f03.Main_Program)):
        exec_part = atmost_one(children_of_type(scope, f03.Execution_Part))
        if exec_part is None:
            continue
        # Enclosing module: USE goes there (host association); else the scope itself.
        host = scope.parent
        while host is not None and not isinstance(host, f03.Module):
            host = host.parent
        cmod, host_spec, target = None, None, scope
        if isinstance(host, f03.Module):
            cmod = utils.find_name_of_stmt(atmost_one(children_of_type(host, f03.Module_Stmt)))
            cmod = cmod.lower() if cmod else None
            host_spec = atmost_one(children_of_type(host, f03.Specification_Part))
            target = host

        visible, whole_use_mods = _scope_visible_names(scope, host_spec)
        # Procedure names referenced in this scope: CALL targets + function/array
        # references (the latter guarded by ``visible`` so a local array of the
        # same name is excluded).
        refs: Set[str] = set()
        for call in walk(exec_part, f03.Call_Stmt):
            nm = next(iter(children_of_type(call, f03.Name)), None)
            if nm:
                refs.add(nm.string.lower())
        for ref in walk(exec_part, (f03.Function_Reference, f03.Part_Ref)):
            nm = next(iter(children_of_type(ref, f03.Name)), None)
            if nm:
                refs.add(nm.string.lower())

        tid = id(target)
        seen = planned.setdefault(tid, set())
        for rn in sorted(refs):
            if rn in visible or rn in seen:
                continue
            tgt = unique_proc_mod.get(rn)
            if not tgt or tgt == cmod or tgt in whole_use_mods:
                continue
            plan.setdefault(tid, (target, {}))[1].setdefault(tgt, set()).add(rn)
            seen.add(rn)

    for target, mod_procs in plan.values():
        for mod, procs in sorted(mod_procs.items()):
            _prepend_use(target, f"use {mod}, only: {', '.join(sorted(procs))}")
    return ast


def _resolve_dont_inline_names(keep_external: Iterable[str], external_functions: Iterable[ExternalFunction],
                               do_not_emit: Iterable[str]) -> Set[str]:
    """Union the (lower-cased) don't-inline names the inliner must stub.

    The external-function policy (see :mod:`dace_fortran.external_functions`)
    is two collections: ``external_functions`` (don't-inline + the bridge EMITs
    an external call) and ``do_not_emit`` (don't-inline + the bridge DROPs the
    call).  The inliner treats both the same -- it only needs the *names* not to
    inline -- so this returns their validated union (:func:`dont_inline_names`).

    ``keep_external`` is the deprecated predecessor parameter; it is kept as a
    thin backward-compatible shim meaning exactly ``do_not_emit`` (the bridge,
    not the inliner, decides emit-vs-drop), and warns when used."""
    validate(external_functions, do_not_emit)
    names = dont_inline_names(external_functions, do_not_emit)
    keep_external = list(keep_external)
    if keep_external:
        warnings.warn(
            "inline_to_ast/inline_to_single_tu(keep_external=...) is deprecated; "
            "pass do_not_emit=[names] (or external_functions=[ExternalFunction(...)]) instead.",
            DeprecationWarning,
            stacklevel=3)
        names |= {n.lower() for n in keep_external}
    return names


def _keep_external_noop_specs(ast: f03.Program, names: Iterable[str]) -> List[types.SPEC]:
    """Resolve a caller-supplied list of *external* procedure names to the
    ``make_noop`` specs that stub them.

    A name matches a subprogram either exactly or as the generic it belongs to
    -- ICON's generic interfaces expand to ``<generic>_<suffix>`` specifics
    (``sync_patch_array`` -> ``sync_patch_array_3d_dp`` ...), so a single
    ``sync_patch_array`` entry stubs the whole family.  Stubbing (emptying the
    body) keeps the call site valid while dropping the procedure's internals --
    e.g. the halo-exchange ``%exchange_data`` type-bound call that has no
    Fortran source -- so they are never inlined.  The list is passed in by the
    caller (not hardcoded here)."""
    targets = {n.lower() for n in names}
    if not targets:
        return []
    specs: List[types.SPEC] = []
    for fn in walk(ast, (f03.Subroutine_Stmt, f03.Function_Stmt)):
        spec = analysis.ident_spec(fn)
        nm = spec[-1].lower()
        if nm in targets or any(nm.startswith(t + "_") for t in targets):
            specs.append(spec)
    return specs


def _function_result_name(fn: f03.Function_Stmt) -> str:
    """The name of a function's result variable: the ``RESULT(name)`` suffix
    when present, else the function's own name."""
    suffix = atmost_one(children_of_type(fn, f03.Suffix))
    if suffix is not None:
        result = atmost_one(children_of_type(suffix, f03.Name))
        if result is not None:
            return result.string
    return utils.find_name_of_stmt(fn)


#: ``/group/ obj, obj, ...`` segment of a ``NAMELIST`` statement.
_NAMELIST_GROUP_RE = re.compile(r"/\s*(\w+)\s*/\s*([^/]+)")


def _prune_namelists_to_declared(ast: f03.Program) -> None:
    """Rewrite each surviving ``NAMELIST`` statement to reference only the
    variables still declared after pruning; drop a namelist that loses all of
    its objects.

    A namelist is a supported construct (its READ lowers to an I/O node, and a
    namelist variable is an ordinary variable populated by that read).  But
    entity-level pruning legitimately drops the namelist parameters a kernel
    does not use (a module declares hundreds of config variables across many
    namelist groups; the kernel touches only a few).  Without this pass the
    emitted TU would still *name* the dropped variables in their namelist
    statements -- "No explicit type declared for ...".  Run AFTER pruning so
    the surviving-declaration set is final; the namelists that retain a live
    variable (e.g. ``/ocean_dynamics_nml/ n_zlev``) are kept intact."""
    declared = {
        nm.string.lower()
        for ed in walk(ast, f03.Entity_Decl)
        for nm in (next(iter(children_of_type(ed, f03.Name)), None), ) if nm is not None
    }
    for nml in list(walk(ast, f03.Namelist_Stmt)):
        groups = []
        for grp, objs in _NAMELIST_GROUP_RE.findall(str(nml)):
            kept = [o.strip() for o in objs.split(",") if o.strip().lower() in declared]
            if kept:
                groups.append(f"/{grp}/ {', '.join(kept)}")
        if not groups:
            utils.remove_self(nml)
        elif " ".join(groups) != str(nml).split("NAMELIST", 1)[-1].strip():
            utils.replace_node(nml, f03.Namelist_Stmt(get_reader("NAMELIST " + " ".join(groups))))


def create_fparser_ast(cfg: ParseConfig) -> f03.Program:
    """Parse the configured sources into one combined, lowercased fparser
    AST (the first stage of the inliner pipeline)."""
    parser = ParserFactory().create(std="f2008")
    ast = construct_full_ast(cfg.sources, parser, cfg.entry_points or None)
    ast = cleanup.lower_identifier_names(ast)
    assert isinstance(ast, f03.Program)
    return ast


def _checkpoint_ast(cfg: ParseConfig, name: str, ast: f03.Program):
    """Dump an intermediate AST as Fortran into the checkpoint dir, if set."""
    if cfg.ast_checkpoint_dir:
        cfg.ast_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg.ast_checkpoint_dir.joinpath(name), 'w') as f:
            f.write(ast.tofortran())


def run_fparser_transformations(ast: f03.Program, cfg: ParseConfig, *, optimize: bool = True) -> f03.Program:
    """Run the desugaring / pruning / optimization pipeline that turns the
    raw combined AST into a simplified, self-contained single-TU AST.

    Ported faithfully from ``fortran_parser.run_fparser_transformations``;
    the only omitted stages are the ones that have no effect on a
    source-text product (none -- the full pipeline is reproduced).

    ``optimize=False`` runs the structural passes only -- procedure /
    interface inlining, coarse pruning, entity-level prune-to-used, and
    ``USE`` consolidation -- and SKIPS the constant-propagation /
    branch-pruning optimizations (``inject_const_evals``,
    ``make_practically_constant_arguments_constants``,
    ``exploit_locally_constant_variables``, ``const_eval_nodes``,
    ``prune_branches``).  This is the mode the HLFIR build path
    (:func:`dace_fortran.preprocess._fparser_merge`) uses: flang and the
    bridge do their own constant-folding / dead-branch elimination, so the
    merge only needs a valid inlined single TU -- and skipping the
    optimizers both matches the legacy regex merge's "splice and let flang
    inline" semantics and sidesteps optimizer fragilities (e.g. the
    ``exploit_locally_constant_variables`` local-alias assertion) on
    inlined-call patterns that never reach the f2dace numpy backend."""
    if not cfg.entry_points:
        cfg.set_all_possible_entry_points_from(ast)
    if cfg.do_not_prune_type_components:
        cfg.avoid_pruning_type_components(ast)
    # Resolve + preserve the specific named union components (if any) BEFORE the
    # first prune, so a sibling kernel's struct members survive even though this
    # kernel's body never references them.  Runs after the all-components flag so
    # both can coexist; a no-op when ``keep_type_components`` is empty.
    cfg.keep_named_type_components(ast)
    # Intrinsic-module ``USE``s (``iso_c_binding`` for C-interop kinds/types, ...)
    # and external-procedure ``INTERFACE`` blocks (C-library declarations) are
    # stripped by the pipeline as "resolved internally" / "no candidate"; capture
    # them now and restore them at the end so the serialised Fortran compiles.
    captured_intrinsic_uses = collect_intrinsic_uses(ast)
    captured_external_interfaces = collect_external_interfaces(ast)
    _checkpoint_ast(cfg, 'ast_v0.f90', ast)

    if cfg.make_noop:
        logger.debug("FParser Op: Making certain functions no-op in the AST...")
        noop_missed: Set[types.SPEC] = set(cfg.make_noop)
        for fn in walk(ast, (f03.Function_Stmt, f03.Subroutine_Stmt)):
            fnspec = analysis.ident_spec(fn)
            if fnspec not in cfg.make_noop:
                continue
            noop_missed.discard(fnspec)
            expart = atmost_one(children_of_type(fn.parent, f03.Execution_Part))
            if expart:
                utils.remove_self(expart)
            # A stubbed body never dispatches, so demote polymorphic CLASS(t)
            # dummies to TYPE(t): callers pass non-polymorphic actuals either
            # way, and a module procedure with a CLASS dummy makes the whole
            # f2py-compiled TU segfault at import (numpy f2py inserts a NULL
            # into the module dict for it) -- the reference leg of every
            # numerical test imports exactly such a TU.
            spec_part = atmost_one(children_of_type(fn.parent, f03.Specification_Part))
            if spec_part is not None:
                for dts in walk(spec_part, f03.Declaration_Type_Spec):
                    kw, tname = dts.children
                    if str(kw).upper() == 'CLASS' and isinstance(tname, f03.Type_Name):
                        utils.replace_node(dts, f03.Declaration_Type_Spec(f"TYPE({tname})"))
            # A return-false stub keeps a valid body so the (kept) call sites
            # still bind: replace the emptied body with `<result> = .FALSE.`
            # instead of leaving the LOGICAL result undefined.
            if isinstance(fn, f03.Function_Stmt) and fnspec[-1].lower() in cfg.make_return_false:
                result = _function_result_name(fn)
                end = fn.parent.children[-1]  # End_Function_Stmt stays last
                utils.replace_node(end, (f03.Execution_Part(get_reader(f"{result} = .FALSE.\n")), end))
        if noop_missed:
            logger.warning("The following functions could not be found for making no-op: %s", noop_missed)

    if cfg.monomorphize:
        # Collapse single-level abstract type-bound dispatch (ICON's halo
        # ``t_comm_pattern``) into static calls BEFORE the procedure-call
        # deconstruction resolves them and BEFORE pruning (which, seeing a
        # still-polymorphic dispatch, would drop the concrete arm overrides as
        # unreferenced): a retyped concrete passed-object makes every
        # ``p_pat%exchange_data_*`` a static bind the inliner then inlines,
        # instead of a ``fir.dispatch`` the bridge rejects.  A precise no-op
        # unless the unit has live abstract dispatch with a concrete arm present
        # (a kernel whose halo is externalised, or whose arm is built by an
        # externalised factory, has no arm to retype to).  Runs always; only the
        # fparser path devirtualises (the regex merge cannot, and the bridge's
        # polymorphism reject is the loud backstop if dispatch reaches it from
        # any other path).
        stats = monomorphize_auto(ast)
        if any((stats.locals_rewritten, stats.components_rewritten, stats.interposers_cloned,
                stats.declarations_retyped, stats.pointer_constructors_cloned, stats.dummy_dispatch_cloned)):
            logger.debug("FParser Op: monomorphised abstract dispatch: %s", stats)
            _checkpoint_ast(cfg, 'ast_v0b.f90', ast)

    logger.debug("FParser Op: Removing local indirections from AST...")
    # fparser splits a scope's spec/exec parts at a decl-after-statement-function
    # boundary (ECMWF fcttre/fccld includes); fold them back before any pass
    # that assumes a single Specification_Part / Execution_Part.
    ast = desugaring.coalesce_split_specification_parts(ast)
    ast = desugaring.deconstruct_enums(ast)
    ast = desugaring.deconstruct_associations(ast)
    ast = cleanup.remove_access_and_bind_statements(ast)
    ast = desugaring.deconstruct_goto_statements(ast)
    ast = desugaring.deconstruct_external_statements(ast)
    # NOTE: We need a coarse pruning as early (and as often) as reasonably
    # possible to make it easier on the operations that rely on full
    # resolution (e.g., building an alias map).  After this pruning, a full
    # resolution is expected.
    ast = pruning.prune_coarsely(ast, cfg.do_not_prune)
    ast.init([n for n in ast.children if n is not None])
    _checkpoint_ast(cfg, 'ast_v1.f90', ast)

    logger.debug("FParser Op: Removing remote indirections from AST...")
    ast = desugaring.convert_data_statements_into_assignments(ast)
    ast = cleanup.correct_for_function_calls(ast)
    ast = desugaring.deconstruct_statement_functions(ast)
    ast = desugaring.deconstruct_procedure_calls(ast)

    # An EXPLICIT make_noop SUBROUTINE has an empty body, so a CALL to it is a
    # pure no-op -- drop the call statements outright (post
    # deconstruct_procedure_calls, so former type-bound calls are plain CALLs
    # under the procedure's real name).  The stubs then lose their last
    # references and prune away with their scaffolding: CLOUDSC's
    # PERFORMANCE_TIMER stubs otherwise survive as module procedures with
    # derived-type dummies, which numpy f2py cannot wrap -- it emits a NULL
    # module entry and the import of the reference leg segfaults.  Scoped to
    # ``cfg.drop_noop_calls`` (the caller's explicit make_noop): the
    # keep_external stubs merged into ``cfg.make_noop`` later MUST keep their
    # call sites -- those calls are real (the bridge / an external
    # implementation serves them).
    def drop_explicit_noop_calls(a: f03.Program) -> None:
        # ``deconstruct_procedure_calls`` clones a type-bound target per call
        # site under ``<name>_deconproc_<n>`` -- a clone of a no-op is a no-op,
        # so match the clone naming alongside the base name.
        def is_noop_name(nm: str) -> bool:
            nm = nm.lower()
            if nm in cfg.drop_noop_calls:
                return True
            base = nm.rsplit("_deconproc_", 1)[0]
            return base != nm and base in cfg.drop_noop_calls

        for call in walk(a, f03.Call_Stmt):
            callee, _ = call.children
            if isinstance(callee, f03.Name) and is_noop_name(callee.string):
                utils.remove_self(call)

    if cfg.drop_noop_calls:
        drop_explicit_noop_calls(ast)
    ast = pruning.prune_coarsely(ast, cfg.do_not_prune)
    ast_f90_old, ast_f90_new = None, ast.tofortran()
    while not ast_f90_old or ast_f90_old != ast_f90_new:
        ast = cleanup.correct_for_function_calls(ast)
        ast = desugaring.deconstruct_interface_calls(ast)
        # Late-resolved calls (a type-bound ``timer%thread_start`` only
        # becomes a plain CALL once its interface/procedure indirection is
        # deconstructed inside this loop) get the same no-op treatment.
        if cfg.drop_noop_calls:
            drop_explicit_noop_calls(ast)
        ast = pruning.prune_coarsely(ast, cfg.do_not_prune)
        ast_f90_old, ast_f90_new = ast_f90_new, ast.tofortran()
    if walk(ast, f03.Interface_Stmt):
        _checkpoint_ast(cfg, 'ast_v1.error.f90', ast)
        if not analysis.TOLERATE_EXTERNAL_USES:
            raise RuntimeError("Could not remove all the interfaces from AST")
        # Tolerating externals: a generic interface whose calls could not all be
        # resolved to a specific is left in place.  This happens for a
        # kept-external halo generic (e.g. ``sync_patch_array_mult``) called only
        # from code unreachable from the entry, with operands whose type the
        # matcher cannot infer -- the call survives coarse (module-level) pruning
        # but the fine reachability prune below drops it.  An unresolved generic
        # is still valid Fortran (the interface plus its stubbed module
        # procedures resolve at the call site), and the final gfortran gate
        # rejects a genuinely uncompilable TU, so this is safe to leave.
        surviving = sorted(
            {utils.find_name_of_stmt(i)
             for i in walk(ast, f03.Interface_Stmt) if utils.find_name_of_stmt(i)})
        logger.warning("Left %d generic interface(s) unresolved while tolerating externals: %s", len(surviving),
                       ", ".join(surviving))
    ast = cleanup.correct_for_function_calls(ast)
    _checkpoint_ast(cfg, 'ast_v2.f90', ast)

    # Specialize the configured targets (ICON's halo ``sync_patch_array`` family)
    # to their call sites by source-level inlining.  Runs AFTER generic-interface /
    # type-bound resolution (so call names are the concrete specifics) and BEFORE
    # the constant-fold / branch-prune loop below, which then collapses the
    # now-constant ``typ`` ladder the wrappers carry into a single-source pointer
    # rebind the bridge can lower.  A no-op when ``specialize_at_source`` is empty.
    if cfg.specialize_at_source:
        n_sub, n_fun = specialize_at_source_mod.specialize_at_source(ast, cfg.specialize_at_source)
        if n_sub or n_fun:
            logger.debug("FParser Op: specialized-at-source %d subprogram call(s) + %d function ref(s): %s", n_sub,
                         n_fun, cfg.specialize_at_source)
            ast = pruning.prune_coarsely(ast, cfg.do_not_prune)
            _checkpoint_ast(cfg, 'ast_v2b.f90', ast)

    ast_f90_old, ast_f90_new = None, ast.tofortran()
    while not ast_f90_old or ast_f90_old != ast_f90_new:
        logger.debug("FParser Op: Coarsely pruning the AST...")
        ast = pruning.prune_coarsely(ast, cfg.do_not_prune)
        if optimize:
            ast = optimizations.inject_const_evals(ast, cfg.config_injections)
            ast = optimizations.make_practically_constant_arguments_constants(ast, cfg.entry_points)
            ast = optimizations.exploit_locally_constant_variables(ast)
            ast = optimizations.const_eval_nodes(ast)
            ast = pruning.prune_branches(ast)
            # ``prune_unused_objects`` trims unused entities WITHIN kept scopes
            # (unused locals / parameters).  It is skipped in merge mode
            # (``optimize=False``): the coarse pruning above already drops
            # unreferenced procedures / modules (the real size win), keeping
            # unused locals is always safe for flang, and the entity-level
            # prune has an upstream bug -- for an unused F77-style named
            # constant (separate ``DOUBLE PRECISION x`` + ``PARAMETER(x=..)``
            # statements) it removes the type declaration but leaves the
            # orphaned ``PARAMETER`` statement, which then fails to compile
            # (hit by NPB LU's unused ``tolrsd*_def``).
            ast = pruning.prune_unused_objects(ast, cfg.do_not_prune, f2py_safe_empty_types=cfg.f2py_safe_empty_types)
        ast = pruning.consolidate_uses(ast)
        ast_f90_old, ast_f90_new = ast_f90_new, ast.tofortran()
    logger.debug("FParser Op: AST-size settled at %d lines.", len(ast_f90_new.splitlines()))
    _checkpoint_ast(cfg, 'ast_v3.f90', ast)

    if analysis.TOLERATE_EXTERNAL_USES:
        # Drop interface bodies left dangling by pruning external baggage -- the
        # halo-exchange comm-pattern abstract interfaces whose IMPORTed comm
        # types were pruned away.  A no-op under full resolution.
        ast = pruning.prune_dangling_interface_bodies(ast)
        _checkpoint_ast(cfg, 'ast_v3b.f90', ast)

    if cfg.consolidate_global_data:
        logger.debug("FParser Op: Consolidating the global variables of the AST...")
        ast = cleanup.consolidate_global_data_into_arg(ast)
        ast = pruning.prune_coarsely(ast, cfg.do_not_prune)
        _checkpoint_ast(cfg, 'ast_v4.f90', ast)

    if cfg.rename_uniquely:
        logger.debug("FParser Op: Rename uniquely...")
        ast = cleanup.assign_globally_unique_subprogram_names(ast, set(cfg.do_not_rename))
        ast = cleanup.assign_globally_unique_variable_names(ast, set(cfg.do_not_rename))
        ast = pruning.consolidate_uses(ast)
        _checkpoint_ast(cfg, 'ast_v5.f90', ast)

    # The pipeline resolves symbols through its own alias map and drops the
    # inter-module ``USE`` statements as redundant -- fine for the native
    # AST->SDFG path, but the serialised Fortran is handed to a real compiler
    # that needs those ``USE``s for legal scoping.  Restore them so the output
    # is a single self-contained, compilable translation unit.  (Do NOT run
    # ``consolidate_uses`` afterwards: it re-applies the same redundant-USE
    # pruning and would strip exactly what we just restored.)
    ast = restore_cross_module_uses(ast)
    ast = restore_intrinsic_uses(ast, captured_intrinsic_uses)
    ast = restore_external_interfaces(ast, captured_external_interfaces)
    _checkpoint_ast(cfg, 'ast_v6.f90', ast)

    return ast


def parse_and_improve(sources: Dict[str, str], entry_points: Optional[Iterable[types.SPEC]] = None) -> f03.Program:
    """Parse ``sources`` into one combined AST and apply the
    ``correct_for_function_calls`` improver -- the light-weight entry used
    by the ported unit tests (matches the upstream
    ``fortran_test_helper.parse_and_improve``)."""
    parser = ParserFactory().create(std="f2008")
    ast = construct_full_ast(sources, parser, entry_points=entry_points)
    ast = cleanup.correct_for_function_calls(ast)
    assert isinstance(ast, f03.Program)
    return ast


def _demangle_spec(mangled: str) -> types.SPEC:
    """Turn a flang-mangled symbol (``_QP<proc>`` / ``_QM<mod>P<proc>``)
    into an fparser entry SPEC ``(proc,)`` / ``(mod, proc)``.

    Self-contained so the inliner's unit tests need not import the C++
    bridge (which ``dace_fortran.builder`` pulls in eagerly).  Kept in
    lock-step with ``dace_fortran.builder._demangle_fortran_proc`` /
    ``_module_of_fortran_sym``: flang lower-cases every identifier, so the
    only upper-case markers are the structural ``M`` / ``P`` / ``F``."""
    if not mangled.startswith("_Q"):
        return (mangled.lower(), )
    p = mangled.rfind("P")
    proc = mangled[p + 1:].lower() if p > 1 else mangled.lower()
    if mangled.startswith("_QM"):
        body = mangled[3:]
        for i, ch in enumerate(body):
            if ch in ("P", "F"):
                if i > 0:
                    return (body[:i].lower(), proc)
                break
    return (proc, )


def _entry_to_spec(source: str, entry: Optional[str]) -> Optional[types.SPEC]:
    """Resolve ``entry`` (plain name / ``module::proc`` / mangled ``_Q...``)
    to an fparser entry-point SPEC ``(module, proc)`` or ``(proc,)``.

    Resolution goes through dace-fortran's own ``_resolve_entry`` (so the
    inliner agrees byte-for-byte with the HLFIR build path on which
    procedure is the root); the result is demangled locally to avoid
    importing the bridge-heavy ``dace_fortran.builder``.  ``None`` passes
    through (every top-level subprogram is kept as an entry point)."""
    if entry is None:
        return None
    from dace_fortran.build import _resolve_entry

    return _demangle_spec(_resolve_entry(source, entry))


#: A standalone ``CONTIGUOUS :: a, b`` attribute statement.  fparser's f2008
#: grammar does not accept it as a stand-alone declaration, so it aborts the
#: whole file parse.  DaCe assumes contiguous storage, so the attribute is
#: semantically inert -- :func:`_cpp_expand_one` comments these lines out (kept
#: as comments to preserve line numbers) after preprocessing so the module
#: stays parseable.
_STANDALONE_CONTIGUOUS_RE = re.compile(r"(?im)^([ \t]*)CONTIGUOUS([ \t]*::.*)$")


def _strip_unparseable_attrs(text: str) -> str:
    """Comment out Fortran attribute statements that are inert under DaCe's
    contiguous-storage assumption but that fparser cannot parse (currently
    the standalone ``CONTIGUOUS :: ...`` statement ICON emits behind its
    ``USE_CONTIGUOUS`` cpp guard)."""
    return _STANDALONE_CONTIGUOUS_RE.sub(r"\1! CONTIGUOUS\2", text)


def _cpp_expand_one(name: str, content: str, *, defines: List[str], include_dirs: List[Path], flang: str) -> str:
    """Run the C preprocessor (via flang ``-cpp -E -P``) over one source's
    text and return the expanded Fortran.

    fparser parses standard Fortran and does *not* run cpp, so a source
    that carries cpp ``#include`` directives (e.g. ICON's
    ``#include "icon_definitions.inc"`` / the DSL macros in
    ``iconfor_dsl_definitions.inc``) or ``#ifdef`` arms cannot be parsed
    directly -- it raises before the inliner gets a chance to resolve the
    ``USE`` graph.  Expanding cpp up front turns the source into pure
    Fortran, mirroring the ``-cpp -U_OPENMP -U_OPENACC -I... -D...`` flags
    flang gets in the regex/codebase compile path
    (:data:`dace_fortran.flang_codebase` / ``emit_hlfir.py:328``) -- except
    here flang only preprocesses (``-E``) instead of compiling.  ``-P``
    suppresses ``# <line> "<file>"`` linemarkers, which fparser would also
    reject.
    """
    with tempfile.TemporaryDirectory() as td:
        # Capital ``.F90`` makes cpp unambiguous; keep the basename so flang
        # diagnostics name the right source.
        stem = Path(name).stem or "src"
        srcf = Path(td) / f"{stem}.F90"
        srcf.write_text(content)
        # ``#include "x.inc"`` resolves relative to the source dir first, then
        # ``-I`` dirs.  The source's real directory (when ``name`` is a path)
        # carries co-located includes, so add it to the search path too.
        local = Path(name).parent
        inc = ([local] if local.is_dir() else []) + list(include_dirs)
        cmd = [flang, "-cpp", "-E", "-P", "-U_OPENMP", "-U_OPENACC"]
        cmd += [f"-D{d}" for d in defines]
        cmd += [f"-I{Path(d)}" for d in inc]
        cmd += [str(srcf)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"cpp preprocessing failed for {name!r} (exit {proc.returncode}):\n"
                               f"{proc.stderr.strip()}")
        return _strip_unparseable_attrs(proc.stdout)


def cpp_expand_sources(src_map: Dict[str, str],
                       *,
                       defines: Iterable[str] = (),
                       include_dirs: Iterable[Union[str, Path]] = (),
                       flang: str = "flang-new-21") -> Dict[str, str]:
    """Preprocess every source in a ``{name: content}`` map through the C
    preprocessor and return the expanded map.

    This is the cpp pre-pass that lets :func:`inline_to_single_tu` /
    :func:`inline_to_ast` ingest real ICON sources (which ``#include`` the
    DSL macro headers) -- pass ``expand_cpp=True`` to those entry points to
    apply it automatically.  ``defines`` selects ``#ifdef`` arms (e.g. the
    ``__LVECTOR__`` cpp twin); ``include_dirs`` must contain the directory
    holding the ``.inc`` headers (ICON: ``src/include``).
    """
    defs = list(defines)
    incs = [Path(d) for d in include_dirs]
    return {
        name: _cpp_expand_one(name, content, defines=defs, include_dirs=incs, flang=flang)
        for name, content in src_map.items()
    }


def inline_to_ast(sources: Union[Dict[str, str], Iterable[Union[str, Path]]],
                  entry: Optional[str] = None,
                  *,
                  expand_cpp: bool = False,
                  force_double_precision: bool = False,
                  defines: Iterable[str] = (),
                  include_dirs: Iterable[Union[str, Path]] = (),
                  flang: str = "flang-new-21",
                  make_noop: Union[None, types.SPEC, List[types.SPEC]] = None,
                  make_return_false: Iterable[str] = (),
                  keep_external: Iterable[str] = (),
                  external_functions: Iterable[ExternalFunction] = (),
                  do_not_emit: Iterable[str] = (),
                  consolidate_global_data: bool = False,
                  rename_uniquely: bool = False,
                  do_not_prune_type_components: bool = False,
                  keep_type_components: Optional[Dict[str, Iterable[str]]] = None,
                  checkpoint_dir: Union[None, str, Path] = None,
                  include_builtins: bool = True,
                  tolerate_external_uses: bool = False,
                  monomorphize: bool = True,
                  rename_specifics: Optional[Dict[str, str]] = None,
                  specialize_at_source: Iterable[str] = (),
                  f2py_safe_empty_types: bool = False,
                  optimize: bool = True) -> f03.Program:
    """Run the full inliner pipeline and return the combined fparser AST.

    ``sources`` is either a ``{filename: content}`` mapping or an iterable
    of file paths / directories (each directory is globbed for Fortran
    sources).  ``entry`` selects the root procedure (and thus what is kept
    by pruning); ``None`` keeps every top-level subprogram.

    ``external_functions`` / ``do_not_emit`` declare the external-function
    policy (see :mod:`dace_fortran.external_functions`): procedures that are NOT
    inlined.  ``external_functions`` are :class:`ExternalFunction` specs the
    bridge later emits as external calls; ``do_not_emit`` are plain names whose
    calls the bridge drops.  The inliner treats both identically -- it only
    needs the *names* (and their generic-interface specifics, e.g.
    ``sync_patch_array`` -> ``sync_patch_array_3d_dp``) -- stubbing each to an
    empty body so its internals (the halo-exchange ``%exchange_data`` type-bound
    call, MPI, I/O) never enter the TU.  Nothing ICON-specific is hardcoded.

    ``keep_external`` is the deprecated predecessor of ``do_not_emit`` -- a plain
    name list, kept as a backward-compatible shim (it warns).

    ``tolerate_external_uses`` lets the pipeline ingest a kernel whose
    enclosing module ``USE``s an external library with no Fortran source on
    the search path (ICON: ``netcdf`` / ``mpi`` / ``cdi``): such imports are
    left unresolved and the reachability pruning drops the procedures that
    referenced them (see
    :data:`dace_fortran.inliner.ast_desugaring.analysis.TOLERATE_EXTERNAL_USES`).

    ``optimize=False`` skips the constant-propagation / branch-pruning
    optimization passes (see :func:`run_fparser_transformations`) -- used by
    the HLFIR build-path merge, which only needs a valid inlined single TU.
    """
    src_map = _normalize_sources(sources)
    if expand_cpp:
        src_map = cpp_expand_sources(src_map, defines=defines, include_dirs=include_dirs, flang=flang)
    if force_double_precision:
        # Force every parametrized real kind to fp64: the model's precision
        # kinds are all defined via ``SELECTED_REAL_KIND(...)`` (the parkind1
        # JPRB/JPRL/JPRM/... family), so rewriting each to the fp64 kind (8)
        # collapses the whole kind graph to double.  This is the caller's
        # explicit "parametrized precision -> fp64" contract: a kernel written
        # for mixed / reduced precision (SC2026 CLOUDSC's FP16/FP32 variants)
        # is lowered and compared at uniform fp64 on both legs.  Integer kinds
        # (``SELECTED_INT_KIND``) are untouched.
        src_map = {
            name: re.sub(r'SELECTED_REAL_KIND\s*\([^)]*\)', '8', src, flags=re.IGNORECASE)
            for name, src in src_map.items()
        }
    spec = _entry_to_spec(_concat_sources(src_map), entry)
    cfg = ParseConfig(
        sources=dict(src_map),
        entry_points=spec,
        make_noop=make_noop,
        ast_checkpoint_dir=checkpoint_dir,
        consolidate_global_data=consolidate_global_data,
        rename_uniquely=rename_uniquely,
        do_not_prune_type_components=do_not_prune_type_components,
        keep_type_components=keep_type_components,
        monomorphize=monomorphize,
        rename_specifics=rename_specifics,
        specialize_at_source=specialize_at_source,
        f2py_safe_empty_types=f2py_safe_empty_types,
    )
    if include_builtins:
        cfg.sources.setdefault("_builtins.f90", BUILTINS)
    dont_inline = _resolve_dont_inline_names(keep_external, external_functions, do_not_emit)
    # Return-false stubs are also non-inlined (stubbed), then assigned .FALSE.
    cfg.make_return_false = {n.lower() for n in make_return_false}
    dont_inline |= cfg.make_return_false
    with analysis.tolerate_external_uses(tolerate_external_uses):
        ast = create_fparser_ast(cfg)
        if cfg.rename_specifics:
            # Disambiguate a specific procedure that shares its name with its
            # generic interface BEFORE the make-noop specs are resolved below --
            # so externalising the generic stubs the RENAMED specific (otherwise
            # its body, and any external it reaches, would survive).
            n = cleanup.rename_clashing_specifics(ast, cfg.rename_specifics)
            if n:
                logger.debug("FParser Op: renamed %d generic/specific name-clash specific(s)", n)
        if dont_inline:
            # Stub the policy's non-inlined procedures (and their generic
            # specifics) BEFORE the transformations inline them.  NEVER stub an
            # ENTRY POINT, even if it is also named as an external: a kernel
            # extracted AS its own single-TU (e.g. ICON's ``velocity_tendencies``
            # is both the extraction entry and a registered external of
            # ``solve_nh``) must keep its real body -- an empty entry would
            # unreference every struct it reads, so pruning would drop those
            # types (and any ``keep_type_components`` members riding on them),
            # yielding a body-less, member-stripped TU.  Entry specs are
            # subtracted from the resolved noop set here.
            noop_specs = [s for s in _keep_external_noop_specs(ast, dont_inline) if s not in cfg.entry_points]
            cfg.make_noop = list(cfg.make_noop or []) + noop_specs
        ast = run_fparser_transformations(ast, cfg, optimize=optimize)
        # NAMELIST statements survive pruning but may name variables pruning
        # dropped; rewrite them to the surviving declarations (or drop empties).
        _prune_namelists_to_declared(ast)
    assert ast.children, "Nothing remains in this AST after pruning."
    return ast


def inline_to_single_tu(sources: Union[Dict[str, str], Iterable[Union[str, Path]]],
                        entry: Optional[str] = None,
                        *,
                        output: Union[None, str, Path] = None,
                        out_dir: Union[None, str, Path] = None,
                        name: str = "inlined",
                        expand_cpp: bool = False,
                        force_double_precision: bool = False,
                        defines: Iterable[str] = (),
                        include_dirs: Iterable[Union[str, Path]] = (),
                        flang: str = "flang-new-21",
                        make_noop: Union[None, types.SPEC, List[types.SPEC]] = None,
                        make_return_false: Iterable[str] = (),
                        keep_external: Iterable[str] = (),
                        external_functions: Iterable[ExternalFunction] = (),
                        do_not_emit: Iterable[str] = (),
                        consolidate_global_data: bool = False,
                        rename_uniquely: bool = False,
                        do_not_prune_type_components: bool = False,
                        keep_type_components: Optional[Dict[str, Iterable[str]]] = None,
                        checkpoint_dir: Union[None, str, Path] = None,
                        include_builtins: bool = True,
                        tolerate_external_uses: bool = False,
                        monomorphize: bool = True,
                        rename_specifics: Optional[Dict[str, str]] = None,
                        specialize_at_source: Iterable[str] = (),
                        f2py_safe_empty_types: bool = False) -> Path:
    """Inline a multi-file Fortran project into ONE self-contained ``.f90``
    and return the path to it.

    This is the headline public entry point.  It parses ``sources``,
    resolves ``USE`` statements, inlines the needed modules, prunes
    everything unreachable from ``entry``, runs the desugaring pipeline,
    and serialises the resulting fparser AST back to Fortran text.

    :param sources: ``{filename: content}`` mapping, or an iterable of
        file / directory paths.
    :param entry: target procedure -- plain Fortran name (``graupel_run``),
        ``module::proc``, or a mangled ``_Q...`` symbol.  ``None`` keeps
        every top-level subprogram.
    :param output: explicit output ``.f90`` path.  When omitted, the file
        is written to ``<out_dir>/<name>.f90`` (``out_dir`` defaults to the
        current directory).
    :param out_dir: directory for the default output filename.
    :param name: base name of the default output file.
    :param make_noop: subprogram specs to replace with an empty body.
    :param make_return_false: names of LOGICAL functions to stub so their body
        is ``<result> = .FALSE.`` (NOT inlined; the kept call sites still bind).
    :param consolidate_global_data: gather module-level variables into one
        derived type passed by argument.
    :param rename_uniquely: rename subprograms / variables to globally
        unique identifiers.
    :param do_not_prune_type_components: keep unused derived-type
        components.
    :param keep_type_components: ``{typename: [component, ...]}`` -- keep
        EXACTLY these derived-type components through pruning even when the
        entry never references them (targeted counterpart of
        ``do_not_prune_type_components``, which keeps all components of all
        types).  Kept members retain their source declaration order.  Used to
        make one kernel's single-TU carry the union of struct members a sibling
        kernel also consumes, so a per-member-SoA callback ABI aligns on both
        sides.  Type / component names are matched case-insensitively.
    :param checkpoint_dir: dump intermediate ASTs (``ast_v*.f90``) here.
    :param include_builtins: inject intrinsic-module stubs so ``USE
        iso_c_binding`` / ``iso_fortran_env`` resolve during parsing.
    :returns: the path to the written single-TU ``.f90``.
    """
    ast = inline_to_ast(sources,
                        entry,
                        expand_cpp=expand_cpp,
                        force_double_precision=force_double_precision,
                        defines=defines,
                        include_dirs=include_dirs,
                        flang=flang,
                        make_noop=make_noop,
                        make_return_false=make_return_false,
                        keep_external=keep_external,
                        external_functions=external_functions,
                        do_not_emit=do_not_emit,
                        consolidate_global_data=consolidate_global_data,
                        rename_uniquely=rename_uniquely,
                        do_not_prune_type_components=do_not_prune_type_components,
                        keep_type_components=keep_type_components,
                        checkpoint_dir=checkpoint_dir,
                        include_builtins=include_builtins,
                        tolerate_external_uses=tolerate_external_uses,
                        monomorphize=monomorphize,
                        rename_specifics=rename_specifics,
                        specialize_at_source=specialize_at_source,
                        f2py_safe_empty_types=f2py_safe_empty_types)
    # Drop the injected intrinsic-module stubs (``iso_c_binding`` / ``iso_fortran_env``)
    # before serialising: they exist only so ``USE`` resolves during parsing, but a
    # PARTIAL stub (the ``iso_c_binding`` one defines just ``c_int``) shadows the real
    # intrinsic at compile time, so a kept C-interop ``USE iso_c_binding, ONLY: c_ptr``
    # would fail to resolve ``c_ptr``.  Removed, a plain ``USE`` falls back to the
    # compiler's intrinsic module -- exactly as the whole-project merge path does.
    ast = strip_builtin_stub_modules(ast)
    f90 = ast.tofortran()
    # fparser serialises a no-argument ``SUBROUTINE foo() BIND(C)`` without its
    # (mandatory) empty parentheses; restore them on the final text so an emitted
    # external C interface (ICON's ``util_abort``) compiles.
    f90 = re.sub(r'(?im)^(\s*SUBROUTINE\s+\w+)\s+(BIND\s*\()', r'\1() \2', f90)

    if output is not None:
        out_path = Path(output)
    else:
        base = Path(out_dir) if out_dir is not None else Path.cwd()
        out_path = base / f"{name}.f90"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f90)
    return out_path


def _normalize_sources(sources: Union[Dict[str, str], Iterable[Union[str, Path]]]) -> Dict[str, str]:
    """Accept a ``{name: content}`` mapping or an iterable of file /
    directory paths and return a ``{name: content}`` mapping."""
    if isinstance(sources, dict):
        return dict(sources)
    out: Dict[str, str] = {}
    for item in sources:
        p = Path(item)
        for f in find_all_f90_files(p):
            out[str(f)] = f.read_text()
    return out


def _concat_sources(src_map: Dict[str, str]) -> str:
    """Join all source texts (for the entry-name scan)."""
    return "\n".join(src_map.values())


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: ``python -m dace_fortran.fparser_inliner -i SRC -k ENTRY -o OUT.f90``."""
    argp = argparse.ArgumentParser(
        prog="dace_fortran.fparser_inliner",
        description="Inline a multi-file Fortran project into one self-contained .f90 (fparser engine).")
    argp.add_argument("-i",
                      "--in_src",
                      action="append",
                      default=[],
                      required=True,
                      help="A Fortran source file or directory (repeatable).")
    argp.add_argument("-k",
                      "--entry_point",
                      default=None,
                      help="Entry procedure: plain name, module::proc, or mangled _Q... symbol.")
    argp.add_argument("-o",
                      "--output",
                      default=None,
                      help="Output .f90 path (default: ./inlined.f90; '-' writes to stdout).")
    argp.add_argument("--noop",
                      action="append",
                      default=[],
                      help="Function/subroutine to make no-op, as dot-separated spec (repeatable).")
    argp.add_argument("-d", "--checkpoint_dir", default=None, help="Dump intermediate ASTs here.")
    argp.add_argument("--consolidate_global_data",
                      action="store_true",
                      help="Consolidate module-level globals into one structure.")
    argp.add_argument("--rename_uniquely",
                      action="store_true",
                      help="Rename variables/functions to globally unique names.")
    args = argp.parse_args(argv)

    src_map = _normalize_sources(args.in_src)
    noops = [tuple(n.split(".")) for n in args.noop]

    if args.output == "-":
        ast = inline_to_ast(src_map,
                            args.entry_point,
                            make_noop=noops or None,
                            consolidate_global_data=args.consolidate_global_data,
                            rename_uniquely=args.rename_uniquely,
                            checkpoint_dir=args.checkpoint_dir)
        print(ast.tofortran())
        return 0

    out = inline_to_single_tu(src_map,
                              args.entry_point,
                              output=args.output,
                              name="inlined",
                              make_noop=noops or None,
                              consolidate_global_data=args.consolidate_global_data,
                              rename_uniquely=args.rename_uniquely,
                              checkpoint_dir=args.checkpoint_dir)
    print(f"Wrote single-TU Fortran to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
