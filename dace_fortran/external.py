"""Registry for external **ISO_C_BINDING ``bind(c)``** function calls.

A registered external function's ``CALL`` in Fortran lowers (via
``builder.emit_library.emit_call``) to an :class:`ExternalCall`
``LibraryNode`` whose single expansion produces a side-effecting CPP
tasklet invoking the function's ``extern "C"`` symbol.  The library
node carries the call body and its connector layout; the linker
flags for the registered ``.so`` libraries are injected into
``compiler.linker.args`` (``-Wl,--no-as-needed <abs .so>
-Wl,-rpath,<dir>``) so the SDFG ``.so`` links the library directly
and resolves at load time with no ``LD_PRELOAD``.

**Contract: the target MUST present a ``bind(c, name=...)`` symbol**,
either natively or via a hand-written shim.  Fortran name mangling is
compiler-specific (``__<mod>_MOD_<name>`` / ``<mod>_mp_<name>`` / ...)
and a ``.mod`` is compiler-version binary metadata that is not
C-consumable, so a stable ``bind(c)`` symbol is the only portable way
to call a Fortran routine from the generated C++.  When only a
``.mod`` exists for a non-``bind(c)`` routine, write a thin
``bind(c)`` shim that ``USE``s the module and forwards, compile it
against that ``.mod``, and register the shim's name.

This module also exports :func:`keep_external`, a thin convenience
wrapper that surfaces the intent ("leave this procedure external;
do not inline its body").  It registers the same way -- the bridge
treats every entry uniformly.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import dace
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation

#: ``Arg.dtype`` -> the C scalar type used in the ``extern "C"``
#: declaration.  Array args take the pointer form (``<ctype> *``).
_C_TYPES = {
    "float64": "double",
    "float32": "float",
    "int32": "int",
    "int64": "long long",
    "bool": "bool",
}

#: C type emitted for an ``Arg(kind="comm")`` parameter.  Full
#: contract (opaque-retype, who calls ``MPI_Comm_f2c``) is on
#: ``Arg`` 's ``kind="comm"`` docstring.
_OPAQUE_COMM_DTYPE = "MPI_Comm"


@dataclass(frozen=True)
class Arg:
    """One argument of a registered external function.

    Two orthogonal axes describe the arg, deliberately decoupled per
    the external-call design (the user can have a Fortran derived
    type cross the C ABI either as a packed AoS struct pointer *or*
    as the per-member SoA slot list the bridge already produces, and
    both are valid -- they just route to different callees):

    :ivar kind: Fortran-side shape of the arg.

        * ``'array'`` -- a flat array dummy.  C ABI defaults to a
          pointer (``<ctype> *``).
        * ``'scalar'`` -- a scalar dummy.  C ABI defaults to by-value.
        * ``'aos'`` -- a whole derived-type dummy that
          ``hlfir-marshal-external-structs`` expanded into per-member
          slots.  The :ivar:`c_abi` choice picks how those slots
          reach the external (see below).
        * ``'comm'`` -- a C ``MPI_Comm`` handle.  ``c_abi`` is forced
          opaque-by-value; ``intent`` is forced ``'in'``.  The
          SDFG-side container is retyped to
          ``dace.dtypes.opaque("MPI_Comm")`` so DaCe codegen emits
          the parameter as ``MPI_Comm`` directly -- the ``bind(c)``
          shim (or DaCe's MPI integration) is responsible for
          ``MPI_Comm_f2c`` so the external sees a C ``MPI_Comm``.

    :ivar c_abi: how the arg crosses the C ABI to the external.
        ``None`` (the default) picks the natural mapping for the
        arg's :ivar:`kind`:

        * ``'value'`` -- pass-by-value (the natural default for
          ``kind='scalar'``).
        * ``'pointer'`` -- pass-by-pointer (the natural default for
          ``kind='array'``).
        * ``'aos_struct_ptr'`` -- the natural default for
          ``kind='aos'``: emit_call locally re-packs the SoA flats
          into a stack AoS struct, passes ``&buf``, then unpacks out
          (the inline pack/unpack body that has shipped in this
          tree since v1).  ``intent`` drives the pack-in / unpack-out
          directions.  The C parameter is ``void *`` (the callee
          casts to its concrete struct).
        * ``'per_member_soa'`` -- for ``kind='aos'`` only.  The
          per-member SoA slots the marshal-expansion produced are
          forwarded *verbatim* to the external (no AoS buffer, no
          pack / unpack copy).  This is the shape a *sibling SDFG*
          built from the same Fortran source expects -- both sides
          already speak per-member SoA, so the AoS round-trip is dead
          work.  The C parameters expand to one pointer per leaf
          member, in marshal-expansion order.

    :ivar dtype: element dtype string -- a key of :data:`_C_TYPES`
        (``'float64'`` / ``'int32'`` / ...).  Ignored when ``kind``
        is ``'aos'`` or ``'comm'``.
    :ivar intent: ``'in'`` | ``'out'`` | ``'inout'``.  Defaults to
        ``'inout'`` -- an external function is opaque, so the safe
        conservative assumption is that it both reads and writes an
        array arg: a missed write is a correctness bug (the mutation
        is invisible to dataflow -> wrong results / illegal
        reordering / DCE), an over-declared read/write only costs
        optimisation.  Narrow to ``'in'`` / ``'out'`` only when the
        true behaviour is known.  A by-value scalar is read-only
        regardless of this field (the callee gets a copy -- an ABI
        fact, not a choice); ``'comm'`` is always read-only.
    """

    kind: str
    dtype: str = ""  # ignored when kind == "comm" or kind == "aos"
    intent: str = "inout"
    c_abi: Optional[str] = None

    def resolved_c_abi(self) -> str:
        """The :ivar:`c_abi` choice resolved to its concrete value
        with the per-:ivar:`kind` natural default applied."""
        if self.c_abi is not None:
            return self.c_abi
        if self.kind == "scalar":
            return "value"
        if self.kind == "array":
            return "pointer"
        if self.kind == "aos":
            return "aos_struct_ptr"
        if self.kind == "comm":
            return "value"
        raise ValueError(f"external Arg: unknown kind {self.kind!r}; "
                         f"expected one of array / scalar / aos / comm")

    def c_decl_type(self) -> str:
        """C parameter type for this arg's ``extern "C"`` declaration.

        For ``kind='aos'`` with ``c_abi='per_member_soa'`` the decl
        expands to multiple parameters (one per leaf member); that
        expansion is per-call-site and lives in
        :func:`builder.emit_library.emit_call`.  This method returns
        only the *single-parameter* C-decl shape; ``per_member_soa``
        signals that to the caller with a sentinel.

        :raises ValueError: unsupported ``kind`` or ``dtype``.
        """
        if self.kind == "comm":
            return _OPAQUE_COMM_DTYPE
        if self.kind == "aos":
            abi = self.resolved_c_abi()
            if abi == "aos_struct_ptr":
                return "void *"  # address of the re-packed AoS buffer
            if abi == "per_member_soa":
                # The leaf-expanded decl is rendered by the caller
                # from the marshal-expansion groups; nothing
                # single-parameter to surface here.
                return ""
            raise ValueError(f"external Arg(kind='aos'): unsupported "
                             f"c_abi {abi!r}; expected aos_struct_ptr "
                             f"or per_member_soa")
        base = _C_TYPES.get(self.dtype)
        if base is None:
            raise ValueError(f"external Arg: unsupported dtype {self.dtype!r}; "
                             f"known: {sorted(_C_TYPES)}")
        return f"{base} *" if self.kind == "array" else base


@dataclass(frozen=True)
class ExternalSignature:
    """Signature of a registered external ``bind(c)`` function.

    :ivar c_name: the stable ``bind(c, name=...)`` symbol the SDFG
        calls (and that ``libraries`` must export).
    :ivar args: ordered positional arguments.
    :ivar libraries: shared libraries that export ``c_name``.  Each is
        linked into the SDFG ``.so`` (``-L<dir> -l:<name>``) with an
        automatic ``-Wl,-rpath,<dir>`` so it also resolves at load
        time -- the SDFG library is self-contained, no ``LD_PRELOAD``.
    :ivar stub: when true the call is DROPPED entirely (no library node) -- the
        procedure is still left external (body stripped) so its unlowerable
        internals never reach the bridge, but the call site becomes a no-op.
        Used to excise infrastructure a kernel pulls in only structurally
        (config / registry / metadata helpers with ``class(*)`` ``select_type``
        bodies) so the kernel can build; a real run needs a proper shim instead.
    """

    c_name: str
    args: Tuple[Arg, ...] = field(default_factory=tuple)
    libraries: Tuple[str, ...] = field(default_factory=tuple)
    stub: bool = False
    # Fortran module globals to forward into the callee's library
    # before each call.  Each tuple = ``(module, member, dtype, rank)``:
    #
    #   * ``module``  -- defining module (``"mo_parallel_config"``).
    #   * ``member``  -- member name within that module (``"nproma"``).
    #   * ``dtype``   -- SDFG dtype string (``"int32"`` / ``"bool"`` /
    #                    ``"float64"`` etc.).
    #   * ``rank``    -- 0 for a scalar, N for a rank-N fixed-shape
    #                    array (the array's declared extents are
    #                    spelled in the inner shim via a literal
    #                    shape list captured at shim-emit time).
    #
    # When non-empty, ``emit_call`` reads ``__<module>_MOD_<member>``
    # directly from the OUTER library's BSS (the bridge has the
    # outer's wrapper populate that copy from the caller's args via
    # the existing ``use <module>, only: ...`` import path) and
    # appends the values to the C ABI call AFTER every other arg.
    # The matching :func:`dace_fortran.bindings.emit_bind_c_shim`
    # accepts those same args in the same order and writes them to
    # the INNER library's ``use <module>`` import alias, so the
    # callee's ``velocity_tendencies_dace`` reads the same value the
    # outer's caller wrote.  See the velocity e2e ASan ODR-violation
    # diagnostic for the per-library Fortran-module-globals issue
    # this contract addresses.
    module_symbol_forward: Tuple[Tuple[str, str, str, int], ...] = field(default_factory=tuple)
    # When true, ``emit_call`` prepends one ``int`` extent per
    # dynamic-shape dim ahead of every dynamic-shape leaf -- the C
    # ABI :func:`dace_fortran.bindings.emit_bind_c_shim` exports
    # ("dynamic-shape" = ``per_member_soa`` AoS member with ``nel ==
    # 0``, or ``kind='array'`` whose connected SDFG array has any
    # symbolic shape entry).  Set this on every registration whose
    # callee was produced by ``build_fortran_library(...,
    # bind_c_shim=True)``: the shim needs the runtime extents to
    # build ``c_f_pointer`` aliases.  Default ``False`` matches the
    # pre-shim convention (the caller hand-authors a C external that
    # accepts raw pointers).
    dynamic_extents_abi: bool = False
    # Flat names of rank-0 struct members the CALLEE's ``bind_c_shim`` forwards
    # as a ``type(c_ptr), value`` POINTER (it dereferences them via
    # ``c_f_pointer``) rather than by value -- every rank-0 member that is NOT a
    # read-only flat-array-dummy extent (see
    # :func:`dace_fortran.bindings.bind_c_shim.scalar_pointer_members`, which
    # derives this from the callee interface).  ``emit_call`` passes the ADDRESS
    # of a scratch cell for each such member even when the CALLER holds it only
    # as an SDFG symbol (a promoted grid extent) -- otherwise the raw integer
    # value would be reinterpreted as a pointer.  The caller thus conforms to
    # the callee shim's per-member ABI (an SDFG-to-SDFG sibling-callee fact the
    # marshal cannot infer from the caller's own symbol view).  Empty for
    # hand-authored C externals.
    callee_ptr_scalar_members: frozenset = field(default_factory=frozenset)

    def c_declaration(self) -> str:
        """The ``extern "C" void <c_name>(<types>);`` declaration."""
        params = ", ".join(a.c_decl_type() for a in self.args) or "void"
        return f'extern "C" void {self.c_name}({params});'


_REGISTRY: Dict[str, ExternalSignature] = {}

#: Snapshot of ``compiler.linker.args`` before the first registration
#: that contributes libraries -- restored by
#: :func:`clear_external_registry` so the global config mutation does
#: not leak past the registry's lifetime.
_ORIG_LINKER_ARGS: Optional[str] = None


def _link_flags(libraries: Tuple[str, ...]) -> List[str]:
    """Build the shared-linker flags for ``libraries``.

    For each registered ``.so`` the absolute path is passed verbatim
    (robust vs ``-l`` name guessing); the parent directories are
    collected (deduped) into ``-Wl,-rpath,<dir>`` entries so the SDFG
    ``.so`` resolves the symbols at load time with no
    ``LD_PRELOAD``.  ``-Wl,--no-as-needed`` precedes the libraries
    because shared-linker flags land *before* the SDFG objects on
    the link line and the default ``--as-needed`` would drop a
    library that no yet-seen object references.

    :param libraries: paths to the registered ``.so`` libraries.
    :returns: a flat list of linker tokens.
    """
    so_paths = [Path(lib).resolve() for lib in libraries]
    rpath_dirs = list(dict.fromkeys(p.parent for p in so_paths))
    return (["-Wl,--no-as-needed"] + [str(p) for p in so_paths] + [f"-Wl,-rpath,{d}" for d in rpath_dirs])


def _apply_linker_config():
    """Recompute ``compiler.linker.args`` = original + the dedup'd link
    flags of every registration's libraries.  This is the global,
    register/clear-scoped config mutation the chosen design accepts
    (verbatim shared-linker flags -- not the CMake-list ``DACE_LIBS``)."""
    import dace

    global _ORIG_LINKER_ARGS
    if _ORIG_LINKER_ARGS is None:
        _ORIG_LINKER_ARGS = dace.Config.get("compiler", "linker", "args") or ""
    flags: List[str] = []
    for sig in _REGISTRY.values():
        flags += _link_flags(sig.libraries)
    merged = (_ORIG_LINKER_ARGS + " " + " ".join(dict.fromkeys(flags))).strip()
    dace.Config.set("compiler", "linker", "args", value=merged)


def register_external(name: str, signature: ExternalSignature):
    """Register ``name`` (the Fortran call-site name) as an external
    ``bind(c)`` function with ``signature``.

    If ``signature.libraries`` is non-empty their link + rpath flags
    are merged into ``compiler.linker.args`` so the SDFG ``.so`` links
    and resolves the symbol with no ``LD_PRELOAD`` (restore with
    :func:`clear_external_registry`).

    :param name: name as it appears at the Fortran ``CALL`` site
        (for a ``bind(c, name=foo)`` interface this is ``foo``).
    :param signature: its :class:`ExternalSignature`.
    """
    _REGISTRY[name] = signature
    if signature.libraries:
        _apply_linker_config()


def keep_external(name: str,
                  *,
                  c_name: Optional[str] = None,
                  args: Tuple[Arg, ...] = (),
                  libraries: Tuple[str, ...] = (),
                  stub: bool = False,
                  dynamic_extents_abi: bool = False,
                  module_symbol_forward: Tuple[Tuple[str, str, str, int], ...] = (),
                  callee_ptr_scalar_members: frozenset = frozenset()):
    """Mark ``name`` to be left external -- the bridge emits an
    :class:`ExternalCall` library node for every ``CALL name(...)``
    instead of inlining ``name`` 's body.

    Functionally a convenience wrapper around :func:`register_external`:
    same registry, same lookup, same library-link injection.  The
    distinct name surfaces the intent ("don't lower this callee into a
    kernel; emit it as an opaque call") and lets callers omit
    ``ExternalSignature`` boilerplate at the call site.

    :param name: name as it appears at the Fortran ``CALL`` site.
    :param c_name: ``bind(c, name=...)`` symbol the C call invokes;
        defaults to ``name`` when omitted (the common case where the
        Fortran name is also the ``extern "C"`` symbol).
    :param args: the ``bind(c)`` parameter list -- same shape as
        :class:`ExternalSignature.args`.
    :param libraries: shared libraries that export ``c_name`` -- merged
        into ``compiler.linker.args`` so the SDFG ``.so`` resolves the
        symbol at load time with no ``LD_PRELOAD``.

    For procedures whose Fortran body still lives in the bridge's
    source bundle, also strip the body (or USE-import only the
    interface) so flang does not inline ``name`` ahead of dispatch.

    :param stub: drop the call entirely (no library node) while still leaving
        the procedure external -- excise infrastructure a kernel pulls in only
        structurally (helpers whose unlowerable bodies would otherwise block
        the build).  A real run needs a proper shim instead.
    :param dynamic_extents_abi: when ``True``, every dynamic-shape leaf
        crosses the C ABI with one ``int`` extent per dim prepended to
        the pointer.  Set this when the callee was produced by
        ``build_fortran_library(..., bind_c_shim=True)`` -- the shim's
        ``c_f_pointer`` aliases need the runtime extents (see
        :attr:`ExternalSignature.dynamic_extents_abi`).
    :param module_symbol_forward: Fortran module globals to forward
        across the library boundary.  Each tuple is ``(module,
        member, dtype, rank)`` -- see
        :attr:`ExternalSignature.module_symbol_forward` for the
        rationale (per-library Fortran-module-globals issue exposed
        by the velocity dycore + external e2e ASan diagnostic).
    :param callee_ptr_scalar_members: flat names of rank-0 struct members
        the callee's ``bind_c_shim`` takes as a ``type(c_ptr), value``
        pointer -- pass
        ``bind_c_shim.scalar_pointer_members(callee_iface)`` when wiring an
        SDFG-to-SDFG sibling callee so ``emit_call`` passes each such member's
        ADDRESS (not its value) even when the caller holds it as a symbol (see
        :attr:`ExternalSignature.callee_ptr_scalar_members`).
    """
    register_external(
        name,
        ExternalSignature(c_name=c_name or name,
                          args=tuple(args),
                          libraries=tuple(libraries),
                          stub=stub,
                          dynamic_extents_abi=dynamic_extents_abi,
                          module_symbol_forward=tuple(module_symbol_forward),
                          callee_ptr_scalar_members=frozenset(callee_ptr_scalar_members)))


def apply_external_functions(external_functions: Iterable["ExternalFunction"] = (),
                             do_not_emit: Iterable[str] = ()) -> None:
    """Register the bridge half of the unified external-function policy.

    This is the bridge-side mirror of the inliner's
    :func:`dace_fortran.fparser_inliner.inline_to_ast` /
    :func:`dace_fortran.preprocess.merge_used_modules` ``external_functions`` /
    ``do_not_emit`` parameters -- ONE policy, declared once per target, drives
    both halves: the inliner stubs each named procedure's body (so its
    unlowerable internals never enter the TU) and this populates the registry
    that ``builder.emit_library.emit_call`` reads to lower the surviving call:

    * each :class:`~dace_fortran.external_functions.ExternalFunction` becomes an
      **emitted** external -- a ``CALL`` lowers to an :class:`ExternalCall`
      library node bound to ``f.symbol`` (its ``c_function`` or ``name``), with
      ``f.library`` linked in.  The argument order/identity comes from the HLFIR
      call site (``args=()`` -> ``emit_call`` derives a conservative plan), so a
      minimal ``ExternalFunction(name, c_function, library)`` is enough.  Rich
      ABIs (AoS structs, ``MPI_Comm``, dynamic extents, intent narrowing) still
      register an authored :class:`ExternalSignature` via :func:`keep_external`.
    * each ``do_not_emit`` name becomes an **ignored** external -- ``stub=True``,
      so the call is dropped (no node).  ``ignore`` is a subset of don't-inline.

    Validation (no duplicate names, no name in both lists) runs first via
    :func:`dace_fortran.external_functions.validate`, so an inconsistent policy
    is rejected before any registration mutates the registry.

    :param external_functions: procedures to emit as external calls.
    :param do_not_emit: procedure names whose calls are dropped entirely.
    """
    from dace_fortran.external_functions import validate

    external_functions = list(external_functions)
    do_not_emit = list(do_not_emit)
    validate(external_functions, do_not_emit)
    for f in external_functions:
        keep_external(f.name, c_name=f.symbol, libraries=(f.library, ) if f.library else ())
    for name in do_not_emit:
        keep_external(name, stub=True)


def lookup_external(name: str) -> Optional[ExternalSignature]:
    """Return the registered signature for ``name``, or ``None``."""
    return _REGISTRY.get(name)


def registered_names() -> List[str]:
    """Names registered as external (``keep_external`` / ``register_external``).

    The builder passes these to ``HLFIRModule.externalize_symbols`` so a
    registered callee whose Fortran body is in the (merged) translation unit
    stays an external declaration through ``hlfir-inline-all`` instead of being
    inlined ahead of the ``ExternalCall`` lowering.

    :returns: the registry keys (Fortran call-site names).
    """
    return list(_REGISTRY)


def inline_external(sdfg: 'dace.SDFG', name: str, callee_sdfg: 'dace.SDFG') -> int:
    """Swap every ``ExternalCall`` library node for ``name`` in ``sdfg``
    with a :class:`dace.sdfg.nodes.NestedSDFG` wrapping ``callee_sdfg``.

    Both kernels (caller + callee) must have gone through the SAME
    ``hlfir-flatten-structs`` pipeline on the SAME merged source, so the
    flat-leaf signatures agree by position: the ExternalCall's connector
    layout (``_a0``, ``_a1``, ...) matches ``callee_sdfg.arglist()`` in
    order.  The lookup is by ``c_name`` (the Fortran procedure name
    appearing in the ``CALL`` site, normalised by ``emit_call``).

    No bind(c) shim is generated and no ``.mod`` files are needed --
    the callee is embedded directly into the caller's SDFG.  This is
    the recommended path for the in-tree ICON
    ``solve_nh -> velocity_tendencies`` case.  Standalone-``.so``
    deployment uses the shim-generation path instead.

    :param sdfg: The caller SDFG that holds the ExternalCall(s).
    :param name: The Fortran procedure name (matches ``ExternalCall.c_name``
        after the registry's ``c_name or name`` defaulting).
    :param callee_sdfg: The callee's pre-built SDFG.
    :returns: The number of ExternalCall sites replaced.
    """
    from dace.sdfg.nodes import NestedSDFG
    sig = lookup_external(name)
    if sig is None:
        raise ValueError(f"inline_external: {name!r} is not registered "
                         f"as an external")
    target_c_name = sig.c_name
    replaced = 0
    # Walk every state for ExternalCall nodes; we mutate as we go but
    # collect first so the walk isn't disturbed.
    targets = []
    for state in sdfg.all_states():
        for node in list(state.nodes()):
            if (isinstance(node, ExternalCall) and node.c_name == target_c_name):
                targets.append((state, node))
    if not targets:
        return 0
    callee_args = list(callee_sdfg.arglist().keys())
    for state, node in targets:
        in_edges = list(state.in_edges(node))
        out_edges = list(state.out_edges(node))
        # ExternalCall connectors are ``_aI`` (in) and ``_aI_o`` (out);
        # map them to the matching arglist entry by I.
        in_map: dict = {}
        out_map: dict = {}
        for e in in_edges:
            i = int(e.dst_conn[2:])  # strip ``_a``
            in_map.setdefault(callee_args[i], e)
        for e in out_edges:
            # ``_a{i}_o`` -> strip ``_a`` prefix and the trailing ``_o``.
            tail = e.src_conn[2:]
            if tail.endswith('_o'):
                tail = tail[:-2]
            i = int(tail)
            out_map.setdefault(callee_args[i], e)
        # Symbol mapping: every callee free symbol that's also live in
        # the caller passes through identity-named.
        symbol_mapping = {s: s for s in callee_sdfg.free_symbols if s in sdfg.symbols or s in sdfg.arrays}
        nested = state.add_nested_sdfg(
            callee_sdfg,
            inputs=set(in_map.keys()),
            outputs=set(out_map.keys()),
            symbol_mapping=symbol_mapping,
            name=f"{name}_inlined_{replaced}",
        )
        for conn, e in in_map.items():
            state.add_memlet_path(e.src, nested, dst_conn=conn, memlet=e.data)
        for conn, e in out_map.items():
            state.add_memlet_path(nested, e.dst, src_conn=conn, memlet=e.data)
        for e in in_edges + out_edges:
            state.remove_edge(e)
        state.remove_node(node)
        replaced += 1
    return replaced


def clear_external_registry():
    """Drop all registrations and restore ``compiler.linker.args`` to
    its pre-registration value (test isolation / no global leak)."""
    global _ORIG_LINKER_ARGS
    _REGISTRY.clear()
    if _ORIG_LINKER_ARGS is not None:
        import dace
        dace.Config.set("compiler", "linker", "args", value=_ORIG_LINKER_ARGS)
        _ORIG_LINKER_ARGS = None


@dace.library.expansion
class ExpandExternalCallPure(ExpandTransformation):
    """Lower :class:`ExternalCall` to a side-effecting CPP tasklet.

    The expansion appends the ``extern "C"`` declaration to the
    parent SDFG's global code (duplicate identical declarations are
    legal C++) and returns a tasklet that wraps the C call -- the
    library node's connectors are inherited verbatim, so the
    surrounding dataflow is unchanged.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **_kwargs):
        if node.c_decl:
            parent_sdfg.append_global_code(node.c_decl)
        tasklet = dace.sdfg.nodes.Tasklet(node.label,
                                          dict(node.in_connectors),
                                          dict(node.out_connectors),
                                          node.body,
                                          language=dace.dtypes.Language.CPP,
                                          side_effects=True)
        return tasklet


@dace.library.node
class ExternalCall(dace.sdfg.nodes.LibraryNode):
    """SDFG library node calling a separately-compiled external
    ``bind(c)`` function.

    The node carries the ``extern "C"`` declaration and the call
    statement as properties; its :class:`ExpandExternalCallPure`
    expansion lowers it to a side-effecting CPP tasklet at code-gen
    time.  Array-arg connectors are pointers (read / written per
    ``intent``) and the corresponding edges are whole-array memlets
    against the same data container -- the C call writes through the
    out-connector pointer (which aliases the same storage as the in
    pointer) so the dataflow sees both the read dependency and the
    write-back.
    """

    implementations = {"pure": ExpandExternalCallPure}
    default_implementation = "pure"

    #: ``bind(c, name=...)`` symbol the C call invokes.
    c_name = dace.properties.Property(dtype=str, default="")
    #: ``extern "C" void <c_name>(...);`` -- appended to SDFG global code.
    c_decl = dace.properties.Property(dtype=str, default="")
    #: The CPP statement(s) the expanded tasklet runs (the call line).
    body = dace.properties.Property(dtype=str, default="")

    def __init__(self, name, *, c_name="", c_decl="", body="", inputs=None, outputs=None, **kwargs):
        super().__init__(name=name, inputs=inputs or set(), outputs=outputs or set(), **kwargs)
        self.c_name = c_name
        self.c_decl = c_decl
        self.body = body

    def validate(self, sdfg, state):
        """Reject the node if the C body does not actually call ``c_name``
        or leaves a connector unreferenced.

        The body is verbatim C: at minimum a call ``<c_name>(<args>);``, or
        (for an array-of-structs argument) that call wrapped by a local
        ``struct`` buffer pack / unpack.  The corruption to guard against is a
        renamed / stale call identifier binding the wrong external, so require
        a call to ``c_name`` to be present.  (Connector use is not required per
        connector: an ``inout`` array's read connector intentionally aliases
        its write connector, and only the writable one appears in the body.)"""
        if not re.search(r"\b" + re.escape(self.c_name) + r"\s*\(", self.body):
            raise ValueError(f"ExternalCall {self.label!r}: body {self.body!r} does not "
                             f"call {self.c_name!r}")
