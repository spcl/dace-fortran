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
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


@dataclass(frozen=True)
class Arg:
    """One argument of a registered external function.

    :ivar kind: ``'array'`` (passed as a pointer) or ``'scalar'``
        (passed by value, i.e. the ``VALUE`` attribute on the Fortran
        ``bind(c)`` dummy).
    :ivar dtype: element dtype string -- a key of :data:`_C_TYPES`
        (``'float64'`` / ``'int32'`` / ...).
    :ivar intent: ``'in'`` | ``'out'`` | ``'inout'``.  **Defaults to
        ``'inout'``** -- an external function is opaque, so the safe
        conservative assumption is that it both reads and writes an
        array arg: a *missed* write is a correctness bug (the mutation
        is invisible to dataflow -> wrong results / illegal
        reordering / DCE), whereas an over-declared read/write only
        costs optimization.  Narrow to ``'in'`` / ``'out'`` only when
        the true behaviour is known.  A by-value ``'scalar'`` is
        read-only regardless of this field (the callee gets a copy --
        it physically cannot write back; an ABI fact, not a choice).
    """

    kind: str
    dtype: str
    intent: str = "inout"

    def c_decl_type(self) -> str:
        """C parameter type for this arg's ``extern "C"`` declaration."""
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
    """

    c_name: str
    args: Tuple[Arg, ...] = field(default_factory=tuple)
    libraries: Tuple[str, ...] = field(default_factory=tuple)

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
    return (["-Wl,--no-as-needed"]
            + [str(p) for p in so_paths]
            + [f"-Wl,-rpath,{d}" for d in rpath_dirs])


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


def lookup_external(name: str) -> Optional[ExternalSignature]:
    """Return the registered signature for ``name``, or ``None``."""
    return _REGISTRY.get(name)


def clear_external_registry():
    """Drop all registrations and restore ``compiler.linker.args`` to
    its pre-registration value (test isolation / no global leak)."""
    global _ORIG_LINKER_ARGS
    _REGISTRY.clear()
    if _ORIG_LINKER_ARGS is not None:
        import dace
        dace.Config.set("compiler", "linker", "args", value=_ORIG_LINKER_ARGS)
        _ORIG_LINKER_ARGS = None


_CALL_ARGS_RE = re.compile(r"(\w+)\s*\(([^)]*)\)\s*;")


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
        """Reject the node if the C call body references a name that
        is not a current connector or an SDFG symbol.

        The body is a verbatim C statement (``<c_name>(<args>);``) and
        the argument identifiers carry meaning: a connector name
        rename (or a stale identifier left in the body) would silently
        bind ``foo`` to the wrong storage at codegen time.  Each
        identifier inside the call is required to be an existing
        ``in_connector`` / ``out_connector`` or a symbol declared on
        the SDFG (free symbols flow into tasklet scope by name)."""
        m = _CALL_ARGS_RE.search(self.body)
        if not m or m.group(1) != self.c_name:
            raise ValueError(f"ExternalCall {self.label!r}: body {self.body!r} is not a call "
                              f"to {self.c_name!r}")
        bound = set(self.in_connectors) | set(self.out_connectors) | {str(s) for s in sdfg.symbols}
        for tok in (t.strip() for t in m.group(2).split(",")):
            if tok and tok not in bound:
                raise ValueError(f"ExternalCall {self.label!r}: body references {tok!r} which "
                                  f"is neither a connector ({sorted(self.in_connectors | self.out_connectors)}) "
                                  f"nor an SDFG symbol ({sorted(str(s) for s in sdfg.symbols)})")
