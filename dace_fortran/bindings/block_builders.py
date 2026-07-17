"""Named block builders  --  one function per Fortran section of the
generated wrapper module.  Each takes the canonical bundle
``(frozen, iface, plan)`` (or a subset) and returns one string
representing that block.

The builders are deliberately thin: they consume a template from
``templates/*.f90.in``, substitute the section-specific variables,
and return the rendered text.  All Fortran-construction logic that
depends on the flattening plan lives in ``loop_copy.py`` and is
called from ``build_wrapper_body`` / ``build_wrapper_tail``.
"""

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: gfortran (and the Fortran 2003+ standard) caps an identifier at 63 characters.
#: There is no compiler flag to lift it, so any generated name longer than this
#: must be renamed before the binding will compile.
_FORTRAN_IDENT_LIMIT = 63


def _shorten_long_idents(src: str, limit: int = _FORTRAN_IDENT_LIMIT) -> str:
    """Rename every identifier longer than ``limit`` to a unique, deterministic
    ``<=limit`` form, consistently across the whole binding.

    The bridge flattens a deeply-inlined struct-member extent into a name like
    ``verticalderiv_vec_midlevel_on_block_inv_prism_center_distance_d0`` (64
    chars), which gfortran rejects.  These names are all binding-INTERNAL -- the
    C ABI is positional and the ``bind(c)`` entry names are separate string
    literals -- so a whole-word token rename cannot perturb linkage or the
    reconstructed ``use`` imports (every real Fortran/ICON entity is already
    ``<=limit``).  The rename keys off a blake2b digest of the full name, so it
    is stable across runs and disambiguates names that share a truncated stem
    (``..._distance_d0`` vs ``..._distance_d1``)."""
    all_idents = set(re.findall(r"[A-Za-z_]\w*", src))
    long_idents = sorted(i for i in all_idents if len(i) > limit)
    if not long_idents:
        return src
    used = set(all_idents)
    rename: Dict[str, str] = {}
    for name in long_idents:
        digest = hashlib.blake2b(name.encode(), digest_size=4).hexdigest()  # 8 hex chars
        short = f"{name[:limit - 9]}_{digest}"
        while short in used:  # digest collision -- vanishingly unlikely; re-hash
            digest = hashlib.blake2b((short + name).encode(), digest_size=4).hexdigest()
            short = f"{name[:limit - 9]}_{digest}"
        rename[name] = short
        used.add(short)
    for name, short in rename.items():
        src = re.sub(rf"\b{re.escape(name)}\b", short, src)
    return src


from dace_fortran.bindings.flatten_plan import (
    FlattenPlan,
    strip_index_args,
)
from dace_fortran.bindings.fortran_interface import DerivedType, OriginalInterface
from dace_fortran.bindings.frozen_signature import FrozenSignature
from dace_fortran.bindings.loop_copy import (
    _fortran_type,
    render_alias_calls,
    render_aos_alloc_pack_in,
    render_aos_alloc_pack_out,
    render_copy_in_loop,
    render_copy_out_loop,
)

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _load(name: str) -> str:
    return (_TEMPLATE_DIR / name).read_text()


# ---------------------------------------------------------------------------
# bind(c) interface block
# ---------------------------------------------------------------------------


def build_c_interface(frozen: FrozenSignature, iface: OriginalInterface, dace_arglist: tuple = ()) -> str:
    """Render the ``interface ... end interface`` block declaring the
    three C entry points that the compiled SDFG exports.

    Template: ``templates/c_interface.f90.in``.

    :param frozen: frozen signature  --  drives the per-arg declarations.
    :param iface: outer interface  --  only ``iface.entry`` is read,
                  used in the ``bind(c, name='...')`` attribute.
    :returns: one rendered string containing the full ``interface``
              block.

    Example fragment::

        interface
          function dace_init_kernel() bind(c, name='__dace_init_kernel') result(h)
            type(c_ptr) :: h
          end function
          subroutine dace_program_kernel(h, fld_a, fld_b) bind(c, name='__program_kernel')
            type(c_ptr), value :: h
            type(c_ptr), value :: fld_a
            type(c_ptr), value :: fld_b
          end subroutine
          function dace_exit_kernel(h) bind(c, name='__dace_exit_kernel') result(err)
            ...
          end function
        end interface
    """
    tpl = _load("c_interface.f90.in")
    header_lines: List[str] = []
    body_lines: List[str] = []
    for a in _dace_call_order(frozen, dace_arglist):
        if isinstance(a, str):
            # A shape-only free symbol DaCe folds into the ``__program``
            # signature (sorted with the scalars) -- pass-by-value int
            # by default, with overrides for the pgrid-driving symbols
            # ``dace_user_comm`` / ``dace_user_comm_size`` that have
            # non-int dtypes.
            header_lines.append(f"      {a}")
            body_lines.append(f"      {_init_symbol_decl(a, frozen)} :: {a}")
            continue
        header_lines.append(f"      {a.sdfg_name}")
        if a.rank > 0:
            # Real array, or length-1 wrapper for a scalar OUTPUT
            # (``intent(out)`` / ``intent(inout)``).  Either way DaCe
            # passes a pointer.
            body_lines.append(f"      type(c_ptr), value :: {a.sdfg_name}")
        elif a.kind == 'symbol':
            # Free symbol -- pass-by-value integer of its own width.
            # int32 by default; int64 for e.g. the AoS-allocatable
            # ``cap_<base>_<member>`` extent.  Must match the
            # wrapper-local decl ``build_wrapper_head`` emits for it.
            body_lines.append(f"      {_fortran_c_value_type(a.dtype)}, value :: {a.sdfg_name}")
        elif a.kind == 'mpi_comm':
            # ``emit_mpi`` retyped the Fortran ``integer`` communicator
            # to an ``opaque(MPI_Comm)`` SDFG scalar; DaCe codegen emits
            # an ``MPI_Comm`` by-value parameter.  ``MPI_Comm`` is a
            # pointer-sized handle (OpenMPI ``ompi_communicator_t*``),
            # so it binds as ``type(c_ptr), value``; the wrapper feeds
            # it the ``MPI_Comm_f2c`` result.
            body_lines.append(f"      type(c_ptr), value :: {a.sdfg_name}")
        elif a.kind == 'scalar':
            # Scalar INPUT (``intent(in)`` or ``REAL(8), VALUE``) lives
            # as a non-transient ``Scalar`` on the SDFG -- DaCe codegen
            # emits a pass-by-value parameter, so the Fortran interface
            # must also bind by value (not via ``c_ptr``).
            body_lines.append(f"      {_fortran_c_value_type(a.dtype)}, value :: {a.sdfg_name}")
        else:
            body_lines.append(f"      type(c_ptr), value :: {a.sdfg_name}")
    init_syms = _init_sym_names(frozen)
    init_arg_decls = "".join(f"      {_init_symbol_decl(s, frozen)} :: {s}\n" for s in init_syms)
    rendered = tpl.format(entry=iface.entry,
                          c_arg_decls=",  &\n".join(header_lines),
                          c_arg_decls_body="\n".join(body_lines),
                          init_args=", ".join(init_syms),
                          init_arg_decls=init_arg_decls)
    if any(a.kind == 'mpi_comm' for a in frozen.args) or frozen.user_comm_source:
        # Splice the ``MPI_Comm_f2c`` C binding into the same interface
        # block (before its ``end interface``) so the wrapper can turn
        # the Fortran integer handle into the C ``MPI_Comm`` the SDFG
        # entry expects.
        rendered = rendered.replace("  end interface", _MPI_COMM_F2C_IFACE + _MPI_COMM_SIZE_IFACE + "  end interface")
    return rendered


# Symbols introduced by ``emit_mpi._install_user_pgrid`` -- their
# init-signature decls are special-cased here rather than via a
# generic dtype lookup because :class:`FrozenSignature.free_symbols`
# carries names only.  Keep in lockstep with the names in
# ``emit_library._USER_*``.
_USER_COMM_SYMBOL_NAME = "dace_user_comm"
_USER_COMM_SIZE_SYMBOL_NAME = "dace_user_comm_size"

_INIT_SYMBOL_DECL_OVERRIDES = {
    _USER_COMM_SYMBOL_NAME: "type(c_ptr), value",
    _USER_COMM_SIZE_SYMBOL_NAME: "integer(c_long_long), value",
}


def _init_symbol_decl(sym: str, frozen=None) -> str:
    """Fortran ``bind(c)`` decl for one ``__dace_init`` free-symbol
    parameter.

    Resolution order:

    1. :data:`_INIT_SYMBOL_DECL_OVERRIDES` for the pgrid-driving symbols
       introduced by ``emit_mpi._install_user_pgrid``.
    2. The matching ``frozen.args`` entry's dtype when the symbol is
       *also* a kernel argument (e.g. an ``int64`` AoS-allocatable
       extent ``cap_<base>_<member>``, or an ``int64`` lower-bound
       ``offset_<arr>_d<i>`` -- the init param dtype must match the
       arg dtype DaCe codegen emits, regardless of whether the bridge
       categorised it as ``kind='symbol'`` or ``kind='scalar'``).
    3. Default ``integer(c_int), value`` (ordinary shape / bound symbol).

    :param sym: SDFG free-symbol name.
    :param frozen: frozen signature consulted for (2).  May be ``None``
                   when only the override map and default are needed.
    """
    if sym in _INIT_SYMBOL_DECL_OVERRIDES:
        return _INIT_SYMBOL_DECL_OVERRIDES[sym]
    if frozen is not None:
        for a in frozen.args:
            if a.sdfg_name == sym and a.kind in ('symbol', 'scalar'):
                return f"{_fortran_c_value_type(a.dtype)}, value"
    return "integer(c_int), value"


# ``MPI_Comm_f2c(MPI_Fint) -> MPI_Comm``: converts a Fortran integer
# communicator handle to the C handle.  ``MPI_Fint`` is ``int``;
# ``MPI_Comm`` is pointer-sized on OpenMPI, returned as ``type(c_ptr)``.
_MPI_COMM_F2C_IFACE = """
    function dace_mpi_comm_f2c(fcomm) bind(c, name='MPI_Comm_f2c')
      import :: c_int, c_ptr
      integer(c_int), value :: fcomm
      type(c_ptr) :: dace_mpi_comm_f2c
    end function
"""

# ``MPI_Comm_size(comm, &size)`` -- needed so the bindings wrapper can
# size the ``__user_pgrid`` (1-D cartesian comm) before ``dace_init``.
# Bound to a static-shape ``size_buf(1)`` so the parameter is a
# normal Fortran pointer (no ``MPI_Comm`` <-> ``c_ptr`` confusion).
_MPI_COMM_SIZE_IFACE = """
    function dace_mpi_comm_size(comm, size) bind(c, name='MPI_Comm_size')
      import :: c_ptr, c_int
      type(c_ptr), value :: comm
      integer(c_int) :: size
      integer(c_int) :: dace_mpi_comm_size
    end function
"""


def _mpi_comm_local(sdfg_name: str) -> str:
    """Wrapper-local ``c_ptr`` name holding the ``MPI_Comm_f2c`` result
    for a communicator arg (kept distinct from the caller's integer
    dummy of the same Fortran name)."""
    return f"{sdfg_name}__commc"


def _free_sym_names(frozen) -> list:
    """Free symbols DaCe folds into ``__program`` (shape symbols like
    ``n``), sorted, excluding any that are already an explicit
    ``frozen.args`` entry or DaCe-internal.

    For the ``__dace_init`` argument list use :func:`_init_sym_names`
    instead -- DaCe codegen passes *every* SDFG free symbol to init,
    including those that are also kernel args (e.g. a scalar input
    that doubles as an interstate-edge condition operand)."""
    argnames = {a.sdfg_name for a in frozen.args}
    return sorted(s for s in frozen.free_symbols if s not in argnames and not s.startswith('__dace'))


def _init_sym_names(frozen) -> list:
    """Symbol list for the ``__dace_init_<entry>`` call -- DaCe's init
    routine takes every SDFG free symbol (alphabetically), regardless
    of whether the same name is also a ``__program`` kernel arg."""
    return sorted(s for s in frozen.free_symbols if not s.startswith('__dace'))


def _dace_call_order(frozen, dace_arglist) -> list:
    """The exact ``__program_<entry>`` argument order DaCe codegen
    emitted (``dace_arglist`` = ``CompiledSDFG._sig``, passed in live
    from ``build_fortran_library``).  Each name resolves to its
    ``FrozenArg``; a name with no matching arg is a free symbol,
    yielded as ``str``.

    No ``dace_arglist`` (direct ``emit_bindings`` callers / simple
    kernels) -> fall back to ``frozen.args`` order then the sorted
    free symbols."""
    by_name = {a.sdfg_name: a for a in frozen.args}
    if dace_arglist:
        return [by_name.get(n, n) for n in dace_arglist]
    return list(frozen.args) + _free_sym_names(frozen)


def _render_logical_bridge_copy_in(recipe, outer_expr: str) -> List[str]:
    """Copy-in for a ``source_logical_kind > 1`` flat companion.

    The struct member is a Fortran ``LOGICAL(KIND=N)`` (N = 2 / 4 /
    8 bytes) but the SDFG storage is 1-byte ``bool``.  Allocate the
    scratch flat at the source extents, then a whole-array
    assignment (``<flat> = <outer>%<mem>``) -- Fortran's intrinsic
    LOGICAL-kind conversion handles the per-element width change.

    Rank-0 members declare the scratch as a *scalar allocatable*
    (``logical(c_bool), allocatable, target :: <flat>``) and
    allocate / assign as a scalar; ``c_loc(<flat>)`` gives the
    address of the single byte the SDFG-side ``bool *`` reads.

    :param recipe: an ``aliasable=True`` recipe whose
        ``source_logical_kind`` is 2 / 4 / 8.  Single-flat by
        construction (struct member -> one companion).
    :param outer_expr: the entry's ``outer_expr`` (e.g.
        ``p_diag%ddt_vn_adv_is_associated``); used as the Fortran-
        side source path on the RHS.
    """
    flat = recipe.flat_names[0]
    if recipe.rank == 0:
        return [f"    allocate({flat})", f"    {flat} = {outer_expr}"]
    shape_args = ", ".join(recipe.shape_exprs)
    return [f"    allocate({flat}({shape_args}))", f"    {flat} = {outer_expr}"]


def _render_logical_bridge_copy_out(recipe, outer_expr: str) -> List[str]:
    """Inverse of :func:`_render_logical_bridge_copy_in`: pack the
    SDFG-side bool flat back into the source struct slot for
    ``intent(out)/inout`` entries, then release the scratch."""
    flat = recipe.flat_names[0]
    return [f"    {outer_expr} = {flat}", f"    deallocate({flat})"]


def _fortran_c_value_type(dtype: str) -> str:
    """Map a frozen-arg ``dtype`` string to its ``iso_c_binding`` form
    for a pass-by-value Fortran dummy."""
    table = {
        'int32': 'integer(c_int)',
        'int64': 'integer(c_long_long)',
        'float32': 'real(c_float)',
        'float64': 'real(c_double)',
        'bool': 'logical(c_bool)',
        'complex64': 'complex(c_float_complex)',
        'complex128': 'complex(c_double_complex)',
    }
    if dtype not in table:
        raise ValueError(f"_fortran_c_value_type: unsupported scalar dtype {dtype!r} -- "
                         "extend the dtype map for new pass-by-value scalar shapes.")
    return table[dtype]


# ---------------------------------------------------------------------------
# Ref-counted handle state
# ---------------------------------------------------------------------------


def build_handle_state(iface: OriginalInterface) -> str:
    """Render the module-level ``dace_handle`` + ``init_count``
    declarations.

    Template: ``templates/handle_state.f90.in``.

    :param iface: only ``iface.entry`` is read (for comment text).
    :returns: the rendered block  --  the two ``save``-scoped
              variables that ``<entry>_dace`` and
              ``<entry>_dace_finalize`` share.
    """
    return _load("handle_state.f90.in").format(entry=iface.entry)


# ---------------------------------------------------------------------------
# Wrapper head  --  dummy decls, flat pointer / scratch decls, symbol / iter locals
# ---------------------------------------------------------------------------


def _enum_args(iface: OriginalInterface, enum_maps: dict) -> dict:
    """Filter ``enum_maps`` down to the iface's actual outer dummies.

    ``enum_maps`` is the ``{arg_name: {literal_lower: int}}`` table
    surfaced by :func:`rewrite_string_enum_to_integer`.  Returns the
    subset whose keys appear in ``iface.args`` (matched
    case-insensitively against ``OriginalArg.name``), preserving each
    entry's literal -> int mapping verbatim.  An ``enum_maps`` key
    that doesn't match any dummy is silently dropped: the dummy may
    have been renamed by ``hlfir-flatten-structs`` or eliminated by
    a later bridge pass.
    """
    if not enum_maps:
        return {}
    iface_names = {a.name.lower() for a in iface.args}
    return {a: m for a, m in enum_maps.items() if a.lower() in iface_names}


def _enum_local_name(arg_name: str) -> str:
    """Local INTEGER scratch name for an enum-mapped CHARACTER dummy.

    Uses the ``dace_enum_<arg>`` prefix the rest of the binding layer
    already follows for its synthesised names (``dace_handle``,
    ``dace_program_<entry>``, ``dace_init_<entry>``, ...).  The
    ``dace_`` namespace is reserved for bridge-emitted identifiers --
    a user-source variable starting with ``dace_`` is by convention
    out-of-bounds, so this avoids the collision risk of a
    ``<arg>__enum`` shape that a kernel could plausibly use itself
    (``flag__enum``, ``column__enum``, ...).  When even that's
    insufficient (a user kernel that defines its own
    ``dace_enum_<arg>``) the synthesised SELECT CASE will fail to
    parse loudly under flang rather than silently shadow.
    """
    return f"dace_enum_{arg_name}"


def _enum_literal_case_clause(literal: str) -> str:
    """Render the ``CASE ('lower', 'UPPER')`` Fortran list for one
    enum literal, matching the lowercase + uppercase variants -- the
    QE ``flag == 'c' .OR. flag == 'C'`` shape the preprocess pass
    collapses to a single integer entry."""
    lower = literal.lower()
    upper = literal.upper()
    if lower == upper:  # digits / symbols
        return f"CASE ('{lower}')"
    return f"CASE ('{lower}', '{upper}')"


def _optional_local_name(sdfg_name: str) -> str:
    """Wrapper-local that holds a forwarded OPTIONAL outer dummy's data.

    The outer dummy is declared ``optional`` so the caller may omit it;
    referencing it when absent (by-value scalar, or ``c_loc`` on an array)
    is undefined.  We route the data through this guarded local instead --
    set from the actual when ``present(...)``, else a degenerate zero -- so
    the value/pointer the SDFG receives is always valid storage.
    """
    return f"{sdfg_name}__opt"


def _optional_outer_dummies(frozen: FrozenSignature, iface: OriginalInterface) -> list:
    """``(OriginalArg, FrozenArg)`` for every wrapper outer dummy declared
    OPTIONAL whose presence the SDFG branches on via ``<name>_present``.

    Scoped to SCALAR optionals: their value reaches the SDFG by value, so a
    guarded local (see :func:`_optional_local_name`) -- set from the actual
    when ``present(...)``, degenerate otherwise -- is all that is needed to
    never reference an omitted optional.  ARRAY optionals additionally need
    their ``<name>_d<i>`` extent symbols guarded (``size(absent)`` is UB), a
    larger change left as follow-up; until then an array optional keeps the
    default-absent presence.  Empty for the common no-optional kernel."""
    optional_names = {a.name.lower() for a in iface.args if getattr(a, 'optional', False)}
    if not optional_names:
        return []
    by_name = {a.name.lower(): a for a in iface.args}
    pairs = []
    for fa in frozen.args:
        if fa.kind != 'scalar' or getattr(fa, 'rank', 0) != 0:
            continue
        fn = (fa.fortran_name or '').lower()
        if fn in optional_names:
            pairs.append((by_name[fn], fa))
    return pairs


def build_wrapper_head(frozen: FrozenSignature,
                       iface: OriginalInterface,
                       plan: FlattenPlan,
                       enum_maps: dict = None) -> str:
    """Render the ``<entry>_dace`` subroutine header + declaration
    section.

    Walks ``plan.entries``:
        * aliasable recipes -> ``<type>, pointer :: <flat>(:,:,...)``
        * non-aliasable     -> ``<type>, allocatable, target :: <flat>(:,:)``
    Free symbols that aren't already outer dummies become local
    ``integer(c_int)`` scalars; ``i1..iN`` loop iters are declared
    whenever any non-aliasable recipe exists.

    :param frozen: for the free-symbol list.
    :param iface: outer dummies drive the subroutine signature.
    :param plan: drives pointer-vs-scratch decisions per flat.
    :returns: everything from ``subroutine <entry>_dace(...)`` through
              the final declaration line; does NOT include the body.

    Template: ``templates/wrapper_head.f90.in``.  Example fragment::

        subroutine kernel_dace(fld, n, m)
          type(t_fields), intent(inout), target :: fld
          integer(c_int), intent(in),    target :: n
          integer(c_int), intent(in),    target :: m
          real(c_double), pointer :: fld_a(:,:)
          real(c_double), pointer :: fld_b(:,:)
          integer(c_int) :: dace_err
    """
    tpl = _load("wrapper_head.f90.in")
    outer_dummy_names = [a.name for a in iface.args]
    outer_dummy_set = set(outer_dummy_names)
    enum_args = _enum_args(iface, enum_maps or {})

    # An arg the ORIGINAL entry declared POINTER (e.g. ICON ocean
    # ``REAL(8), POINTER, INTENT(INOUT) :: vn(:,:,:)``) carries a presence
    # guard ``<name>_allocated`` in the SDFG free symbols, and its presence
    # fold is emitted as ``merge(1, 0, associated(<name>))`` -- legal ONLY on a
    # POINTER dummy.  Declaring such an arg plain ``target`` makes gfortran
    # reject ``associated`` (``'pointer' argument ... must be a POINTER``), so
    # mirror the pointer attribute.  ``c_loc`` accepts an associated POINTER, so
    # the wrapper body is unaffected.  This matches the presence fold's own
    # pointer-default (``arg_is_pointer_local.get(base, True)``).
    _free_syms = set(frozen.free_symbols)
    ptr_outer_args = {a.name for a in iface.args if f"{a.name}_allocated" in _free_syms}

    # An enum-mapped arg comes through ``rewrite_string_enum_to_integer``
    # as an ``INTEGER, INTENT(IN)`` dummy on the SDFG side, but the
    # caller's surface is still ``CHARACTER(LEN=N)`` -- the binding's
    # job is to bridge the two by accepting the string outside and
    # converting it to the integer inside.  Override the outer
    # ``fortran_type`` accordingly and remove the ``target`` attr
    # (a character dummy needs no ``c_loc``, the converted INTEGER
    # local is what reaches the SDFG).
    def _outer_decl(a) -> str:
        if a.name.lower() in enum_args:
            literals = enum_args[a.name.lower()]
            max_len = max((len(lit) for lit in literals), default=1)
            return (f"    character(len={max_len}),"
                    f" intent({a.intent or 'in'}) :: {a.name}")
        opt = ", optional" if getattr(a, 'optional', False) else ""
        attr = "pointer" if a.name in ptr_outer_args else "target"
        return (f"    {a.fortran_type},"
                f" intent({a.intent or 'inout'}), {attr}{opt} :: {a.name}"
                f"{_dim_spec(a.shape, {d.lower() for d in outer_dummy_names})}")

    outer_dummy_decls = "\n".join(_outer_decl(a) for a in iface.args)

    # One ``integer(c_int)`` scratch per enum-mapped arg holds the
    # ``SELECT CASE``-translated value the SDFG actually receives.
    # Declared in the head, populated in the body, passed to the call
    # in the tail.
    enum_local_decls = "\n".join(f"    integer(c_int) :: {_enum_local_name(a.name)}" for a in iface.args
                                 if a.name.lower() in enum_args)

    flat_ptr_lines: List[str] = []
    scratch_lines: List[str] = []
    # Element dtypes needing a ``presence_scratch_<dtype>(1)`` target: the
    # degenerate binding an ABSENT deferred-storage member's flat POINTER is
    # bounds-remapped onto (copy-in guard's ELSE branch).
    guard_scratch_dtypes: set = set()
    max_loop_rank = 0
    for entry in live_entries(frozen, plan):
        r = entry.recipe
        ftype = _fortran_type(r.scratch_dtype)
        # Rank-0 (a scalar struct member, e.g. ``s%scal``) takes no
        # array spec  --  ``real :: x()`` is invalid Fortran; it must
        # be declared as a plain scalar ``pointer``.
        shape_dims = ("(" + ", ".join(":" for _ in range(r.rank)) + ")") if r.rank > 0 else ""
        # A LOGICAL(KIND=N) struct member with N > 1 (default
        # Fortran LOGICAL is N=4) needs a width-bridging scratch
        # even though its access pattern would otherwise alias:
        # ``c_loc(<outer>%<mem>) + c_f_pointer(..., logical(c_bool))``
        # reinterprets a 4-byte slot as 1 byte, so a SDFG write
        # leaves the upper 3 bytes garbage and an adjacent struct
        # field's heap chunk metadata can be clobbered (the
        # ``free(): invalid next size`` diagnostic the ICON
        # velocity e2e surfaced).  Force scratch + element copy
        # so Fortran's intrinsic LOGICAL-kind conversion handles
        # the per-element width change exactly like the existing
        # top-level ``_build_logical_bridges`` does for outer
        # LOGICAL dummies.
        if r.aliasable and r.source_logical_kind in (0, 1):
            for flat in r.flat_names:
                flat_ptr_lines.append(f"    {ftype}, pointer :: {flat}{shape_dims}")
            if _recipe_presence_guard(iface, r):
                guard_scratch_dtypes.add(r.scratch_dtype)
        else:
            if r.rank > 0:
                max_loop_rank = max(max_loop_rank, r.rank)
            for flat in r.flat_names:
                scratch_lines.append(f"    {ftype}, allocatable, target :: {flat}{shape_dims}")

    for dt in sorted(guard_scratch_dtypes):
        flat_ptr_lines.append(f"    {_fortran_type(dt)}, target :: {_presence_scratch_name(dt)}(1)")

    # A free symbol that is also a frozen arg (an ``<arr>_d<i>``
    # extent / ``offset_...`` lower bound the bridge passes by value)
    # must be declared with *that arg's* C type  --  the bridge sizes
    # these as ``int64`` on real ICON signatures, and the C interface
    # block already binds them ``integer(c_long_long), value``.  A
    # hardcoded ``integer(c_int)`` local mismatches the bind(c) dummy.
    sym_dtype = {a.sdfg_name: a.dtype for a in frozen.args if a.kind in ('scalar', 'symbol')}
    # A rank-0 scalar struct member (``p_patch%nlev``) is BOTH a flat
    # companion (declared above as a ``pointer``) and a free symbol
    # the SDFG wants by value.  It already has storage from the
    # flat-companion decl  --  re-declaring it here is a duplicate
    # ``already has basic type`` error.  Skip any free symbol that is
    # also a flat companion.
    # LIVE entries only: a dead entry declares no companion (see
    # ``live_entries``), so a free symbol that shares its name must get its own
    # scalar declaration here rather than be skipped as "already declared".
    flat_names = {f for entry in live_entries(frozen, plan) for f in entry.recipe.flat_names}
    # A symbol that is ALSO an orphan / AoS module-global arg is already given a
    # ``target`` local by those paths (``uspp_param::lmaxq`` -- a module global the
    # bridge ALSO lifted into a dimension free-symbol).  Re-declaring it here is a
    # duplicate ``already has basic type`` error, so skip it (the orphan decl wins).
    module_arg_names = ({a.sdfg_name
                         for a, _m, _mem in _orphan_module_args(frozen, iface, plan)}
                        | {a.sdfg_name
                           for a in _aos_module_args(frozen)})

    def _local_decl(s: str) -> str:
        """Wrapper-local Fortran decl for one free symbol.  Mirrors the
        ``init_arg_decls`` overrides from ``build_c_interface`` so the
        ``__user_comm`` / ``__user_comm_size`` pgrid params are typed
        consistently across the interface block and the wrapper scope."""
        if s == _USER_COMM_SYMBOL_NAME:
            return "type(c_ptr)"
        if s == _USER_COMM_SIZE_SYMBOL_NAME:
            return "integer(c_long_long)"
        return _fortran_c_value_type(sym_dtype.get(s, 'int32'))

    symbol_decls = "\n".join(f"    {_local_decl(s)} :: {s}" for s in frozen.free_symbols
                             if s not in outer_dummy_set and s not in flat_names and s not in module_arg_names)
    if max_loop_rank:
        iter_decl = "    integer(c_int) :: " + ", ".join(f"i{d + 1}" for d in range(max_loop_rank))
        symbol_decls = (symbol_decls + "\n" + iter_decl) if symbol_decls else iter_decl

    # Orphan module-global args: a wrapper-local ``target`` per arg,
    # filled from the renamed module import in build_wrapper_body.
    for a, _mod, _member in _orphan_module_args(frozen, iface, plan):
        ftype = _fortran_c_value_type(a.dtype)
        spec = "(" + ", ".join(":" for _ in range(a.rank)) + ")" if a.rank > 0 else ""
        kw = "allocatable, target" if a.rank > 0 else "target"
        scratch_lines.append(f"    {ftype}, {kw} :: {a.sdfg_name}{spec}")

    # Module-global AoS-struct component SoA buffers: a wrapper-local
    # ``allocatable, target`` per arg, plus a loop index + one cap per member
    # dim (filled in build_wrapper_body / drained in build_wrapper_tail).
    for a in _aos_module_args(frozen):
        ftype = _fortran_c_value_type(a.dtype)
        spec = "(" + ", ".join(":" for _ in range(a.rank)) + ")" if a.rank else ""
        scratch_lines.append(f"    {ftype}, allocatable, target :: {a.sdfg_name}{spec}")
        its, caps, _mrank, _elem = _aos_loop_pieces(a)
        int_names = its + caps  # scalar-struct (outer_rank 0): no loop vars
        if int_names:
            scratch_lines.append(f"    integer(c_int) :: {', '.join(int_names)}")

    # Absent-optional data buffers with no host source: a degenerate local.
    for a in _unsourced_array_args(frozen, iface, plan):
        ftype = _fortran_c_value_type(a.dtype)
        spec = "(" + ", ".join(":" for _ in range(a.rank)) + ")"
        scratch_lines.append(f"    {ftype}, allocatable, target :: {a.sdfg_name}{spec}")

    # Unsourced scalar optionals + undeclared shape symbols (declarations only;
    # values are written in build_wrapper_body).
    for name, ftype, _rhs in _extra_local_symbols(frozen, iface, plan):
        scratch_lines.append(f"    {ftype} :: {name}")

    # Guarded scalar locals for FORWARDED optional outer dummies: the by-value
    # data the SDFG reads routes through these so an omitted optional is never
    # referenced.  Filled in build_wrapper_body, passed in build_wrapper_tail.
    for _oa, fa in _optional_outer_dummies(frozen, iface):
        local = _optional_local_name(fa.sdfg_name)
        scratch_lines.append(f"    {_fortran_c_value_type(fa.dtype)} :: {local}")

    bridge_decls, _, _, _ = _build_logical_bridges(frozen, iface)
    if bridge_decls:
        scratch_lines = scratch_lines + bridge_decls

    # One ``type(c_ptr)`` local per communicator arg, holding the
    # ``MPI_Comm_f2c`` result fed to the SDFG call (the outer dummy
    # itself stays the caller's Fortran ``integer`` handle).
    for a in frozen.args:
        if a.kind == 'mpi_comm':
            scratch_lines.append(f"    type(c_ptr) :: {_mpi_comm_local(a.sdfg_name)}")
    # Pgrid path (the modern replacement for the opaque-MPI_Comm scalar
    # ``mpi_comm`` arg above): one ``integer(c_int)`` scratch for the
    # ``MPI_Comm_size`` return + one ``integer(c_int)`` for the call's
    # error code.  ``__user_comm`` / ``__user_comm_size`` themselves
    # are declared above in ``symbol_decls`` (as free symbols).
    if frozen.user_comm_source:
        scratch_lines.append("    integer(c_int) :: dace_user_comm_size_buf")
        scratch_lines.append("    integer(c_int) :: dace_user_comm_size_err")
    if enum_local_decls:
        scratch_lines.append(enum_local_decls)

    return tpl.format(
        entry=iface.entry,
        outer_dummy_list=", ".join(outer_dummy_names),
        outer_dummy_decls=outer_dummy_decls or "    ! (no dummies)",
        flat_ptr_decls="\n".join(flat_ptr_lines) or "    ! (no flat pointers)",
        scratch_decls="\n".join(scratch_lines) or "    ! (no scratch)",
        symbol_decls=symbol_decls or "    ! (no free symbols)",
    )


# ---------------------------------------------------------------------------
# Wrapper body  --  per-entry alias calls / copy-in loops, symbol population
# ---------------------------------------------------------------------------


def build_wrapper_body(frozen: FrozenSignature,
                       iface: OriginalInterface,
                       plan: FlattenPlan,
                       enum_maps: dict = None) -> str:
    """Render the between-declaration-and-SDFG-call block  --  for each
    ``FlattenEntry`` either alias it (zero-copy) or allocate + copy
    in, then populate SDFG free symbols from ``size(...)`` on the
    outer storage.

    :param frozen: for the free-symbol set.
    :param iface: to skip symbols that are already outer dummies.
    :param plan: drives per-entry alias-vs-copy emission.
    :returns: indented Fortran lines ready to slot between the
              wrapper head's declarations and the SDFG call.
    """
    outer_dummy_set = {a.name for a in iface.args}
    enum_args = _enum_args(iface, enum_maps or {})
    body: List[str] = []
    # Enum-mapped CHARACTER dummy -> ``SELECT CASE`` -> INTEGER scratch.
    # Runs FIRST so the converted value is ready by the time the
    # symbol-population block (which may reference it) executes.
    if enum_args:
        body.append("    ! ----- String-enum CHARACTER -> integer conversion -----")
        for a in iface.args:
            tbl = enum_args.get(a.name.lower())
            if not tbl:
                continue
            local = _enum_local_name(a.name)
            body.append(f"    select case ({a.name})")
            for literal, value in sorted(tbl.items(), key=lambda kv: kv[1]):
                body.append(f"    {_enum_literal_case_clause(literal)}")
                body.append(f"      {local} = {value}")
            body.append("    case default")
            # ``-1`` is the bridge's "unknown enum value" sentinel;
            # the SDFG's IF chain has no branch for it so the kernel
            # falls through to the source's ``ELSE`` (or to a no-op
            # if there isn't one).  An explicit error stop would
            # change observable behaviour, so the bridge sticks to
            # the source's already-permissive default-fallthrough.
            body.append(f"      {local} = -1")
            body.append("    end select")
        body.append("")
    # Copy-in / alias for each flatten entry (+ the LOGICAL-kind bridge copy-in)
    # is BUILT here but DEFERRED -- appended to ``body`` only after the
    # module-global reconstruction blocks below.  A double-buffer AoR alias
    # (``c_f_pointer(c_loc(ocean_state%p_prog(nold(1))%h), ...)``) subscripts the
    # record array with a module-global time-level symbol (``nold``/``nnew``);
    # those are orphan module args, allocated + seeded only by the
    # ``Module-global args sourced from use-imports`` block.  Emitting the alias
    # first would subscript an unallocated ``nold`` -> SIGSEGV.  Copy-in reads no
    # symbol / buffer this wrapper builds (its shapes are ``size(<member>)`` on
    # the caller's storage), so it has no forward dependency on the blocks it now
    # trails; the buffer-derived symbol block still follows it.
    copyin: List[str] = []
    copyin.append("    ! ----- Copy-in / alias per flatten entry -----")
    for entry in live_entries(frozen, plan):
        r = entry.recipe
        # Four mutually exclusive emitter shapes  --  see FlattenRecipe
        # for the flag matrix.  ``source_logical_kind > 1`` overrides
        # the aliasable path with a width-bridging scratch + Fortran
        # intrinsic LOGICAL-kind conversion (see ``build_wrapper_head``
        # for the rationale).
        #
        # A deferred-storage member (POINTER / ALLOCATABLE) may be absent
        # at run time -- its descriptor bounds are then undefined, and the
        # unguarded ``c_loc``/``size`` marshal reads garbage (gfortran's
        # ``internal_pack`` at the alias site smashes the stack).  Guard
        # the marshal; the ABSENT branch gives the flat a defined
        # degenerate binding so the SDFG call's ``c_loc(<flat>)`` stays
        # valid (the kernel takes the member's absent branch and never
        # dereferences it).
        guard = _recipe_presence_guard(iface, r)
        if r.aos_alloc:
            copyin.extend(render_aos_alloc_pack_in(r, entry.outer_expr))
        elif r.aliasable and r.source_logical_kind in (0, 1):
            lines = render_alias_calls(r)
            if guard:
                copyin.append(f"    if ({guard}) then")
                copyin.extend("  " + ln for ln in lines)
                copyin.append("    else")
                scratch = _presence_scratch_name(r.scratch_dtype)
                for flat in r.flat_names:
                    if r.rank > 0:
                        bounds = ", ".join("1:1" for _ in range(r.rank))
                        copyin.append(f"      {flat}({bounds}) => {scratch}")
                    else:
                        copyin.append(f"      {flat} => {scratch}(1)")
                copyin.append("    end if")
            else:
                copyin.extend(lines)
        else:
            if r.aliasable and r.source_logical_kind > 1:
                lines = _render_logical_bridge_copy_in(r, entry.outer_expr)
            else:
                lines = render_copy_in_loop(r)
            if guard:
                copyin.append(f"    if ({guard}) then")
                copyin.extend("  " + ln for ln in lines)
                copyin.append("    else")
                for flat in r.flat_names:
                    degen = "(" + ", ".join("1" for _ in range(r.rank)) + ")" if r.rank > 0 else ""
                    copyin.append(f"      allocate({flat}{degen})")
                    copyin.append(f"      {flat} = {_zero_literal(r.scratch_dtype)}")
                copyin.append("    end if")
            else:
                copyin.extend(lines)

    _, copy_in_lines, _, _ = _build_logical_bridges(frozen, iface)
    if copy_in_lines:
        copyin.append("")
        copyin.append("    ! ----- LOGICAL -> logical(c_bool) bridge (copy-in) -----")
        copyin.extend(copy_in_lines)

    # Symbol population is SPLIT by data dependency.  The orphan / AoS /
    # unsourced blocks below emit ``allocate(buf(<shape syms>))``; a shape sym
    # sourced from a module global / dummy / constant (``nqs``, ``nks``,
    # ``dfftt_ngm``, ``qvan_init_nij`` ...) must be assigned BEFORE those
    # allocates, while a buffer-EXTENT sym (``becxx_k_d0 = size(becxx_k)``) can
    # only be assigned AFTER its buffer exists.  So input-derived assignments go
    # up here; buffer-derived ones stay at the tail (emitted after the allocates).
    sym_lines = _build_symbol_assigns(frozen, plan, outer_dummy_set, iface)
    extra_syms = _extra_local_symbols(frozen, iface, plan)
    _buffer_names = {a.sdfg_name for a, _m, _mem in _orphan_module_args(frozen, iface, plan) if a.rank > 0}
    _buffer_names |= {a.sdfg_name for a in _aos_module_args(frozen)}
    _buffer_names |= {a.sdfg_name for a in _unsourced_array_args(frozen, iface, plan)}
    # Copy-in / alias companions (``ocean_state_p_prog_nold_vn`` and friends) are
    # also buffer-derived: a symbol reading their ``lbound`` / ``size`` -- notably a
    # double-buffer lane's ``offset_<companion>_d<i> = lbound(<companion>, dim=i)``
    # -- can only run AFTER the copy-in that ``c_f_pointer``-associates the
    # companion, which now trails the module-global block.  Sorting those offsets
    # into the late (post-copy-in) symbol block keeps them from reading an
    # unassociated pointer (garbage lbound -> out-of-bounds SDFG index).  A symbol
    # reading the STRUCT MEMBER instead (``size(ocean_state%p_prog(nold(1))%vn)``)
    # does not textually contain the companion name, so it is unaffected.
    if plan is not None:
        _buffer_names |= {f for e in plan.entries for f in e.recipe.flat_names}

    def _buffer_derived(rhs: str) -> bool:
        # A symbol whose RHS reads ``size``/``lbound`` of a buffer this wrapper
        # allocates below cannot be assigned until that buffer exists.
        return any(re.search(r'\b' + re.escape(b) + r'\b', rhs) for b in _buffer_names)

    early_syms = [ln for ln in sym_lines if not _buffer_derived(ln)]
    late_syms = [ln for ln in sym_lines if _buffer_derived(ln)]
    early_extra = [(n, ft, r) for (n, ft, r) in extra_syms if not _buffer_derived(r)]
    late_extra = [(n, ft, r) for (n, ft, r) in extra_syms if _buffer_derived(r)]
    if early_syms or early_extra:
        body.append("")
        body.append("    ! ----- Symbol population (input-derived; before allocates) -----")
        body.extend(early_syms)
        for name, _ftype, rhs in early_extra:
            body.append(f"    {name} = {rhs}")

    orphans = _orphan_module_args(frozen, iface, plan)
    if orphans:
        body.append("")
        body.append("    ! ----- Module-global args sourced from use-imports -----")
        for a, _mod, _member in orphans:
            alias = _module_symbol_alias(a.sdfg_name)
            alloc_inside = getattr(a, 'global_alloc_inside', False)
            is_alloc = getattr(a, 'module_origin_allocatable', False)
            is_ptr = getattr(a, 'module_origin_pointer', False)
            # A host global with DEFERRED storage (ALLOCATABLE / POINTER) may be
            # unallocated on entry: the kernel only reads it under a condition
            # that the no-op path doesn't take, so the caller need not allocate
            # it.  Its declared SDFG extents are its OWN size symbols
            # (``egrp_pairs(egrp_pairs_d0, ...)`` where ``egrp_pairs_d0 ==
            # size(egrp_pairs)``), so an explicit allocate is also circular.
            # Handle both: guard the copy with ``allocated``/``associated`` and
            # auto-(re)allocate from the host when present, else fall back to a
            # degenerate buffer so ``c_loc`` stays valid.
            deferred = (is_alloc or is_ptr) and not alloc_inside and a.rank > 0
            if alloc_inside:
                # The kernel ALLOCATEs this global itself: the host alias holds
                # no data on entry (reading it would be UB).  The allocate gives
                # the SDFG its buffer; the kernel fills it; the write-back
                # assigns it to the host global on exit.
                if a.rank > 0:
                    dims = ", ".join(a.shape) if a.shape else "1"
                    body.append(f"    allocate({a.sdfg_name}({dims}))")
                body.append(f"    ! {a.sdfg_name}: kernel-allocated, no copy-in "
                            f"(host alias unallocated on entry)")
            elif deferred:
                degen = ", ".join("1" for _ in range(a.rank))
                body.append(f"    if ({_present(alias, is_ptr)}) then")
                body.append(f"      {a.sdfg_name} = {alias}")
                body.append(f"    else")
                body.append(f"      allocate({a.sdfg_name}({degen}))  "
                            f"! host unallocated (absent on no-op path)")
                body.append(f"    end if")
            else:
                # Static / explicit-shape array (always present) or a scalar
                # module global.  Allocate to the SDFG arg's own concrete extent
                # (``FrozenArg.shape``) -- NOT ``size(alias)``: the source can be
                # a *scalar* module global the bridge lifted to a length-1 array,
                # and ``size(scalar)`` is a hard Fortran error.
                if a.rank > 0:
                    dims = ", ".join(a.shape) if a.shape else "1"
                    body.append(f"    allocate({a.sdfg_name}({dims}))")
                body.append(f"    {a.sdfg_name} = {alias}")

    aos_args = _aos_module_args(frozen)
    if aos_args:
        body.append("")
        body.append("    ! ----- Module-global AoS-struct components (AoS->SoA) -----")
        for a in aos_args:
            body.extend(_render_aos_copy_in(a))

    unsourced = _unsourced_array_args(frozen, iface, plan)
    if unsourced:
        body.append("")
        body.append("    ! ----- Absent-optional buffers (degenerate, no host source) -----")
        for a in unsourced:
            # Size the placeholder at the arg's REAL SDFG extent when the shape is
            # fully recoverable: the SDFG may WRITE an absent-optional buffer that is
            # actually an inlined-callee LOCAL exposed as a program arg (``vort_v`` /
            # ``vort_flux`` zero-fills), and a degenerate ``(1,1,..)`` local overruns
            # on any mesh-sized write.  The shape symbols (``nproma``/``n_zlev``/
            # ``nblks_*``) are assigned above by ``_build_symbol_assigns``.  Fall back
            # to ``(1,..)`` only when the shape is not fully symbol-recoverable.
            shape = tuple(str(s) for s in (a.shape or ()))
            if len(shape) == a.rank and all(shape):
                dims = ", ".join(shape)
            else:
                dims = ", ".join("1" for _ in range(a.rank))
            body.append(f"    allocate({a.sdfg_name}({dims}))")
            body.append(f"    {a.sdfg_name} = {_zero_literal(a.dtype)}")

    optional_dummies = _optional_outer_dummies(frozen, iface)
    if optional_dummies:
        body.append("")
        body.append("    ! ----- Forwarded optional dummies: guarded data local -----")
        for oa, fa in optional_dummies:
            local = _optional_local_name(fa.sdfg_name)
            body.append(f"    if (present({oa.name})) then")
            body.append(f"      {local} = {oa.name}")
            body.append(f"    else")
            body.append(f"      {local} = {_zero_literal(fa.dtype)}")
            body.append(f"    end if")

    comm_args = [a for a in frozen.args if a.kind == 'mpi_comm']
    if comm_args:
        body.append("")
        body.append("    ! ----- Fortran integer comm -> C MPI_Comm -----")
        for a in comm_args:
            body.append(f"    {_mpi_comm_local(a.sdfg_name)} = dace_mpi_comm_f2c({a.fortran_name})")

    if frozen.user_comm_source:
        body.append("")
        body.append("    ! ----- Fortran integer comm -> __user_comm / __user_comm_size -----")
        body.append("    ! Convert the caller's MPI_Fint to a C MPI_Comm, query its")
        body.append("    ! size, and hand both to ``dace_init_<entry>`` so the")
        body.append("    ! FortranProcessGrid's MPI_Cart_create runs against the")
        body.append("    ! user's communicator (not MPI_COMM_WORLD).")
        body.append(f"    {_USER_COMM_SYMBOL_NAME} = dace_mpi_comm_f2c({frozen.user_comm_source})")
        body.append(
            f"    dace_user_comm_size_err = dace_mpi_comm_size({_USER_COMM_SYMBOL_NAME}, dace_user_comm_size_buf)")
        body.append(f"    {_USER_COMM_SIZE_SYMBOL_NAME} = int(dace_user_comm_size_buf, c_long_long)")

    # ----- Copy-in / alias (deferred from the top of the body): now safe -- the
    # module-global time-level indices (``nold``/``nnew``) its double-buffer
    # aliases subscript have been allocated + seeded by the blocks above. -----
    body.append("")
    body.extend(copyin)

    # Buffer-derived symbols: the EXTENT of a wrapper-allocated buffer
    # (``becxx_k_d0 = size(becxx_k)``), valid only now that the allocates above
    # have run.  Input-derived symbols were emitted before the allocates.
    if late_syms or late_extra:
        body.append("")
        body.append("    ! ----- Symbol population (buffer-derived; after allocates) -----")
        body.extend(late_syms)
        for name, _ftype, rhs in late_extra:
            body.append(f"    {name} = {rhs}")
    return "\n".join(body)


# ---------------------------------------------------------------------------
# Wrapper tail  --  init-count bump, SDFG call, copy-back, deallocate, end sub
# ---------------------------------------------------------------------------


def build_wrapper_tail(frozen: FrozenSignature,
                       iface: OriginalInterface,
                       plan: FlattenPlan,
                       dace_arglist: tuple = (),
                       enum_maps: dict = None) -> str:
    """Render the tail of the wrapper: init-count bump + ``call
    dace_program_<entry>`` + copy-back for every non-aliased
    writeable entry, then deallocate scratch, then close the
    subroutine.

    :param frozen: for the SDFG call argument list.
    :param iface: for the entry name.
    :param plan: for per-entry writeback decisions.
    :returns: everything after ``build_wrapper_body`` through the
              final ``end subroutine <entry>_dace``.

    Template:
        ``templates/wrapper_call.f90.in`` supplies the init bump,
        the SDFG call, and the ``end subroutine`` + finalize marker.
        We splice the copy-back block in before the end marker.
    """
    tpl = _load("wrapper_call.f90.in")
    _, _, bridge_copy_out, name_override = _build_logical_bridges(frozen, iface)

    # Enum-mapped CHARACTER dummies pass their ``SELECT CASE``-converted
    # INTEGER scratch to the SDFG, not the outer ``CHARACTER`` itself.
    # Extend ``name_override`` so ``_call_actual`` picks up the swap
    # alongside the existing LOGICAL-bridge overrides.
    enum_args = _enum_args(iface, enum_maps or {})
    name_override = dict(name_override)
    if enum_args:
        # ``_call_actual`` keys ``name_override`` by ``FrozenArg.
        # sdfg_name``; map each iface arg's outer name to its frozen
        # entry's sdfg_name (typically identical for plain scalar
        # dummies, but plumbed defensively for any future flatten-
        # induced rename).
        frozen_by_fortran = {fa.fortran_name.lower(): fa for fa in frozen.args if fa.fortran_name}
        for a in iface.args:
            if a.name.lower() not in enum_args:
                continue
            fa = frozen_by_fortran.get(a.name.lower())
            sdfg_name = fa.sdfg_name if fa is not None else a.name
            name_override[sdfg_name] = _enum_local_name(a.name)

    # Forwarded optional dummies pass their guarded local (set from the actual
    # when present, degenerate otherwise) -- never the outer dummy directly,
    # which is undefined to reference when the caller omitted it.
    for _oa, fa in _optional_outer_dummies(frozen, iface):
        name_override[fa.sdfg_name] = _optional_local_name(fa.sdfg_name)

    # The C interface declares every array arg as ``type(c_ptr), value``
    # (see ``build_c_interface``).  Wrap each array actual with
    # ``c_loc(...)`` so Fortran's bind(c) conversion is explicit -- the
    # implicit conversion from a Fortran pointer / target array to
    # ``c_ptr`` only kicks in for ``intent``-typed dummies, NOT for
    # ``type(c_ptr), value`` dummies (gfortran rejects with "type
    # mismatch ... passed REAL/LOGICAL to TYPE(c_ptr)").  Scalars and
    # symbols pass by value -- no wrapping.
    def _call_actual(a) -> str:
        if isinstance(a, str):
            # Free symbol -- the wrapper-local DaCe sees by value
            # (declared + populated from ``size(...)`` earlier).
            return a
        actual = name_override.get(a.sdfg_name, a.sdfg_name)
        if a.kind == 'mpi_comm':
            # The C ``MPI_Comm`` (f2c result), not the integer dummy.
            return _mpi_comm_local(a.sdfg_name)
        if a.kind == 'array' or a.rank > 0:
            return f"c_loc({actual})"
        return actual

    call_args = ",  &\n".join(f"      {_call_actual(a)}" for a in _dace_call_order(frozen, dace_arglist))
    call_block = tpl.format(entry=iface.entry,
                            call_arg_list=call_args,
                            init_call_args=", ".join(_init_sym_names(frozen)))

    copy_out_lines: List[str] = []
    for entry in live_entries(frozen, plan):
        r = entry.recipe
        if r.aos_alloc:
            if entry.writeback_intent in ('out', 'inout'):
                copy_out_lines.extend(render_aos_alloc_pack_out(r, entry.outer_expr))
            else:
                # intent(in): no copy-back, but the scratch buffer
                # was allocated unconditionally in pack-in and still
                # needs releasing.
                copy_out_lines.append(f"    deallocate({r.flat_names[0]})")
            continue
        # ``source_logical_kind > 1`` overrides the aliasable path
        # with a width-bridging scratch (see ``build_wrapper_head``);
        # the scratch was allocated unconditionally in copy-in and
        # needs releasing.  For ``out`` / ``inout`` add the
        # ``<outer> = <flat>`` assignment ahead of the deallocate.
        # Presence guard mirrors the copy-in: an ABSENT deferred-storage
        # member was never marshalled (its flat holds a degenerate scratch),
        # so the writeback must not touch the member -- but scratch the
        # copy-in DID allocate in its ELSE branch still needs releasing.
        guard = _recipe_presence_guard(iface, r)
        if r.aliasable and r.source_logical_kind > 1:
            if entry.writeback_intent in ('out', 'inout'):
                lines = _render_logical_bridge_copy_out(r, entry.outer_expr)
            else:
                lines = [f"    deallocate({r.flat_names[0]})"]
            copy_out_lines.extend(_guarded_copy_out(lines, guard, r))
            continue
        if r.aliasable:
            continue
        if entry.writeback_intent not in ('out', 'inout'):
            continue
        # Writeable non-aliasable member -> copy the flats back.  A
        # reconstruction recipe carries ``write_expr`` (e.g. complex
        # re/im -> ``cmplx``); a plain single-flat member (e.g. an AoS
        # scalar member ``pts(i)%x``) has no ``write_expr`` and is the
        # exact inverse of its copy-in, scattered back into
        # ``read_exprs[0]`` by ``render_copy_out_loop``.
        if not r.write_expr and not (len(r.flat_names) == 1 and r.read_exprs):
            continue
        copy_out_lines.extend(_guarded_copy_out(render_copy_out_loop(r, entry.outer_expr), guard, r))

    # Module globals the kernel WRITES (``FrozenArg.is_written``) are
    # host-shared inout state: after the call, copy the SDFG arg's final
    # value back to the host module variable (the ``=> member`` use-import
    # alias) so the update is visible to the caller -- symmetric to the
    # copy-in in ``build_wrapper_body``.  A scalar source was lifted to a
    # length-1 array, so write back its first element; an array source
    # assigns whole.  ``name_override`` resolves a LOGICAL arg to its
    # ``logical(c_bool)`` bridge buffer (intrinsic kind conversion on the
    # assignment back to the LOGICAL host member).
    module_writeback_lines: List[str] = []
    for a, _mod, _member in _orphan_module_args(frozen, iface, plan):
        if not a.is_written:
            continue
        alias = _module_symbol_alias(a.sdfg_name)
        actual = name_override.get(a.sdfg_name, a.sdfg_name)
        rhs = f"{actual}(1)" if tuple(a.shape) == ('1', ) else actual
        module_writeback_lines.append(f"    {alias} = {rhs}")

    # Module-global AoS-struct components: pack the SoA buffer back into the
    # host struct (only if WRITTEN) and always deallocate the buffer allocated
    # in the body's copy-in.
    aos_out_lines: List[str] = []
    for a in _aos_module_args(frozen):
        if a.is_written:
            aos_out_lines.extend(_render_aos_copy_out(a))
        else:
            aos_out_lines.append(f"    deallocate({a.sdfg_name})")
    for a in _unsourced_array_args(frozen, iface, plan):
        aos_out_lines.append(f"    deallocate({a.sdfg_name})")

    bridge_block = ""
    if bridge_copy_out:
        bridge_block = "\n    ! ----- logical(c_bool) -> LOGICAL bridge (copy-out + dealloc) -----\n" + "\n".join(
            bridge_copy_out)

    writeback_block = ""
    if module_writeback_lines:
        writeback_block = "\n    ! ----- Write-back for kernel-written module globals -----\n" + "\n".join(
            module_writeback_lines)

    if (not copy_out_lines and not bridge_copy_out and not module_writeback_lines and not aos_out_lines):
        return call_block

    copy_out_block = ""
    if copy_out_lines:
        copy_out_block = "\n    ! ----- Copy-out for writeable deep-copy entries -----\n" + "\n".join(copy_out_lines)
    aos_out_block = ""
    if aos_out_lines:
        aos_out_block = "\n    ! ----- AoS-struct component copy-out / dealloc -----\n" + "\n".join(aos_out_lines)
    marker = f"  end subroutine {iface.entry}_dace"
    pre, post = call_block.split(marker, 1)
    return pre + copy_out_block + bridge_block + writeback_block + aos_out_block + "\n" + marker + post


# ---------------------------------------------------------------------------
# Finalize subroutine
# ---------------------------------------------------------------------------


def build_finalize(iface: OriginalInterface) -> str:
    """Placeholder  --  the finalize subroutine is baked into
    ``templates/wrapper_call.f90.in`` and emitted together with the
    main wrapper tail.  Kept as a named function so the coordinator
    has a uniform shape.

    :param iface: unused today  --  kept for API symmetry.
    :returns: the empty string; reserved for a future split that
              moves the finalize body out of ``wrapper_call.f90.in``.
    """
    del iface  # unused
    return ""


# ---------------------------------------------------------------------------
# Module assembler
# ---------------------------------------------------------------------------


def assemble_module(iface: OriginalInterface, frozen: FrozenSignature, blocks: dict) -> str:
    """Stitch the rendered blocks into the complete Fortran module.

    :param iface: for ``iface.used_modules`` (use-only statements).
    :param frozen: for the schema_version stamped in the header.
    :param blocks: dict of ``'c_interface'`` / ``'handle_state'`` /
                   ``'wrapper_head'`` / ``'wrapper_body'`` /
                   ``'wrapper_tail'`` / ``'finalize'`` -> str.
    :returns: the complete Fortran module source.

    Template:
        ``templates/module.f90.in``  --  three placeholders (use
        statements, c-interface, handle state, wrapper body,
        finalize body) plus the entry name + schema version.
    """
    use_lines = [f"  use {mod}, only: {', '.join(syms)}" for mod, syms in sorted(iface.used_modules.items())]
    # Module-sourced free symbols: import each member under a
    # ``<sym>__mod`` alias so it doesn't clash with the wrapper's own
    # local ``<sym>`` (declared for the by-value SDFG call).  Group by
    # module to keep one ``use`` per module.
    by_mod: dict = {}
    for sym, (mod, member) in sorted(effective_module_sources(frozen, iface).items()):
        by_mod.setdefault(mod, []).append(f"{_module_symbol_alias(sym)} => {member}")
    for mod, renames in sorted(by_mod.items()):
        use_lines.append(f"  use {mod}, only: {', '.join(sorted(set(renames)))}")
    # Module-global AoS-struct components: import the HOST STRUCT plainly
    # (the copy loop references ``becxx(i)%k`` directly, no ``__mod`` alias).
    # A DUMMY-rooted AoS component (``patch_3d % p_patch_1d(i) % dolic_e``, with
    # an EMPTY ``aos_origin_mod``) needs no import -- its root struct is already
    # a wrapper argument -- so it is skipped here.
    aos_by_mod: dict = {}
    for a in _aos_module_args(frozen):
        if a.aos_origin_mod:
            aos_by_mod.setdefault(a.aos_origin_mod, set()).add(a.aos_origin_struct)
    for mod, structs in sorted(aos_by_mod.items()):
        use_lines.append(f"  use {mod}, only: {', '.join(sorted(structs))}")
    use_statements = "\n".join(use_lines)
    wrapper_body = (blocks['wrapper_head'] + "\n" + blocks['wrapper_body'] + "\n" + blocks['wrapper_tail'])
    module_src = _load("module.f90.in").format(
        entry=iface.entry,
        schema_version=frozen.schema_version,
        use_statements=use_statements,
        c_interface=blocks['c_interface'],
        handle_state=blocks['handle_state'],
        wrapper_body=wrapper_body,
        finalize_body=blocks['finalize'],
    )
    # gfortran rejects identifiers over 63 chars; the bridge can flatten a
    # deeply-inlined member extent past that.  Rename any such name uniquely.
    return _shorten_long_idents(module_src)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _shape_references_non_dummy(shape, dummy_set_lower: set) -> bool:
    """True iff an explicit extent in ``shape`` names an identifier that is NOT
    one of this wrapper's dummies.

    A Fortran explicit-shape *dummy* bound may reference only other dummies (and
    literals / intrinsics) -- NOT a local variable.  The binding localizes some
    module globals (e.g. ICON/QE ``npol``, ``max_ibands``) as wrapper locals,
    so a dummy whose extent uses one (``psi(lda*npol, max_ibands)``) is illegal:
    gfortran rejects "Variable 'npol' cannot appear in the expression".  Such a
    dummy is rendered assumed-size instead (the wrapper only ``c_loc``-s it, so
    the declared extents are unused; see ``_dim_spec``).
    """
    for s in shape or ():
        if not s or s == '?':
            continue  # assumed-shape placeholder -- a separate, legal form
        # Identifiers that are variable references (not ``name(`` calls).
        for m in re.finditer(r"[A-Za-z_]\w*", s):
            ident, end = m.group(0), m.end()
            if end < len(s) and s[end] == '(':
                continue  # intrinsic / function call, not a bound variable
            if ident.lower() not in dummy_set_lower:
                return True
    return False


def _dim_spec(shape, dummy_set_lower: set = None) -> str:
    """Render the dimension spec suffix for an outer dummy's
    declaration as a *postfix* shape (``name(d1,d2)``), not a
    ``dimension(...)`` attribute.  Postfix is the only form that
    works when the suffix lands after the ``::`` -- a leading comma
    plus ``dimension(...)`` after the ``::`` is read by Fortran as
    a SECOND variable declaration (so ``:: mask, dimension(n)``
    silently declares ``mask`` AND ``dimension`` of unknown rank).

    Assumed-shape dummies render the surviving ``?`` placeholders as
    ``:``; explicit extents pass through.  An empty shape leaves the
    declaration as a scalar (no suffix).  When ``dummy_set_lower`` is
    given and an explicit extent references a non-dummy (an illegal
    dummy bound), the whole array collapses to assumed-SHAPE ``(:,:,:)``
    -- which drops the illegal explicit bounds but KEEPS the rank (and
    carries the actual's bounds, so ``size(dummy, dim=k)`` and a rank-N
    element reference ``dummy(i,j,k)%m`` both stay valid).  Assumed-size
    ``(*)`` would collapse to rank 1 and break a deep-copied AoS member
    (``t_cartesian_coordinates%x`` over an AoS) whose gather loop indexes
    the dummy by its full rank and sizes it per dim.
    """
    if not shape:
        return ""
    if dummy_set_lower is not None and _shape_references_non_dummy(shape, dummy_set_lower):
        return "(" + ",".join(":" for _ in shape) + ")"
    return f"({','.join(s if s != '?' else ':' for s in shape)})"


def _is_default_logical(fortran_type: str) -> bool:
    """Recognise a caller-visible Fortran LOGICAL declaration whose
    storage layout differs from ``logical(c_bool)``.

    Default ``logical`` is 4 bytes (``LOGICAL(KIND=4)``); ``logical(1)``
    /  ``logical(8)`` are different sizes again.  Only ``logical(c_bool)``
    matches the SDFG's bool storage directly  --  every other LOGICAL kind
    needs a copy-via-Fortran-intrinsic-cast at the wrapper boundary so
    the SDFG sees the correct 1-byte ``bool`` layout.
    """
    s = fortran_type.strip().lower()
    if s == 'logical':
        return True
    if s.startswith('logical(') and 'c_bool' not in s:
        return True
    return False


def _build_logical_bridges(frozen: FrozenSignature, iface: OriginalInterface):
    """Emit scratch buffers + entry/exit copies for any LOGICAL outer
    dummy that the SDFG sees as ``bool``.

    The C ABI binds the wrapper's outer ``logical`` (default 4-byte)
    dummy to a ``T*`` whose elements are 4 bytes wide; the SDFG expects
    1-byte ``bool*``.  Passing the outer dummy's address straight
    through corrupts every other element's read.  The fix is a
    ``logical(c_bool)`` scratch buffer with the same shape  --  Fortran's
    intrinsic LOGICAL-kind-conversion (``cbool_buf = outer``) handles
    the bit-fiddling, and ``c_loc(cbool_buf)`` is then safely passed
    to the SDFG.

    :returns: ``(decl_lines, copy_in_lines, copy_out_lines,
              name_override)``:

              * ``decl_lines``: scratch buffer declarations (one per
                affected dummy).  Empty if no dummy needs bridging.
              * ``copy_in_lines``: Fortran-intrinsic cast assignments
                run before the SDFG call.
              * ``copy_out_lines``: reverse-direction casts for
                ``intent(out)/inout`` dummies, run after the SDFG call.
              * ``name_override``: ``{sdfg_name: scratch_name}``
                mapping the SDFG-call-side name should pass instead of
                the outer dummy when this dummy needs bridging.

    Bool dummies whose outer Fortran declaration is already
    ``logical(c_bool)`` need no bridge  --  pass-through is correct.
    Bool ``intent(in)`` scalars are pass-by-value; the C interface
    builder takes a ``logical(c_bool), value`` so the Fortran
    intrinsic cast happens at the call site instead of through a
    scratch buffer (handled separately in ``build_wrapper_tail``).
    """
    decl_lines: List[str] = []
    copy_in_lines: List[str] = []
    copy_out_lines: List[str] = []
    name_override: dict = {}

    iface_by_name = {a.name: a for a in iface.args}
    for fa in frozen.args:
        if fa.dtype != 'bool':
            continue
        oa = iface_by_name.get(fa.fortran_name)
        if oa is None:
            continue
        if not _is_default_logical(oa.fortran_type):
            continue
        # Array dummy  --  explicit scratch buffer + element-wise cast.
        if fa.rank > 0:
            scratch = f"{fa.fortran_name}_cbool"
            shape_dim = "(" + ",".join(":" for _ in range(fa.rank)) + ")"
            decl_lines.append(f"    logical(c_bool), allocatable, target :: {scratch}{shape_dim}")
            # A scalar ``intent(out)/inout`` LOGICAL is a *scalar* on the
            # caller side (``oa.rank == 0``) but the bridge lifts it to a
            # length-1 ``Array`` on the SDFG signature (the scalar-output
            # convention -- see ``descriptors.py``).  ``size(oa.name)`` on
            # the outer scalar is a hard Fortran error and a bare
            # ``scratch = oa.name`` is a rank-0/rank-1 mismatch, so the
            # scratch must be allocated to the SDFG arg's own concrete
            # extent (``FrozenArg.shape``) and the scalar bridged through
            # element ``(1)`` -- mirroring the orphan-module-global path.
            if oa.rank == 0:
                dims = ", ".join(fa.shape) if fa.shape else "1"
                copy_in_lines.append(f"    allocate({scratch}({dims}))")
                copy_in_lines.append(f"    {scratch}(1) = {oa.name}")
                if oa.intent in ('out', 'inout', ''):
                    copy_out_lines.append(f"    {oa.name} = {scratch}(1)")
            else:
                shape_args = ", ".join(f"size({oa.name}, dim={d + 1})" for d in range(fa.rank))
                copy_in_lines.append(f"    allocate({scratch}({shape_args}))")
                copy_in_lines.append(f"    {scratch} = {oa.name}")
                if oa.intent in ('out', 'inout', ''):
                    copy_out_lines.append(f"    {oa.name} = {scratch}")
            copy_out_lines.append(f"    deallocate({scratch})")
            name_override[fa.sdfg_name] = scratch
        # Scalar bool dummy (fa.rank == 0): the SDFG declares the C
        # interface as ``logical(c_bool), value :: <name>``, but the
        # outer Fortran dummy is default ``logical`` (4 bytes).  A
        # direct call ``dace_program_X(flag, ...)`` makes gfortran
        # reject with ``Type mismatch ... passed LOGICAL(4) to
        # LOGICAL(1)`` -- there is no implicit kind conversion at the
        # call expression for a pass-by-value bind(c) dummy.
        #
        # The fix mirrors the array path: declare a local
        # ``logical(c_bool)`` temporary, run the Fortran-intrinsic
        # LOGICAL-kind cast on it (``flag_cbool = flag``), then pass
        # the temp.  ``name_override`` redirects the SDFG-call name to
        # the temp; the call-arg renderer in ``build_wrapper_tail``
        # already knows ``kind == 'scalar'`` means pass-by-value, so it
        # won't wrap with ``c_loc`` (correct: the interface wants the
        # value itself, not a c_ptr).
        else:
            scratch = f"{fa.fortran_name}_cbool"
            decl_lines.append(f"    logical(c_bool) :: {scratch}")
            copy_in_lines.append(f"    {scratch} = {oa.name}")
            if oa.intent in ('out', 'inout', ''):
                # Symmetric copy-back for intent(out)/inout scalars.
                # No deallocate -- this is a stack temporary, not
                # allocatable.
                copy_out_lines.append(f"    {oa.name} = {scratch}")
            name_override[fa.sdfg_name] = scratch
            continue

    return decl_lines, copy_in_lines, copy_out_lines, name_override


_OFFSET_SYM_RE = re.compile(r"^offset_(.+)_d(\d+)$")
_EXTENT_SYM_RE = re.compile(r"^(.+)_d(\d+)$")


def _sym_from_intrinsic(sym: str, frozen: FrozenSignature) -> Optional[Tuple[str, str, int]]:
    """Map a free SDFG symbol to the Fortran intrinsic that populates
    it from the caller's actual storage.

    ``offset_<arr>_d<i>``  -> ``("lbound", <fortran-expr>, i+1)``
    ``<arr>_d<i>`` (extent) -> ``("size",   <fortran-expr>, i+1)``

    ``<arr>`` is matched to a ``FrozenArg`` by ``sdfg_name``; the
    Fortran expression is the original dummy (or, for a flattened
    struct member, the ``st%u`` outer expression) so ``lbound`` /
    ``size`` query the array the caller actually passed.

    :param sym: a free symbol name.
    :param frozen: the frozen signature (arg metadata).
    :returns: ``(intrinsic, fortran_expr, dim)`` or ``None`` when the
        symbol isn't an offset/extent of a known array arg.
    """
    by_sdfg = {a.sdfg_name: a for a in frozen.args}

    def _expr(arr: str) -> Optional[str]:
        a = by_sdfg.get(arr)
        if a is None or a.kind != "array":
            return None
        return a.from_struct_member or a.fortran_name

    m = _OFFSET_SYM_RE.match(sym)
    if m:
        e = _expr(m.group(1))
        return ("lbound", e, int(m.group(2)) + 1) if e else None
    m = _EXTENT_SYM_RE.match(sym)
    if m:
        e = _expr(m.group(1))
        return ("size", e, int(m.group(2)) + 1) if e else None
    return None


def _sym_from_array_extent(sym: str, frozen: FrozenSignature, exclude=None) -> Optional[Tuple[str, int]]:
    """A free symbol that is a NAMED extent of an array arg -- e.g. ``n_zlev``
    is the 2nd dim of ``vn(nproma, n_zlev, nblks_e)``.

    Unlike :func:`_sym_from_intrinsic` (which matches only the bridge-minted
    ``<arr>_d<i>`` extent names), this matches a symbol that carries its own
    Fortran name and appears verbatim in some array arg's ``shape``.  The array
    dimension is the GROUND TRUTH for the SDFG's memory layout, so sourcing the
    symbol from ``size(<arr>, dim=i+1)`` MUST take precedence over a same-named
    module global: ICON's ``mo_ocean_nml::n_zlev`` is unset (0) in an extracted
    kernel, and reading it would size SDFG transients (``z_vort_internal(n_zlev)``)
    to zero -> out-of-bounds writes.  Returns ``(fortran_expr, dim)`` or None.

    ``exclude`` (arg ``sdfg_name`` set) are the ABSENT-OPTIONAL args the binding
    allocates as a degenerate ``(1,1,...)`` placeholder (no host source): their
    extent is always 1, so deriving a blocking symbol (``nproma``, ``nblks_e``)
    from one pins it to 1 and every ``size(<arr>, dim)`` transient collapses.
    Skip them so the symbol falls to a PRESENT array's real extent (or the seeded
    module global), the true SDFG layout.
    """
    exclude = exclude or set()
    for a in frozen.args:
        if a.kind != "array":
            continue
        if a.sdfg_name in exclude:
            continue
        shape = tuple(str(s) for s in (a.shape or ()))
        if sym in shape:
            expr = a.from_struct_member or a.fortran_name
            if expr:
                return (expr, shape.index(sym) + 1)
    return None


def _module_symbol_alias(sym: str) -> str:
    """Local rename for a module-sourced free symbol's import.

    The wrapper already declares its own local ``<sym>`` (passed
    by value to the SDFG), so the module member is imported under a
    distinct ``<sym>__mod`` alias to avoid the name clash.
    """
    return f"{sym}__mod"


def effective_module_sources(frozen: FrozenSignature, iface: OriginalInterface) -> Dict[str, Tuple[str, str]]:
    """Merge bridge-auto-detected module-global provenance with any
    hand-authored ``OriginalInterface.module_symbol_sources``.

    The bridge tags every SDFG symbol / arg that traces to a Fortran
    module global (``_QM<mod>E<entity>``) with its ``(module, entity)``
    origin (``FrozenSignature.module_symbol_origins``).  This is the
    primary source: kernels need NO hand-authored list.  The explicit
    ``iface.module_symbol_sources`` is kept as an override / fallback
    and wins on conflict  --  it lets a caller correct a mis-decoded
    origin or supply one the bridge could not recover.

    :param frozen: the SDFG-side frozen signature carrying
        auto-detected ``module_symbol_origins``.
    :param iface: the caller-facing interface; its
        ``module_symbol_sources`` is the explicit override layer.
    :returns: ``sdfg_name -> (module, entity)`` for every
        module-global-sourced symbol / arg, auto + explicit merged.
    """
    merged: Dict[str, Tuple[str, str]] = dict(getattr(frozen, 'module_symbol_origins', {}) or {})
    merged.update(iface.module_symbol_sources)  # explicit override wins
    return merged


def _orphan_module_args(frozen: FrozenSignature, iface: OriginalInterface, plan: FlattenPlan):
    """SDFG args that are neither an outer dummy nor a flat companion
    nor an extent / offset symbol  --  Fortran *module* globals the
    kernel reads directly that the bridge lifted onto the SDFG
    signature (ICON's ``nrdmax`` / ``nflatlev`` / ``i_am_accel_node``
    / timer handles).  Each origin is bridge-auto-detected (or
    explicitly overridden via ``iface.module_symbol_sources``); each
    becomes a wrapper-local ``target`` initialised from the renamed
    module import so the ``c_loc(<name>)`` SDFG-call actual resolves.

    :returns: list of ``(FrozenArg, module, member)`` in signature
        order, restricted to those with a known module origin.
    """
    sources = effective_module_sources(frozen, iface)
    dummy = {a.name for a in iface.args}
    flat = {f for e in plan.entries for f in e.recipe.flat_names}
    out = []
    for a in frozen.args:
        n = a.sdfg_name
        if n in dummy or n in flat:
            continue
        if _OFFSET_SYM_RE.match(n) or (a.kind == 'symbol'):
            continue
        if _EXTENT_SYM_RE.match(n) and n not in sources:
            continue
        src = sources.get(n)
        if src is not None:
            out.append((a, src[0], src[1]))
    return out


def _aos_module_args(frozen: FrozenSignature):
    """SDFG args that are the SoA image of an array-of-structs component,
    marshalled with an AoS<->SoA copy loop.  Identified by the bridge-stamped
    ``aos_origin_struct`` provenance on the FrozenArg.  Two origins share this
    path (the copy machinery is origin-agnostic -- it only string-builds
    ``<struct>(i)%<member>``):

      * a MODULE-LEVEL global (``becxx_k`` <- ``becxx(:)%k``): ``aos_origin_mod``
        is set, so the binding ``use``-imports the host struct.
      * a DUMMY-rooted nested member through a pointer-array-of-records
        (``patch_3d_p_patch_1d_dolic_e`` <- ``patch_3d % p_patch_1d(i) % dolic_e``):
        ``aos_origin_struct`` is a ``%``-expression and ``aos_origin_mod`` is
        empty (the root is already a wrapper argument -- no import).

    Restricted to ARRAY args: a rank-0 struct member (``dfftt%ngm``) reaches the
    SDFG as a by-VALUE free symbol, not a ``c_loc`` data buffer -- it keeps its
    free-symbol / scalar-member sourcing and must NOT be re-declared here as an
    ``allocatable`` (a duplicate-declaration error).
    """
    return [a for a in frozen.args if getattr(a, 'aos_origin_struct', '') and getattr(a, 'rank', 0) > 0]


def _unsourced_array_args(frozen: FrozenSignature, iface: OriginalInterface, plan: FlattenPlan):
    """Array SDFG args with NO host source -- not a wrapper dummy, not a
    struct-flatten companion, not a module global, not an AoS component.

    These are the data buffers of ABSENT inlined-callee optionals (QE's
    ``becphi`` -- ``becphi_r`` survives as a program arg even though the
    matching ``becphi_r_present`` folds to 0).  The wrapper must still pass a
    valid ``c_loc`` pointer, so we declare a shape-degenerate, zero-filled local
    -- correct whenever the kernel does not read it (the optional is absent).
    Without this they reach ``c_loc(becphi_r)`` undeclared -> a compile error.
    """
    declared = {f for entry in plan.entries for f in entry.recipe.flat_names}
    declared |= {a.sdfg_name for a, _m, _mem in _orphan_module_args(frozen, iface, plan)}
    declared |= {a.sdfg_name for a in _aos_module_args(frozen)}
    declared |= {a.name for a in iface.args}
    return [
        a for a in frozen.args if a.kind == 'array' and getattr(a, 'rank', 0) > 0 and a.sdfg_name not in declared
        and not getattr(a, 'aos_origin_struct', '')
    ]


def _extra_local_symbols(frozen: FrozenSignature, iface: OriginalInterface, plan: FlattenPlan):
    """Symbols the SDFG call / binding-allocate shape-exprs reference that NO
    existing decl path covers:

    * (a) an unsourced SCALAR / symbol arg -- the VALUE of an absent
      inlined-callee optional (QE ``run_on_gpu_``: ``run_on_gpu__present`` folds
      to 0 but the value slot survives), not a wrapper dummy and not a free symbol.
    * (b) a bare-identifier SHAPE symbol of a binding-allocated arg that is
      neither a free symbol nor a dummy (``nks`` -- a ``klist`` module global used
      only inside a scratch ``allocate``; ``qvan_init_nij`` -- a dim derived in an
      inlined function, with no host source).

    Returns ``(name, fortran_type, rhs)``: a module-origin symbol is sourced from
    its ``__mod`` import; everything else degrades to a degenerate default (1 for a
    dim, ``.false.``/0 for an absent scalar) -- valid because none is READ on the
    no-op path.  (NOTE: the ASSIGNMENT ordering vs the binding-allocates that use
    these is the separate symbol-population-hoist work; this only makes the wrapper
    COMPILE by declaring + sourcing them.)
    """
    sources = effective_module_sources(frozen, iface)
    outer = {a.name for a in iface.args}
    flat = {f for e in plan.entries for f in e.recipe.flat_names}
    declared = set(frozen.free_symbols) | outer | flat
    # Every name with a wrapper-local / dummy decl already: array args (all reach
    # a decl via flat / orphan / aos / unsourced), the orphan + aos SCALAR locals
    # (rank-0 module globals like ``lmaxq`` -- declared ``target`` by the orphan
    # path), the comm pgrid params.
    declared |= {a.sdfg_name for a in frozen.args if a.kind == 'array'}
    declared |= {a.sdfg_name for a, _m, _mem in _orphan_module_args(frozen, iface, plan)}
    declared |= {a.sdfg_name for a in _aos_module_args(frozen)}
    declared |= {a.sdfg_name for a in _unsourced_array_args(frozen, iface, plan)}
    declared |= {_USER_COMM_SYMBOL_NAME, _USER_COMM_SIZE_SYMBOL_NAME}
    ident = re.compile(r'^[A-Za-z_]\w*$')

    def _rhs(name: str, dtype: str, is_dim: bool) -> str:
        if name in sources:
            alias = _module_symbol_alias(name)
            return f"int({alias}, c_int)" if dtype != 'bool' else alias
        return "1" if is_dim else _zero_literal(dtype)

    out: dict = {}
    # (a) unsourced scalar / symbol args
    for a in frozen.args:
        if a.kind in ('scalar', 'symbol') and a.sdfg_name not in declared:
            out[a.sdfg_name] = (_fortran_c_value_type(a.dtype), _rhs(a.sdfg_name, a.dtype, False))
    # (b) bare-identifier shape symbols of any arg
    for a in frozen.args:
        for s in (getattr(a, 'shape', ()) or ()):
            s = str(s)
            if ident.match(s) and s not in declared and s not in out:
                out[s] = ("integer(c_int)", _rhs(s, 'int32', True))
    return [(n, ft, rhs) for n, (ft, rhs) in out.items()]


def _zero_literal(dtype: str) -> str:
    """The neutral fill literal for ``dtype`` -- ``.false.`` for LOGICAL (where
    ``= 0`` is an INTEGER->LOGICAL extension warning), ``0`` for everything else
    (Fortran promotes the integer 0 to real / complex)."""
    return ".false." if dtype == 'bool' else "0"


def _present(expr: str, is_pointer: bool) -> str:
    """Definedness test for a POINTER (``associated``) vs ALLOCATABLE
    (``allocated``) -- using the wrong intrinsic is a hard Fortran type error."""
    return f"associated({expr})" if is_pointer else f"allocated({expr})"


def live_entries(frozen: FrozenSignature, plan: FlattenPlan) -> List:
    """The flatten entries the kernel actually consumes.

    ``hlfir-flatten-structs`` records one entry per struct member it can
    describe -- including members no SDFG argument or symbol ever reads (the
    real ICON ``t_nh_diag`` contributes 668 entries against 508 kernel args).
    Marshalling a member the kernel never takes is not merely dead work: an
    ICON POINTER member that the running configuration never nullifies NOR
    allocates (Held-Suarez leaves ``t_nh_diag%t2m_bias`` -- an NWP-physics
    diagnostic -- with UNDEFINED association status) makes ``ASSOCIATED`` and
    ``c_loc`` undefined behaviour: the "descriptor" read lands in adjacent
    blank CHARACTER data, and gfortran's ``internal_pack`` then walks a rank
    of 0x20 (32) off the end of its own 15-element ``stride`` locals and
    trips the stack canary.

    A member is live iff one of its flat companions is a kernel argument or
    symbol.  Symbol population is unaffected either way: it sources from the
    member's ``%`` path (``size(p_diag%vt, dim=1)``), never from the flat.
    """
    live_names = {a.sdfg_name for a in frozen.args}
    return [e for e in plan.entries if any(f in live_names for f in e.recipe.flat_names)]


def _guarded_copy_out(lines: List[str], guard: str, recipe) -> List[str]:
    """Wrap a non-aliased entry's copy-out in its presence guard.  The
    ABSENT branch releases the degenerate scratch the guarded copy-in
    allocated (the member itself is untouched -- it has no storage)."""
    if not guard:
        return lines
    out = [f"    if ({guard}) then"]
    out.extend("  " + ln for ln in lines)
    out.append("    else")
    out.extend(f"      deallocate({flat})" for flat in recipe.flat_names)
    out.append("    end if")
    return out


def _presence_scratch_name(dtype: str) -> str:
    """Wrapper-local length-1 ``target`` array an ABSENT aliasable member's
    flat POINTER is bounds-remapped onto, so the SDFG call's ``c_loc(<flat>)``
    stays a defined reference (mirrors the orphan-module-global degenerate
    buffer)."""
    return f"presence_scratch_{dtype}"


def _entry_presence_guard(iface: OriginalInterface, base: str) -> str:
    """``associated(<base>)`` / ``allocated(<base>)`` when the dotted member
    path ``base`` (``p_diag%ddt_ua_adv``, subscripted intermediates allowed:
    ``ocean_state%p_prog(nold(1))%vn``) ends in a deferred-storage struct
    member; ``''`` for a plain member / non-member path.

    An unallocated / disassociated member's descriptor bounds are undefined:
    ``c_loc`` / ``size`` on it read garbage, and gfortran's ``internal_pack``
    at the alias site then smashes the stack (ICON Held-Suarez leaves every
    ``t_nh_diag%ddt_ua_* / ddt_va_*`` tendency pointer disassociated).  Every
    marshal of such a member is therefore wrapped in this guard.

    Only the LEAF member's storage class is tested: pointer-to-record handle
    members never reach the plan (all walkers skip them), so intermediates
    are value records; a deferred ARRAY-of-records intermediate keeps its
    current unguarded behaviour (a compound guard needs nested IFs --
    Fortran ``.and.`` does not guarantee short-circuit evaluation).
    """
    parts = [p.strip() for p in base.split('%')]
    if len(parts) < 2:
        return ''
    root = parts[0].split('(')[0].strip().lower()
    arg = next((a for a in iface.args if a.name.lower() == root and a.struct_type), None)
    if arg is None:
        return ''
    st = iface.struct_types.get(arg.struct_type)
    for i, seg in enumerate(parts[1:]):
        if st is None:
            return ''
        mname = seg.split('(')[0].strip().lower()
        m = next((mm for mm in st.members if mm.name.lower() == mname), None)
        if m is None:
            return ''
        if i == len(parts) - 2:
            return _present(base, m.alloc == 'pointer') if m.alloc else ''
        st = iface.struct_types.get(m.struct_name) if m.struct_name else None
    return ''


def _recipe_presence_guard(iface: OriginalInterface, recipe) -> str:
    """The presence guard for a flatten recipe's source member, from its
    first read expression (index placeholders stripped).  ``aos_alloc``
    recipes guard per-row inside their own pack/unpack code -- no outer
    guard."""
    if recipe.aos_alloc or not recipe.read_exprs:
        return ''
    return _entry_presence_guard(iface, strip_index_args(recipe.read_exprs[0]))


def _aos_loop_pieces(a):
    """Per-outer-dim loop vars / per-member-dim cap-var names + the host element
    accessor for an AoS-component arg.  ``aos_outer_rank == N`` (an N-D record
    array): N element-index loop vars ``aos_<base>_i0 .. aos_<base>_i{N-1}`` are
    the SoA buffer's LEADING dims, matching the bridge's ``[element-dims...,
    member-dims...]`` layout (N=1 for a 1-D record array ``becxx(:)``; N>1 for an
    N-D cartesian array member ``p_diag % p_vn_dual(:,:,:)``).  ``aos_outer_rank
    == 0`` (a SCALAR struct global, ``vcut``/``dfftt``): no element loop --
    ``its``/``caps`` are empty and the accessor is the bare ``struct%member``."""
    base = a.sdfg_name
    member_rank = a.rank - a.aos_outer_rank
    if a.aos_outer_rank == 0:
        return [], [], member_rank, f"{a.aos_origin_struct}%{a.aos_member_path}"
    # Fortran identifiers must start with a letter (no leading ``_``).
    its = [f"aos_{base}_i{k}" for k in range(a.aos_outer_rank)]
    caps = [f"aos_{base}_c{j}" for j in range(member_rank)]
    # ``becxx(i0)%k`` / ``p_diag % p_vn_dual(i0,i1,i2)%x`` (member_path is
    # ``%``-joined: ``k`` / ``x`` / ``a%b``).
    elem = f"{a.aos_origin_struct}({', '.join(its)})%{a.aos_member_path}"
    return its, caps, member_rank, elem


def _aos_member_is_static(a) -> bool:
    """True when the AoS component is a fixed-shape VALUE member -- every member
    dim a compile-time literal (``t_cartesian_coordinates%x(3)``) -- rather than
    an ALLOCATABLE / POINTER component (whose extents the bridge always registers
    as symbols).  A static member is unconditionally present (no
    ``allocated``/``associated`` guard) and its extents ARE the literals, so the
    copy skips the per-element cap-max scan and the member-present guard."""
    member_dims = a.shape[a.aos_outer_rank:]
    return bool(member_dims) and all(str(d).isdigit() for d in member_dims)


def _render_aos_copy_in(a) -> List[str]:
    """``allocate`` the SoA buffer (per-member-dim cap = max over elements) +
    pack the host AoS component into it.  Skips the data copy when the kernel
    allocates the component itself (``global_alloc_inside`` -- host has no
    data yet) but still allocates a non-degenerate buffer."""
    its, caps, mrank, elem = _aos_loop_pieces(a)
    if a.aos_outer_rank == 0:
        # SCALAR struct global member (``vcut%corrected``, ``dfftt%nl``,
        # ICON-O's ``free_sfc_solver%x_loc_wp``): a POINTER / ALLOCATABLE member
        # that may be unassociated / unallocated on entry.  When it IS defined,
        # size the SoA companion from the LIVE member and copy it in, so a
        # kernel that WRITES it with the member's real (mesh) extents stays in
        # bounds -- the SDFG's descriptor dims are set from ``size(companion)``,
        # so a shape-degenerate buffer under-allocates while the kernel's own
        # (mesh-bounded) write loops overrun it and smash the heap.  When the
        # member is NOT defined (a POINTER the kernel never reads -- the vexx
        # no-op path, or a member whose allocator was stubbed out of the TU and
        # the caller never rebuilt), fall back to a shape-degenerate,
        # zero-filled buffer: valid ``c_loc`` storage of the right rank.
        member = f"{a.aos_origin_struct}%{a.aos_member_path}"
        zero = _zero_literal(a.dtype)
        out = [f"    ! ----- scalar-struct member: {a.sdfg_name} <- {member} -----"]
        if a.rank and _aos_member_is_static(a):
            # Fixed-shape VALUE member (``vcut%a`` = ``a(3, 3)``): unconditionally
            # present with literal extents, so it takes no presence guard --
            # ``allocated`` is illegal on a non-ALLOCATABLE and ``associated`` on a
            # non-POINTER.  Allocate the literal shape and copy straight in.
            static_dims = ", ".join(str(d) for d in a.shape[a.aos_outer_rank:])
            out.append(f"    allocate({a.sdfg_name}({static_dims})); {a.sdfg_name} = {member}")
        elif a.rank:
            mp = _present(member, a.aos_member_pointer)
            live_dims = ", ".join(f"size({member}, {k + 1})" for k in range(a.rank))
            degen_dims = ", ".join("1" for _ in range(a.rank))
            out.append(f"    if ({mp}) then")
            out.append(f"      allocate({a.sdfg_name}({live_dims})); {a.sdfg_name} = {member}")
            out.append(f"    else")
            out.append(f"      allocate({a.sdfg_name}({degen_dims})); {a.sdfg_name} = {zero}")
            out.append(f"    end if")
        else:
            out.append(f"    {a.sdfg_name} = {zero}")
        return out
    struct = a.aos_origin_struct
    sp = _present(struct, a.aos_struct_pointer)  # struct allocated/associated?
    zero = _zero_literal(a.dtype)
    # N nested element loops + matching closers; outer-dim allocate specs.
    do_open = [f"      do {itv} = 1, size({struct}, {k + 1})" for k, itv in enumerate(its)]
    do_close = ["      end do" for _ in its]
    outer_dims = ", ".join(f"size({struct}, {k + 1})" for k in range(len(its)))
    one_dims = ", ".join("1" for _ in its)
    idx = ", ".join(its)
    if mrank == 0:
        # SCALAR member of a record array (``upf(:)%tvanp`` -- a plain LOGICAL /
        # INTEGER per element, NOT allocatable).  Gather one value per element
        # into the N-D SoA buffer.  No member ``allocated`` guard (the member is
        # a fixed scalar); the OUTER struct may itself be unallocated on entry
        # (no-op path) -> degenerate size-1 buffer.
        out = [
            f"    ! ----- AoS->SoA gather (scalar member): {a.sdfg_name} <- "
            f"{struct}(:)%{a.aos_member_path} -----"
        ]
        out.append(f"    if ({sp}) then")
        out.append(f"      allocate({a.sdfg_name}({outer_dims}))")
        out += do_open
        out.append(f"        {a.sdfg_name}({idx}) = {elem}")
        out += do_close
        out.append(f"    else")
        out.append(f"      allocate({a.sdfg_name}({one_dims})); {a.sdfg_name} = {zero}")
        out.append(f"    end if")
        return out
    if _aos_member_is_static(a):
        # Fixed-shape VALUE member (``t_cartesian_coordinates%x(3)``): always
        # present (no allocated/associated guard) with literal extents, so skip
        # the cap-max scan and gather every element directly.  Buffer is
        # [outer-dims..., literal-member-dims...].
        member_dims = list(a.shape[a.aos_outer_rank:])
        cap_dims = ", ".join(member_dims)
        slc = ", ".join(f"1:{d}" for d in member_dims)
        out = [
            f"    ! ----- AoS->SoA copy-in (static value member): {a.sdfg_name} <- "
            f"{struct}(:)%{a.aos_member_path} -----"
        ]
        out.append(f"    if ({sp}) then")
        out.append(f"      allocate({a.sdfg_name}({outer_dims}, {cap_dims})); {a.sdfg_name} = {zero}")
        out += do_open
        out.append(f"        {a.sdfg_name}({idx}, {slc}) = {elem}")
        out += do_close
        out.append(f"    else")
        out.append(f"      allocate({a.sdfg_name}({one_dims}, {cap_dims})); {a.sdfg_name} = {zero}")
        out.append(f"    end if")
        return out
    # member_rank > 0: a 2D+ ALLOCATABLE / POINTER component per element
    # (``becxx(:)%k``, ``ke(:)%k``).  Cap = max member extent over elements;
    # both the struct and each element's component are guarded (no-op path
    # leaves them unallocated / unassociated).
    mp = _present(elem, a.aos_member_pointer)  # element's component defined?
    out = [f"    ! ----- AoS->SoA copy-in: {a.sdfg_name} <- {struct}(:)%{a.aos_member_path} -----"]
    for c in caps:
        out.append(f"    {c} = 0")
    out.append(f"    if ({sp}) then")
    out += do_open
    if not a.global_alloc_inside:
        out.append(f"        if ({mp}) then")
        for j, c in enumerate(caps):
            out.append(f"          {c} = max({c}, size({elem}, {j + 1}))")
        out.append(f"        end if")
    out += do_close
    out.append(f"    end if")
    for c in caps:
        out.append(f"    if ({c} == 0) {c} = 1")
    cap_dims = ", ".join(caps)
    out.append(f"    if ({sp}) then")
    out.append(f"      allocate({a.sdfg_name}({outer_dims}, {cap_dims}))")
    out.append(f"    else")
    out.append(f"      allocate({a.sdfg_name}({one_dims}, {cap_dims}))")
    out.append(f"    end if")
    out.append(f"    {a.sdfg_name} = {zero}")
    if not a.global_alloc_inside:
        slc = ", ".join(f"1:size({elem}, {j + 1})" for j in range(mrank))
        out.append(f"    if ({sp}) then")
        out += do_open
        out.append(f"        if ({mp}) &")
        out.append(f"          {a.sdfg_name}({idx}, {slc}) = {elem}")
        out += do_close
        out.append(f"    end if")
    return out


def _render_aos_copy_out(a) -> List[str]:
    """Pack the SoA buffer back into the host AoS component (only when the arg
    is WRITTEN), allocating each component first if the kernel created it."""
    its, caps, mrank, elem = _aos_loop_pieces(a)
    if a.aos_outer_rank == 0:
        # Scalar-struct member: when the host member is defined, the companion
        # was sized from it (copy-in above), so pack the kernel's writes back
        # into it (conformable) before releasing the buffer.  When the member
        # is undefined the companion is the shape-degenerate no-op buffer with
        # nothing to write back.
        out = []
        member = f"{a.aos_origin_struct}%{a.aos_member_path}"
        if a.rank and _aos_member_is_static(a):
            # Fixed-shape VALUE member (``vcut%a`` = ``a(3, 3)``): unconditionally
            # present, so it takes no presence guard (``allocated``/``associated``
            # are illegal on a non-ALLOCATABLE/non-POINTER); pack back directly.
            out.append(f"    {member} = {a.sdfg_name}")
        elif a.rank:
            mp = _present(member, a.aos_member_pointer)
            out.append(f"    if ({mp}) {member} = {a.sdfg_name}")
        out.append(f"    deallocate({a.sdfg_name})")
        return out
    struct = a.aos_origin_struct
    sp = _present(struct, a.aos_struct_pointer)
    do_open = [f"      do {itv} = 1, size({struct}, {k + 1})" for k, itv in enumerate(its)]
    do_close = ["      end do" for _ in its]
    idx = ", ".join(its)
    if mrank == 0:
        # Scalar member of a record array: scatter one value per element back.
        out = [
            f"    ! ----- SoA->AoS scatter (scalar member): {struct}(:)%"
            f"{a.aos_member_path} <- {a.sdfg_name} -----"
        ]
        out.append(f"    if ({sp}) then")
        out += do_open
        out.append(f"        {elem} = {a.sdfg_name}({idx})")
        out += do_close
        out.append(f"    end if")
        out.append(f"    deallocate({a.sdfg_name})")
        return out
    if _aos_member_is_static(a):
        # Fixed-shape VALUE member (``%x(3)``): always present, literal extents;
        # scatter every element back unconditionally.
        member_dims = list(a.shape[a.aos_outer_rank:])
        slc = ", ".join(f"1:{d}" for d in member_dims)
        out = [
            f"    ! ----- SoA->AoS copy-out (static value member): {struct}(:)%"
            f"{a.aos_member_path} <- {a.sdfg_name} -----"
        ]
        out.append(f"    if ({sp}) then")
        out += do_open
        out.append(f"        {elem} = {a.sdfg_name}({idx}, {slc})")
        out += do_close
        out.append(f"    end if")
        out.append(f"    deallocate({a.sdfg_name})")
        return out
    mp = _present(elem, a.aos_member_pointer)
    out = [f"    ! ----- SoA->AoS copy-out: {struct}(:)%{a.aos_member_path} <- {a.sdfg_name} -----"]
    out.append(f"    if ({sp}) then")
    out += do_open
    if a.global_alloc_inside:
        alloc_dims = ", ".join(f"size({a.sdfg_name}, {a.aos_outer_rank + j + 1})" for j in range(mrank))
        out.append(f"        if (.not. {mp}) allocate({elem}({alloc_dims}))")
    slc = ", ".join(f"1:size({elem}, {j + 1})" for j in range(mrank))
    out.append(f"        if ({mp}) &")
    out.append(f"          {elem} = {a.sdfg_name}({idx}, {slc})")
    out += do_close
    out.append(f"    end if")
    out.append(f"    deallocate({a.sdfg_name})")
    return out


def _struct_member_symbol_sources(iface: OriginalInterface) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Map a struct dummy's member free-symbols to the Fortran
    expression that reads them from the caller's actual struct.

    A struct member used ONLY symbolically  --  a loop bound
    (``do i = 1, dfftt%ngm``) or an array extent
    (``g(3, dfftt%ngm)``)  --  is lifted by ``hlfir-flatten-structs``
    to a free SDFG symbol with NO ``FlattenEntry`` (only members read
    as *values* get a data entry, cf. ``struct_of_scalars`` ``cst%rg``).
    The plan-driven ``scalar_member`` / ``flat_shapes`` paths therefore
    never source it, and it falls through to the unresolved-symbol
    TODO.  This rebuilds the bridge's member-symbol names from the
    static ``iface.struct_types`` layout and pairs each with its ``%``
    read so the symbol-population block can assign it directly.

    The bridge joins an access path with ``_`` (``FlattenStructs.cpp``
    ``designateChainPath``; the ``member.replace('%', '_')`` convention
    pinned by ``struct_of_scalars_test``), so for struct dummy ``d``:

      * scalar member ``d%m`` (rank 0)      -> ``d_m``             = ``d%m``
      * array  member ``d%a`` extent dim i  -> ``d_a_d<i>``        = ``size(d%a, dim=i+1)``
      * array  member ``d%a`` lbound dim i  -> ``offset_d_a_d<i>`` = ``lbound(d%a, dim=i+1)``

    Nested derived-type members recurse with the joined path
    (``d_inner_m`` <- ``d%inner%m``).

    Returns ``(sources, member_paths)``.  ``member_paths`` maps EVERY leaf
    member's symbol name (scalar or array, any rank) to its bare caller-side
    ``%``-path -- the ``_allocated`` fold in :func:`_build_symbol_assigns` reads
    it to spell an ``ASSOCIATED``/``ALLOCATED`` on a nested struct-member
    POINTER/ALLOCATABLE the kernel branches on (no flat array arg backs such a
    member, so its host path is only recoverable from the static layout).
    """
    sources: Dict[str, str] = {}
    member_paths: Dict[str, str] = {}

    def walk(st: DerivedType, sym_prefix: str, access: str) -> None:
        for m in st.members:
            msym = f"{sym_prefix}_{m.name}"
            macc = f"{access}%{m.name}"
            if m.struct_name:
                nested = iface.struct_types.get(m.struct_name)
                if nested is not None:
                    # An array-of-records member (a pointer/allocatable array of
                    # a derived type -- ICON ``patch_3d%p_patch_2d(:)`` rank 1, or
                    # a value-record array ``edge2vert_coeff_cc_t(:,:,:,:)`` rank 4)
                    # must be INDEXED to reach a single record before descending
                    # into its members, else ``patch_3d%p_patch_2d%edges%...`` is a
                    # rank-1 reference assigned to a rank-0 symbol.  Index EVERY
                    # dim (``(1)`` for rank 1, ``(1,1,1,1)`` for rank 4) -- a single
                    # ``(1)`` on a multi-dim array-of-records is a rank mismatch.
                    # The kernel reads the (single-domain) first element -- the
                    # same constant record index the access generator prepends.
                    idx1 = ", ".join(["1"] * m.rank)
                    walk(nested, msym, f"{macc}({idx1})" if m.rank > 0 else macc)
                continue
            member_paths[msym] = macc
            if m.rank == 0:
                sources[msym] = macc
                continue
            for i in range(m.rank):
                sources[f"{msym}_d{i}"] = f"size({macc}, dim={i + 1})"
                sources[f"offset_{msym}_d{i}"] = f"lbound({macc}, dim={i + 1})"

    for a in iface.args:
        if a.struct_type:
            st = iface.struct_types.get(a.struct_type)
            if st is not None:
                walk(st, a.name, a.name)
    return sources, member_paths


def _build_symbol_assigns(frozen: FrozenSignature, plan: FlattenPlan, outer_dummy_set: set,
                          iface: OriginalInterface) -> List[str]:
    """Emit one assignment per free SDFG symbol from the caller's
    actual Fortran storage.

    A free symbol is either an array's per-dim lower bound
    (``offset_<arr>_d<i>`` -> ``lbound``) or extent
    (``<arr>_d<i>`` -> ``size``).  The struct-flatten plan supplies a
    precise ``size(st%a, dim=d)`` expression where it has one;
    otherwise we fall back to ``lbound``/``size`` on the arg's own
    Fortran expression (covers plain assumed-shape and non-default
    lower-bound dummies, which have no flatten entry).  Symbols that
    are themselves outer dummies are left for the caller to pass.
    """
    _module_sources = effective_module_sources(frozen, iface)
    # Struct-member free symbols (a member used only as a loop bound /
    # array extent, so the flatten pass lifted it to a free symbol with
    # no FlattenEntry  --  QE ``dfftt%ngm`` / ``dfftt%nnr`` /
    # ``size(dfftt%nl_d)``).  Sourced from the static ``struct_types``
    # layout as the last resort below.
    _struct_member_sources, _struct_member_paths = _struct_member_symbol_sources(iface)
    # Absent-optional args are allocated as a degenerate ``(1,1,...)`` placeholder
    # (no host source), so their extent is 1 -- a named blocking symbol
    # (``nproma`` / ``nblks_e``) must NOT be sourced from one or every
    # ``size(<arr>, dim)`` transient collapses to 1 and mesh-bounded writes smash.
    _unsourced_names = {a.sdfg_name for a in _unsourced_array_args(frozen, iface, plan)}
    arg_by_sdfg = {a.sdfg_name: a for a in frozen.args}
    # SCALAR outer dummies declared OPTIONAL -- their ``<name>_present``
    # companion is forwarded from the caller's actual ``present(<name>)`` (M1),
    # not defaulted.  Same scoping as the guarded-local data path.
    outer_optional_set = {oa.name for oa, _fa in _optional_outer_dummies(frozen, iface)}
    # Cap symbols of aos_alloc recipes are populated by the pack-in
    # code (``render_aos_alloc_pack_in`` writes ``cap_<m> = max_i(...)``)
    # before the SDFG call  --  skip them here so we don't emit a stray
    # TODO line or duplicate assignment.
    aos_cap_syms = {
        entry.recipe.cap_symbol
        for entry in plan.entries if entry.recipe.aos_alloc and entry.recipe.cap_symbol
    }
    # An extent symbol ``<flat>_d<i>`` is the i-th extent of exactly
    # the flat companion named ``<flat>``, so it must take *that*
    # entry's ``shape_exprs[i]`` (substring-scanning the shape list
    # mis-binds every symbol to the first entry's dim-1 on a
    # multi-member struct).
    flat_shapes: dict = {}
    # A rank-0 plan entry whose single flat companion *is* the symbol
    # name is a scalar struct member (``p_patch%nlev``) lifted to a
    # free symbol.  It carries no ``size(...)`` shape  --  its value is
    # the member itself.  ``read_exprs[0]`` is the dotted access with
    # ``$i`` placeholders (none for rank 0); strip them to the bare
    # path so the assignment reads ``p_patch%nlev`` directly.
    scalar_member: dict = {}
    # Whether the binding declares each flat arg's local as a POINTER vs an
    # ``allocatable, target`` scratch -- mirrors the struct-arg decl rule
    # (``aliasable and source_logical_kind in (0, 1)`` -> pointer, else
    # ``allocatable, target``).  Selects ``associated`` vs ``allocated`` for an
    # ``_allocated`` presence fold so the emitted test is legal for BOTH storage
    # classes (``allocated`` on a POINTER / ``associated`` on an ALLOCATABLE are
    # hard gfortran type errors).  Args with no recipe (double-buffer lanes,
    # which are ``c_f_pointer``'d pointers) default to POINTER.
    arg_is_pointer_local: dict = {}
    # Presence guard of each flat's SOURCE member (``associated(p_diag%
    # ddt_ua_adv)`` / ``allocated(...)``; '' for plain members).  Extents /
    # scalar values read the member's descriptor, which is undefined while
    # the member is absent -- guard the assignment and fall back to 0.
    flat_guard: dict = {}
    for entry in plan.entries:
        r = entry.recipe
        is_ptr_local = bool(r.aliasable and r.source_logical_kind in (0, 1))
        guard = _recipe_presence_guard(iface, r)
        for flat in r.flat_names:
            flat_shapes[flat] = r.shape_exprs
            arg_is_pointer_local[flat] = is_ptr_local
            flat_guard[flat] = guard
        if r.rank == 0 and len(r.flat_names) == 1 and r.read_exprs:
            scalar_member[r.flat_names[0]] = strip_index_args(r.read_exprs[0])

    def _guarded_assign(sym: str, rhs: str, guard: str, absent: str = "0") -> List[str]:
        if not guard:
            return [f"    {sym} = int({rhs}, c_int)"]
        return [
            f"    if ({guard}) then",
            f"      {sym} = int({rhs}, c_int)",
            f"    else",
            f"      {sym} = {absent}",
            f"    end if",
        ]

    def _member_sym_guard(sym: str) -> str:
        # ``offset_<flat>_d<i>`` / ``<flat>_d<i>`` / ``<flat>`` -> the leaf
        # member path recorded by ``_struct_member_symbol_sources``.
        base = sym[len("offset_"):] if sym.startswith("offset_") else sym
        m = _EXTENT_SYM_RE.match(base)
        if m and not _OFFSET_SYM_RE.match(base):
            base = m.group(1)
        path = _struct_member_paths.get(base)
        return _entry_presence_guard(iface, path) if path else ''

    out: List[str] = []
    for sym in frozen.free_symbols:
        if sym in outer_dummy_set:
            continue
        if sym in aos_cap_syms:
            continue
        m = _EXTENT_SYM_RE.match(sym)
        if m and not _OFFSET_SYM_RE.match(sym):
            flat, dim = m.group(1), int(m.group(2))
            shapes = flat_shapes.get(flat)
            if shapes is not None and dim < len(shapes):
                out.extend(_guarded_assign(sym, shapes[dim], flat_guard.get(flat, '')))
                continue
        if sym in scalar_member:
            out.extend(_guarded_assign(sym, scalar_member[sym], flat_guard.get(sym, '')))
            continue
        # A struct-member array's lower-bound offset must be taken from the MEMBER
        # itself (``lbound(p_patch%verts%end_block)``), NOT from the binding's
        # ``c_f_pointer`` flat companion of the same name -- that alias is built
        # ``c_f_pointer(c_loc(member), flat, [size])`` so its lbound is always 1.
        # ``_sym_from_intrinsic`` below resolves the SDFG name to that flat local,
        # so for a struct-member offset the member source (which spells the ``%``
        # access, block_builders ``_struct_member_symbol_sources``) takes priority:
        # an array ICON allocates with a non-default lower bound (the
        # refinement-control index arrays ``verts/cells/edges%{start,end}_
        # {block,index}``, allocated ``(min_rl : max_rl)``) otherwise gets offset 1
        # and the SDFG's ``arr[(idx) - offset]`` reads out of bounds at negative
        # ``rl``.  Extent symbols already read the member (via the flatten shape).
        if _OFFSET_SYM_RE.match(sym) and sym in _struct_member_sources:
            # Absent member -> neutral lower bound 1 (the kernel's absent
            # branch never evaluates the ``arr[(idx) - offset]`` math).
            out.extend(_guarded_assign(sym, _struct_member_sources[sym], _member_sym_guard(sym), absent="1"))
            continue
        # No flatten-plan size expr: derive the value directly from the
        # caller's array via lbound/size (closes the gap for plain
        # assumed-shape / non-default-lower-bound dummies, and the fallback
        # path that populates a NON-struct ``offset_<arr>_d<i>``).
        intr = _sym_from_intrinsic(sym, frozen)
        if intr is not None:
            fn, expr, dim = intr
            out.append(f"    {sym} = int({fn}({expr}, dim={dim}), c_int)")
            continue
        # A NAMED array extent (``n_zlev`` = ``size(vn, dim=2)``).  Must beat the
        # module-global fallback below: the SDFG sized its arrays/transients by
        # this symbol, so it has to equal the actual allocation, not a (possibly
        # unset) namelist global of the same name.
        ext = _sym_from_array_extent(sym, frozen, exclude=_unsourced_names)
        if ext is not None:
            expr, dim = ext
            out.append(f"    {sym} = int(size({expr}, dim={dim}), c_int)")
            continue
        # Last resort: a Fortran module global the kernel reads
        # directly (no dummy to query).  Bridge-auto-detected (or
        # explicitly overridden); ``use``-imported under the
        # ``__mod`` alias  --  assign from that import.
        if sym in _module_sources:
            out.append(f"    {sym} = int({_module_symbol_alias(sym)}, c_int)")
            continue
        # Presence of a deferred-storage MODULE global the kernel branches on:
        # ``ALLOCATED(g)`` -> ``g_allocated``, ``ASSOCIATED(g)`` -> ``g_allocated``,
        # ``PRESENT(g)`` -> ``g_present`` (the bridge folds these to a symbol).
        # Source it from the REAL host (``allocated``/``associated``) so a
        # conditionally-used global the caller left unallocated takes the
        # kernel's ABSENT branch: the defensive copy-in passes a degenerate data
        # buffer (which would otherwise look "present"), but the kernel must see
        # the host's TRUE presence to branch correctly.
        pres_base = next((sym[:-len(suf)] for suf in ("_allocated", "_present") if sym.endswith(suf)), None)
        if pres_base is not None:
            fa = arg_by_sdfg.get(pres_base)
            if fa is not None and (getattr(fa, 'module_origin_allocatable', False)
                                   or getattr(fa, 'module_origin_pointer', False)):
                present = _present(_module_symbol_alias(pres_base), getattr(fa, 'module_origin_pointer', False))
                out.append(f"    {sym} = int(merge(1, 0, {present}), c_int)")
                continue
            # An AoS-MATERIALISED companion (``aos_outer_rank > 0``: a nested
            # record-array member -- ``patch_3d%p_patch_2d(:)%cells%owned%
            # vertical_levels`` -- copied element-by-element into an
            # ``allocatable, target`` SoA scratch the wrapper unconditionally
            # allocates).  Its OWN local is therefore always allocated (so a
            # test on it is meaningless) AND is an ALLOCATABLE, so
            # ``associated(<local>)`` is a hard gfortran type error.  Source the
            # presence from the HOST member via the caller-side dotted path,
            # with the member's real storage class (``associated`` for a POINTER
            # member, ``allocated`` for an ALLOCATABLE).
            if fa is not None and fa.aos_outer_rank > 0 and sym.endswith("_allocated") \
                    and pres_base in _struct_member_paths:
                present = _present(_struct_member_paths[pres_base], fa.aos_member_pointer)
                out.append(f"    {sym} = int(merge(1, 0, {present}), c_int)")
                continue
            # A STRUCT-MEMBER (or other aliased array) ``ASSOCIATED``/``ALLOCATED``
            # fold the kernel branches on -- ICON gates the ``p_nh%prog(nnew)%w``
            # output store on ``associated(...)``, and the pg / gradp copy-ins on
            # the pg-array presence.  The binding ``c_f_pointer``s every DIRECTLY
            # aliased array arg to a POINTER local from the reconstructed struct,
            # so ``associated(<base>)`` is true exactly when the caller provided
            # the member (the shim allocates it).  Leaving the flag unset defaults
            # the branch OFF, dropping the output nondeterministically
            # (uninitialised local read).  Only ``_allocated`` folds resolve this
            # way; ``_present`` (OPTIONAL dummy) keeps its own branch below.
            if fa is not None and fa.kind == 'array' and sym.endswith("_allocated"):
                # A presence-GUARDED entry's flat is bound either way (alias
                # or degenerate scratch), so ``associated(<flat>)`` is
                # always true -- test the HOST member instead: the guard
                # expression IS the member's true definedness.
                host_guard = flat_guard.get(pres_base, '')
                present = host_guard or _present(pres_base, arg_is_pointer_local.get(pres_base, True))
                out.append(f"    {sym} = int(merge(1, 0, {present}), c_int)")
                continue
            # A NESTED struct-member POINTER/ALLOCATABLE the kernel branches on
            # (ICON's inlined minmaxmean gates on ``ASSOCIATED(in_subset %
            # vertical_levels)`` where ``in_subset`` is bound to ``patch_3d %
            # p_patch_2d(1) % cells % owned``).  No flat array arg backs the
            # member, so ``fa is None`` -- but the static struct layout knows how
            # to spell the caller-side dotted path.  Source the tracker from that
            # host member's definedness.  Pointer default (``associated``) mirrors
            # the array-arg fold above: a struct-member pointer is the
            # ``ASSOCIATED`` case the bridge folded into ``_allocated``.
            if sym.endswith("_allocated") and pres_base in _struct_member_paths:
                present = _present(_struct_member_paths[pres_base], True)
                out.append(f"    {sym} = int(merge(1, 0, {present}), c_int)")
                continue
        # An OPTIONAL-presence flag (``<dummy>_present``, registered by the
        # bridge for every ``present(x)`` fold).
        if sym.endswith("_present"):
            base = sym[:-len("_present")]
            # The wrapper's OWN optional dummy: forward the caller's actual
            # ``present(base)`` (the outer dummy is declared OPTIONAL, see
            # ``_outer_decl``) so a PROVIDED optional reads as present and the
            # kernel takes the right branch -- not the hardwired absent default.
            if base in outer_optional_set:
                out.append(f"    {sym} = int(merge(1, 0, present({base})), c_int)")
                continue
            # No provider and not a wrapper dummy: an inlined-callee optional
            # (QE ``becphi``/``becpsi``, ``run_on_gpu``) bottoming out here.
            # Fortran ``present()`` of an omitted optional is ``.false.`` -> 0,
            # which also matches the no-op call path.
            out.append(f"    {sym} = 0  ! optional absent (not forwarded by wrapper)")
            continue
        # A struct dummy's member used only symbolically: no plan entry,
        # but the static struct layout names it (``dfftt%ngm`` ->
        # ``dfftt_ngm``).  Read it straight from the caller's struct.
        if sym in _struct_member_sources:
            out.extend(_guarded_assign(sym, _struct_member_sources[sym], _member_sym_guard(sym)))
            continue
        out.append(f"    ! TODO: no plan entry gives size for free symbol {sym!r}")
    return out
