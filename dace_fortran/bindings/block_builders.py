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

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dace_fortran.bindings.flatten_plan import (
    FlattenPlan,
    strip_index_args,
)
from dace_fortran.bindings.fortran_interface import OriginalInterface
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
    init_arg_decls = "".join(
        f"      {_init_symbol_decl(s, frozen)} :: {s}\n" for s in init_syms)
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
        rendered = rendered.replace("  end interface",
                                    _MPI_COMM_F2C_IFACE + _MPI_COMM_SIZE_IFACE + "  end interface")
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
        return [f"    allocate({flat})",
                f"    {flat} = {outer_expr}"]
    shape_args = ", ".join(recipe.shape_exprs)
    return [f"    allocate({flat}({shape_args}))",
            f"    {flat} = {outer_expr}"]


def _render_logical_bridge_copy_out(recipe, outer_expr: str) -> List[str]:
    """Inverse of :func:`_render_logical_bridge_copy_in`: pack the
    SDFG-side bool flat back into the source struct slot for
    ``intent(out)/inout`` entries, then release the scratch."""
    flat = recipe.flat_names[0]
    return [f"    {outer_expr} = {flat}",
            f"    deallocate({flat})"]


def _fortran_c_value_type(dtype: str) -> str:
    """Map a frozen-arg ``dtype`` string to its ``iso_c_binding`` form
    for a pass-by-value Fortran dummy."""
    table = {
        'int32': 'integer(c_int)',
        'int64': 'integer(c_long_long)',
        'float32': 'real(c_float)',
        'float64': 'real(c_double)',
        'bool': 'logical(c_bool)',
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


def build_wrapper_head(frozen: FrozenSignature, iface: OriginalInterface, plan: FlattenPlan) -> str:
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
    outer_dummy_decls = "\n".join(f"    {a.fortran_type}, intent({a.intent or 'inout'}), target :: {a.name}"
                                  f"{_dim_spec(a.shape)}" for a in iface.args)

    flat_ptr_lines: List[str] = []
    scratch_lines: List[str] = []
    max_loop_rank = 0
    for entry in plan.entries:
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
        else:
            if r.rank > 0:
                max_loop_rank = max(max_loop_rank, r.rank)
            for flat in r.flat_names:
                scratch_lines.append(f"    {ftype}, allocatable, target :: {flat}{shape_dims}")

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
    flat_names = {f for entry in plan.entries for f in entry.recipe.flat_names}

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

    symbol_decls = "\n".join(
        f"    {_local_decl(s)} :: {s}"
        for s in frozen.free_symbols
        if s not in outer_dummy_set and s not in flat_names)
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


def build_wrapper_body(frozen: FrozenSignature, iface: OriginalInterface, plan: FlattenPlan) -> str:
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
    body: List[str] = ["    ! ----- Copy-in / alias per flatten entry -----"]
    for entry in plan.entries:
        r = entry.recipe
        # Four mutually exclusive emitter shapes  --  see FlattenRecipe
        # for the flag matrix.  ``source_logical_kind > 1`` overrides
        # the aliasable path with a width-bridging scratch + Fortran
        # intrinsic LOGICAL-kind conversion (see ``build_wrapper_head``
        # for the rationale).
        if r.aos_alloc:
            body.extend(render_aos_alloc_pack_in(r, entry.outer_expr))
        elif r.aliasable and r.source_logical_kind in (0, 1):
            body.extend(render_alias_calls(r))
        elif r.aliasable and r.source_logical_kind > 1:
            body.extend(_render_logical_bridge_copy_in(r, entry.outer_expr))
        else:
            body.extend(render_copy_in_loop(r))

    _, copy_in_lines, _, _ = _build_logical_bridges(frozen, iface)
    if copy_in_lines:
        body.append("")
        body.append("    ! ----- LOGICAL -> logical(c_bool) bridge (copy-in) -----")
        body.extend(copy_in_lines)

    orphans = _orphan_module_args(frozen, iface, plan)
    if orphans:
        body.append("")
        body.append("    ! ----- Module-global args sourced from use-imports -----")
        for a, _mod, _member in orphans:
            alias = _module_symbol_alias(a.sdfg_name)
            if a.rank > 0:
                # Allocate to the SDFG arg's own concrete extent
                # (``FrozenArg.shape``)  --  NOT ``size(alias)``: the
                # source can be a *scalar* module global (ICON timer
                # handles, ``i_am_accel_node``) the bridge lifted to a
                # length-1 array, and ``size(scalar)`` is a hard
                # Fortran error.  Scalar-source -> scalar broadcast
                # into the length-1 buffer; array-source -> conformant
                # array assign.
                dims = ", ".join(a.shape) if a.shape else "1"
                body.append(f"    allocate({a.sdfg_name}({dims}))")
            body.append(f"    {a.sdfg_name} = {alias}")

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
        body.append(f"    dace_user_comm_size_err = dace_mpi_comm_size({_USER_COMM_SYMBOL_NAME}, dace_user_comm_size_buf)")
        body.append(f"    {_USER_COMM_SIZE_SYMBOL_NAME} = int(dace_user_comm_size_buf, c_long_long)")

    sym_lines = _build_symbol_assigns(frozen, plan, outer_dummy_set, iface)
    if sym_lines:
        body.append("")
        body.append("    ! ----- Symbol population -----")
        body.extend(sym_lines)
    return "\n".join(body)


# ---------------------------------------------------------------------------
# Wrapper tail  --  init-count bump, SDFG call, copy-back, deallocate, end sub
# ---------------------------------------------------------------------------


def build_wrapper_tail(frozen: FrozenSignature, iface: OriginalInterface, plan: FlattenPlan,
                       dace_arglist: tuple = ()) -> str:
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
    for entry in plan.entries:
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
        if r.aliasable and r.source_logical_kind > 1:
            if entry.writeback_intent in ('out', 'inout'):
                copy_out_lines.extend(
                    _render_logical_bridge_copy_out(r, entry.outer_expr))
            else:
                copy_out_lines.append(f"    deallocate({r.flat_names[0]})")
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
        copy_out_lines.extend(render_copy_out_loop(r, entry.outer_expr))

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

    bridge_block = ""
    if bridge_copy_out:
        bridge_block = "\n    ! ----- logical(c_bool) -> LOGICAL bridge (copy-out + dealloc) -----\n" + "\n".join(
            bridge_copy_out)

    writeback_block = ""
    if module_writeback_lines:
        writeback_block = "\n    ! ----- Write-back for kernel-written module globals -----\n" + "\n".join(
            module_writeback_lines)

    if not copy_out_lines and not bridge_copy_out and not module_writeback_lines:
        return call_block

    copy_out_block = ""
    if copy_out_lines:
        copy_out_block = "\n    ! ----- Copy-out for writeable deep-copy entries -----\n" + "\n".join(copy_out_lines)
    marker = f"  end subroutine {iface.entry}_dace"
    pre, post = call_block.split(marker, 1)
    return pre + copy_out_block + bridge_block + writeback_block + "\n" + marker + post


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
    use_statements = "\n".join(use_lines)
    wrapper_body = (blocks['wrapper_head'] + "\n" + blocks['wrapper_body'] + "\n" + blocks['wrapper_tail'])
    return _load("module.f90.in").format(
        entry=iface.entry,
        schema_version=frozen.schema_version,
        use_statements=use_statements,
        c_interface=blocks['c_interface'],
        handle_state=blocks['handle_state'],
        wrapper_body=wrapper_body,
        finalize_body=blocks['finalize'],
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dim_spec(shape) -> str:
    """Render the dimension spec suffix for an outer dummy's
    declaration as a *postfix* shape (``name(d1,d2)``), not a
    ``dimension(...)`` attribute.  Postfix is the only form that
    works when the suffix lands after the ``::`` -- a leading comma
    plus ``dimension(...)`` after the ``::`` is read by Fortran as
    a SECOND variable declaration (so ``:: mask, dimension(n)``
    silently declares ``mask`` AND ``dimension`` of unknown rank).

    Assumed-shape dummies render the surviving ``?`` placeholders as
    ``:``; explicit extents pass through.  An empty shape leaves the
    declaration as a scalar (no suffix).
    """
    if not shape:
        return ""
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
    for entry in plan.entries:
        r = entry.recipe
        for flat in r.flat_names:
            flat_shapes[flat] = r.shape_exprs
        if r.rank == 0 and len(r.flat_names) == 1 and r.read_exprs:
            scalar_member[r.flat_names[0]] = strip_index_args(r.read_exprs[0])

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
                out.append(f"    {sym} = int({shapes[dim]}, c_int)")
                continue
        if sym in scalar_member:
            out.append(f"    {sym} = int({scalar_member[sym]}, c_int)")
            continue
        # No flatten-plan size expr: derive the value directly from the
        # caller's array via lbound/size (closes the gap for plain
        # assumed-shape / non-default-lower-bound dummies, and is the
        # ONLY path that ever populates an ``offset_<arr>_d<i>``).
        intr = _sym_from_intrinsic(sym, frozen)
        if intr is not None:
            fn, expr, dim = intr
            out.append(f"    {sym} = int({fn}({expr}, dim={dim}), c_int)")
            continue
        # Last resort: a Fortran module global the kernel reads
        # directly (no dummy to query).  Bridge-auto-detected (or
        # explicitly overridden); ``use``-imported under the
        # ``__mod`` alias  --  assign from that import.
        if sym in _module_sources:
            out.append(f"    {sym} = int({_module_symbol_alias(sym)}, c_int)")
            continue
        out.append(f"    ! TODO: no plan entry gives size for free symbol {sym!r}")
    return out
