"""Auto-generate a ``bind(c, name='<entry>_c')`` wrapper around the
emitted ``<entry>_dace`` Fortran module procedure.

Why a separate shim?  The ``<entry>_dace`` wrapper emitted by
:func:`dace_fortran.bindings.emit_bindings` is a *Fortran module
procedure* -- its symbol is mangled by gfortran (e.g.
``__velocity_tendencies_dace_bindings_MOD_velocity_tendencies_dace``)
and its dummies are Fortran shape descriptors, not flat C pointers.
A C / ``ctypes`` / Python caller cannot reach it without writing a
hand-authored Fortran shim that ``USE``\\s the binding module and
re-exports the call under a known ``bind(c)`` symbol with flat C-ABI
dummies (the ``run_sr`` pattern in ``tests/mpi_comm_e2e_test.py``, the
``run_velocity_flat_c`` pattern in ``tests/icon_full/...``).

This emitter writes that shim mechanically from the same
:class:`OriginalInterface` the bindings emitter already consumes, so
downstream callers get a standalone ``.so`` with one stable C entry
per kernel and no per-kernel hand-written Fortran glue.

Supported dummy shapes:

  * **flat scalar / array dummies** -- the MVP shape; the dummy maps
    to one C-ABI slot (by-value for ``intent(in)`` scalars, ``c_ptr``
    + ``c_f_pointer`` alias otherwise).
  * **derived-type dummies whose every member is inline-flat**
    (scalar or static-shape array of scalar) -- the Phase 2.4 struct
    extension.  The dummy expands to one C-ABI slot per member; the
    shim allocates a local instance of the derived type, copies each
    member in, calls ``<entry>_dace`` with the whole struct, and (for
    ``out``/``inout``) copies each member back out.

Non-supported shapes that today raise
:class:`UnsupportedShimInterfaceError`:

  * derived types with nested derived-type members
  * derived types with ``allocatable`` / ``pointer`` / dynamic-shape
    members (the same v2 boundary that ``MarshalExternalStructs``
    rejects with its strict ``isInlineFlatMember`` check).
"""
from pathlib import Path
from typing import List

from dace_fortran.bindings.fortran_interface import (
    DerivedType,
    Member,
    OriginalArg,
    OriginalInterface,
)


class UnsupportedShimInterfaceError(NotImplementedError):
    """Raised when an :class:`OriginalInterface` carries a dummy the
    shim emitter cannot yet handle (nested struct members,
    ``allocatable`` / ``pointer`` members, dynamic shape, or any
    derived type the :class:`OriginalInterface` did not record a
    layout for in :attr:`OriginalInterface.struct_types`)."""


def _dim_spec(shape) -> str:
    """``(:,:)`` for rank-N, empty for scalars."""
    if not shape:
        return ""
    return "(" + ", ".join(":" for _ in shape) + ")"


def _shape_literal(shape) -> str:
    """``[d1, d2, ...]`` Fortran array constructor for the
    ``c_f_pointer`` extent argument."""
    return "[" + ", ".join(str(s) for s in shape) + "]"


def _is_inline_flat_member(m: Member) -> bool:
    """``True`` iff ``m`` is the inline-flat shape the shim's struct
    expansion can build a c_f_pointer alias for: a scalar (rank 0),
    or a static-shape array of scalar (rank > 0, no ``'?'`` /
    ``'*'`` / ``':'`` in shape, no nested struct -- the
    :class:`Member.fortran_type` already carries an ``iso_c_binding``
    type for scalar elements only)."""
    if m.rank == 0:
        return True
    return all(s not in ('?', '*', ':') for s in m.shape)


def _struct_module_use(iface: OriginalInterface, struct_name: str) -> str:
    """``use <mod>, only: <struct_name>[, <shape_const_1>, ...]`` for
    the module that defines ``struct_name``.  The static-shape
    constants the struct's member shapes reference (e.g. ``NX`` /
    ``NY``) live in the same module and must ride the import so the
    shim can spell them in its ``c_f_pointer`` calls."""
    for mod, syms in iface.used_modules.items():
        if struct_name in syms:
            return f"  use {mod}, only: {', '.join(syms)}"
    return ""


def _emit_flat_arg(a: OriginalArg, header_args: List[str],
                   decls_value: List[str], decls_ptr: List[str],
                   decls_local: List[str], c_f_calls: List[str],
                   call_args: List[str]):
    """Append the per-dummy split for a non-struct argument: scalar
    inputs ride by value, scalar outputs and arrays ride as ``c_ptr``
    + ``c_f_pointer`` alias.  Mutates the parallel lists in place."""
    if a.rank == 0 and a.intent in ('in', ''):
        header_args.append(a.name)
        decls_value.append(f"  {a.fortran_type}, value :: {a.name}")
        call_args.append(a.name)
        return
    ptr_name = f"{a.name}_p"
    header_args.append(ptr_name)
    decls_ptr.append(f"  type(c_ptr), value :: {ptr_name}")
    if a.rank == 0:
        # Length-1 array alias for scalar I/O (matches
        # ``feedback_scalar_io_convention``).
        decls_local.append(f"  {a.fortran_type}, pointer :: {a.name}(:)")
        c_f_calls.append(f"  call c_f_pointer({ptr_name}, {a.name}, [1])")
    else:
        decls_local.append(
            f"  {a.fortran_type}, pointer :: {a.name}{_dim_spec(a.shape)}")
        c_f_calls.append(f"  call c_f_pointer({ptr_name}, {a.name}, "
                         f"{_shape_literal(a.shape)})")
    call_args.append(a.name)


def _emit_struct_arg(a: OriginalArg, st: DerivedType,
                     header_args: List[str], decls_ptr: List[str],
                     decls_local: List[str], c_f_calls: List[str],
                     copy_in: List[str], copy_out: List[str],
                     call_args: List[str]):
    """Append the per-member split for a derived-type argument with
    inline-flat members only.

    The dummy itself becomes a local ``type(<struct>), target ::
    <name>``; each member rides as its own C-ABI slot
    ``<name>_<member>_p`` (``c_ptr, value``) aliased through
    ``c_f_pointer`` to the member's declared shape.  Per
    :attr:`OriginalArg.intent` the shim copies values in before the
    ``<entry>_dace`` call and (for ``out`` / ``inout``) copies them
    back after.  The intent contract holds at the *whole-struct*
    granularity since the member layout / intent split is not
    surfaced on :class:`Member` today.
    """
    for m in st.members:
        flat_name = f"{a.name}_{m.name}"
        ptr_name = f"{flat_name}_p"
        header_args.append(ptr_name)
        decls_ptr.append(f"  type(c_ptr), value :: {ptr_name}")
        if m.rank == 0:
            decls_local.append(f"  {m.fortran_type}, pointer :: {flat_name}(:)")
            c_f_calls.append(f"  call c_f_pointer({ptr_name}, {flat_name}, [1])")
        else:
            decls_local.append(
                f"  {m.fortran_type}, pointer :: {flat_name}{_dim_spec(m.shape)}")
            c_f_calls.append(f"  call c_f_pointer({ptr_name}, {flat_name}, "
                             f"{_shape_literal(m.shape)})")
        if a.intent in ('in', 'inout', ''):
            if m.rank == 0:
                copy_in.append(f"  {a.name}%{m.name} = {flat_name}(1)")
            else:
                copy_in.append(f"  {a.name}%{m.name} = {flat_name}")
        if a.intent in ('out', 'inout'):
            if m.rank == 0:
                copy_out.append(f"  {flat_name}(1) = {a.name}%{m.name}")
            else:
                copy_out.append(f"  {flat_name} = {a.name}%{m.name}")
    decls_local.append(f"  {a.fortran_type}, target :: {a.name}")
    call_args.append(a.name)


def emit_bind_c_shim(iface: OriginalInterface, out_path: str) -> Path:
    """Emit ``<entry>_c.f90`` -- a thin ``bind(c)`` wrapper around the
    binding module's ``<entry>_dace`` procedure.

    Per dummy:

    * **scalar input** (``rank == 0``, ``intent in / ''``): declared
      ``<fortran_type>, value`` -- the C-side passes the value
      directly, no pointer indirection.
    * **scalar output** (``rank == 0``, ``intent out / inout``):
      declared as a ``c_ptr, value`` and aliased through
      ``c_f_pointer`` to a length-1 array.  Matches
      ``feedback_scalar_io_convention`` -- inputs by value, outputs
      via pointer to a length-1 buffer.
    * **array** (``rank > 0``): declared as a ``c_ptr, value`` and
      aliased through ``c_f_pointer`` to the dummy's declared shape.
      The shape extents reference the scalar-input dummies preceding
      the array in C-ABI order, so the C caller passes dims first.
    * **derived-type dummy whose every member is inline-flat**: one
      C-ABI slot per member (named ``<dummy>_<member>_p``); a local
      ``type(<struct>), target :: <dummy>`` instance is assembled
      from the flat aliases (copy-in), passed whole to
      ``<entry>_dace``, and (for ``out``/``inout``) copied back out
      per member.

    After all aliases are set the shim calls
    ``<entry>_dace(...)`` with the *Fortran-side* names (the local
    aliases for flat dummies, the local instance for struct dummies)
    and finalises with ``<entry>_dace_finalize()`` so the DaCe handle
    is reference-counted out on the last call.

    :param iface: caller-facing Fortran interface.
    :param out_path: where to write ``<entry>_c.f90``.  Parent dirs
                     are created as needed; any existing file at the
                     path is overwritten.
    :returns: ``out_path`` as a :class:`~pathlib.Path` (just written).
    :raises UnsupportedShimInterfaceError: a struct dummy has a
            member shape the emitter cannot handle (nested struct,
            ``allocatable`` / ``pointer``, dynamic shape, or no
            recorded layout in
            :attr:`OriginalInterface.struct_types`).
    """
    # Validate every struct dummy has an inline-flat-only layout.
    for a in iface.args:
        if a.struct_type is None:
            continue
        st = iface.struct_types.get(a.struct_type)
        if st is None:
            raise UnsupportedShimInterfaceError(
                f"bind(c) shim: dummy {a.name!r} is type {a.fortran_type} "
                f"but no layout was recorded for {a.struct_type!r} in "
                f"OriginalInterface.struct_types.  Supply a hand-authored "
                f"interface with the member list.")
        for m in st.members:
            if not _is_inline_flat_member(m):
                raise UnsupportedShimInterfaceError(
                    f"bind(c) shim: struct {a.struct_type!r} has a "
                    f"non-inline-flat member {m.name!r} ({m.fortran_type}, "
                    f"rank={m.rank}, shape={m.shape}).  Only scalar and "
                    f"static-shape-array-of-scalar members are supported; "
                    f"allocatable / pointer / dynamic-shape / nested "
                    f"derived-type members need a hand-authored shim "
                    f"(see ``run_velocity_flat_c`` in "
                    f"tests/icon_full/velocity_full_caller.f90 for the "
                    f"ICON-velocity pattern).")

    entry = iface.entry
    c_name = f"{entry}_c"
    bind_mod = f"{entry}_dace_bindings"

    header_args: List[str] = []
    decls_value: List[str] = []
    decls_ptr: List[str] = []
    decls_local: List[str] = []
    c_f_calls: List[str] = []
    copy_in: List[str] = []
    copy_out: List[str] = []
    call_args: List[str] = []
    struct_use_lines: List[str] = []
    seen_struct_uses = set()
    for a in iface.args:
        if a.struct_type is None:
            _emit_flat_arg(a, header_args, decls_value, decls_ptr,
                           decls_local, c_f_calls, call_args)
            continue
        st = iface.struct_types[a.struct_type]
        _emit_struct_arg(a, st, header_args, decls_ptr, decls_local,
                         c_f_calls, copy_in, copy_out, call_args)
        # Pick up the ``use`` line for the struct's defining module so
        # the shim can spell ``type(<struct>)`` and any shape constants
        # the member declarations reference.
        u = _struct_module_use(iface, a.struct_type)
        if u and u not in seen_struct_uses:
            struct_use_lines.append(u)
            seen_struct_uses.add(u)

    decl_block = "\n".join(decls_value + decls_ptr + decls_local)
    body_parts: List[str] = []
    if c_f_calls:
        body_parts.append("\n".join(c_f_calls))
    if copy_in:
        body_parts.append("\n".join(copy_in))
    body_parts.append(f"  call {entry}_dace({', '.join(call_args)})")
    if copy_out:
        body_parts.append("\n".join(copy_out))
    body_parts.append(f"  call {entry}_dace_finalize()")
    body_block = "\n".join(body_parts)

    use_lines = ["  use iso_c_binding", *struct_use_lines,
                 f"  use {bind_mod}, only: {entry}_dace, {entry}_dace_finalize"]
    lines = [
        "! AUTO-GENERATED by dace_fortran.bindings.bind_c_shim -- do not edit.",
        f"! bind(c) shim around module procedure {bind_mod}::{entry}_dace.",
        f"subroutine {c_name}({', '.join(header_args)}) "
        f"bind(c, name='{c_name}')",
        *use_lines,
        "  implicit none",
        decl_block,
        body_block,
        f"end subroutine {c_name}",
        "",
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path
