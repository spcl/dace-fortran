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

Scope (MVP): kernels whose :class:`OriginalInterface` declares only
*flat* dummies (no ``struct_type``).  Derived-type interfaces (the
ICON ``velocity_tendencies`` shape with ``t_patch`` / ``t_int_state``
/ ...) need per-member struct construction inside the shim from the
``FlattenPlan`` recipes; that extension is a follow-up.
"""
from pathlib import Path
from typing import List

from dace_fortran.bindings.fortran_interface import OriginalInterface


class UnsupportedShimInterfaceError(NotImplementedError):
    """Raised when an :class:`OriginalInterface` shape the MVP shim
    emitter can't handle (today: any derived-type dummy)."""


def _dim_spec(shape) -> str:
    """``(:,:)`` for rank-N, empty for scalars."""
    if not shape:
        return ""
    return "(" + ", ".join(":" for _ in shape) + ")"


def _shape_literal(shape) -> str:
    """``[d1, d2, ...]`` Fortran array constructor for the
    ``c_f_pointer`` extent argument."""
    return "[" + ", ".join(str(s) for s in shape) + "]"


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

    After all aliases are set the shim calls
    ``<entry>_dace(...)`` with the *Fortran-side* names (the local
    aliases) and finalises with ``<entry>_dace_finalize()`` so the
    DaCe handle is reference-counted out on the last call.

    :param iface: caller-facing Fortran interface -- only flat dummies
                  supported in this MVP.
    :param out_path: where to write ``<entry>_c.f90``.  Parent dirs
                     are created as needed; any existing file at the
                     path is overwritten.
    :returns: ``out_path`` as a :class:`~pathlib.Path` (just written).
    :raises UnsupportedShimInterfaceError: any dummy is a derived type.
    """
    for a in iface.args:
        if a.struct_type is not None:
            raise UnsupportedShimInterfaceError(
                f"bind(c) shim auto-gen does not yet support derived-type "
                f"dummies ({a.name!r}: {a.fortran_type}).  Use a "
                f"hand-authored shim, or wait for the FlattenPlan-driven "
                f"struct-construction extension.")

    entry = iface.entry
    c_name = f"{entry}_c"
    bind_mod = f"{entry}_dace_bindings"

    # Per-dummy split: scalar-by-value vs c_ptr+c_f_pointer alias.  We
    # rename the c_ptr dummies to ``<name>_p`` so the Fortran-side local
    # alias keeps the original name (and the call to ``<entry>_dace``
    # reads naturally).  Declarations and aliases are kept in separate
    # lists so the rendered subroutine has all decls before any
    # executable statement -- gfortran's strict F2003+ ordering.
    header_args: List[str] = []
    decls_value: List[str] = []
    decls_ptr: List[str] = []
    decls_local: List[str] = []
    c_f_calls: List[str] = []
    call_args: List[str] = []
    for a in iface.args:
        # Scalar input -- pass-by-value, no rename.
        if a.rank == 0 and a.intent in ('in', ''):
            header_args.append(a.name)
            decls_value.append(f"  {a.fortran_type}, value :: {a.name}")
            call_args.append(a.name)
            continue
        # Scalar output / array -- pass-by-pointer, alias inside.
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

    decl_block = "\n".join(decls_value + decls_ptr + decls_local)
    body_block = "\n".join(c_f_calls)
    lines = [
        "! AUTO-GENERATED by dace_fortran.bindings.bind_c_shim -- do not edit.",
        f"! bind(c) shim around module procedure {bind_mod}::{entry}_dace.",
        f"subroutine {c_name}({', '.join(header_args)}) "
        f"bind(c, name='{c_name}')",
        "  use iso_c_binding",
        f"  use {bind_mod}, only: {entry}_dace, {entry}_dace_finalize",
        "  implicit none",
        decl_block,
        body_block,
        f"  call {entry}_dace({', '.join(call_args)})",
        f"  call {entry}_dace_finalize()",
        f"end subroutine {c_name}",
        "",
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path
