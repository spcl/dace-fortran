"""Thin coordinator -- turns ``(FrozenSignature, OriginalInterface,
FlattenPlan)`` into a ``<entry>_bindings.f90`` file.

Real work lives in sibling modules (``flatten_plan.py`` data model,
``loop_copy.py`` renderers, ``block_builders.py`` builders +
``assemble_module``); this file is pure orchestration.
"""

from pathlib import Path

from dace_fortran.bindings.block_builders import (
    assemble_module,
    build_c_interface,
    build_finalize,
    build_handle_state,
    build_wrapper_body,
    build_wrapper_head,
    build_wrapper_tail,
)
from dace_fortran.bindings.flatten_plan import FlattenPlan
from dace_fortran.bindings.fortran_interface import OriginalInterface
from dace_fortran.bindings.frozen_signature import FrozenSignature


def emit_bindings(
        frozen: FrozenSignature,
        iface: OriginalInterface,
        plan: FlattenPlan,
        out_path: str,
        dace_arglist: tuple = (),
        enum_maps: dict = None,
) -> Path:
    """Emit a Fortran binding module for the built SDFG.

    Creates ``out_path``'s parent dir if missing; overwrites any existing
    file.  ``dace_arglist`` is live codegen output (``CompiledSDFG._sig``),
    not snapshotted in ``FrozenSignature`` -- empty falls back to
    ``frozen.args`` order.  ``enum_maps`` (from
    :func:`rewrite_string_enum_to_integer`) makes the binding accept a
    ``CHARACTER`` dummy and ``SELECT CASE``-translate it to the integer
    the SDFG expects; the SDFG itself only ever sees the integer.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enum_maps = enum_maps or {}

    blocks = {
        'c_interface': build_c_interface(frozen, iface, dace_arglist),
        'handle_state': build_handle_state(iface),
        'wrapper_head': build_wrapper_head(frozen, iface, plan, enum_maps=enum_maps),
        'wrapper_body': build_wrapper_body(frozen, iface, plan, enum_maps=enum_maps),
        'wrapper_tail': build_wrapper_tail(frozen, iface, plan, dace_arglist, enum_maps=enum_maps),
        'finalize': build_finalize(iface),
    }
    out_path.write_text(assemble_module(iface, frozen, blocks))
    return out_path
