"""Thin coordinator  --  turns ``(FrozenSignature, OriginalInterface,
FlattenPlan)`` into a ``<entry>_bindings.f90`` file.

The real work is in sibling modules:
    * ``flatten_plan.py``     --  data model for the pass's output.
    * ``loop_copy.py``        --  per-recipe renderers.
    * ``block_builders.py``   --  one builder per Fortran section +
                               ``assemble_module`` stitcher.

``emit_bindings`` is ~10 lines of orchestration.  No Fortran-
construction logic lives here; it all routes through the named
builders so each concern is test-isolated.
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

    Side effect: creates ``out_path``'s parent directory if missing and
    overwrites any existing file there.

    :param frozen: ``FrozenSignature`` snapshot  --  the SDFG-facing arg
                   list + free symbols, from ``SDFGBuilder.build()``
                   (drift-checked by ``build_fortran_library``).
    :param iface: ``OriginalInterface``  --  the caller-facing Fortran
                  surface of the entry subroutine (dummies + struct
                  layouts, snapshotted from the HLFIR pre-pass).
    :param plan: ``FlattenPlan``  --  the record of every unpack
                 ``hlfir-flatten-structs`` performed; one source of
                 truth for copy-in / copy-out code.
    :param out_path: where to write ``<entry>_bindings.f90``.
    :param dace_arglist: the live ``__program_<entry>`` argument-name
                  order DaCe codegen emitted (``CompiledSDFG._sig``),
                  supplied by ``build_fortran_library`` from the
                  just-compiled SDFG.  This is codegen output -- not a
                  stable contract -- so it is passed in rather than
                  snapshotted in ``FrozenSignature``.  Empty -> the
                  emitter falls back to ``frozen.args`` order.
    :param enum_maps: ``{arg_name: {literal_lower: int}}`` from
                  :func:`rewrite_string_enum_to_integer`.  When an
                  arg name matches one of ``iface.args``, the emitter
                  generates the binding with a ``CHARACTER(LEN=N)``
                  outer dummy (length sized by the longest literal)
                  plus an internal ``SELECT CASE`` that translates
                  the string to the integer the SDFG expects.  The
                  SDFG itself only ever receives the integer; the
                  binding is the only place the string is accepted.
                  Empty / ``None`` -> emitter behaves identically to
                  the pre-enum-maps path.
    :returns: ``out_path`` as a ``Path`` (just materialised).
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
