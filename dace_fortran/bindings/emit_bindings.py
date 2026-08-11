# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
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
    splice_acc_staging,
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
        acc_residency=None,
) -> Path:
    """Emit a Fortran binding module for the built SDFG.

    Creates ``out_path``'s parent dir if missing; overwrites any existing
    file.  ``dace_arglist`` is live codegen output (``CompiledSDFG._sig``),
    not snapshotted in ``FrozenSignature`` -- empty falls back to
    ``frozen.args`` order.  ``enum_maps`` (from
    :func:`rewrite_string_enum_to_integer`) makes the binding accept a
    ``CHARACTER`` dummy and ``SELECT CASE``-translate it to the integer
    the SDFG expects; the SDFG itself only ever sees the integer.

    ``acc_residency`` (optional) is an
    :class:`dace_fortran.bindings.acc_transfers.AccTransferPlan` -- the sidecar
    x SDFG-storage crossing from :func:`plan_acc_transfers` -- and makes the
    wrapper stage ICON's device-resident arguments itself: ``UPDATE HOST
    ASYNC(1)`` before the copy-in, ``UPDATE DEVICE ASYNC(1)`` + ``WAIT(1)``
    after the copy-out, ``HOST_DATA USE_DEVICE`` around the SDFG invocation
    (see :func:`block_builders.splice_acc_staging`).  ``None`` (the default)
    emits today's byte-identical CPU-only module.
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
    if acc_residency is not None:
        blocks = splice_acc_staging(blocks, iface.entry, acc_residency)
    out_path.write_text(assemble_module(iface, frozen, blocks, plan))
    return out_path
