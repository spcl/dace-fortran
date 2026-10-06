# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
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
from dace_fortran.bindings.acc_transfers import AccTransferPlan, Directive
from dace_fortran.bindings.flatten_plan import FlattenPlan
from dace_fortran.bindings.fortran_interface import OriginalInterface
from dace_fortran.bindings.frozen_signature import FrozenSignature


def emit_bindings(
    frozen: FrozenSignature,
    iface: OriginalInterface,
    plan: FlattenPlan,
    out_path: str | Path,
    dace_arglist: tuple[str, ...] = (),
    enum_maps: dict | None = None,
    acc_residency: AccTransferPlan | None = None,
    init_symbols: tuple[str, ...] | None = None,
    directive: Directive = Directive.OPENACC,
) -> Path:
    """Emit a Fortran binding module for the built SDFG.

    Creates ``out_path``'s parent dir if missing; overwrites any existing
    file.  ``dace_arglist`` is live codegen output (``CompiledSDFG._sig``),
    not snapshotted in ``FrozenSignature`` -- empty falls back to
    ``frozen.args`` order.  ``enum_maps`` (from
    :func:`rewrite_string_enum_to_integer`) makes the binding accept a
    ``CHARACTER`` dummy and ``SELECT CASE``-translate it to the integer
    the SDFG expects; the SDFG itself only ever sees the integer.

    ``init_symbols`` (optional) is the parameter list the compiled ``__dace_init`` declares, from
    :func:`dace_fortran.bindings.build_fortran_library.compiled_init_symbols`; ``None`` assumes every free symbol.

    ``acc_residency`` (optional) is an
    :class:`dace_fortran.bindings.acc_transfers.AccTransferPlan` -- the sidecar
    x SDFG-storage crossing from :func:`plan_acc_transfers` -- and makes the
    wrapper stage ICON's device-resident arguments itself: ``UPDATE HOST
    ASYNC(1)`` before the copy-in, ``UPDATE DEVICE ASYNC(1)`` + ``WAIT(1)``
    after the copy-out, ``HOST_DATA USE_DEVICE`` around the SDFG invocation
    (see :func:`block_builders.splice_acc_staging`).  ``None`` (the default)
    falls back to the standalone direction: if ``frozen`` records arguments an
    offload pass moved to the device, the wrapper gets an ``!$ACC DATA`` region
    staging the host caller's buffers up; if it records none, the emitted module
    is byte-identical to today's CPU-only one.

    ``directive`` picks the language the staging is written in: OpenACC (nvfortran, ICON) or OpenMP offload
    (e.g. ROCm amdflang); the plan is the same.
    """
    from dace_fortran.bindings.acc_transfers import plan_frozen_transfers

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    enum_maps = enum_maps or {}
    if acc_residency is None:
        acc_residency = plan_frozen_transfers(frozen)

    blocks = {
        "c_interface": build_c_interface(frozen, iface, dace_arglist, init_symbols),
        "handle_state": build_handle_state(iface),
        "wrapper_head": build_wrapper_head(frozen, iface, plan, enum_maps=enum_maps),
        "wrapper_body": build_wrapper_body(frozen, iface, plan, enum_maps=enum_maps),
        "wrapper_tail": build_wrapper_tail(
            frozen, iface, plan, dace_arglist, enum_maps=enum_maps, init_symbols=init_symbols
        ),
        "finalize": build_finalize(iface),
    }
    blocks = splice_acc_staging(blocks, iface.entry, acc_residency, directive)
    out_file.write_text(assemble_module(iface, frozen, blocks, plan))
    return out_file
