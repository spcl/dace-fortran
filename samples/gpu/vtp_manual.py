# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""dace-gpu-manual lane: the human-written VelocityTendenciesPipeline (VTP) flow.

Loads VTP's stage-3 SDFG artifact and runs its stage-4 GPU entry point
(``utils.stages.stage4:optimization_action`` -- offload, persistent transients,
body timer) live from the checkout at ``VTP_DIR``. Stages 1-3 cannot be re-run
here: they import passes that only exist on VTP's own DaCe branch. The VTP
checkout is read-only; everything it needs at compile time (reduction-kernel
sources, headers) is injected into the SDFG's global code as in-memory text.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

VTP_DEFAULT = "/capstor/scratch/cscs/ybudanaz/aarch64/VelocityTendenciesPipeline"
DEFAULT_VARIANT = "velocity_no_nproma_if_prop_lvn_only_0_istep_1"
_HEADERS = ("reductions_cpu.h", "reductions_kernel.cuh", "reductions_device.cuh")
_GLOBAL_FLAG_LITERALS = {"i_am_accel_node": 1, "timers_level": 0, "timer_intp": 0, "timer_solve_nh_veltend": 0}


def vtp_dir() -> Path:
    return Path(os.environ.get("VTP_DIR", VTP_DEFAULT))


def vtp_variant() -> str:
    return os.environ.get("VTP_VARIANT", DEFAULT_VARIANT)


def vtp_describe() -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(vtp_dir()), "describe", "--always", "--dirty"],
                                      text=True,
                                      stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception as exc:  # provenance is best-effort, never fatal
        return f"unknown({type(exc).__name__})"


def build_manual_sdfg():
    root = vtp_dir()
    print(f"VTP: {root} @ {vtp_describe()} variant={vtp_variant()}", flush=True)
    if not root.is_dir():
        raise SystemExit(f"no VTP checkout at {root} (set VTP_DIR)")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from utils.stages.stage4 import optimization_action
    import dace
    stage3 = root / "codegen" / "stage3" / f"{vtp_variant()}.sdfgz"
    if not stage3.is_file():
        raise SystemExit(f"no stage-3 artifact at {stage3} (VTP_DIR/VTP_VARIANT wrong?)")
    try:
        # Not our cache to rebuild -- name the truncated file instead of dying on its gzip magic.
        sdfg = dace.SDFG.from_file(str(stage3))
    except Exception as exc:
        raise SystemExit(f"stage-3 artifact {stage3} is unreadable ({type(exc).__name__}: {exc}); "
                         "re-export it from VTP") from exc
    sdfg = optimization_action(sdfg)
    _demote_persistent_scalars(sdfg)
    _demote_host_write_gpu_maps(sdfg)
    _absolutize_includes(sdfg)
    _shim_gpu_aliases(sdfg)
    _inline_reduction_kernels(sdfg)
    return sdfg


def _shim_gpu_aliases(sdfg) -> None:
    """VTP's sync tasklets say gpuStreamSynchronize; the user's DaCe branch aliases
    gpu* to cuda* everywhere, this one nowhere -- shim it into the frame."""
    from dace.properties import CodeBlock
    import dace
    shim = ("#ifndef gpuStreamSynchronize\n"
            "#define gpuStreamSynchronize cudaStreamSynchronize\n"
            "#endif\n")
    existing = sdfg.global_code.get("frame")
    prior = existing.code if existing is not None and isinstance(existing.code, str) else ""
    sdfg.global_code["frame"] = CodeBlock(shim + prior, dace.dtypes.Language.CPP)


def check_variant_matches_deck(meta: dict) -> None:
    """The deck holds ONE timestep; the lane must run exactly the variant it specializes."""
    m = re.search(r"lvn_only_(\d)_istep_(\d)", vtp_variant())
    if m is None:
        raise SystemExit(f"cannot parse lvn/istep from VTP_VARIANT {vtp_variant()!r}")
    lvn, istep = int(m.group(1)), int(m.group(2))
    if (meta.get("lvn_only"), meta.get("istep")) != (lvn, istep):
        raise SystemExit(f"deck is lvn_only={meta.get('lvn_only')} istep={meta.get('istep')} but "
                         f"VTP_VARIANT={vtp_variant()} specializes lvn_only={lvn} istep={istep}")


def _deck_key_of_symbol(sym: str):
    """(deck key, dim, kind) behind one VTP size/offset symbol; None if the name is not one.

    Three f2dace spellings: struct members (``__CG_p_diag__m_SOA_ddt_vn_apc_pc_d_1``),
    grid-struct members (``SOA_end_block_d_0_cells_p_patch_2``), locals (``OA_z_kin_hor_e_d_0``).
    """
    import run_velocity_perf as cpu
    flat = cpu.flat_name(sym)
    m = re.match(r"^(SA|SOA)_(.+)_d_(\d+)_(cells|edges|verts)_p_patch_\d+$", flat)
    if m:
        return f"p_patch_{m.group(4)}_{m.group(2)}", int(m.group(3)), m.group(1)
    m = re.match(r"^(?:(.*)_)?(SA|SOA)_(.+)_d_(\d+)$", flat)
    if m:
        prefix, kind, arr, dim = m.group(1), m.group(2), m.group(3), int(m.group(4))
        return (f"{prefix}_{arr}" if prefix else arr), dim, kind
    m = re.match(r"^(A|OA)_(.+)_d_(\d+)$", flat)
    if m:
        return m.group(2), int(m.group(3)), {"A": "SA", "OA": "SOA"}[m.group(1)]
    return None


def bind_manual_call(sdfg, arrays: dict, meta: dict) -> dict:
    """The harness deck (frontend ABI) bound to VTP's SoA signature: same flat names,
    sizes read off the deck arrays, lower bounds off meta['lbounds']."""
    import numpy as np
    import dace.data as dt
    import run_velocity_perf as cpu
    check_variant_matches_deck(meta)
    lbounds = meta["lbounds"]
    scalars = cpu.meta_scalars(meta)
    call: dict = {}
    sizes: dict = {}
    missing: list = []

    for name, desc in sdfg.arglist().items():
        if not isinstance(desc, dt.Array) or name in ("__gpu_t_begin", "__gpu_t_end"):
            continue
        flat = cpu.flat_name(name)
        npdt = desc.dtype.as_numpy_dtype()
        arr = arrays.get(flat)
        if arr is None:
            # global_data members ride the deck under their bare frontend name/seed
            bare = flat.removeprefix("global_data_")
            seed = cpu.seed_value(bare, scalars)
            if seed is None:
                missing.append(f"array {name} (deck key {flat}): no data and no seed")
                continue
            shape = tuple(int(s) if str(s).isdigit() else 1 for s in desc.shape)
            arr = np.full(shape, seed, dtype=npdt, order="F")
        elif arr.dtype != npdt:
            arr = arr.astype(npdt, order="F")
        for dim, expr in enumerate(desc.shape):
            if str(expr).isdigit():
                if int(str(expr)) != arr.shape[dim]:
                    missing.append(f"array {name} dim {dim}: signature wants {expr}, deck has {arr.shape[dim]}")
            else:
                prev = sizes.setdefault(str(expr), arr.shape[dim])
                if prev != arr.shape[dim]:
                    missing.append(f"symbol {expr}: {prev} (earlier array) vs {arr.shape[dim]} ({name})")
        call[name] = arr

    for name, desc in sdfg.arglist().items():
        if isinstance(desc, dt.Array) or name in call:
            continue
        if name in sizes:
            call[name] = int(sizes[name])
            continue
        hit = _deck_key_of_symbol(name)
        if hit is not None:
            key, dim, kind = hit
            src = arrays.get(key)
            if kind == "SA" and src is not None and dim < src.ndim:
                call[name] = int(src.shape[dim])
                continue
            if kind == "SOA":
                lb = lbounds.get(key)
                if lb is not None and dim < len(lb):
                    call[name] = int(lb[dim])
                    continue
                if src is not None and dim < src.ndim:
                    call[name] = 1
                    continue
            missing.append(f"symbol {name}: no deck entry {key!r} dim {dim} ({kind})")
            continue
        flat = cpu.flat_name(name)
        bare = flat.removeprefix("global_data_")
        value = cpu.seed_value(flat, scalars)
        if value is None:
            value = cpu.seed_value(bare, scalars)
        if value is None:
            value = _GLOBAL_FLAG_LITERALS.get(bare)
        if value is None:
            missing.append(f"scalar {name} (deck key {flat}): no recorded/seeded value")
            continue
        call[name] = desc.dtype.as_numpy_dtype().type(value)

    if missing:
        raise RuntimeError("cannot bind the harness deck to the VTP signature:\n  " + "\n  ".join(missing))
    return call


def _demote_persistent_scalars(sdfg) -> None:
    """Stage 4 promotes Default-storage Scalars to Persistent; this DaCe's CUDA
    preprocess infers them to Register, and Register+Persistent fails validation.
    Demote just those -- the GPU/heap buffers keep their Persistent promotion."""
    import dace
    from dace import data
    demotable = (dace.StorageType.Default, dace.StorageType.Register)
    count = 0
    for g in sdfg.all_sdfgs_recursive():
        for desc in g.arrays.values():
            if (desc.transient and isinstance(desc, data.Scalar) and desc.lifetime == dace.AllocationLifetime.Persistent
                    and desc.storage in demotable):
                desc.lifetime = dace.AllocationLifetime.SDFG
                count += 1
    if count:
        print(f"vtp_manual: demoted {count} Default/Register scalar(s) from Persistent to SDFG lifetime", flush=True)


def _demote_host_write_gpu_maps(sdfg) -> None:
    """VTP's reduction pass leaves seed tasklets (constant writes to its host-side
    async scratch) inside 1-iteration GPU maps; on the user's coherent-memory DaCe
    branch that is fine, on this one it is an IllegalCopy. A GPU map that reads
    nothing and writes only host-storage data runs on the host instead."""
    import dace
    from dace.sdfg import nodes as nd
    count = 0
    for g in sdfg.all_sdfgs_recursive():
        for state in g.states():
            for node in state.nodes():
                if not (isinstance(node, nd.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Device):
                    continue
                reads = [e for e in state.in_edges(node) if e.data is not None and e.data.data]
                writes = [
                    e.data.data for e in state.out_edges(state.exit_node(node)) if e.data is not None and e.data.data
                ]
                if reads or not writes:
                    continue
                if all(g.arrays[w].storage != dace.StorageType.GPU_Global for w in writes):
                    node.map.schedule = dace.ScheduleType.Sequential
                    count += 1
    if count:
        print(f"vtp_manual: demoted {count} host-writing GPU map(s) to Sequential", flush=True)


def _absolutize_includes(sdfg) -> None:
    """VTP's global code quotes headers relative to its include/; the harness build
    runs from a dacecache dir, so point them at the checkout."""
    from dace.properties import CodeBlock
    inc = vtp_dir() / "include"
    for key, block in list(sdfg.global_code.items()):
        code = block.code
        if not isinstance(code, str):
            continue
        for header in _HEADERS:
            code = code.replace(f'"{header}"', f'"{inc / header}"')
        sdfg.global_code[key] = CodeBlock(code, block.language)


def _inline_reduction_kernels(sdfg) -> None:
    """VTP compiles src/reductions{_kernel.cu,.cpp} as extra sources; sdfg.compile()
    has no hook for that, so their text goes into the global code instead."""
    inc = vtp_dir() / "include"
    for name, location in (("reductions_kernel.cu", "cuda"), ("reductions.cpp", "frame")):
        src = (vtp_dir() / "src" / name).read_text()
        for header in _HEADERS:
            src = src.replace(f'"{header}"', f'"{inc / header}"')
        sdfg.append_global_code("\n" + src, location)
