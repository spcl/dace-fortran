"""Stage 4: GPU offload.

Applies ``OffloadVelocityToGPU`` to each stage-3 SDFG, which drives
DaCe's ``GPUTransformSDFG`` over the host-level SDFG tree:
- block maps (structurally identified) stay ``Sequential`` on the host
  and launch the kernels nested inside them,
- every other top-level map at a host level becomes ``GPU_Device``,
  inner maps become ``Sequential``,
- kernel-touched non-transients get ``gpu_<name>`` clones with
  copy-in/copy-out states, transients move to ``GPU_Global``,
- host-only arrays stay CPU-side.

**Signature freeze.** The outer (non-transient) argument list of the
SDFG is snapshotted before ``OffloadVelocityToGPU`` runs and asserted
unchanged after. Any pass step that would flip an arg's type/shape
(e.g. a length-1 Array → Scalar rewrite) is rejected here -- stage 4
must leave the ABI that stage 3 produced intact so existing callers
link without a rebuild.

No ``offload_cpu``, no ``pre_gpu_fix``, no ``move_lib_schedules`` --
each of those was redundant given the structural contract stages 1-3
already produce; ``ToGPU`` + ``GPUKernelLaunchRestructure`` cover the
schedule + storage transitions on their own.

Run:
    python -m utils.stages.stage4              # optimise + compile (GPU)
    python -m utils.stages.stage4 --optimize   # codegen stage4 .sdfgz only
    python -m utils.stages.stage4 --compile    # compile existing stage4 sdfgz
"""

import argparse
import os
from pathlib import Path

# DaCe's CUDA codegen emits ``cudaDeviceSynchronize`` /
# ``cudaStreamSynchronize`` / ``cudaEventSynchronize`` calls whose
# placement is currently unreliable for velocity (the new GPU codegen
# rewrite addresses this; until then we run without emission). The
# ``__DACE_NO_SYNC`` env var in ``dace.codegen.common.no_sync_emission``
# gates every sync emission site. Default to enabled for this stage;
# callers can set ``__DACE_NO_SYNC=0`` explicitly to re-enable sync
# emission if needed.
os.environ.setdefault("__DACE_NO_SYNC", "1")

import dace

from utils.passes.offload_velocity_to_gpu import OffloadVelocityToGPU
from utils.stages import common


STAGE_ID = 4


# No keep-CPU list. ``OffloadVelocityToGPU`` derives the host-only set
# structurally (arrays no kernel scope touches -- patch index/block
# descriptors, the max_vcfl_dyn round-trip output, host-only fields)
# instead of matching names, which drifted every time the frontend
# renamed a container.


def _signature_fingerprint(sdfg: dace.SDFG):
    """A hashable summary of the outer SDFG's argument list -- every
    non-transient entry in ``sdfg.arrays`` plus each free symbol.
    Compared across the pass to detect ABI drift."""
    args = []
    for name, desc in sdfg.arrays.items():
        if desc.transient:
            continue
        args.append((name,
                     type(desc).__name__,
                     str(desc.dtype),
                     tuple(str(d) for d in desc.shape)))
    symbols = [(s, str(t)) for s, t in sdfg.symbols.items()]
    args.sort()
    symbols.sort()
    return tuple(args), tuple(symbols)


def optimization_action(sdfg: dace.SDFG) -> dace.SDFG:
    # Snapshot the outer (caller-visible) signature before any pass
    # step runs; we assert it unchanged after OffloadVelocityToGPU so
    # downstream callers (e.g. main.cpp) keep linking without a
    # rebuild. Any type/shape drift surfaces here loudly.
    before = _signature_fingerprint(sdfg)

    OffloadVelocityToGPU(sdfg)

    after = _signature_fingerprint(sdfg)
    if before != after:
        diff = _describe_signature_diff(before, after)
        raise RuntimeError(
            f"Stage #{STAGE_ID}: OffloadVelocityToGPU changed the outer "
            f"SDFG signature (ABI break). Diff:\n{diff}")

    # Watchtower: if the segmented-reduction workaround was ever
    # secretly needed, an ``out_val_0`` reference would leak into the
    # post-GPU SDFG. Stage 1's ``remove_clip_count`` + no `reduce_scan`
    # libnode means this should always be empty. Loud if not.
    leaks = _find_out_val_0(sdfg)
    if leaks:
        raise RuntimeError(
            f"Stage #{STAGE_ID}: {len(leaks)} unexpected 'out_val_0' "
            f"reference(s) survived OffloadVelocityToGPU: "
            f"{sorted(leaks)[:5]} ... -- revisit the dropped "
            f"pre_gpu_fixes.step_1 (segmented reduction pipelining).")

    # Promote every transient with ``AllocationLifetime.SDFG`` to
    # ``Persistent`` so its allocation lives across timesteps. Velocity
    # is invoked once per timestep from a long-lived host loop;
    # ``SDFG`` lifetime would re-alloc per call, ``Persistent`` keeps
    # the buffers warm and skips the malloc traffic.
    promoted = _promote_transients_persistent(sdfg)
    if promoted:
        print(f"Stage #{STAGE_ID}: promoted {promoted} transient(s) to "
              f"AllocationLifetime.Persistent")

    # Host-side body timer. Start immediately after the copy-in sync,
    # stop just before copy-out. Lives at stage 4 so the
    # OffloadVelocityToGPU output already carries the instrumentation
    # downstream stages inherit.
    _insert_body_timer(sdfg)

    sdfg.validate()
    return sdfg


def _promote_transients_persistent(sdfg: dace.SDFG) -> int:
    """Flip every transient array in ``sdfg`` (root + nested) on
    ``AllocationLifetime.SDFG`` over to ``Persistent``. Skips
    ``StorageType.Register`` -- those are stack-local in the
    generated function body and can't outlive a single call -- and
    ``StorageType.Default``, which codegen's own inference turns into
    ``Register`` for exactly these scalars; promoting one produces the
    persistent+Register pair ``validate_sdfg`` rejects.
    ``OffloadVelocityToGPU`` already stamps every device buffer
    ``GPU_Global`` explicitly, so nothing device-side is left at
    ``Default`` for this to miss. Returns the count flipped."""
    from dace.sdfg import nodes as _nodes
    sdfgs = [sdfg]
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, _nodes.NestedSDFG):
            sdfgs.append(n.sdfg)
    count = 0
    for g in sdfgs:
        for desc in g.arrays.values():
            if not getattr(desc, 'transient', False):
                continue
            if desc.lifetime != dace.AllocationLifetime.SDFG:
                continue
            if desc.storage in (dace.StorageType.Register, dace.StorageType.Default):
                continue
            desc.lifetime = dace.AllocationLifetime.Persistent
            count += 1
    return count


def _insert_body_timer(sdfg: dace.SDFG) -> None:
    """Wrap the body with two host-side timer tasklets. Start state is
    inserted as a successor of ``_sync_after_copy_in``; stop state as
    a predecessor of ``_gpu_to_cpu_copy_out``. Reduction-induced sync
    inside the body suffices for accurate timing because the
    velocity body always contains a scalar-return reduction whose
    deferred sync flushes stream 0 before timer stop.
    """
    from dace.properties import CodeBlock as _CodeBlock
    existing = sdfg.global_code.get('frame')
    prior = existing.code if existing is not None \
        and isinstance(existing.code, str) else ''
    if '_stage4_t0' not in prior:
        sdfg.global_code['frame'] = _CodeBlock(
            prior
            + '#include <chrono>\n'
              '#include <cstdio>\n'
              'static std::chrono::steady_clock::time_point _stage4_t0;\n',
            dace.dtypes.Language.CPP)

    sync_in = _find_state_by_label(sdfg, '_sync_after_copy_in') \
        or sdfg.start_state
    copy_out = _find_state_by_label(sdfg, '_gpu_to_cpu_copy_out')
    if copy_out is None:
        ends = [s for s in sdfg.nodes() if sdfg.out_degree(s) == 0]
        copy_out = ends[-1] if ends else None

    start_state = sdfg.add_state_after(
        sync_in, label='_stage4_timer_start')
    start_state.add_tasklet(
        name='_stage4_timer_start',
        inputs={}, outputs={},
        code='_stage4_t0 = std::chrono::steady_clock::now();',
        language=dace.dtypes.Language.CPP,
        side_effects=True,
    )
    if copy_out is not None:
        stop_state = sdfg.add_state_before(
            copy_out, label='_stage4_timer_stop')
        stop_state.add_tasklet(
            name='_stage4_timer_stop',
            inputs={}, outputs={},
            code=(
                'auto _stage4_t1 = std::chrono::steady_clock::now();\n'
                'std::chrono::duration<double, std::milli> _stage4_dt = '
                '_stage4_t1 - _stage4_t0;\n'
                'std::printf("velocity body: %.6f ms\\n", _stage4_dt.count());'
            ),
            language=dace.dtypes.Language.CPP,
            side_effects=True,
        )


def _find_state_by_label(sdfg: dace.SDFG, label: str):
    for block in sdfg.nodes():
        if isinstance(block, dace.SDFGState) and block.label == label:
            return block
    return None


def _describe_signature_diff(before, after):
    before_args, before_syms = before
    after_args, after_syms = after
    removed = set(before_args) - set(after_args)
    added = set(after_args) - set(before_args)
    lines = []
    for r in sorted(removed):
        lines.append(f"  - REMOVED: {r}")
    for a in sorted(added):
        lines.append(f"  + ADDED:   {a}")
    if before_syms != after_syms:
        lines.append(f"  ! symbols: before={before_syms} after={after_syms}")
    return "\n".join(lines) or "  (no diff -- tuples unequal but elements match?)"


def _find_out_val_0(sdfg: dace.SDFG):
    hits = set()
    for name in sdfg.arrays:
        if 'out_val_0' in name:
            hits.add(('array', name))
    for n, _ in sdfg.all_nodes_recursive():
        for attr in ('data', 'label'):
            v = getattr(n, attr, None)
            if isinstance(v, str) and 'out_val_0' in v:
                hits.add(('node', v))
    for e, _ in sdfg.all_edges_recursive():
        if e.data is not None and getattr(e.data, 'data', None):
            if 'out_val_0' in e.data.data:
                hits.add(('memlet', e.data.data))
    return hits


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--optimize", action=argparse.BooleanOptionalAction, default=False)
    argp.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    # Release is the default (-O3 + ``--use_fast_math``). Opt into
    # debug with ``--debug``; debug also flips fp to IEEE-compliant
    # (``--fmad=false``, ``--prec-div=true``, ``--prec-sqrt=true``,
    # ``--ftz=false``), which matches the CPU reference bit-for-bit.
    argp.add_argument("--debug", dest="release", action="store_false", default=True,
                      help="build with -O0 + IEEE fp (default: release + fast math)")
    args = argp.parse_args()
    if not args.optimize and not args.compile:
        args.optimize, args.compile = True, True

    names = common.sdfg_names()

    if args.optimize:
        for name in names:
            infile = common.stage_input(name, STAGE_ID)
            outfile = common.stage_output(name, STAGE_ID)
            print(f"Stage #{STAGE_ID}: Optimising {name} from {infile}")

            sdfg = dace.SDFG.from_file(infile)
            sdfg.name = name
            sdfg.validate()

            sdfg = optimization_action(sdfg)

            Path(outfile).parent.mkdir(parents=True, exist_ok=True)
            sdfg.save(outfile, compress=True)
            print(f"Stage #{STAGE_ID}: Saved as {outfile}")

    if args.compile:
        sdfgs = {
            name: dace.SDFG.from_file(common.stage_output(name, STAGE_ID))
            for name in names
        }
        # Velocity-owned reduction helpers.
        engine_srcs, engine_defs = common.engine_sources()
        extra_sources = ["src/reductions.cpp", "src/reductions_kernel.cu"] + engine_srcs
        extra_include_dirs = ["include"]
        common.compile_action(STAGE_ID, sdfgs, gpu=True, release=args.release,
                              extra_sources=extra_sources,
                              extra_include_dirs=extra_include_dirs,
                              extra_defines=engine_defs)


if __name__ == "__main__":
    main()
