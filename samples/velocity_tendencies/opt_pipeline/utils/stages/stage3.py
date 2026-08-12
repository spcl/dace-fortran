"""Stage 3: pre-GPU structural prep.

Four small steps, in order:

1. ``set_transient_sdfg_lifetime`` -- every transient array gets
   ``AllocationLifetime.SDFG`` so its storage is bound to the SDFG
   invocation (not a narrower scope). Persistent lifetime is deferred
   until the replacement GPU codegen provides init/exit bookends.
2. ``int64_to_int32`` -- demote 64-bit integer symbols, array dtypes
   and nested-SDFG connector types to 32-bit. Fixes the
   ``long int -> int`` truncation warnings emitted by the stage-2 code.
3. ``lift_transients`` -- multi-element transient arrays that live in
   nested SDFGs are promoted to the outermost SDFG's array table and
   piped back in as input/output connectors (routed through any
   enclosing MapEntry / MapExit). Keeps storage at a scope that GPU
   allocation can hook into.
4. ``verify_no_nested_transients`` -- explicit post-check that every
   multi-element transient now lives at the root. Scalars are fine.

No schedule assignment here (that's stage 4), no Persistent lifetime,
no symbol-propagation / reassign-vars.

Run:
    python -m utils.stages.stage3              # optimise + compile
    python -m utils.stages.stage3 --optimize   # codegen stage3 .sdfgz only
    python -m utils.stages.stage3 --compile    # compile stage3 .sdfgz only
"""

import argparse
from pathlib import Path

import dace

from dace.transformation.passes.lift_transients import lift_transients
from dace.transformation.passes.loop_invariant_code_motion import (
    LoopInvariantCodeMotion)
from dace.transformation.passes.loop_local_memory_reduction import (
    LoopLocalMemoryReduction)
from dace.transformation.passes.verify_no_nested_transients import (
    verify_no_nested_transients)
from utils.passes.int64_to_int32 import int64_to_int32
from utils.passes.set_transient_sdfg_lifetime import set_transient_sdfg_lifetime
from utils.stages import common


STAGE_ID = 3


# Block-loop bounds in velocity_tendencies are computed by interstate
# edges (``i_startblk = p_patch%<grid>%start_block(rl)``), so lifting a
# transient through one of those maps would leave the lifted shape
# dependent on a symbol unknown at SDFG entry. These pairs identify
# each block loop's (begin, end) symbols and tell ``lift_transients``
# to size the prepended dim by the grid's ``nblks_*`` instead. Source:
# velocity.f90 -- the cell block loop (jb_var_84 over
# p_patch%cells%start_block(4)..end_block(-5)) hosts all 4 lifted
# arrays; the edge block loops are included so any future transient
# lifted through them is handled without another edit.
_MAP_RANGE_SUGGESTIONS = {
    ("i_startblk_var_146", "i_endblk_var_147"): "__CG_p_patch__m_nblks_c",
    ("i_startblk_var_140_0", "i_endblk_var_141_0"): "__CG_p_patch__m_nblks_e",
    ("i_startblk_var_118_0", "i_endblk_var_119_0"): "__CG_p_patch__m_nblks_e",
}


def optimization_action(sdfg: dace.SDFG) -> dace.SDFG:
    # 1. Transient lifetime.
    flipped = set_transient_sdfg_lifetime(sdfg)
    if flipped:
        print(f"Stage #{STAGE_ID}: set_transient_sdfg_lifetime updated {flipped} descriptor(s)")
    sdfg.validate()

    # 2. Integer width demotion.
    demoted = int64_to_int32(sdfg)
    if demoted:
        print(f"Stage #{STAGE_ID}: int64_to_int32 demoted {demoted} type(s)")
    sdfg.validate()

    # 3. Lift cross-scope / multi-element transients. The pass checks
    # its own post-conditions at the end and raises on violation.
    lifted = lift_transients(sdfg, map_range_suggestions=_MAP_RANGE_SUGGESTIONS)
    if lifted:
        print(f"Stage #{STAGE_ID}: lift_transients promoted {lifted} array(s) to top level")
    sdfg.validate()

    # 4. Independent invariant check: every multi-element transient
    # must now live at the root SDFG. Scalars (shape == (1,)) are left
    # alone. Redundant with lift_transients's internal check, but
    # explicit here so the pipeline stays guarded even if a future
    # step re-introduces a nested transient (e.g. a GPU pass adding a
    # scratch array at the wrong level).
    verify_no_nested_transients(sdfg)

    # 5. Loop-invariant code motion. Hoists tasklets whose inputs don't
    # change across iterations out of enclosing LoopRegions (into the
    # preheader) and out of Map scopes (into the enclosing state).
    # Runs before GPU offloading so the hoisted tasklets are scheduled
    # on the host and the per-thread kernel work shrinks.
    hoisted = LoopInvariantCodeMotion().apply_pass(sdfg, {})
    if hoisted:
        print(f"Stage #{STAGE_ID}: LICM hoisted {hoisted} tasklet/region(s)")
    sdfg.validate()

    # 6. Loop-local memory reduction. Shrinks thread-local arrays whose
    # access pattern is a sliding window (``a[i + k]``) to a circular
    # buffer of size ``max(k) - min(k) + 1``. Rounds up to the next
    # power of two so modulo becomes a bitmask. Per-iteration memory
    # drops from O(N) to O(window size); a win for any stencil-style
    # transient that survived earlier simplification.
    shrunk = LoopLocalMemoryReduction().apply_pass(sdfg, {})
    if shrunk:
        print(f"Stage #{STAGE_ID}: LLMR reduced {len(shrunk)} array(s)")
    sdfg.validate()

    return sdfg


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--optimize", action=argparse.BooleanOptionalAction, default=False)
    argp.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    argp.add_argument("--debug", dest="release", action="store_false", default=True,
                      help="build with -O0 + DACE_VELOCITY_DEBUG (default: release)")
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
        common.compile_action(STAGE_ID, sdfgs, release=args.release)


if __name__ == "__main__":
    main()
