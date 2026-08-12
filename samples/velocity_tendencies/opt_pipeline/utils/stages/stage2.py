"""Stage 2: post maximally-parallel cleanup.

Runs three cleanup steps on the stage-1 output:

1. ``LiftTrivialIf`` -- the utility transformation from DaCe PR #2138,
   lifts ``ConditionalBlock``s whose guard folds to a constant out of the
   control flow. Trims the dead conditionals exposed once LoopToMap /
   PerfLoopNesting / MoveIfIntoMap have reshuffled the structure.
2. ``OffsetLoopsAndMaps`` (PR #2156) -- shifts every loop and map by +1
   regardless of its starting expression, converts ``<=`` conditions to
   ``<``, and normalises when possible. Aligns the kernel's index space
   with the upstream fused form.
3. ``simplify_expressions`` -- walks every memlet subset/volume and every
   interstate edge assignment/condition in the SDFG tree, runs
   ``sympy.simplify`` on each expression, and re-renders via DaCe's
   ``symstr`` printer.

Run:
    python -m utils.stages.stage2              # optimise + compile
    python -m utils.stages.stage2 --optimize   # codegen stage2 .sdfgz only
    python -m utils.stages.stage2 --compile    # compile stage2 .sdfgz only
"""

import argparse
from pathlib import Path

import dace
from dace.transformation.passes.lift_trivial_if import LiftTrivialIf
from dace.transformation.passes.offset_loop_and_maps import OffsetLoopsAndMaps
from dace.transformation.passes.simplify_expressions import simplify_expressions

from utils.stages import common


STAGE_ID = 2


def optimization_action(sdfg: dace.SDFG) -> dace.SDFG:
    # 1. Utility transformation (PR #2138): lift ``if``-blocks whose
    #    condition is trivially constant-folded out of the control flow.
    lifted = LiftTrivialIf().apply_pass(sdfg, {}) or 0
    if lifted:
        print(f"Stage #{STAGE_ID}: LiftTrivialIf lifted {lifted} trivial if(s)")
    sdfg.validate()

    # 2. Offset every loop and map by -1, shifting Fortran-style
    #    ``1..N+1`` iteration spaces to C-style ``0..N``. Converts ``<=``
    #    conditions to ``<`` and normalises where possible.
    OffsetLoopsAndMaps(offset_expr="-1",
                       begin_expr=None,
                       convert_leq_to_lt=True,
                       normalize_loops=True).apply_pass(sdfg, {})
    # In case some arrays were already 0 begin
    OffsetLoopsAndMaps(offset_expr="1",
                       begin_expr="-1",
                       convert_leq_to_lt=False,
                       normalize_loops=False).apply_pass(sdfg, {})
    sdfg.validate()

    # 3. Re-render every symbolic expression in the SDFG through
    #    ``sympy.simplify`` + ``dace.symbolic.symstr``.
    rewrote = simplify_expressions(sdfg)
    if rewrote:
        print(f"Stage #{STAGE_ID}: simplify_expressions rewrote {rewrote} expression(s)")
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
