"""Stage 5: neighborhood-list / block-array compression.

This stage carries **only** the velocity-tendencies compression work.
Every other GPU-side fix (offload, async reductions, persistent
transient lifetimes, body timer instrumentation) lives in stage 4 so
the stage-4 binary inherits them whether or not stage 5 runs.

Three compression call sites:

1. :func:`fold_array_access_to_expression` on the ``*_blk`` arrays
   known to be single-valued under the runtime precondition
   ``nblks_c == 1 && nblks_v == 1``. Every subscript access is
   rewritten to the literal 1 (Fortran 1-indexed block ID); arrays
   themselves stay in the function signature, just unused.
2. :func:`generate_compressed_variant` on the ``*_idx`` arrays with
   ``uint16`` under the bound ``nproma * nblks_* < 65536``. Adds a
   compressed clone of the body and dispatches at runtime.
3. :func:`generate_compressed_variant` on the multi-valued ``*_blk``
   arrays with ``uint8`` under the bound ``nblks_* < 256``.

Run:
    python -m utils.stages.stage5              # optimise + compile
    python -m utils.stages.stage5 --optimize   # codegen stage5 .sdfgz
    python -m utils.stages.stage5 --compile    # compile existing stage5
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("__DACE_NO_SYNC", "1")

import dace

from utils.passes.compress_indices import (
    SymbolicConstraint,
    fold_array_access_to_expression,
    generate_compressed_variant,
)
from utils.stages import common


STAGE_ID = 5


# Neighborhood-list indices demoted to ``uint16`` under the bound
# ``nproma * nblks_* < 65536`` at runtime.
_IDX_ARRAYS_GPU = (
    "gpu___CG_p_patch__CG_cells__m_edge_idx",
    "gpu___CG_p_patch__CG_cells__m_neighbor_idx",
    "gpu___CG_p_patch__CG_edges__m_cell_idx",
    "gpu___CG_p_patch__CG_edges__m_quad_idx",
    "gpu___CG_p_patch__CG_edges__m_vertex_idx",
    "gpu___CG_p_patch__CG_verts__m_cell_idx",
    "gpu___CG_p_patch__CG_verts__m_edge_idx",
)

# Block-index arrays whose every value equals the enclosing block-loop
# iterator ``jb`` (structural on single-block grids). Folded to the
# subscript expression -- arrays dropped entirely post-rewrite.
_BLK_ARRAYS_SINGLE_VAL_GPU = (
    "gpu___CG_p_patch__CG_cells__m_neighbor_blk",
    "gpu___CG_p_patch__CG_edges__m_cell_blk",
    "gpu___CG_p_patch__CG_edges__m_vertex_blk",
    "gpu___CG_p_patch__CG_verts__m_cell_blk",
)

# Block-index arrays that do carry distinct block numbers. Demoted to
# ``uint8`` under ``nblks_* < 256``.
_BLK_ARRAYS_MULTI_VAL_GPU = (
    "gpu___CG_p_patch__CG_cells__m_edge_blk",
    "gpu___CG_p_patch__CG_edges__m_quad_blk",
    "gpu___CG_p_patch__CG_verts__m_edge_blk",
)

# Runtime precondition: single-block patch. Every stage-5 compression
# uses this via a ``SymbolicConstraint`` wrapper so non-velocity
# callers can swap to ``ArrayMaxBelow`` / ``ArrayAllEqual`` etc.
_SINGLE_BLOCK = SymbolicConstraint(
    "__CG_p_patch__m_nblks_c == 1 && "
    "__CG_p_patch__m_nblks_v == 1")

# Independent runtime bounds for the dtype-compression dispatch.
_IDX_FITS_UINT16 = SymbolicConstraint(
    "__CG_global_data__m_nproma * __CG_p_patch__m_nblks_c < 65536 && "
    "__CG_global_data__m_nproma * __CG_p_patch__m_nblks_e < 65536 && "
    "__CG_global_data__m_nproma * __CG_p_patch__m_nblks_v < 65536")

_BLK_FITS_UINT8 = SymbolicConstraint(
    "__CG_p_patch__m_nblks_c < 256 && "
    "__CG_p_patch__m_nblks_e < 256 && "
    "__CG_p_patch__m_nblks_v < 256")

def optimization_action(sdfg: dace.SDFG) -> dace.SDFG:
    # Call 1: single-val blk elimination via fold.
    #
    # Under the ``_SINGLE_BLOCK`` precondition every listed ``*_blk``
    # array stores the constant 1 (Fortran 1-indexed block ID of the
    # sole block). Consumers subtract the matching ``SOA_d_2`` offset
    # to convert to 0-indexed, so folding the array read to the
    # literal 1 produces the same arithmetic result the consumer
    # would compute from an actual memory load.
    eliminated = fold_array_access_to_expression(
        sdfg,
        array_names=_BLK_ARRAYS_SINGLE_VAL_GPU,
        rewrite_rule=lambda _name, _idxs: 1,
        constraints=[_SINGLE_BLOCK],
    )
    if eliminated:
        print(f"Stage #{STAGE_ID}: fold_array_access_to_expression "
              f"eliminated {eliminated} blk array(s)")
    sdfg.validate()

    # Call 2: neighbor-index uint16 demotion.
    generate_compressed_variant(
        sdfg, array_names=_IDX_ARRAYS_GPU,
        target_dtype=dace.uint16,
        constraints=[_SINGLE_BLOCK, _IDX_FITS_UINT16],
        name_suffix='uint16')
    sdfg.validate()

    # Call 3: multi-val blk uint8 demotion.
    generate_compressed_variant(
        sdfg, array_names=_BLK_ARRAYS_MULTI_VAL_GPU,
        target_dtype=dace.uint8,
        constraints=[_SINGLE_BLOCK, _BLK_FITS_UINT8],
        name_suffix='uint8')
    sdfg.validate()

    return sdfg


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--optimize", action=argparse.BooleanOptionalAction, default=False)
    argp.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
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
        extra_sources = ["src/reductions.cpp", "src/reductions_kernel.cu"]
        extra_include_dirs = ["include"]
        common.compile_action(STAGE_ID, sdfgs, gpu=True, release=args.release,
                              extra_sources=extra_sources,
                              extra_include_dirs=extra_include_dirs)


if __name__ == "__main__":
    main()
