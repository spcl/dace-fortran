#!/usr/bin/env bash
# regenerate_baselines.sh -- F90 -> stage 5 from inside VelocityTendenciesPipeline.
#
# This script is the single entry point for rebuilding baselines from
# scratch. It does NOT pull anything from icon-artifacts at runtime -- the
# Fortran AST snapshot we use lives at::
#
#     baseline_inputs/velocity_modified.f90
#
# Pipeline (5 phases):
#
#   0. f2dace (stage 2): velocity_modified.f90 -> baseline/velocity_no_nproma.sdfgz
#      (driver: tools/sdfg_from_velocity_f90.py)
#   1. generate_baselines.py: AoS -> SoA + symbol resolution + 4 specialised
#      variants -> baseline/velocity_no_nproma_if_prop_lvn_only_{0,1}_istep_{1,2}.sdfgz
#   2. utils.stages.stage1 --optimize -> codegen/stage1/<variant>.sdfgz
#   3. utils.stages.stage2 --optimize -> codegen/stage2/<variant>.sdfgz
#   4. utils.stages.stage3 --optimize -> codegen/stage3/<variant>.sdfgz
#   5. utils.stages.stage4 --optimize -> codegen/stage4/<variant>.sdfgz
#   6. utils.stages.stage5 --optimize -> codegen/stage5/<variant>.sdfgz
#
# Phase 0 needs DaCe checked out on a branch that ships the Fortran
# frontend (typically ``f2dace/staging``). The remaining phases run on
# the day-to-day branch (``yakup/dev``). The DACE_BRANCH env switching is
# handled by the caller -- this script does NOT mutate your DaCe checkout.
#
# Env overrides:
#   PIPELINE_DIR        path to this checkout
#                       (default: directory containing this script's parent)
#   PYTHON              python interpreter (default: python)
#   SKIP_F2DACE         set to 1 to skip phase 0 (use existing
#                       baseline/velocity_no_nproma.sdfgz)
#   ONLY_PHASE          run a single phase: 0,1,2,3,4,5,6 (default: all)
#   STAGE_FLAGS         extra args forwarded to each utils.stages.stage*
#                       invocation (default: --optimize)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="${PIPELINE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-python}"
SKIP_F2DACE="${SKIP_F2DACE:-0}"
ONLY_PHASE="${ONLY_PHASE:-all}"
STAGE_FLAGS="${STAGE_FLAGS:---optimize}"

cd "${PIPELINE_DIR}"

run_phase() {
  local phase="$1"
  if [[ "${ONLY_PHASE}" != "all" && "${ONLY_PHASE}" != "${phase}" ]]; then
    return 1
  fi
  return 0
}

if run_phase 0 && [[ "${SKIP_F2DACE}" != "1" ]]; then
  echo "[regenerate_baselines] phase 0: f2dace velocity_modified.f90 -> baseline/velocity_no_nproma.sdfgz"
  "${PYTHON}" tools/sdfg_from_velocity_f90.py \
      --input  baseline_inputs/velocity_modified.f90 \
      --output baseline/velocity_no_nproma.sdfgz
fi

if run_phase 1; then
  echo "[regenerate_baselines] phase 1: generate_baselines.py (AoS->SoA + 4 variants)"
  "${PYTHON}" generate_baselines.py \
      --input      baseline/velocity_no_nproma.sdfgz \
      --output-dir baseline
fi

for STAGE in 1 2 3 4 5; do
  if run_phase "$((STAGE + 1))"; then
    echo "[regenerate_baselines] phase $((STAGE + 1)): utils.stages.stage${STAGE} ${STAGE_FLAGS}"
    "${PYTHON}" -m utils.stages.stage${STAGE} ${STAGE_FLAGS}
  fi
done

echo "[regenerate_baselines] done. Latest SDFGs under codegen/stage5/"
