#!/bin/bash
# End-to-end ICON-with-DaCe integration test, with a side-by-side
# diff against an unpatched (stock-Fortran) ICON.
#
#   1. Builds STOCK ICON  (pristine mo_velocity_advection.f90, no DaCe link)
#      into ``${LANE_BUILD}`` as ``bin/icon.stock`` -- FIRST, so its
#      config-matched .mod + -D defines are available for the lib build.
#   2. Builds the DaCe velocity library from ICON's REAL
#      ``mo_velocity_advection`` source, lowered against STOCK's config.
#   3. Patches mo_velocity_advection.f90 to dispatch into the DaCe wrapper
#      and relinks the SAME tree as ``bin/icon.dace``.
#   4. Caches the R02B05 grid.
#   5. Generates the short Held-Suarez R02B05 experiment.
#   6. Generates the runscript for the lane build dir.
#   7. Runs both ICON binaries on the SAME exp (NRANKS ranks each), parking
#      each run's output as ``experiments/${EXP}.{stock,dace}``.
#   8. Calls ``compare_icon_runs.py`` to diff every overlapping
#      ``*_{ml,hl,pl}_*.nc`` variable-by-variable.
#
# The DaCe velocity SDFG is now lowered from ICON's REAL
# ``mo_velocity_advection.f90`` (real ``t_patch`` / ``t_nh_prog`` layout), not
# the stub-typed ``velocity_full.f90`` that SIGSEGV'd inside the first
# ``velocity_tendencies`` call -- so the DaCe run should progress past t=0 and
# the comparison is meaningful beyond the initial dump.
#
# RUN THIS FIRST:  STOCK_ONLY=1 bash run_icon_e2e.sh
# It builds + runs ONLY stock ICON at NRANKS ranks and asserts the run is real
# (right rank count) and non-vacuous (a dump after t=0).  That proves the 2-node
# run works INDEPENDENTLY of any DaCe integration -- so if the integrated run
# later fails, it cannot be confused with a broken grid / experiment / rank
# setup.  Then re-run without STOCK_ONLY for the full stock-vs-DaCe differential.
#
# Tunables:
#   ICON_SRC, DACE_FORTRAN, DACE_LIBS, GRID_DIR, LANE_BUILD, GPU,
#   EXP, NRANKS, PY, RTOL, STOCK_ONLY, CAP
set -euo pipefail

# Every work root is derived from this script's own location: the repo is two
# levels above tests/icon/full, and everything ICON needs (clone, grids, DaCe
# libs) lives under the gitignored samples/_work root inside it.  No default is
# an absolute path; env overrides still win.
_SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)
DACE_FORTRAN=${DACE_FORTRAN:-$(cd "${_SELF_DIR}/../../.."; pwd)}
WORK_ROOT=${WORK_ROOT:-${DACE_FORTRAN}/samples/_work}
ICON_SRC=${ICON_SRC:-${WORK_ROOT}/icon-model}
DACE_LIBS=${DACE_LIBS:-${WORK_ROOT}/dace-icon-libs}
GRID_DIR=${GRID_DIR:-${WORK_ROOT}/icon-grids}
PY=${PY:-$(command -v python3)}
# DAINT=1 switches the TOOLCHAIN defaults to the Alps/daint aarch64 lane
# (auto-detected on aarch64; force DAINT=0 to keep the x86 ones).  Paths are
# no longer part of this split -- only the lane script and the build budget.
DAINT=${DAINT:-$([[ $(uname -m) == aarch64 ]] && echo 1 || echo 0)}
if [[ "${DAINT}" == 1 ]]; then
  CONFIGURE_SH=${CONFIGURE_SH:-configure_icon_nvhpc_daint.sh}
  CAP=${CAP:-none}
  MAKE_J=${MAKE_J:-16}
fi
CONFIGURE_SH=${CONFIGURE_SH:-configure_icon_dace_cpu.sh}
MAKE_J=${MAKE_J:-1}
GPU=${GPU:-0}
# ONE build tree, named exactly as scripts/icon_daint_common.sh derives it from
# (compiler lane, GPU flag): cpu_nvhpc | gpu_nvhpc, never a third name.  Stock
# and DaCe-linked ICON live side by side inside it as bin/icon.stock and
# bin/icon.dace -- they differ only by the icon.mk link rule and the velocity
# patch, so a separate tree buys nothing and would be a third directory.  Each
# run's output is parked under its own experiments/ suffix.
_LANE=nvhpc
[[ ${GPU} == 1 ]] && _PFX=gpu || _PFX=cpu
LANE_BUILD=${LANE_BUILD:-${ICON_SRC}/build/${_PFX}_${_LANE}}
EXP=${EXP:-atm_heldsuarez_dace_r02b05}
NRANKS=${NRANKS:-2}
# STOCK_ONLY=1: build + run ONLY stock ICON at NRANKS ranks and verify the run is
# real (right rank count) and non-vacuous (a dump after t=0).  No DaCe lib, no
# DaCe ICON, no comparison.  Run this FIRST: it proves the 2-node run itself
# works independently, so an integration failure later can't be confused with a
# broken experiment / grid / rank setup.
STOCK_ONLY=${STOCK_ONLY:-0}
RTOL=${RTOL:-1e-12}

GRID_ID=0014
GRID_NAME=icon_grid_${GRID_ID}_R02B05_G
GRID_URL=http://icon-downloads.mpimet.mpg.de/grids/public/edzw/${GRID_NAME}.nc

VELOCITY_F90=${ICON_SRC}/src/atm_dyn_iconam/mo_velocity_advection.f90
RUN=${ICON_SRC}/run
COMPARE=${DACE_FORTRAN}/tests/icon/full/compare_icon_runs.py

step() { printf '\n=== %s ===\n' "$1"; }


# Apply the DaCe forwarding patch to mo_velocity_advection.f90.
apply_dace_patch() {
  cp "${VELOCITY_F90}.bak" "${VELOCITY_F90}"
  "${PY}" "${DACE_FORTRAN}/tests/icon/full/apply_velocity_dace_patch.py" "${VELOCITY_F90}"
}


# Serial, memory-capped runner.  This box has 12GB RAM; a parallel ``make`` or an
# uncapped model run thrashes swap.  Run every heavy step in a transient systemd
# scope so it is OOM-killed at ${CAP} instead of swap-crawling.  No fallback: a
# silent uncapped run is exactly what the cap exists to prevent, so if
# systemd-run is unavailable this fails loudly.
CAP=${CAP:-8G}
capped() {
  # CAP=none (daint default): the exclusive GH200 node has RAM to spare and
  # compute nodes have no user systemd session -- run uncapped.
  if [[ "${CAP}" == none ]]; then "$@"; return $?; fi
  systemd-run --user --scope -p MemoryMax="${CAP}" -p MemorySwapMax=0 --quiet "$@"
}

# Configure + build the lane tree, keeping the result as bin/icon.$2.
# $3=1 wipes the tree first (the stock pass); the DaCe pass reconfigures in
# place so make recompiles and relinks only what the velocity patch touched.
# make -j1 + 8GB cap (single build at a time -- see the box's RAM budget).
build_icon() {
  local dace_libs_dir=$1 label=$2 fresh=${3:-0}
  echo "  -> ${LANE_BUILD} (bin/icon.${label}, DACE_LIBS_DIR='${dace_libs_dir}')"
  [[ "${fresh}" == 1 ]] && rm -rf "${LANE_BUILD}"
  mkdir -p "${LANE_BUILD}"
  # The daint lane scripts own their build dir via BUILD_DIR/ICON_SRC; the x86
  # script still infers it from the cwd, so pass both.
  ( cd "${LANE_BUILD}" && GPU="${GPU}" DACE_LIBS_DIR="${dace_libs_dir}" \
      BUILD_DIR="${LANE_BUILD}" ICON_SRC="${ICON_SRC}" \
      bash "${DACE_FORTRAN}/scripts/${CONFIGURE_SH}" )
  # The sub-configures and cmake externals run from make, a shell that never
  # sourced the lane script; build-env.sh replays the compiler env it used.
  # shellcheck disable=SC1091
  [[ -f "${LANE_BUILD}/build-env.sh" ]] && source "${LANE_BUILD}/build-env.sh"
  capped make -C "${LANE_BUILD}" -j"${MAKE_J}" >/dev/null
  cp "${LANE_BUILD}/bin/icon" "${LANE_BUILD}/bin/icon.${label}"
}


# Make the runscript helpers in source/run/ accessible from the build's
# own run/ (so the runscript can ``${basedir}/run/add_run_routines`` etc).
stage_runscript_helpers() {
  local build_dir=$1
  ln -sfn "${build_dir}/run/set-up.info" "${RUN}/set-up.info"
  for entry in "${RUN}"/*; do
    base=$(basename "${entry}")
    [[ -e "${build_dir}/run/${base}" ]] || ln -sn "${entry}" "${build_dir}/run/${base}"
  done
}


# Swap bin/icon to the labelled binary, run it, then park the output as
# experiments/${EXP}.${label} so the next label's run cannot clobber it --
# both binaries share one tree, so the outputs must not share one exp dir.
run_icon() {
  local label=$1
  local exp_dir="${LANE_BUILD}/experiments/${EXP}"
  cp "${LANE_BUILD}/bin/icon.${label}" "${LANE_BUILD}/bin/icon"
  rm -rf "${exp_dir}" "${exp_dir}.${label}"
  ln -sfn "${RUN}/exp.${EXP}.run" "${LANE_BUILD}/run/exp.${EXP}.run"
  set +e
  # The generated ``exp.<EXP>.run`` RECOMPUTES the rank count as
  # ``: ${no_of_nodes:=1} ${mpi_procs_pernode:=4}; ((mpi_total_procs = no_of_nodes
  # * mpi_procs_pernode))`` -- so exporting ``mpi_total_procs`` is ignored (default
  # 4 ranks).  The ``:=`` honours these two if already set, so export them (via
  # ``env`` so they survive the ``capped`` systemd scope): 1 node x NRANKS =
  # NRANKS ranks.  Both compute (num_io_procs=0), a genuine NRANKS-rank run.
  capped env no_of_nodes=1 mpi_procs_pernode="${NRANKS}" bash -c \
    "cd '${LANE_BUILD}/run' && bash 'exp.${EXP}.run'" > "${LANE_BUILD}/icon_run.${label}.log" 2>&1
  local rc=$?
  set -e
  mv "${exp_dir}" "${exp_dir}.${label}" 2>/dev/null || true
  echo "  ${label} run rc=${rc}, exp_dir=${exp_dir}.${label}"
}


step "0) Apply ICON source guards (idempotent)"
# pinit_seed segfault guard: with inwp_surface=0 (no TERRA land, e.g. the EXCLAIM
# aquaplanet) prog_lnd%t_so_t is unallocated, so the unconditional soil-temp
# perturbation loop reads SIZE(...,4) and segfaults during construct. The guard
# skips it when pinit_seed==0 (which step 5 forces in the namelist). Applied to
# the shared ICON tree BEFORE both builds so STOCK and DACE carry it identically
# -- the comparison stays valid. Idempotent: skip if already applied.
PINIT_PATCH=${PINIT_PATCH:-${DACE_FORTRAN}/scripts/icon_patches/icon_pinit_seed_guard.patch}
( cd "${ICON_SRC}"
  if git apply --reverse --check "${PINIT_PATCH}" 2>/dev/null; then
    echo "(pinit_seed guard already applied)"
  else
    git apply "${PINIT_PATCH}" && echo "applied pinit_seed guard"
  fi )


step "1) Build STOCK ICON (no patch, no DaCe link)"
# STOCK is built FIRST: the DaCe velocity lib is now lowered from ICON's REAL
# mo_velocity_advection source (real t_patch / t_nh_prog layout -- the stub-typed
# velocity_full.f90 SIGSEGVs in a real run), and that lowering needs the TARGET
# ICON config's -D defines + compiled .mod, which only exist after a build.
# STOCK and DACE differ ONLY by the icon.mk link patch + the velocity source
# patch, so the tree's mod/ + defines are valid for the lib the DACE relink
# picks up.
# Preserve the pristine source the first time through, and build from it.
[[ -f "${VELOCITY_F90}.bak" ]] || cp "${VELOCITY_F90}" "${VELOCITY_F90}.bak"
cp "${VELOCITY_F90}.bak" "${VELOCITY_F90}"
# STOCK_REUSE=1: skip the stock rebuild when a binary already exists.  Safe
# because stock is always built from the pristine ``.bak`` (restored just
# above), so an existing ``bin/icon`` was built from the same source; the
# ~1h clean rebuild buys nothing.  Leave unset for the authoritative
# from-scratch run.
if [[ "${STOCK_REUSE:-0}" == 1 && -x "${LANE_BUILD}/bin/icon.stock" ]]; then
  echo "(reusing existing ${LANE_BUILD}/bin/icon.stock -- STOCK_REUSE=1)"
else
  build_icon "" stock 1
fi


if [[ "${STOCK_ONLY}" == 1 ]]; then
  step "2-3) SKIPPED (STOCK_ONLY): no DaCe lib, no DaCe ICON"
else

step "2) Build DaCe velocity lib from ICON's REAL source (vs STOCK config)"
# --icon-src/--icon-build select the real-source route: the SDFG is lowered from
# the pristine mo_velocity_advection.f90.bak (STOCK's source is pristine right
# now) and the bind_c shim resolves its USEs against STOCK_BUILD/mod via -I.
capped "${PY}" "${DACE_FORTRAN}/scripts/build_icon_dace_libs.py" \
  --icon-src "${ICON_SRC}" \
  --icon-build "${LANE_BUILD}" \
  --out-dir "${DACE_LIBS}"


step "3) Patch mo_velocity_advection.f90 + build DACE ICON"
# DACE_REUSE=1: skip the DaCe ICON rebuild when a binary already exists (symmetric
# to STOCK_REUSE).  The DaCe lib is always regenerated in step 2, and ICON links
# it by rpath/soname, so a reused binary picks up the fresh lib at load time --
# valid whenever only the emitter / lib changed and the patch + ICON tree did not.
if [[ "${DACE_REUSE:-0}" == 1 && -x "${LANE_BUILD}/bin/icon.dace" ]]; then
  echo "(reusing existing ${LANE_BUILD}/bin/icon.dace -- DACE_REUSE=1; lib relinked via rpath)"
else
  apply_dace_patch
  build_icon "${DACE_LIBS}" dace
fi

fi


step "4) Fetch R02B05 grid"
mkdir -p "${GRID_DIR}/${GRID_ID}"
GRID_FILE="${GRID_DIR}/${GRID_ID}/${GRID_NAME}.nc"
if [[ ! -f "${GRID_FILE}" ]]; then
  wget -q --show-progress -O "${GRID_FILE}" "${GRID_URL}"
else
  echo "(grid already cached at ${GRID_FILE})"
fi
ls -lh "${GRID_FILE}"


step "5) Generate Held-Suarez R02B05 experiment file"
EXP_FILE="${RUN}/exp.${EXP}"
cp "${RUN}/exp.atm_heldsuarez" "${EXP_FILE}"
sed -i \
  -e "s|^grid_id=.*|grid_id=${GRID_ID}|" \
  -e 's|^grid_refinement=.*|grid_refinement=R02B05|' \
  -e "s|^icon_data_poolFolder=.*|icon_data_poolFolder=\"${GRID_DIR}\"|" \
  -e 's|0011-01-01T00:00:00Z|0000-01-01T00:00:30Z|' \
  -e 's|modelTimeStep *= *"PT10M"|modelTimeStep    = "PT10S"|' \
  "${EXP_FILE}"
sed -i \
  -e 's|inwp_radiation *= *[0-9]*|inwp_radiation = 0|g' \
  -e 's|ecrad_iconfig *= *[0-9]*|ecrad_iconfig = 0|g' \
  -e 's|llockedmode *= *\.TRUE\.|llockedmode = .FALSE.|g' \
  -e 's|init_seed *= *-*[0-9]*|init_seed = 0|g' \
  -e 's|pinit_seed *= *-*[0-9]*|pinit_seed = 0|g' \
  -e 's|seed *= *-*[0-9]*|seed = 0|g' \
  "${EXP_FILE}"
# Make the run NON-VACUOUS + genuinely 2-rank:
#  - output_interval=PT10S + include_last: without this the P1D interval emits
#    ONLY the t=0 dump (written BEFORE the first velocity_tendencies call), so
#    stock and DaCe would be compared only at t=0 -- a vacuous test.  PT10S lands
#    a record after step 1 (t>=10s), which is the first output that reflects a
#    velocity_tendencies result.
#  - num_io_procs=0: with a dedicated async I/O PE (=1) and NRANKS=2 only ONE PE
#    computes, so the horizontal halo exchange is never exercised.  Zero I/O PEs
#    puts BOTH ranks on the compute decomposition (output still gathered to one
#    global file) -- a real 2-rank dycore run.
sed -i \
  -e 's|output_interval="P1D"|output_interval="PT10S"|g' \
  -e 's|include_last *= *\.FALSE\.|include_last = .TRUE.|g' \
  -e 's|num_io_procs *= *1|num_io_procs = 0|g' \
  "${EXP_FILE}"


step "6) Stage runscripts in the lane build dir"
# One tree, so the runscript's hardcoded ``basedir`` is right for both
# binaries and is generated exactly once.
stage_runscript_helpers "${LANE_BUILD}"
( cd "${ICON_SRC}" && ./make_runscripts "${EXP}" )
ls -lh "${RUN}/exp.${EXP}.run"


step "7) Run ICON on the exp (${NRANKS} ranks)"
echo "Stock:"
run_icon stock

STOCK_EXP="${LANE_BUILD}/experiments/${EXP}.stock"
DACE_EXP="${LANE_BUILD}/experiments/${EXP}.dace"

if [[ "${STOCK_ONLY}" == 1 ]]; then
  # INDEPENDENT 2-node check: prove the plain (un-integrated) ICON run works at
  # NRANKS ranks and is worth comparing, BEFORE any DaCe integration is layered
  # on.  Everything asserted here is DaCe-independent -- grid, experiment
  # validity, num_io_procs=0, the rank count, and non-vacuous t>0 output.
  step "8) Verify the STOCK ${NRANKS}-rank run (independent of DaCe)"
  ls -lh "${STOCK_EXP}/" 2>/dev/null | head -8
  echo
  "${PY}" - "${STOCK_EXP}" "${NRANKS}" <<'PYEOF'
import glob
import sys
from pathlib import Path

from netCDF4 import Dataset

exp_dir, nranks = Path(sys.argv[1]), int(sys.argv[2])
ncs = sorted(glob.glob(str(exp_dir / "*_ml_*.nc")) + glob.glob(str(exp_dir / "*_hl_*.nc")) +
             glob.glob(str(exp_dir / "*_pl_*.nc")))
if not ncs:
    sys.exit(f"FAIL: no *_{{ml,hl,pl}}_*.nc output in {exp_dir} -- the run produced nothing")
worst = 0
for nc in ncs:
    with Dataset(nc) as ds:
        n = len(ds.dimensions["time"]) if "time" in ds.dimensions else 0
    print(f"  {Path(nc).name}: {n} time record(s)")
    worst = max(worst, n)
# >1 record means at least one dump AFTER the first velocity_tendencies call:
# a t=0-only run is vacuous -- nothing to compare that exercises the kernel.
if worst < 2:
    sys.exit(f"FAIL: only {worst} time record(s) -- vacuous (t=0 dump precedes the first "
             f"velocity_tendencies call); fix output_interval/include_last")
print(f"OK: stock run emitted {worst} time records (>1 => a post-step-1 dump exists)")
PYEOF
  echo
  # ASSERT the run really used NRANKS COMPUTE ranks -- the runscript recomputes
  # mpi_total_procs, so a silent default (or a mis-set num_io_procs) would make
  # "2-node" a lie.  Parse ICON's own report; ``work: N`` is the compute-PE count.
  work=$(grep -oE "work: *[0-9]+" "${STOCK_BUILD}/icon_run.log" 2>/dev/null | grep -oE "[0-9]+" | head -1)
  echo "  ICON reports: $(grep -m1 'mpi processes' "${STOCK_BUILD}/icon_run.log" 2>/dev/null | sed 's/^ *//')"
  if [[ "${work}" != "${NRANKS}" ]]; then
    echo "FAIL: ICON used ${work:-?} compute ranks, expected ${NRANKS} (see ${STOCK_BUILD}/icon_run.log)" >&2
    exit 1
  fi
  echo "=== STOCK e2e OK: ${NRANKS} compute ranks + a post-t=0 dump, DaCe-independent ==="
  exit 0
fi

echo "DaCe:"
# Re-point set-up.info at the dace build, regen the runscript so it
# hardcodes ``basedir=${DACE_BUILD}``, then run it.
stage_runscript_helpers "${DACE_BUILD}"
( cd "${ICON_SRC}" && ./make_runscripts "${EXP}" )
run_icon "${DACE_BUILD}"


step "8) Compare output, variable-by-variable, across ALL dumps"
# With output_interval=PT10S the run emits records at t=10/20/30s -- each AFTER a
# velocity_tendencies call -- so the comparison exercises the DaCe kernel, not
# just the t=0 module-init dump.  Stock-Fortran and DaCe-patched ICON must agree
# bit-closely at every dump.
ls -lh "${STOCK_EXP}/" 2>/dev/null | head -8
echo
ls -lh "${DACE_EXP}/" 2>/dev/null | head -8
echo

set +e
"${PY}" "${COMPARE}" "${STOCK_EXP}" "${DACE_EXP}" --rtol "${RTOL}"
cmp_rc=$?
set -e

echo
echo "=== e2e run complete (compare rc=${cmp_rc}) ==="
# Propagate the verdict: 0 = bit-close within rtol, 1 = divergence, 2 = no
# overlapping output (a vacuous run -- treat as failure).  CI / callers gate on
# this exit code instead of parsing stdout.
exit "${cmp_rc}"
