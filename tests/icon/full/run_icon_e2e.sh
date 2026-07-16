#!/bin/bash
# End-to-end ICON-with-DaCe integration test, with a side-by-side
# diff against an unpatched (stock-Fortran) ICON.
#
#   1. Builds STOCK ICON  (pristine mo_velocity_advection.f90, no DaCe link)
#      into ``${STOCK_BUILD}`` -- FIRST, so its config-matched .mod + -D
#      defines are available for the lib build.
#   2. Builds the DaCe velocity library from ICON's REAL
#      ``mo_velocity_advection`` source, lowered against STOCK's config.
#   3. Patches mo_velocity_advection.f90 to dispatch into the DaCe wrapper
#      and builds DACE ICON into ``${DACE_BUILD}``.
#   4. Caches the R02B05 grid.
#   5. Generates the short Held-Suarez R02B05 experiment.
#   6. Generates the runscript for each build dir.
#   7. Runs both ICON binaries on the SAME exp (NRANKS ranks each).
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
#   ICON_SRC, DACE_FORTRAN, DACE_LIBS, GRID_DIR, STOCK_BUILD,
#   DACE_BUILD, EXP, NRANKS, PY, RTOL, STOCK_ONLY, CAP
set -euo pipefail

# Default ICON_SRC to the in-tree submodule (icon-2026.04-public); override to
# point at a separate checkout.  The old default
# ``/home/primrose/Work/icon-model-public`` does not exist on this box.
_SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)
ICON_SRC=${ICON_SRC:-${_SELF_DIR}/icon-model}
DACE_FORTRAN=${DACE_FORTRAN:-/home/primrose/Work/dace-fortran}
DACE_LIBS=${DACE_LIBS:-/home/primrose/Work/dace-icon-libs}
GRID_DIR=${GRID_DIR:-/home/primrose/Work/icon-grids}
STOCK_BUILD=${STOCK_BUILD:-${ICON_SRC}/build/stock_cpu}
DACE_BUILD=${DACE_BUILD:-${ICON_SRC}/build/dace_cpu}
EXP=${EXP:-atm_heldsuarez_dace_r02b05}
NRANKS=${NRANKS:-2}
# STOCK_ONLY=1: build + run ONLY stock ICON at NRANKS ranks and verify the run is
# real (right rank count) and non-vacuous (a dump after t=0).  No DaCe lib, no
# DaCe ICON, no comparison.  Run this FIRST: it proves the 2-node run itself
# works independently, so an integration failure later can't be confused with a
# broken experiment / grid / rank setup.
STOCK_ONLY=${STOCK_ONLY:-0}
PY=${PY:-/home/primrose/.pyenv/versions/py13/bin/python3}
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
  "${PY}" - "${VELOCITY_F90}" <<'PYEOF'
import sys
from pathlib import Path

p = Path(sys.argv[1])
lines = p.read_text().splitlines()
subr_start = next(i for i, ln in enumerate(lines)
                  if "SUBROUTINE velocity_tendencies " in ln and "(" in ln)
header_end = subr_start
while lines[header_end].rstrip().endswith("&"):
    header_end += 1
end_subr = next(i for i, ln in enumerate(lines[header_end + 1:],
                                         start=header_end + 1)
                if "END SUBROUTINE velocity_tendencies" in ln)
last_intent = header_end
for i, ln in enumerate(lines[header_end + 1:end_subr], start=header_end + 1):
    if "INTENT" in ln.upper() and "::" in ln:
        last_intent = i

iface_block = [
    "    ! DACE INTEGRATION: dispatch the velocity tendencies kernel to the",
    "    ! SDFG-generated implementation in libvelocity_inner_wrap.so.  The",
    "    ! INTERFACE block declares a FREE-STANDING wrapper symbol so we do",
    "    ! NOT USE the bindings module's .mod (its stub types would conflict",
    "    ! with mo_model_domain / mo_nonhydro_types).  The original body is",
    "    ! removed -- recover via mo_velocity_advection.f90.bak.",
    "    INTERFACE",
    "      SUBROUTINE velocity_tendencies_dace_icon(p_prog, p_patch, p_int, p_metrics, p_diag, &",
    "                                               z_w_concorr_me, z_kin_hor_e, z_vt_ie, &",
    "                                               ntnd, istep, lvn_only, &",
    "                                               dtime, dt_linintp_ubc, ldeepatmo)",
    "        USE iso_c_binding,        ONLY: c_int, c_double, c_bool",
    "        USE mo_model_domain,      ONLY: t_patch",
    "        USE mo_intp_data_strc,    ONLY: t_int_state",
    "        USE mo_nonhydro_types,    ONLY: t_nh_prog, t_nh_metrics, t_nh_diag",
    "        TYPE(t_nh_prog),    INTENT(INOUT), TARGET :: p_prog",
    "        TYPE(t_patch),      INTENT(IN),    TARGET :: p_patch",
    "        TYPE(t_int_state),  INTENT(IN),    TARGET :: p_int",
    "        TYPE(t_nh_metrics), INTENT(INOUT), TARGET :: p_metrics",
    "        TYPE(t_nh_diag),    INTENT(INOUT), TARGET :: p_diag",
    "        REAL(c_double),     INTENT(INOUT), TARGET :: z_w_concorr_me(:,:,:)",
    "        REAL(c_double),     INTENT(INOUT), TARGET :: z_kin_hor_e(:,:,:)",
    "        REAL(c_double),     INTENT(INOUT), TARGET :: z_vt_ie(:,:,:)",
    "        INTEGER(c_int),     INTENT(IN),    TARGET :: ntnd",
    "        INTEGER(c_int),     INTENT(IN),    TARGET :: istep",
    "        LOGICAL(c_bool),    INTENT(IN),    TARGET :: lvn_only",
    "        REAL(c_double),     INTENT(IN),    TARGET :: dtime",
    "        REAL(c_double),     INTENT(IN),    TARGET :: dt_linintp_ubc",
    "        LOGICAL(c_bool),    INTENT(IN),    TARGET :: ldeepatmo",
    "      END SUBROUTINE velocity_tendencies_dace_icon",
    "    END INTERFACE",
    "    CALL velocity_tendencies_dace_icon(p_prog, p_patch, p_int, p_metrics, p_diag, &",
    "                                       z_w_concorr_me, z_kin_hor_e, z_vt_ie, &",
    "                                       ntnd, istep, &",
    "                                       LOGICAL(lvn_only, kind=1), &",
    "                                       dtime, dt_linintp_ubc, &",
    "                                       LOGICAL(ldeepatmo, kind=1))",
    "",
]
extra_top_use = ["    USE iso_c_binding, ONLY: c_bool"]
new = (lines[:header_end + 1]
       + extra_top_use
       + lines[header_end + 1:last_intent + 1]
       + iface_block
       + lines[end_subr:])
p.write_text("\n".join(new) + "\n")
PYEOF
}


# Serial, memory-capped runner.  This box has 12GB RAM; a parallel ``make`` or an
# uncapped model run thrashes swap.  Run every heavy step in a transient systemd
# scope so it is OOM-killed at ${CAP} instead of swap-crawling.  No fallback: a
# silent uncapped run is exactly what the cap exists to prevent, so if
# systemd-run is unavailable this fails loudly.
CAP=${CAP:-8G}
capped() {
  systemd-run --user --scope -p MemoryMax="${CAP}" -p MemorySwapMax=0 --quiet "$@"
}

# Configure + clean rebuild ICON.  Pass DACE_LIBS_DIR="" for stock.
# make -j1 + 8GB cap (single build at a time -- see the box's RAM budget).
build_icon() {
  local build_dir=$1
  local dace_libs_dir=$2
  echo "  -> ${build_dir} (DACE_LIBS_DIR='${dace_libs_dir}')"
  rm -rf "${build_dir}"
  mkdir -p "${build_dir}"
  ( cd "${build_dir}" && DACE_LIBS_DIR="${dace_libs_dir}" \
      bash "${DACE_FORTRAN}/scripts/configure_icon_dace_cpu.sh" )
  capped make -C "${build_dir}" -j1 >/dev/null
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


# Run ICON under the generated runscript and capture rc.
run_icon() {
  local build_dir=$1
  local exp_dir="${build_dir}/experiments/${EXP}"
  rm -rf "${exp_dir}"
  ln -sfn "${RUN}/exp.${EXP}.run" "${build_dir}/run/exp.${EXP}.run"
  set +e
  # The generated ``exp.<EXP>.run`` RECOMPUTES the rank count as
  # ``: ${no_of_nodes:=1} ${mpi_procs_pernode:=4}; ((mpi_total_procs = no_of_nodes
  # * mpi_procs_pernode))`` -- so exporting ``mpi_total_procs`` is ignored (default
  # 4 ranks).  The ``:=`` honours these two if already set, so export them (via
  # ``env`` so they survive the ``capped`` systemd scope): 1 node x NRANKS =
  # NRANKS ranks.  Both compute (num_io_procs=0), a genuine NRANKS-rank run.
  capped env no_of_nodes=1 mpi_procs_pernode="${NRANKS}" bash -c \
    "cd '${build_dir}/run' && bash 'exp.${EXP}.run'" > "${build_dir}/icon_run.log" 2>&1
  local rc=$?
  set -e
  echo "  ${build_dir##*/} run rc=${rc}, exp_dir=${exp_dir}"
}


step "1) Build STOCK ICON (no patch, no DaCe link)"
# STOCK is built FIRST: the DaCe velocity lib is now lowered from ICON's REAL
# mo_velocity_advection source (real t_patch / t_nh_prog layout -- the stub-typed
# velocity_full.f90 SIGSEGVs in a real run), and that lowering needs the TARGET
# ICON config's -D defines + compiled .mod, which only exist after a build.
# STOCK and DACE differ ONLY by the icon.mk link patch + the velocity source
# patch, so STOCK_BUILD's mod/ + defines are valid for the lib the DACE build
# links.
# Preserve the pristine source the first time through, and build from it.
[[ -f "${VELOCITY_F90}.bak" ]] || cp "${VELOCITY_F90}" "${VELOCITY_F90}.bak"
cp "${VELOCITY_F90}.bak" "${VELOCITY_F90}"
build_icon "${STOCK_BUILD}" ""


if [[ "${STOCK_ONLY}" == 1 ]]; then
  step "2-3) SKIPPED (STOCK_ONLY): no DaCe lib, no DaCe ICON"
else

step "2) Build DaCe velocity lib from ICON's REAL source (vs STOCK config)"
# --icon-src/--icon-build select the real-source route: the SDFG is lowered from
# the pristine mo_velocity_advection.f90.bak (STOCK's source is pristine right
# now) and the bind_c shim resolves its USEs against STOCK_BUILD/mod via -I.
capped "${PY}" "${DACE_FORTRAN}/scripts/build_icon_dace_libs.py" \
  --icon-src "${ICON_SRC}" \
  --icon-build "${STOCK_BUILD}" \
  --out-dir "${DACE_LIBS}"


step "3) Patch mo_velocity_advection.f90 + build DACE ICON"
apply_dace_patch
build_icon "${DACE_BUILD}" "${DACE_LIBS}"

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


step "6) Stage runscripts in both build dirs"
stage_runscript_helpers "${STOCK_BUILD}"
( cd "${ICON_SRC}" && ./make_runscripts "${EXP}" )
# The generated runscript hardcodes ``basedir`` to whichever build's
# set-up.info was symlinked at make_runscripts time -- which is the
# stock one now.  We regenerate it for the dace build after switching
# the set-up.info symlink in step 7.
ls -lh "${RUN}/exp.${EXP}.run"


step "7) Run ICON on the exp (${NRANKS} ranks)"
echo "Stock:"
run_icon "${STOCK_BUILD}"

STOCK_EXP="${STOCK_BUILD}/experiments/${EXP}"
DACE_EXP="${DACE_BUILD}/experiments/${EXP}"

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
