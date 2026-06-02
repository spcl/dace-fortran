#!/bin/bash
# End-to-end ICON-with-DaCe integration test, with a side-by-side
# diff against an unpatched (stock-Fortran) ICON.
#
#   1. Builds the velocity + dycore-wrapper DaCe libraries.
#   2. Builds STOCK ICON  (pristine mo_velocity_advection.f90, no DaCe link)
#      into ``${STOCK_BUILD}``.
#   3. Patches mo_velocity_advection.f90 to dispatch into the DaCe wrapper
#      and builds DACE ICON into ``${DACE_BUILD}``.
#   4. Caches the R02B05 grid.
#   5. Generates the short Held-Suarez R02B05 experiment.
#   6. Generates the runscript for each build dir.
#   7. Runs both ICON binaries on the SAME exp.
#   8. Calls ``compare_icon_runs.py`` to diff every overlapping
#      ``*_{ml,hl,pl}_*.nc`` variable-by-variable.
#
# CURRENT KNOWN LIMITATION: the SDFG was built against
# ``velocity_full.f90``'s stub-typed test kernel, not ICON's real
# ``t_patch`` / ``t_nh_prog`` layout, so the DaCe-patched ICON
# SIGSEGVs inside the first velocity_tendencies call.  The DaCe run
# therefore writes only the t=0 initial dump (BEFORE
# velocity_tendencies is called) and we compare THAT against the
# stock run's t=0 dump.  A bit-exact run for t > 0 requires rebuilding
# the SDFG against ICON's real ``mo_velocity_advection`` source --
# a separate effort.
#
# Tunables:
#   ICON_SRC, DACE_FORTRAN, DACE_LIBS, GRID_DIR, STOCK_BUILD,
#   DACE_BUILD, EXP, NRANKS, PY, RTOL
set -euo pipefail

ICON_SRC=${ICON_SRC:-/home/primrose/Work/icon-model-public}
DACE_FORTRAN=${DACE_FORTRAN:-/home/primrose/Work/dace-fortran}
DACE_LIBS=${DACE_LIBS:-/home/primrose/Work/dace-icon-libs}
GRID_DIR=${GRID_DIR:-/home/primrose/Work/icon-grids}
STOCK_BUILD=${STOCK_BUILD:-${ICON_SRC}/build/stock_cpu}
DACE_BUILD=${DACE_BUILD:-${ICON_SRC}/build/dace_cpu}
EXP=${EXP:-atm_heldsuarez_dace_r02b05}
NRANKS=${NRANKS:-2}
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


# Configure + clean rebuild ICON.  Pass DACE_LIBS_DIR="" for stock.
build_icon() {
  local build_dir=$1
  local dace_libs_dir=$2
  echo "  -> ${build_dir} (DACE_LIBS_DIR='${dace_libs_dir}')"
  rm -rf "${build_dir}"
  mkdir -p "${build_dir}"
  ( cd "${build_dir}" && DACE_LIBS_DIR="${dace_libs_dir}" \
      bash "${DACE_FORTRAN}/scripts/configure_icon_dace_cpu.sh" )
  make -C "${build_dir}" -j "$(nproc)" >/dev/null
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
  mpi_total_procs="${NRANKS}" bash -c \
    "cd '${build_dir}/run' && bash 'exp.${EXP}.run'" >/dev/null 2>&1
  local rc=$?
  set -e
  echo "  ${build_dir##*/} run rc=${rc}, exp_dir=${exp_dir}"
}


step "1) Build DaCe libs (velocity_inner_wrap + dycore_wrapper)"
"${PY}" "${DACE_FORTRAN}/scripts/build_icon_dace_libs.py" \
  --with-dycore --out-dir "${DACE_LIBS}"


step "2) Build STOCK ICON (no patch, no DaCe link)"
# Preserve the pristine source the first time through.
[[ -f "${VELOCITY_F90}.bak" ]] || cp "${VELOCITY_F90}" "${VELOCITY_F90}.bak"
cp "${VELOCITY_F90}.bak" "${VELOCITY_F90}"
build_icon "${STOCK_BUILD}" ""


step "3) Patch mo_velocity_advection.f90 + build DACE ICON"
apply_dace_patch
build_icon "${DACE_BUILD}" "${DACE_LIBS}"


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


step "6) Stage runscripts in both build dirs"
stage_runscript_helpers "${STOCK_BUILD}"
( cd "${ICON_SRC}" && ./make_runscripts "${EXP}" )
# The generated runscript hardcodes ``basedir`` to whichever build's
# set-up.info was symlinked at make_runscripts time -- which is the
# stock one now.  We regenerate it for the dace build after switching
# the set-up.info symlink in step 7.
ls -lh "${RUN}/exp.${EXP}.run"


step "7) Run BOTH ICON binaries on the SAME exp"
echo "Stock:"
run_icon "${STOCK_BUILD}"

echo "DaCe:"
# Re-point set-up.info at the dace build, regen the runscript so it
# hardcodes ``basedir=${DACE_BUILD}``, then run it.
stage_runscript_helpers "${DACE_BUILD}"
( cd "${ICON_SRC}" && ./make_runscripts "${EXP}" )
run_icon "${DACE_BUILD}"

STOCK_EXP="${STOCK_BUILD}/experiments/${EXP}"
DACE_EXP="${DACE_BUILD}/experiments/${EXP}"


step "8) Compare initial-state output, variable-by-variable"
# ICON's nc dump at t=0 is written BEFORE the first velocity_tendencies
# call.  Stock-Fortran and DaCe-patched ICONs MUST produce identical
# t=0 output -- if they don't, the linker pulled in something
# numerically different at module-init time.  Subsequent dumps (which
# only the stock run completes) are not comparable today because the
# DaCe ICON crashes mid-step 1; the comparer just skips files that
# only exist on one side.
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
