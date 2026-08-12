#!/bin/bash
# Shared machinery for the Daint/Alps (aarch64 GH200) ICON configure scripts.
# Sourced by configure_icon_nvhpc_daint.sh (the only ICON lane script).
# Lives in one place because the netcdf-c-before-netcdf-fortran ordering, the
# rpath set, the ICON source patches and the reuse fingerprint MUST NOT drift
# between the two lanes; per-compiler flags stay in the lane scripts.
#
# A lane script sets COMPILER_SPEC / CC / CXX / FC / COMMON_FLAGS / FC_EXTRA
# and then calls the functions below in order.

ICON_DAINT_SCRIPTS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)
ICON_DAINT_REPO_DIR=$(cd "${ICON_DAINT_SCRIPTS_DIR}/.."; pwd)
# Every ICON source patch both lanes carry, repo-relative, applied idempotently
# in order.  Each patch file exists exactly once in the repo -- the pinit guard
# is the copy the icon e2e tests already use, deliberately NOT duplicated here.
# Entry form: [<apply-root>::]<repo-relative patch>.  The apply root is
# relative to the ICON checkout and defaults to "." -- externals/* are git
# SUBMODULES, so a patch against one must be applied from inside it; git apply
# run at the superproject root will not cross the submodule boundary.
ICON_DAINT_PATCHES=(
  scripts/icon_patches/icon_acc_device_management_nompi.patch
  scripts/icon_patches/icon_nh_supervise_acc_directives.patch
  scripts/icon_patches/icon_pinit_seed_guard.patch
)

# Split an ICON_DAINT_PATCHES entry into its apply root and its patch path.
icon_daint_patch_root() { case "$1" in *::*) echo "${1%%::*}" ;; *) echo "." ;; esac; }
icon_daint_patch_file() { echo "${1##*::}"; }
# Opt-in, applied only by the DaCe-linked build (not part of a stock lane).
ICON_DAINT_DACE_PATCH=scripts/icon_patches/icon_velocity_dace_dispatch.patch
# The sha fetch_icon_source.sh pins; the patches are diffs against it.
ICON_DAINT_PIN_SHA=${ICON_DAINT_PIN_SHA:-8597da45ef4b86323f3fb844caedc4ae5e1ffc01}

# Everything defaults off the repo root; only the toolchain lives outside it,
# as a sibling of the checkout.  Env overrides win, defaults are never absolute.
ICON_DAINT_WORK_ROOT=${WORK_ROOT:-${ICON_DAINT_REPO_DIR}/samples/_work}
ICON_DAINT_SITE_ROOT=$(cd "${ICON_DAINT_REPO_DIR}/.."; pwd)

icon_daint_load_spack() {
  SPACK_ROOT=${SPACK_ROOT:-${ICON_DAINT_SITE_ROOT}/spack}
  # shellcheck disable=SC1091
  source "${SPACK_ROOT}/share/spack/setup-env.sh"
  spack load "gcc@16.1.0 arch=linux-sles15-neoverse_v2"
}

# The ONLY two ICON build trees.  A name is a pure function of (compiler
# lane, GPU flag) -- no lane script ever spells a directory itself, so a third
# tree cannot be produced by a typo or a copied default.
#
# ICON is nvhpc-only: gfortran's OpenACC lacks the derived-type manual deep
# copy ICON's GPU port relies on (it rejects an aggregate and its own component
# named from one directive), and ICON's directives were written against
# nvfortran and Cray.  gcc keeps its lanes in the STANDALONE velocity sample,
# whose extracted kernel carries none of that surface.
ICON_DAINT_BUILD_DIRS='cpu_nvhpc gpu_nvhpc'

icon_daint_build_dir_name() {
  local compiler=$1 gpu=$2 name
  case "${gpu}" in
    0) name="cpu_${compiler}" ;;
    1) name="gpu_${compiler}" ;;
    *) echo "error: GPU must be 0 or 1, got '${gpu}'" >&2; return 1 ;;
  esac
  case " ${ICON_DAINT_BUILD_DIRS} " in
    *" ${name} "*) echo "${name}" ;;
    *) echo "error: derived build dir '${name}' is not one of: ${ICON_DAINT_BUILD_DIRS}" >&2; return 1 ;;
  esac
}

# $1 = compiler lane (nvhpc), $2 = GPU flag (0|1).  BUILD_DIR still wins so
# a caller can build elsewhere, but an override off the canonical four is
# announced rather than silently accepted.
icon_daint_enter_build_dir() {
  local canonical
  canonical=$(icon_daint_build_dir_name "$1" "$2") || exit 1
  ICON_SRC=${ICON_SRC:-${ICON_DAINT_WORK_ROOT}/icon-model}
  if test -n "${BUILD_DIR:-}" && test "$(basename "${BUILD_DIR}")" != "${canonical}"; then
    echo "[${ICON_DAINT_LANE}] NOTE: BUILD_DIR override '${BUILD_DIR}' is off the canonical ${canonical}" >&2
  fi
  BUILD_DIR=${BUILD_DIR:-${ICON_SRC}/build/${canonical}}
  mkdir -p "${BUILD_DIR}"
  BUILD_DIR=$(cd "${BUILD_DIR}"; pwd)
  icon_dir=$(cd "${ICON_SRC}"; pwd)
  cd "${BUILD_DIR}"
  echo "[${ICON_DAINT_LANE}] icon_dir  = ${icon_dir}"
  echo "[${ICON_DAINT_LANE}] build_dir = ${BUILD_DIR}"
}

# Provenance for the run log: what ICON, which patches, which checksums.
icon_daint_print_provenance() {
  local f
  echo "ICON_SHA=$(git -C "${icon_dir}" rev-parse HEAD) pin=${ICON_DAINT_PIN_SHA}"
  local pf
  for f in "${ICON_DAINT_PATCHES[@]}"; do
    pf=$(icon_daint_patch_file "${f}")
    echo "ICON_PATCH=$(basename "${pf}") sha256=$(sha256sum "${ICON_DAINT_REPO_DIR}/${pf}" | cut -c1-16)"
  done
}

# A clone off the pin builds something other than what the patches describe.
icon_daint_assert_pin() {
  local have
  have=$(git -C "${icon_dir}" rev-parse HEAD)
  [ "${have}" = "${ICON_DAINT_PIN_SHA}" ] || {
    echo "ERROR: ${icon_dir} is at ${have}, pin is ${ICON_DAINT_PIN_SHA}" >&2
    echo "       run scripts/fetch_icon_source.sh on a login node first." >&2
    exit 1
  }
}

# Idempotent: a patch already in the tree is detected by a successful reverse
# check, so a second configure run is a no-op and a fresh clone reproduces it.
icon_daint_apply_patches() {
  local f
  for f in "${ICON_DAINT_PATCHES[@]}"; do
    icon_daint_apply_one_patch "$(icon_daint_patch_file "${f}")" "$(icon_daint_patch_root "${f}")"
  done
}

icon_daint_apply_one_patch() {
  local f=$1 root=${2:-.} p dir
  p="${ICON_DAINT_REPO_DIR}/${f}"
  dir="${icon_dir}/${root}"
  test -f "${p}" || { echo "error: missing patch ${p}" >&2; exit 1; }
  test -d "${dir}" || { echo "error: patch root ${dir} does not exist" >&2; exit 1; }
  if git -C "${dir}" apply --reverse --check "${p}" 2>/dev/null; then
    echo "[${ICON_DAINT_LANE}] patch already applied: ${f}"
  elif git -C "${dir}" apply --check "${p}" 2>/dev/null; then
    git -C "${dir}" apply "${p}"
    echo "[${ICON_DAINT_LANE}] patch applied: ${f} (root ${root})"
  else
    echo "error: ${f} neither applies nor is already applied under ${dir}" >&2
    echo "       the ICON pin may have moved -- see scripts/fetch_icon_source.sh" >&2
    exit 1
  fi
}

icon_daint_resolve_deps() {
  sloc() { spack location -i "$1" "${COMPILER_SPEC}"; }
  # netcdf-c is plain C (ABI-neutral): the nvhpc-lane netcdf-fortran was
  # concretized against the gcc netcdf-c, so fall back to that arm.
  NETCDF_C_ROOT=$(sloc netcdf-c 2>/dev/null || spack location -i netcdf-c %gcc@16.1.0)
  NETCDF_F_ROOT=$(sloc netcdf-fortran)
  HDF5_ROOT=$(sloc "hdf5+hl~mpi")
  # openblas stays the gcc build in both lanes: plain C ABI, own rpath.
  OPENBLAS_ROOT=$(spack location -i openblas %gcc@16.1.0)
  echo "[${ICON_DAINT_LANE}] netcdf-c=${NETCDF_C_ROOT}"
  echo "[${ICON_DAINT_LANE}] netcdf-f=${NETCDF_F_ROOT}"
  echo "[${ICON_DAINT_LANE}] hdf5=${HDF5_ROOT}"
}

# C side: netcdf-c include FIRST in every C search path (CFLAGS, CPPFLAGS,
# CPATH) so ICON's C submodules (cdi, ...) never pick a stray netcdf.h.
# Fortran side: netcdf-fortran's module dir leads FCFLAGS; netcdf-c's include
# follows only for netcdf.inc-style consumers, never ahead.
icon_daint_build_flags() {
  NETCDF_C_INC="-I${NETCDF_C_ROOT}/include"
  NETCDF_F_INC="-I${NETCDF_F_ROOT}/include"
  HDF5_INC="-I${HDF5_ROOT}/include"
  CFLAGS="${NETCDF_C_INC} ${COMMON_FLAGS}"
  CXXFLAGS="${COMMON_FLAGS}"
  FCFLAGS="${COMMON_FLAGS} ${FC_EXTRA} ${NETCDF_F_INC} ${NETCDF_C_INC} ${HDF5_INC}"
  CPPFLAGS="${NETCDF_C_INC} ${HDF5_INC} -I/usr/include/libxml2"
  export CPATH="${NETCDF_C_ROOT}/include:${HDF5_ROOT}/include${CPATH:+:${CPATH}}"
  # Spack lib dirs are NOT on the default LD_LIBRARY_PATH here -- every lib
  # dir gets both -L and an rpath baked into the binaries.
  RPATHS="-Wl,-rpath,${NETCDF_C_ROOT}/lib -Wl,-rpath,${NETCDF_F_ROOT}/lib -Wl,-rpath,${HDF5_ROOT}/lib -Wl,-rpath,${OPENBLAS_ROOT}/lib"
  LDFLAGS="-L${NETCDF_C_ROOT}/lib -L${NETCDF_F_ROOT}/lib -L${HDF5_ROOT}/lib -L${OPENBLAS_ROOT}/lib ${RPATHS}"
  LIBS="-lxml2 -L${OPENBLAS_ROOT}/lib -lopenblas -L${NETCDF_F_ROOT}/lib -lnetcdff -L${NETCDF_C_ROOT}/lib -lnetcdf -L${HDF5_ROOT}/lib -lhdf5_hl -lhdf5 -lstdc++"
}

icon_daint_check_opt() {
  ICON_DACE_OPT=${ICON_DACE_OPT:-O0}
  case "${ICON_DACE_OPT}" in
    O0|O2|O3) ;;
    *) echo "error: ICON_DACE_OPT must be O0|O2|O3, got '${ICON_DACE_OPT}'" >&2; exit 1 ;;
  esac
}

# Dycore-focused set shared by both lanes: no MPI at all (user requirement),
# no grib2 (no eccodes on this machine), yaxt off because it hard-requires an
# MPI Fortran wrapper.
icon_daint_base_config_args() {
  EXTRA_CONFIG_ARGS="\
--disable-mpi \
--disable-yaxt \
--enable-bundled-python=mtime \
--disable-jsbach \
--disable-ocean \
--disable-coupling \
--disable-waves \
"
}

# The make phase runs in a shell that never sourced the configure script; this
# records the exact compiler env it used so the build wrapper can reproduce it.
icon_daint_write_build_env() {
  { echo "# generated by ${ICON_DAINT_LANE}; source before make"
    test -n "${1:-}" && echo "export PATH=\"$1:\${PATH}\""
    test "${2:-0}" = 1 && echo "unset CUDA_HOME NVHPC_CUDA_HOME CUDA_ROOT CUDA_PATH"
    :
  } > build-env.sh
}

# Reuse fingerprint: any change to a lane script, this helper or a patch file
# invalidates the recorded icon.mk, whose recipes bake in compilers and flags.
icon_daint_stamp_id() {
  local lane_key=$1 f
  local acc
  acc=$(sha256sum "${ICON_DAINT_SCRIPTS_DIR}/$2" "${BASH_SOURCE[0]}" | cut -d' ' -f1 | tr '\n' ' ')
  for f in "${ICON_DAINT_PATCHES[@]}"; do
    acc="${acc}$(sha256sum "${ICON_DAINT_REPO_DIR}/$(icon_daint_patch_file "${f}")" | cut -d' ' -f1) "
  done
  echo "${acc}${lane_key}"
}

icon_daint_write_stamp() {
  icon_daint_stamp_id "$1" "$2" > "${BUILD_DIR}/.configure.stamp"
  echo "[${ICON_DAINT_LANE}] configure stamp written: ${BUILD_DIR}/.configure.stamp"
}

# Guard against silent /usr netcdf pickup: the recorded configure line must
# carry the spack prefixes, and no bare -lnetcdf may resolve outside them.
icon_daint_check_config_log() {
  local want
  for want in "${NETCDF_C_ROOT}" "${NETCDF_F_ROOT}"; do
    if ! grep -qF -- "${want}" config.log; then
      echo "WARNING: config.log never mentions ${want} -- netcdf may have resolved elsewhere (check /usr)" >&2
    fi
  done
  if grep -qE -- '-I/usr/include([^/]|$).*netcdf' config.log; then
    echo "WARNING: config.log shows a /usr/include netcdf reference -- check header search order" >&2
  fi
}

# Append the DaCe lib AFTER $(link_files) so ICON module symbols resolve, and
# keep it live under --as-needed.
icon_daint_patch_icon_mk() {
  test -n "${DACE_LIBS_DIR}" || return 0
  local link_old link_new
  link_old='$(silent_FCLD)$(FC) -o $@ $(make_FCFLAGS) $(FCFLAGS) $(ICON_FCFLAGS) $(LDFLAGS) $(link_files) $(shell . ./collect.extra-libs) $(LIBS)'
  link_new="${link_old} -L${DACE_LIBS_DIR} -Wl,-rpath,${DACE_LIBS_DIR} -Wl,--no-as-needed -l:libvelocity_inner_wrap.so"
  if grep -qF -- "${link_old}" icon.mk; then
    if grep -qF -- "${link_new}" icon.mk; then
      echo "[${ICON_DAINT_LANE}] icon.mk link rule already DaCe-patched"
    else
      python3 - "${link_old}" "${link_new}" <<'PYEOF'
import pathlib, sys
p = pathlib.Path("icon.mk")
src = p.read_text()
old, new = sys.argv[1], sys.argv[2]
assert old in src
p.write_text(src.replace(old, new))
PYEOF
      echo "[${ICON_DAINT_LANE}] icon.mk link rule patched to append libvelocity_inner_wrap.so"
    fi
  else
    echo "WARNING: icon.mk link-line anchor not found; ICON did not produce a recognised link rule." >&2
    echo "         The DaCe library will NOT be linked into bin/icon." >&2
  fi
}

icon_daint_report_dace_libs() {
  DACE_LIBS_DIR=${DACE_LIBS_DIR-}
  if test -z "${DACE_LIBS_DIR}"; then
    echo "WARNING: DACE_LIBS_DIR not set; building stock ICON (no DaCe libs linked)" >&2
  else
    DACE_LIBS_DIR=$(cd "${DACE_LIBS_DIR}"; pwd)
    echo "[${ICON_DAINT_LANE}] DaCe libs from ${DACE_LIBS_DIR}"
  fi
}
