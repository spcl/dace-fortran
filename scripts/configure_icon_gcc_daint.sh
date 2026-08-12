#!/bin/bash
# Daint/Alps (aarch64 GH200) ICON configure -- GCC lane (gcc/g++/gfortran).
# CPU ONLY: gcc's nvptx toolchain here targets sm_80, not the GH200's cc90, so
# the OpenACC offload lane is nvhpc's (configure_icon_nvhpc_daint.sh).  The two
# lanes never share a build tree.  Deps come from the user spack tree.
#
#   scripts/configure_icon_gcc_daint.sh                # build/gcc
#   BUILD_DIR=... DACE_LIBS_DIR=... scripts/configure_icon_gcc_daint.sh
#
# Env knobs:
#   ICON_DACE_OPT   optimization level: O0 (default, bit-exact) | O2 | O3
#   BUILD_DIR       build tree (default: $ICON_SRC/build/gcc)
#   ICON_SRC        ICON checkout (default: the aarch64 scratch clone)
#   SPACK_ROOT      spack tree (default: the aarch64 scratch spack)
#   DACE_LIBS_DIR   dir with libvelocity_inner_wrap.so (empty => stock ICON)
#
# NO MPI at all (--disable-mpi, serial compilers -- user requirement).  The
# ICON source patches this lane carries are listed in icon_daint_common.sh and
# applied idempotently below.

set -eu
unset CDPATH

ICON_DAINT_LANE=configure_icon_gcc_daint
# shellcheck source=icon_daint_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)/icon_daint_common.sh"

icon_daint_load_spack
icon_daint_enter_build_dir gcc
icon_daint_assert_pin
icon_daint_apply_patches
icon_daint_print_provenance

CC=$(command -v gcc); CXX=$(command -v g++); FC=$(command -v gfortran)
COMPILER_SPEC='%gcc@16.1.0'
echo "[${ICON_DAINT_LANE}] compilers: ${CC} ${CXX} ${FC}"

icon_daint_resolve_deps
icon_daint_report_dace_libs
icon_daint_check_opt

# FMA allowed (user req): contraction ON, everything else FP-conservative.
# The DaCe lib build must match (DACE_FORTRAN_FP_CONTRACT=fast).
COMMON_FLAGS="-${ICON_DACE_OPT} -g -fno-fast-math -ffp-contract=fast -fPIC"
FC_EXTRA='-fbacktrace -ffree-line-length-none'

icon_daint_build_flags
icon_daint_base_config_args
EXTRA_CONFIG_ARGS="${EXTRA_CONFIG_ARGS} --enable-loop-exchange --enable-openmp"

icon_daint_write_build_env "" 0

echo "[${ICON_DAINT_LANE}] invoking ${icon_dir}/configure ..."

"${icon_dir}/configure" \
  CC="${CC}" \
  CXX="${CXX}" \
  FC="${FC}" \
  CFLAGS="${CFLAGS}" \
  CXXFLAGS="${CXXFLAGS}" \
  FCFLAGS="${FCFLAGS}" \
  CPPFLAGS="${CPPFLAGS}" \
  LDFLAGS="${LDFLAGS}" \
  LIBS="${LIBS}" \
  ${EXTRA_CONFIG_ARGS} \
  "$@"

icon_daint_check_config_log
icon_daint_patch_icon_mk
icon_daint_write_stamp "OPT=${ICON_DACE_OPT} DACE=${DACE_LIBS_DIR:+yes}" \
  "$(basename "${BASH_SOURCE[0]}")"

echo ""
echo "[${ICON_DAINT_LANE}] configure done.  Run: make -C ${BUILD_DIR} -j <n>"
if test -n "${DACE_LIBS_DIR}"; then
  echo "ICON will link against: ${DACE_LIBS_DIR}/libvelocity_inner_wrap.so"
fi
