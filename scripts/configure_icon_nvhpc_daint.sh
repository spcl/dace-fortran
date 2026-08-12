#!/bin/bash
# Daint/Alps (aarch64 GH200) ICON configure -- NVHPC lane (nvc/nvc++/nvfortran).
# This is the only lane that can do OpenACC offload here: GPU=1 adds
# -acc=gpu -gpu=cc90 (arch 90, NOT 90a) and --enable-gpu=openacc+cuda.
# The gcc lane lives in configure_icon_gcc_daint.sh; the two never share a
# build tree.  Deps come from the user spack tree.
#
#   scripts/configure_icon_nvhpc_daint.sh              # CPU, build/nvhpc
#   GPU=1 scripts/configure_icon_nvhpc_daint.sh        # OpenACC GPU
#   BUILD_DIR=... DACE_LIBS_DIR=... scripts/configure_icon_nvhpc_daint.sh
#
# Env knobs:
#   GPU             1 => OpenACC GPU build (-acc=gpu -gpu=cc90 -Minfo=accel,
#                   --enable-gpu=openacc+cuda, nvhpc's bundled CUDA as CUDACXX).
#                   -Minfo=accel is compile-time output -- it lands in the make log.
#   ICON_DACE_OPT   optimization level: O0 (default, bit-exact) | O2 | O3
#   BUILD_DIR       build tree (default: $ICON_SRC/build/nvhpc)
#   ICON_SRC        ICON checkout (default: the aarch64 scratch clone)
#   SPACK_ROOT      spack tree (default: the aarch64 scratch spack)
#   DACE_LIBS_DIR   dir with libvelocity_inner_wrap.so (empty => stock ICON)
#
# NO MPI at all (--disable-mpi, serial compilers -- user requirement).  The
# ICON source patches this lane carries are listed in icon_daint_common.sh and
# applied idempotently below.

set -eu
unset CDPATH

ICON_DAINT_LANE=configure_icon_nvhpc_daint
# shellcheck source=icon_daint_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)/icon_daint_common.sh"

GPU=${GPU:-0}
icon_daint_load_spack
icon_daint_enter_build_dir nvhpc
icon_daint_assert_pin
icon_daint_apply_patches
icon_daint_print_provenance

NVHPC_PREFIX=$(spack location -i nvhpc)
NVHPC_BIN=$(dirname "$(find "${NVHPC_PREFIX}" -name nvfortran -path '*compilers/bin*' | head -1)")
test -x "${NVHPC_BIN}/nvfortran" || { echo "error: nvfortran not found under ${NVHPC_PREFIX}" >&2; exit 1; }
export PATH="${NVHPC_BIN}:${PATH}"
# ABSOLUTE paths: icon.mk re-exports CC/CXX/FC verbatim into the delayed
# sub-configures (cdi, rte-rrtmgp) and the cmake externals, which run from
# make -- a shell that never sourced this script's PATH.
CC="${NVHPC_BIN}/nvc"; CXX="${NVHPC_BIN}/nvc++"; FC="${NVHPC_BIN}/nvfortran"
COMPILER_SPEC='%nvhpc@26.3'
echo "[${ICON_DAINT_LANE}] compilers: ${CC} ${CXX} ${FC}"

icon_daint_resolve_deps
icon_daint_report_dace_libs
icon_daint_check_opt

# FMA allowed (user req): nvhpc keeps its default -Mfma; -Kieee holds the rest
# of the FP model.  The DaCe lib build must match (DACE_FORTRAN_FP_CONTRACT=fast).
COMMON_FLAGS="-${ICON_DACE_OPT} -g -Kieee -fPIC"
FC_EXTRA='-traceback -Mrecursive -Mallocatable=03'
if [ "${GPU}" = 1 ]; then
  FC_EXTRA="${FC_EXTRA} -acc=gpu -gpu=cc90 -Minfo=accel"
fi

icon_daint_build_flags
icon_daint_base_config_args

CONFIG_VARS=()
if [ "${GPU}" = 1 ]; then
  # loop-exchange is a CPU-cache layout, OpenMP off (OpenACC owns the
  # parallelism), CUDA from the NVHPC bundle next to nvfortran.
  # spack-load of gcc16 exports CUDA_HOME/NVHPC_CUDA_HOME=spack cuda-13.3 (its
  # nvptx dep) and makes g++16 nvcc's host compiler -- cuda13.3 rejects gcc>15,
  # so unset the redirects and pin the host to system g++-13.
  unset CUDA_HOME NVHPC_CUDA_HOME CUDA_ROOT CUDA_PATH || true
  EXTRA_CONFIG_ARGS="${EXTRA_CONFIG_ARGS} --enable-gpu=openacc+cuda --disable-loop-exchange --disable-openmp"
  CONFIG_VARS+=("CUDACXX=${NVHPC_BIN}/nvcc" "CUDAFLAGS=-arch=sm_90 -O3 -ccbin /usr/bin/g++-13")
  # nvfortran does not search the bundled CUDA lib dir at link time.
  CUDA_LIB64="${NVHPC_BIN%/compilers/bin}/cuda/lib64"
  LDFLAGS="${LDFLAGS} -L${CUDA_LIB64} -Wl,-rpath,${CUDA_LIB64}"
  LIBS="${LIBS} -lcudart"
else
  EXTRA_CONFIG_ARGS="${EXTRA_CONFIG_ARGS} --enable-loop-exchange --enable-openmp"
fi

icon_daint_write_build_env "${NVHPC_BIN}" "${GPU}"

echo "[${ICON_DAINT_LANE}] invoking ${icon_dir}/configure (GPU=${GPU}) ..."

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
  ${CONFIG_VARS[@]+"${CONFIG_VARS[@]}"} \
  ${EXTRA_CONFIG_ARGS} \
  "$@"

icon_daint_check_config_log
icon_daint_patch_icon_mk
icon_daint_write_stamp "GPU=${GPU} OPT=${ICON_DACE_OPT} DACE=${DACE_LIBS_DIR:+yes}" \
  "$(basename "${BASH_SOURCE[0]}")"

echo ""
echo "[${ICON_DAINT_LANE}] configure done.  Run: make -C ${BUILD_DIR} -j <n>"
echo "[${ICON_DAINT_LANE}] source ${BUILD_DIR}/build-env.sh before make"
if test -n "${DACE_LIBS_DIR}"; then
  echo "ICON will link against: ${DACE_LIBS_DIR}/libvelocity_inner_wrap.so"
fi
