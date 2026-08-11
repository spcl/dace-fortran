#!/bin/bash
# Daint/Alps (aarch64 GH200) sibling of configure_icon_dace_cpu.sh.
# Same philosophy: invoke ICON's own configure unchanged, then append the
# DaCe .so to the final icon.mk link rule when DACE_LIBS_DIR is set.
# Deps come from the user spack tree (gcc@16.1.0 arms) instead of apt paths.
#
#   cd $ICON_SRC && mkdir -p build/dace_cpu && cd build/dace_cpu
#   DACE_LIBS_DIR=/path/to/dace-icon-libs \
#     $DACE_FORTRAN_REPO/scripts/configure_icon_dace_cpu_daint.sh
#
# Env knobs:
#   COMPILER        gcc (default) | nvhpc -- the ICON compiler lane; deps are
#                   resolved from the matching spack arm (%gcc@16.1.0 or
#                   %nvhpc@26.3).  LLVM is NOT an ICON lane (no OpenACC).
#   GPU             1 => OpenACC GPU build: nvhpc only (forces COMPILER=nvhpc;
#                   the gcc nvptx toolchain here targets sm_80, not cc90),
#                   -acc=gpu -gpu=cc90 -Minfo=accel, --enable-gpu=openacc+cuda
#                   with nvhpc's bundled CUDA as CUDACXX.  -Minfo=accel is
#                   compile-time output -- it lands in the make log.
#   ICON_DACE_OPT   optimization level: O0 (default, bit-exact) | O2 | O3
#   SPACK_ROOT      spack tree (default: the aarch64 scratch spack)
#   DACE_LIBS_DIR   dir with libvelocity_inner_wrap.so (empty => stock ICON)
#
# Differences vs the x86 script: NO MPI at all (--disable-mpi, serial
# compilers -- user requirement), spack-resolved netcdf/hdf5/openblas,
# system libxml2 (SLES /usr), no eccodes/fyaml and no --enable-grib2 (not
# installed on this machine; grib2 output unused by the Held-Suarez e2e).

set -eu
unset CDPATH

icon_dir=$(cd ../..; pwd)
echo "[configure_icon_dace_cpu_daint] icon_dir = ${icon_dir}"

SPACK_ROOT=${SPACK_ROOT:-/capstor/scratch/cscs/ybudanaz/aarch64/spack}
# shellcheck disable=SC1091
source "${SPACK_ROOT}/share/spack/setup-env.sh"
spack load "gcc@16.1.0 arch=linux-sles15-neoverse_v2"

GPU=${GPU:-0}
if [ "${GPU}" = 1 ]; then
  COMPILER=${COMPILER:-nvhpc}
  if [ "${COMPILER}" != nvhpc ]; then
    echo "error: GPU=1 supports COMPILER=nvhpc only (gcc nvptx here is sm_80, not cc90)" >&2
    exit 1
  fi
fi
COMPILER=${COMPILER:-gcc}
case "${COMPILER}" in
  gcc)
    CC=gcc; CXX=g++; FC=gfortran
    COMPILER_SPEC='%gcc@16.1.0'
    ;;
  nvhpc)
    # Serial nvc/nvfortran from the spack-installed NVHPC SDK; no MPI here
    # either.  The CPU lane hardcodes no -acc flags so a later GPU variant
    # only has to extend COMMON_FLAGS.
    NVHPC_PREFIX=$(spack location -i nvhpc)
    NVHPC_BIN=$(dirname "$(find "${NVHPC_PREFIX}" -name nvfortran -path '*compilers/bin*' | head -1)")
    test -x "${NVHPC_BIN}/nvfortran" || { echo "error: nvfortran not found under ${NVHPC_PREFIX}" >&2; exit 1; }
    export PATH="${NVHPC_BIN}:${PATH}"
    CC=nvc; CXX=nvc++; FC=nvfortran
    COMPILER_SPEC='%nvhpc@26.3'
    ;;
  *) echo "error: COMPILER must be gcc|nvhpc, got '${COMPILER}'" >&2; exit 1 ;;
esac
echo "[configure_icon_dace_cpu_daint] compiler lane: ${COMPILER}"

sloc() { spack location -i "$1" "${COMPILER_SPEC}"; }
# netcdf-c is plain C (ABI-neutral): the nvhpc-lane netcdf-fortran was
# concretized against the gcc netcdf-c, so fall back to that arm.
NETCDF_C_ROOT=$(sloc netcdf-c 2>/dev/null || spack location -i netcdf-c %gcc@16.1.0)
NETCDF_F_ROOT=$(sloc netcdf-fortran)
HDF5_ROOT=$(sloc "hdf5+hl~mpi")
# openblas stays the gcc build in both lanes: plain C ABI, own rpath.
OPENBLAS_ROOT=$(spack location -i openblas %gcc@16.1.0)
echo "[configure_icon_dace_cpu_daint] netcdf-c=${NETCDF_C_ROOT}"
echo "[configure_icon_dace_cpu_daint] netcdf-f=${NETCDF_F_ROOT}"
echo "[configure_icon_dace_cpu_daint] hdf5=${HDF5_ROOT}"

DACE_LIBS_DIR=${DACE_LIBS_DIR-}
if test -z "${DACE_LIBS_DIR}"; then
  echo "WARNING: DACE_LIBS_DIR not set; building stock ICON (no DaCe libs linked)" >&2
else
  DACE_LIBS_DIR=$(cd "${DACE_LIBS_DIR}"; pwd)
  echo "[configure_icon_dace_cpu_daint] DaCe libs from ${DACE_LIBS_DIR}"
fi

# Cross-check the resolved prefixes against the packages' own config tools
# (both are 'netcdf' to configure but DIFFERENT spack packages -- the C
# submodules must see netcdf-c, the Fortran side netcdf-fortran).
if [ -x "${NETCDF_C_ROOT}/bin/nc-config" ]; then
  echo "[configure_icon_dace_cpu_daint] nc-config --includedir: $(${NETCDF_C_ROOT}/bin/nc-config --includedir)"
fi
if [ -x "${NETCDF_F_ROOT}/bin/nf-config" ]; then
  echo "[configure_icon_dace_cpu_daint] nf-config --includedir: $(${NETCDF_F_ROOT}/bin/nf-config --includedir)"
  echo "[configure_icon_dace_cpu_daint] nf-config --fflags:     $(${NETCDF_F_ROOT}/bin/nf-config --fflags)"
fi

NETCDF_C_INC="-I${NETCDF_C_ROOT}/include"
NETCDF_C_LIB="-L${NETCDF_C_ROOT}/lib -lnetcdf"
NETCDF_F_INC="-I${NETCDF_F_ROOT}/include"
NETCDF_F_LIB="-L${NETCDF_F_ROOT}/lib -lnetcdff"
HDF5_INC="-I${HDF5_ROOT}/include"
HDF5_LIB="-L${HDF5_ROOT}/lib -lhdf5_hl -lhdf5"
XML2_INC='-I/usr/include/libxml2'
XML2_LIB='-lxml2'
BLAS_LIB="-L${OPENBLAS_ROOT}/lib -lopenblas"

# Same FP-conservative philosophy as the x86 script, per compiler lane;
# ICON_DACE_OPT=O2/O3 for production timings (envelope widens to ~1 ULP).
# The nvhpc lane keeps -acc OFF here -- a later GPU variant extends
# COMMON_FLAGS instead of editing this script's structure.
ICON_DACE_OPT=${ICON_DACE_OPT:-O0}
case "${ICON_DACE_OPT}" in
  O0|O2|O3) ;;
  *) echo "error: ICON_DACE_OPT must be O0|O2|O3, got '${ICON_DACE_OPT}'" >&2; exit 1 ;;
esac
# FMA allowed (user req): contraction ON in both lanes -- gcc gets an
# explicit -ffp-contract=fast, nvhpc keeps its default -Mfma.  The DaCe
# lib build must match: DACE_FORTRAN_FP_CONTRACT=fast for these lanes.
if [ "${COMPILER}" = gcc ]; then
  COMMON_FLAGS="-${ICON_DACE_OPT} -g -fno-fast-math -ffp-contract=fast -fPIC"
  FC_EXTRA='-fbacktrace -ffree-line-length-none'
else
  COMMON_FLAGS="-${ICON_DACE_OPT} -g -Kieee -fPIC"
  FC_EXTRA='-traceback -Mrecursive -Mallocatable=03'
  if [ "${GPU}" = 1 ]; then
    FC_EXTRA="${FC_EXTRA} -acc=gpu -gpu=cc90 -Minfo=accel"
  fi
fi

# C side: netcdf-c include FIRST in every C search path (CFLAGS, CPPFLAGS,
# CPATH) so ICON's C submodules (cdi, ...) never pick a stray netcdf.h.
# Fortran side: netcdf-fortran's module dir leads FCFLAGS; netcdf-c's
# include follows only for netcdf.inc-style consumers, never ahead.
CFLAGS="${NETCDF_C_INC} ${COMMON_FLAGS}"
CXXFLAGS="${COMMON_FLAGS}"
FCFLAGS="${COMMON_FLAGS} ${FC_EXTRA} ${NETCDF_F_INC} ${NETCDF_C_INC} ${HDF5_INC}"
CPPFLAGS="${NETCDF_C_INC} ${HDF5_INC} ${XML2_INC}"
export CPATH="${NETCDF_C_ROOT}/include:${HDF5_ROOT}/include${CPATH:+:${CPATH}}"

# Spack lib dirs are NOT on the default LD_LIBRARY_PATH here -- every lib
# dir gets both -L and an rpath baked into the binaries.
RPATHS="-Wl,-rpath,${NETCDF_C_ROOT}/lib -Wl,-rpath,${NETCDF_F_ROOT}/lib -Wl,-rpath,${HDF5_ROOT}/lib -Wl,-rpath,${OPENBLAS_ROOT}/lib"
LDFLAGS="-L${NETCDF_C_ROOT}/lib -L${NETCDF_F_ROOT}/lib -L${HDF5_ROOT}/lib -L${OPENBLAS_ROOT}/lib ${RPATHS}"

LIBS="${XML2_LIB} ${BLAS_LIB} ${NETCDF_F_LIB} ${NETCDF_C_LIB} ${HDF5_LIB} -lstdc++"

# Same dycore-focused set as x86 minus grib2 (no eccodes on this machine),
# plus --disable-mpi --disable-yaxt (yaxt hard-requires an MPI Fortran
# wrapper; configure says to disable both together).
EXTRA_CONFIG_ARGS="\
--disable-mpi \
--disable-yaxt \
--enable-bundled-python=mtime \
--disable-jsbach \
--disable-ocean \
--disable-coupling \
--disable-waves \
"
CONFIG_VARS=()
if [ "${GPU}" = 1 ]; then
  # GPU lane: loop-exchange is a CPU-cache layout, OpenMP off (OpenACC owns
  # the parallelism), CUDA from the NVHPC bundle next to nvfortran.
  # spack-load of gcc16 exports CUDA_HOME/NVHPC_CUDA_HOME=spack cuda-13.3
  # (its nvptx dep) and makes g++16 nvcc's host compiler -- cuda13.3 rejects
  # gcc>15, so unset the redirects and pin the host to system g++-13.
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

echo "[configure_icon_dace_cpu_daint] invoking ${icon_dir}/configure ..."

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

# Guard against silent /usr netcdf pickup: the recorded configure line must
# carry the spack prefixes, and no bare -lnetcdf may resolve outside them.
for want in "${NETCDF_C_ROOT}" "${NETCDF_F_ROOT}"; do
  if ! grep -qF -- "${want}" config.log; then
    echo "WARNING: config.log never mentions ${want} -- netcdf may have resolved elsewhere (check /usr)" >&2
  fi
done
if grep -qE -- '-I/usr/include([^/]|$).*netcdf' config.log; then
  echo "WARNING: config.log shows a /usr/include netcdf reference -- check header search order" >&2
fi

if test -n "${DACE_LIBS_DIR}"; then
  # Identical icon.mk patch logic to configure_icon_dace_cpu.sh: append the
  # DaCe lib AFTER $(link_files) so ICON module symbols resolve, and keep it
  # live under --as-needed.
  link_old='$(silent_FCLD)$(FC) -o $@ $(make_FCFLAGS) $(FCFLAGS) $(ICON_FCFLAGS) $(LDFLAGS) $(link_files) $(shell . ./collect.extra-libs) $(LIBS)'
  link_new="${link_old} -L${DACE_LIBS_DIR} -Wl,-rpath,${DACE_LIBS_DIR} -Wl,--no-as-needed -l:libvelocity_inner_wrap.so"
  if grep -qF -- "${link_old}" icon.mk; then
    if grep -qF -- "${link_new}" icon.mk; then
      echo "[configure_icon_dace_cpu_daint] icon.mk link rule already DaCe-patched"
    else
      python3 - "${link_old}" "${link_new}" <<'PYEOF'
import pathlib, sys
p = pathlib.Path("icon.mk")
src = p.read_text()
old, new = sys.argv[1], sys.argv[2]
assert old in src
p.write_text(src.replace(old, new))
PYEOF
      echo "[configure_icon_dace_cpu_daint] icon.mk link rule patched to append libvelocity_inner_wrap.so"
    fi
  else
    echo "WARNING: icon.mk link-line anchor not found; ICON did not produce a recognised link rule." >&2
    echo "         The DaCe library will NOT be linked into bin/icon." >&2
  fi
fi

echo ""
echo "[configure_icon_dace_cpu_daint] configure done.  Run: make -j <n>"
if test -n "${DACE_LIBS_DIR}"; then
  echo "ICON will link against: ${DACE_LIBS_DIR}/libvelocity_inner_wrap.so"
fi
