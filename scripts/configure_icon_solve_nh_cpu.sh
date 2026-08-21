#!/bin/bash
# Hardcoded ICON configure for the Ubuntu / GCC / OpenMPI / system-NetCDF
# CPU build with the DaCe-generated ``libsolve_nh.so`` linked in.
#
# Run from an out-of-source build directory inside the ICON checkout:
#
#   cd $ICON_SRC
#   mkdir -p build/solve_nh_dace_cpu && cd build/solve_nh_dace_cpu
#   DACE_LIBS_DIR=/path/to/solve-nh-dace-libs \
#     $DACE_FORTRAN_REPO/scripts/configure_icon_solve_nh_cpu.sh
#   make -j 8
#
# Override $DACE_LIBS_DIR to point at the directory produced by
# `scripts/build_icon_solve_nh_libs.py`.

set -eu
unset CDPATH

script_dir=$(cd "$(dirname "$0")"; pwd)
# The build directory may be outside the ICON source tree (e.g. a CI scratch dir),
# so the source root must come from $ICON_SRC when set.
icon_dir=${ICON_SRC:-$(cd ../..; pwd)}
echo "[configure_icon_solve_nh_cpu] icon_dir = ${icon_dir}"

DACE_LIBS_DIR=${DACE_LIBS_DIR-}
if test -z "${DACE_LIBS_DIR}"; then
  echo "WARNING: DACE_LIBS_DIR not set; building stock ICON (no solve_nh DaCe lib linked)" >&2
else
  DACE_LIBS_DIR=$(cd "${DACE_LIBS_DIR}"; pwd)
  echo "[configure_icon_solve_nh_cpu] DaCe libs from ${DACE_LIBS_DIR}"
  # The patched mo_solve_nonhydro.f90 only needs mo_solve_nh_diff.mod from the
  # DaCe binding directory.  Expose ONLY that module to ICON's build; the binding
  # directory also carries stub modules (mo_mpi, mo_nonhydro_types, ...) that
  # would shadow ICON's real modules if added to the global include path.
  DACE_ICON_MOD_DIR="${DACE_LIBS_DIR}/icon_mods"
  mkdir -p "${DACE_ICON_MOD_DIR}"
  if test -f "${DACE_LIBS_DIR}/mo_solve_nh_diff.mod"; then
    cp -p "${DACE_LIBS_DIR}/mo_solve_nh_diff.mod" "${DACE_ICON_MOD_DIR}/"
  fi
fi

# Ubuntu apt-installed deps (hardcoded paths):
NETCDF_C_INC='-I/usr/include'
NETCDF_C_LIB='-L/usr/lib/x86_64-linux-gnu -lnetcdf'
NETCDF_F_INC='-I/usr/include'
NETCDF_F_LIB='-L/usr/lib/x86_64-linux-gnu -lnetcdff'
HDF5_INC='-I/usr/include/hdf5/serial'
HDF5_LIB='-L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_hl -lhdf5'
XML2_INC='-I/usr/include/libxml2'
XML2_LIB='-lxml2'
ECCODES_LIB='-leccodes'
FYAML_LIB='-lfyaml'
LAPACK_LIB='-llapack -lblas'

# MPI wrappers (system OpenMPI):
CC='mpicc'
CXX='mpicxx'
FC='mpifort'

# FP-conservative flags for bit-exact comparison against an unmodified build.
COMMON_FLAGS='-O0 -g -fno-fast-math -ffp-contract=off -fPIC'

CFLAGS="${COMMON_FLAGS}"
CXXFLAGS="${COMMON_FLAGS}"
FCFLAGS="${COMMON_FLAGS} -fbacktrace -ffree-line-length-none ${NETCDF_F_INC} ${NETCDF_C_INC} ${HDF5_INC}"
if test -n "${DACE_ICON_MOD_DIR:-}"; then
  FCFLAGS="${FCFLAGS} -I${DACE_ICON_MOD_DIR}"
fi
CPPFLAGS="${NETCDF_C_INC} ${HDF5_INC} ${XML2_INC}"

LDFLAGS="-L/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu/hdf5/serial"

LIBS="${XML2_LIB} ${FYAML_LIB} ${ECCODES_LIB} ${LAPACK_LIB} ${NETCDF_F_LIB} ${NETCDF_C_LIB} ${HDF5_LIB} -lstdc++"

EXTRA_CONFIG_ARGS="\
--enable-grib2 \
--enable-mpi \
--disable-loop-exchange \
--enable-openmp \
--enable-bundled-python=mtime \
--disable-jsbach \
--disable-ocean \
--disable-coupling \
--disable-waves \
"

echo "[configure_icon_solve_nh_cpu] invoking ${icon_dir}/configure ..."

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
  MPI_LAUNCH='mpiexec' \
  ${EXTRA_CONFIG_ARGS} \
  "$@"

if test -n "${DACE_LIBS_DIR}"; then
  # Append the DaCe solve_nh libraries to the FINAL ICON link rule in icon.mk
  # so they come AFTER ${link_files} (which provides the ICON module symbols
  # the .so's shim forwards reference).  Two artifacts are required:
  #   * libsolve_nh.so           - the Fortran wrapper + bind(c) shim
  #   * _sdfg_build/dacecache/build/libsolve_nh.so - the C++ SDFG implementation
  # --no-as-needed keeps them live under Ubuntu's default --as-needed because
  # the only references to solve_nh_dace_icon live in a single object file that
  # ld may process before the libs.
  link_old='$(silent_FCLD)$(FC) -o $@ $(make_FCFLAGS) $(FCFLAGS) $(ICON_FCFLAGS) $(LDFLAGS) $(link_files) $(shell . ./collect.extra-libs) $(LIBS)'
  cpp_so="${DACE_LIBS_DIR}/_sdfg_build/dacecache/build/libsolve_nh.so"
  link_new="${link_old} -Wl,--no-as-needed ${DACE_LIBS_DIR}/libsolve_nh.so ${cpp_so} -Wl,-rpath,${DACE_LIBS_DIR} -Wl,-rpath,${cpp_so%/*}"
  if grep -qF -- "${link_old}" icon.mk; then
    if grep -qF -- "${link_new}" icon.mk; then
      echo "[configure_icon_solve_nh_cpu] icon.mk link rule already DaCe-patched"
    else
      python3 - "${link_old}" "${link_new}" <<'PYEOF'
import pathlib, sys
p = pathlib.Path("icon.mk")
src = p.read_text()
old, new = sys.argv[1], sys.argv[2]
assert old in src
p.write_text(src.replace(old, new))
PYEOF
      echo "[configure_icon_solve_nh_cpu] icon.mk link rule patched to append DaCe solve_nh libs"
    fi
  else
    echo "WARNING: icon.mk link-line anchor not found; ICON did not produce a recognised link rule." >&2
    echo "         The solve_nh DaCe library will NOT be linked into bin/icon." >&2
  fi
fi

echo ""
echo "[configure_icon_solve_nh_cpu] configure done.  Run: make -j 8"
if test -n "${DACE_LIBS_DIR}"; then
  echo ""
  echo "ICON will link against:"
  echo "  ${DACE_LIBS_DIR}/libsolve_nh.so"
  echo "  ${DACE_LIBS_DIR}/_sdfg_build/dacecache/build/libsolve_nh.so"
  echo ""
  echo "Don't forget to patch"
  echo "  ${icon_dir}/src/atm_dyn_iconam/mo_solve_nonhydro.f90"
  echo "to forward solve_nh to solve_nh_dace_icon"
  echo "(see tests/icon/full/_icon_solve_nh_patch.py)."
fi
