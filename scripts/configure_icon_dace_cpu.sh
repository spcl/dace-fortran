#!/bin/bash
# Hardcoded ICON configure for the Ubuntu / GCC-15 / OpenMPI-5 / system-NetCDF
# CPU build with DaCe-generated `libvelocity_inner_wrap.so` linked in.
# fp64 only -- mixed precision is NOT enabled.
#
# Run from an out-of-source build directory inside the ICON checkout:
#
#   cd $ICON_SRC
#   mkdir -p build/dace_cpu && cd build/dace_cpu
#   DACE_LIBS_DIR=/path/to/dace-icon-libs \
#     $DACE_FORTRAN_REPO/scripts/configure_icon_dace_cpu.sh
#   make -j 8
#
# Override $DACE_LIBS_DIR to point at the directory produced by
# `scripts/build_icon_dace_libs.py`.
#
# All paths are pinned to the Ubuntu apt locations:
#   netcdf-c        /usr/include + /usr/lib/x86_64-linux-gnu
#   netcdf-fortran  /usr/include + /usr/lib/x86_64-linux-gnu
#   hdf5 (serial)   /usr/include/hdf5/serial + /usr/lib/x86_64-linux-gnu/hdf5/serial
#   libxml2-dev     /usr/include/libxml2 + /usr/lib/x86_64-linux-gnu
#   eccodes         /usr/include + /lib/x86_64-linux-gnu
#   fyaml           /usr/include + /lib/x86_64-linux-gnu
#   lapack/blas     system default
#   openmpi         /usr/lib/x86_64-linux-gnu/openmpi (via mpifort wrapper)

set -eu
unset CDPATH

script_dir=$(cd "$(dirname "$0")"; pwd)
icon_dir=$(cd ../..; pwd)
echo "[configure_icon_dace_cpu] icon_dir = ${icon_dir}"

DACE_LIBS_DIR=${DACE_LIBS_DIR-}
if test -z "${DACE_LIBS_DIR}"; then
  echo "WARNING: DACE_LIBS_DIR not set; building stock ICON (no DaCe libs linked)" >&2
else
  DACE_LIBS_DIR=$(cd "${DACE_LIBS_DIR}"; pwd)
  echo "[configure_icon_dace_cpu] DaCe libs from ${DACE_LIBS_DIR}"
fi
# We do NOT plumb ``-L${DACE_LIBS_DIR}`` / ``-l:libvelocity_inner_wrap.so``
# through ${LDFLAGS} -- ICON propagates ${LDFLAGS} to every external
# (rte-rrtmgp, cdi, mtime, ...) at THEIR configure time, and those
# externals don't have the ICON module symbols our .so needs.  Under
# Ubuntu's default ``--as-needed`` the conftest tolerates the unused
# .so, but ``--no-as-needed`` (which the final ICON link needs) would
# make their conftest reject it.  Instead, we append the .so directly
# to the FINAL ICON link rule in ``icon.mk`` AFTER configure -- after
# ``$(link_files)``, so the ICON module symbols are visible when the
# .so's references resolve.  Same with the .mod ``-I`` path: nothing
# in ICON ``USE``s the bindings module (the patch uses an INTERFACE
# block; see ``docs/ICON_INTEGRATION.md``) so adding it to FCFLAGS
# is dead weight.

# Ubuntu apt-installed deps (hardcoded paths -- no spack, no module load):
NETCDF_C_INC='-I/usr/include'
NETCDF_C_LIB='-L/usr/lib/x86_64-linux-gnu -lnetcdf'
NETCDF_F_INC='-I/usr/include'
NETCDF_F_LIB='-L/usr/lib/x86_64-linux-gnu -lnetcdff'
HDF5_INC='-I/usr/include/hdf5/serial'
HDF5_LIB='-L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_hl -lhdf5'
XML2_INC='-I/usr/include/libxml2'
XML2_LIB='-lxml2'
ECCODES_LIB='-leccodes -leccodes_f90'
FYAML_LIB='-lfyaml'
LAPACK_LIB='-llapack -lblas'

# MPI wrappers (system OpenMPI 5):
CC='mpicc'
CXX='mpicxx'
FC='mpifort'

# fp64 only -- no -D__MIXED_PRECISION.  Match the FP-conservative flag
# triple from the e2e tests so the SDFG-vs-ICON arithmetic order is
# identical and the produced binary is bit-exact against an unmodified
# ICON build.  Drop -O0 to -O3 for production timings (numerical
# envelope widens to ~1 ULP).
COMMON_FLAGS='-O0 -g -fno-fast-math -ffp-contract=off -fPIC'

CFLAGS="${COMMON_FLAGS}"
CXXFLAGS="${COMMON_FLAGS}"
FCFLAGS="${COMMON_FLAGS} -fbacktrace -ffree-line-length-none ${NETCDF_F_INC} ${NETCDF_C_INC} ${HDF5_INC}"
CPPFLAGS="${NETCDF_C_INC} ${HDF5_INC} ${XML2_INC}"

LDFLAGS="-L/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu/hdf5/serial"

LIBS="${XML2_LIB} ${FYAML_LIB} ${ECCODES_LIB} ${LAPACK_LIB} ${NETCDF_F_LIB} ${NETCDF_C_LIB} ${HDF5_LIB} -lstdc++"

# Dycore-focused atm-only build.  JSBach (land), ocean, and YAC
# coupling are disabled because (a) we don't need them for the
# velocity / solve_nh integration, and (b) JSBach's
# mo_pheno_process internal split causes link-time symbol
# undefineds with the stock 2026.04 release on stock-Ubuntu
# gfortran-15 -- `--disable-jsbach` makes that error class go
# away cleanly.
EXTRA_CONFIG_ARGS="\
--enable-grib2 \
--enable-loop-exchange \
--enable-openmp \
--enable-bundled-python=mtime \
--disable-jsbach \
--disable-ocean \
--disable-coupling \
--disable-waves \
"

echo "[configure_icon_dace_cpu] invoking ${icon_dir}/configure ..."

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
  # Append the DaCe library to the FINAL ICON link rule in icon.mk so
  # it comes AFTER ${link_files} (which provides the ICON module
  # symbols the .so's bind_c_shim forwards reference).  --no-as-needed
  # keeps the lib live since Ubuntu's default --as-needed would
  # discard it (the .so's symbols are referenced from a single
  # .o file that ld may process before getting to the lib).
  link_old='$(silent_FCLD)$(FC) -o $@ $(make_FCFLAGS) $(FCFLAGS) $(ICON_FCFLAGS) $(LDFLAGS) $(link_files) $(shell . ./collect.extra-libs) $(LIBS)'
  link_new="${link_old} -L${DACE_LIBS_DIR} -Wl,-rpath,${DACE_LIBS_DIR} -Wl,--no-as-needed -l:libvelocity_inner_wrap.so"
  if grep -qF -- "${link_old}" icon.mk; then
    if grep -qF -- "${link_new}" icon.mk; then
      echo "[configure_icon_dace_cpu] icon.mk link rule already DaCe-patched"
    else
      # Use python (always present per ICON's bundled-mtime requirement)
      # rather than sed -- the link line has ``$(...)`` and shell special
      # chars that would need awkward escaping.
      python3 - "${link_old}" "${link_new}" <<'PYEOF'
import pathlib, sys
p = pathlib.Path("icon.mk")
src = p.read_text()
old, new = sys.argv[1], sys.argv[2]
assert old in src
p.write_text(src.replace(old, new))
PYEOF
      echo "[configure_icon_dace_cpu] icon.mk link rule patched to append libvelocity_inner_wrap.so"
    fi
  else
    echo "WARNING: icon.mk link-line anchor not found; ICON did not produce a recognised link rule." >&2
    echo "         The DaCe library will NOT be linked into bin/icon." >&2
  fi
fi

echo ""
echo "[configure_icon_dace_cpu] configure done.  Run: make -j 8"
if test -n "${DACE_LIBS_DIR}"; then
  echo ""
  echo "ICON will link against:"
  echo "  ${DACE_LIBS_DIR}/libvelocity_inner_wrap.so"
  echo ""
  echo "Don't forget to manually patch"
  echo "  ${icon_dir}/src/atm_dyn_iconam/mo_velocity_advection.f90"
  echo "to forward the velocity_tendencies body to velocity_tendencies_dace"
  echo "(see docs/ICON_INTEGRATION.md, Step 3)."
fi
