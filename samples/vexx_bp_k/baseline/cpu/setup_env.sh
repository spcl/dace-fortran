#!/bin/bash
# Toolchain environment for the vexx_bp_k CPU (OpenMP) baseline.
#
#   . ./setup_env.sh        # SOURCE it -- do not execute
#   ./build.sh
#
# Executing this in a subshell would load the modules into that subshell and
# then throw them away, so it guards against that below.
#
# Loads an all-GCC stack: mpif90 -> gfortran 11.2, plus FFTW and OpenBLAS built
# with the same gcc-11.  The point is that the resulting binary carries exactly
# one OpenMP runtime (libgomp) and one ABI -- see the toolchain block in
# build.sh for why mixing in nvfortran breaks the timings.

# ------------------------------------------------------------- sanity: sourced?
# $0 is the script name only when executed; when sourced it stays the shell.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "ERROR: source this script, don't execute it:" >&2
  echo "         . ${BASH_SOURCE[0]}" >&2
  exit 1
fi

# ------------------------------------------- make `module` exist in this shell
# `module` is a shell function from the login profile, so it is absent in
# non-interactive shells -- notably inside a PBS job script.  Re-source its
# init if needed; MODULESHOME points at the Cray PE modules install here.
if ! type module >/dev/null 2>&1; then
  for _init in "${MODULESHOME:-/opt/cray/pe/modules/3.2.11.6}/init/bash" \
               /usr/share/Modules/init/bash \
               /etc/profile.d/modules.sh; do
    if [ -r "$_init" ]; then . "$_init"; break; fi
  done
  unset _init
fi
if ! type module >/dev/null 2>&1; then
  echo "ERROR: no 'module' command and no modules init script found" >&2
  return 1
fi

# ---------------------------------------------------- clear the conflicting set
# Every openmpi/fftw/openblas modulefile declares `conflict <name>`, so loading
# ours while another version of the same name is loaded is REFUSED -- the old
# module simply stays on PATH and the build then picks up the wrong wrapper.
# Unload by bare name first; that is a no-op when nothing is loaded.
#
# nvhpc additionally conflicts with the GNU PrgEnv and -- the trap that cost us
# a build -- exports FC=<prefix>/bin/nvfortran.  Unloading reverses that setenv,
# but unset FC/F77/F90 explicitly too, in case NVHPC was set up by hand.
for _m in nvhpc nvhpc-nompi nvhpc-byo-compiler openmpi fftw openblas; do
  if module list 2>&1 | grep -q "\b$_m/"; then
    echo "== unloading pre-loaded $_m =="
    module unload "$_m" 2>/dev/null || true
  fi
done
unset _m
unset FC F77 F90
# OpenMPI's wrappers let these override the compiler they were configured with;
# a stale OMPI_FC=nvfortran would defeat the gcc-built wrapper just as loudly.
unset OMPI_FC OMPI_F77 OMPI_F90 OMPI_CC

# ------------------------------------------------------------------- the stack
# Order matters only in that openmpi pulls in PrgEnv-gnu and swaps to
# gcc/11.2.0; fftw and openblas are plain library modules.
# NOTE: no `set -e` here -- this script is sourced, so a failing command would
# kill the caller's interactive shell.  Each load is checked explicitly instead.
for _spec in openmpi/4.1.7-gcc11 fftw/3.3.10-gcc11 openblas/0.3.23; do
  module load "$_spec"
  # Trust `module list`, not the exit status: these modulefiles report a
  # refused load on stderr while still exiting 0.
  if ! module list 2>&1 | grep -q "$_spec"; then
    echo "ERROR: '$_spec' failed to load -- run 'module list' and unload whatever" >&2
    echo "       conflicts with it, then source this script again." >&2
    unset _spec
    return 1
  fi
done
unset _spec

# --------------------------------------------------------------- runtime knobs
# The kernel parallelises itself with OpenMP.  FFTW and OpenBLAS are both the
# OpenMP-threaded builds, so left alone they would spawn nested teams inside
# the kernel's own region and oversubscribe the cores -- which shows up as
# noisy, inflated timings rather than an outright failure.  Pin them to 1.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${NCPUS:-1}}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}

# ------------------------------------------- verify the wrapper's real backend
# The failure this guards against: an nvfortran-configured mpif90 left on PATH
# by a previously loaded openmpi module.  `mpif90 -show` prints the compiler it
# will actually exec, which is the only trustworthy check -- the wrapper's own
# name says nothing about which compiler sits behind it.
_backend=$(mpif90 -show 2>/dev/null | awk '{print $1}')
case "$_backend" in
  gfortran|*/gfortran) ;;
  '') echo "ERROR: no working mpif90 on PATH after loading the modules." >&2
      unset _backend; return 1 ;;
  *)  echo "ERROR: mpif90 ($(command -v mpif90)) is backed by '$_backend', not gfortran." >&2
      echo "       A non-GNU openmpi module is still ahead on PATH; 'module list'," >&2
      echo "       unload it, then source this script again." >&2
      unset _backend; return 1 ;;
esac
unset _backend

# ------------------------------------------------------------------ report back
echo "== toolchain =="
echo "  mpif90      : $(command -v mpif90 || echo '** NOT FOUND **')"
echo "  backend     : $(mpif90 -show 2>/dev/null | awk '{print $1}')"
echo "  gfortran    : $(gfortran -dumpversion 2>/dev/null || echo '** NOT FOUND **')"
echo "  FFTW        : ${NSCC_FFTW_DIR:-** NOT SET **}"
echo "  OpenBLAS    : ${NSCC_OPENBLAS_DIR:-** NOT SET **}"
echo "  OMP threads : $OMP_NUM_THREADS (OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS)"
if [ -z "$PBS_JOBID" ]; then
  echo
  echo "NOTE: not inside a PBS job -- build and run on a compute node, e.g."
  echo "      qsub -I -l select=1:ncpus=<N> -l walltime=01:00:00 -q <queue> -P <project>"
fi
