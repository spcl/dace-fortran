#!/usr/bin/env bash
# Explicit OMP_PLACES id-list (not OMP_PLACES=cores): needs a hyperthread-free,
# NUMA-node-major placement that survives unknown socket/NPS configs.
set -euo pipefail

# Site env hook, sourced before any probe below so a module/spack PATH is visible to all of them:
# SAMPLES_ENV=<file>, else samples/env.sh when it exists. An explicit SAMPLES_ENV must resolve.
SAMPLES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SAMPLES_ENV:-}" ]; then
    if [ ! -f "$SAMPLES_ENV" ]; then
        echo "FATAL: SAMPLES_ENV=$SAMPLES_ENV does not exist" >&2
        exit 1
    fi
elif [ -f "$SAMPLES_DIR/env.sh" ]; then
    SAMPLES_ENV="$SAMPLES_DIR/env.sh"
fi
# Once per job: srun steps inherit the exports, re-sourcing would only re-run `spack load`.
if [ -n "${SAMPLES_ENV:-}" ] && [ -z "${SAMPLES_ENV_LOADED:-}" ]; then
    echo "samples: sourcing env hook $SAMPLES_ENV"
    # shellcheck disable=SC1090
    . "$SAMPLES_ENV"
    SAMPLES_ENV_LOADED=1
    export SAMPLES_ENV SAMPLES_ENV_LOADED
fi

# Pooled allocator for every lane (mimalloc by default, MEAS_ALLOC=system opts out). Sourced HERE,
# after the env hook has put spack on PATH but before any lane launches the python that dlopens a
# dacecache .so -- LD_PRELOAD only takes effect at process start, so a later export would miss it.
# Every entry point reaches this file (run_*.sbatch, both baselines.sh, tagcycle_lane.sh), so this
# is the one edit that covers all of them.
# shellcheck disable=SC1091
. "$SAMPLES_DIR/alloc_pool.sh"

# Batch jobs don't inherit the interactive PATH, so a bare `python` may lack ordered_set/dace.
PY="${PYTHON:-$HOME/.pyenv/versions/py13/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
"$PY" -c 'import ordered_set' 2>/dev/null || {
    echo "FATAL: $PY cannot import ordered_set -- wrong interpreter, set PYTHON=" >&2
    exit 1
}
export PY

export PYTHONHASHSEED=0 # dace codegen determinism (stable artifact reuse across invocations)
export PYTHONUNBUFFERED=1
# MPI-inerting: keep any mpi4py import from probing fabrics/GL on a compute node.
export MPI4PY_RC_INITIALIZE=0
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader
export UCX_VFS_ENABLE=n
export HWLOC_COMPONENTS=-gl

THREADS="${THREADS:-1 4 8 16 32 64}"
export THREADS

# --- build root --------------------------------------------------------------
# /tmp is tmpfs; build on disk, in a root private to the given tag (two modes running
# concurrently must not share a dacecache). DACE_cache=name (not unique) ON PURPOSE: one SDFG
# name per root, and reuse across the thread-loop invocations is exactly what Phase A caches.
# BUILD_ROOT_BASE is the job-wide root the per-lane tags hang off; a lane that must not inherit the
# previous lane's per-lane root unsets BUILD_ROOT and lands under the base rather than back in $HOME.
setup_build_root() {
    local tag="$1"
    BUILD_ROOT="${BUILD_ROOT:-${BUILD_ROOT_BASE:-$HOME/.cache/dace-fortran-samples}/$tag}"
    mkdir -p "$BUILD_ROOT/tmp"
    export TMPDIR="$BUILD_ROOT/tmp"
    export DACE_default_build_folder="$BUILD_ROOT/dacecache"
    export DACE_cache=name
    export BUILD_ROOT
}

# Two JOBS can be aimed at one BUILD_ROOT on purpose (cpu_warm.sbatch submits CHUNK=small and
# CHUNK=large as separate concurrent jobs, and the non-DaCe cloudsc lanes share BUILD_ROOT_BASE
# warm->meas so the binary survives).  Compiling twice in one directory races gfortran's temp
# modules ("Cannot delete temporary module file 'parkind1.mod0'") and executing a binary another
# job is relinking gives "Text file busy", so build+run is a critical section.  The lock is held
# on fd 9 (same idiom as tagcycle_lane.sh ensure_npz) for the life of the calling shell, and the
# kernel drops it if that shell dies -- a crashed job cannot wedge the next one.
lock_build_root() {
    local lock="${1:-$BUILD_ROOT/.build.lock}"
    mkdir -p "$(dirname "$lock")" || return 1
    exec 9> "$lock"
    if ! flock -n 9; then
        echo "waiting for $lock (another job is building/running in this root)"
        flock 9
    fi
    echo "holding $lock"
}

# --- topology ----------------------------------------------------------------
# Sets: TOPO_NSOCKETS, TOPO_NNUMA, TOPO_NCORES, TOPO_PHYS_CPUS (one CPU id per physical core,
# lowest sibling, ordered NUMA-node-major then CPU id). Never hardcoded: the target node may be
# a 1- or 2-socket EPYC in any NPS mode, so this runs at job runtime.
probe_topology() {
    local lines
    if ! lines="$(lscpu -p=CPU,CORE,SOCKET,NODE 2>/dev/null | grep -v '^#')"; then
        # Fallback (no parseable -p output): assume linear ids, no SMT, one socket/node.
        echo "probe_topology: lscpu -p failed, assuming $(nproc) flat physical cores" >&2
        lines="$(awk -v n="$(nproc)" 'BEGIN { for (i = 0; i < n; i++) printf "%d,%d,0,0\n", i, i }')"
    fi
    # First occurrence of a (socket,core) pair = lowest CPU id of that core (lscpu lists CPUs
    # ascending), i.e. hyperthread siblings drop out.
    TOPO_PHYS_CPUS="$(printf '%s\n' "$lines" \
        | awk -F, '!seen[$3 ":" $2]++ { print ($4 == "" ? 0 : $4), $1 }' \
        | sort -n -k1,1 -k2,2 | awk '{ printf "%s%s", sep, $2; sep = " " } END { print "" }')"
    TOPO_NSOCKETS="$(printf '%s\n' "$lines" | awk -F, '!s[$3]++ { n++ } END { print n }')"
    TOPO_NNUMA="$(printf '%s\n' "$lines" | awk -F, '!s[$4]++ { n++ } END { print n }')"
    TOPO_NCORES="$(printf '%s\n' "$TOPO_PHYS_CPUS" | wc -w)"
    export TOPO_PHYS_CPUS TOPO_NSOCKETS TOPO_NNUMA TOPO_NCORES
}

# Explicit OMP_PLACES list "{c0},{c1},..." of nthreads physical-core CPU ids. NUMA-node-major
# order means close threads fill one NUMA region before spilling to the next (holds for
# NPS1/2/4 alike -- the order derives from the probed NODE column, not an assumed layout).
omp_places_close() {
    local n="$1" out="" c i=0
    [ -n "${TOPO_PHYS_CPUS:-}" ] || probe_topology
    for c in $TOPO_PHYS_CPUS; do
        [ "$i" -ge "$n" ] && break
        out="${out:+$out,}{$c}"
        i=$((i + 1))
    done
    if [ "$i" -lt "$n" ]; then
        echo "omp_places_close: only $i physical cores for $n threads" >&2
        return 1
    fi
    printf '%s\n' "$out"
}

# Lane-matrix probe: exports <TOOL> + HAVE_<TOOL>=0/1; presets win, flang order mirrors tests/_util.py.
probe_compilers() {
    local spec var name path
    CLANGXX="${CLANGXX:-$(command -v clang++-21 || command -v clang++-22 || command -v clang++ || true)}"
    FLANG="${FLANG:-$(command -v flang-new-21 || command -v flang-21 || command -v flang-new-22 \
        || command -v flang-22 || command -v flang-new || command -v flang || true)}"
    for spec in GFORTRAN:gfortran GCC:gcc CLANGXX:clang++ FLANG:flang NVCC:nvcc \
        NVFORTRAN:nvfortran NVC:nvc H5FC:h5fc H5CC:h5cc; do
        var="${spec%%:*}" name="${spec##*:}"
        path="${!var:-$name}"
        path="$(command -v "$path" 2>/dev/null || true)"
        printf -v "$var" '%s' "$path"
        if [ -n "$path" ]; then
            printf -v "HAVE_$var" '%s' 1
        else
            printf -v "HAVE_$var" '%s' 0
        fi
        export "${var?}" "HAVE_${var?}"
        echo "probe_compilers: $name ${path:-MISSING}"
    done
}

# HDF5 prefix -> the libdir that actually holds libhdf5 (spack uses lib, some distros lib64).
hdf5_libdir() {
    local root="$1" d
    for d in lib64 lib; do
        if compgen -G "$root/$d/libhdf5.*" > /dev/null; then
            printf '%s' "$root/$d"
            return 0
        fi
    done
    printf '%s' "$root/lib"
}

# -L plus rpath for one HDF5 libdir. The lanes link three different HDF5 builds (one per Fortran
# compiler); rpath makes each binary resolve ITS own libhdf5*.so instead of whichever prefix
# happens to sit on LD_LIBRARY_PATH.
hdf5_ldflags() { printf -- '-L%s -Wl,-rpath,%s' "$1" "$1"; }

# Export the full OMP + BLAS thread environment for one sweep point.
set_omp_env() {
    local n="$1"
    OMP_PLACES="$(omp_places_close "$n")"
    export OMP_PLACES
    export OMP_NUM_THREADS="$n"
    export OMP_PROC_BIND=close
    export OPENBLAS_NUM_THREADS="$n" MKL_NUM_THREADS="$n" BLIS_NUM_THREADS="$n"
}
