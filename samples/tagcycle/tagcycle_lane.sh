#!/usr/bin/env bash
# One rank = one VARIANT of the tag cycle. Three modes:
#   warm      phase A only (--build-only); needs a WRITABLE clone (the bridge takes an flock)
#   meas      the timed sweep; needs a FROZEN clone (a cache miss then dies on the lock, loudly)
#   artifacts print the .sdfgz this variant owns at $EXPECTED_TAG, one per line
# Never calls srun itself -- the caller decides whether this runs inside a step or in the batch body.
set -uo pipefail

MODE="${1:?usage: tagcycle_lane.sh <warm|meas|artifacts> <variant>}"
VARIANT="${2:?usage: tagcycle_lane.sh <warm|meas|artifacts> <variant>}"

MEAS="${MEAS:-/capstor/scratch/cscs/ybudanaz/aarch64/dace-fortran-meas}"
BR2="${BR2:-/capstor/scratch/cscs/ybudanaz/aarch64/dace-fortran-samples-meas}"
VELOCITY_VARIANTS="${VELOCITY_VARIANTS:-loopexch noloopexch}"

# variant -> the lane roots under $BR2 that hold its .sdfgz artifacts
lane_roots() {
    case "$1" in
        cloudsc-klon) echo "cloudsc_klon_dace-gcc cloudsc_klon_dace-llvm" ;;
        cloudsc-nblks) echo "cloudsc_nblks_dace-gcc cloudsc_nblks_dace-llvm" ;;
        velocity-gcc) echo "velocity_tendencies_dace-gcc" ;;
        velocity-llvm) echo "velocity_tendencies_dace-llvm" ;;
        *) return 1 ;;
    esac
}

if [ "$MODE" = artifacts ]; then
    tag="${EXPECTED_TAG:?EXPECTED_TAG must be set}"
    roots="$(lane_roots "$VARIANT")" || { echo "unknown variant: $VARIANT" >&2; exit 2; }
    for r in $roots; do
        for f in "$BR2/$r"/*_"$tag".sdfgz; do
            [ -f "$f" ] && printf '%s\n' "$f"
        done
    done
    exit 0
fi

cd "$MEAS" || exit 1
ulimit -c 0 # the clone already carries ~10 GB of core dumps from an earlier abort
# common.sh sources samples/env.sh and turns on -e; the lane keeps going past a red point instead.
# shellcheck disable=SC1091
. "$MEAS/samples/common.sh"
set +e

TAG_NOW="$(git -C "$MEAS" describe --always --dirty)"
[ "$TAG_NOW" = "${EXPECTED_TAG:?EXPECTED_TAG must be set}" ] || {
    echo "ABORT: clone tag is $TAG_NOW, expected $EXPECTED_TAG" >&2
    exit 1
}

# A view argument silently reinterprets a non-contiguous array; never on in a measurement.
export DACE_compiler_allow_view_arguments=false
probe_compilers

# CPU ids this process may actually touch. NOT lscpu (common.sh probe_topology): that reports all
# 288 node cpus, and an OMP_PLACES id outside the step's cpuset is the same EINVAL class of bug
# that numactl --cpunodebind hits inside an srun step.
lane_cpus() {
    local part lo hi
    for part in $(awk '/^Cpus_allowed_list:/ { print $2 }' /proc/self/status | tr ',' ' '); do
        case "$part" in
            *-*)
                lo="${part%-*}" hi="${part#*-}"
                seq "$lo" "$hi"
                ;;
            *) echo "$part" ;;
        esac
    done
}
LANE_CPUS="$(lane_cpus | tr '\n' ' ')"
LANE_NCORES="$(echo "$LANE_CPUS" | wc -w)"
echo "lane $VARIANT: $LANE_NCORES cpus in cpuset: $LANE_CPUS"

set_omp_lane() {
    local n="$1" c i=0 out=""
    for c in $LANE_CPUS; do
        [ "$i" -ge "$n" ] && break
        out="${out:+$out,}{$c}"
        i=$((i + 1))
    done
    [ "$i" -eq "$n" ] || {
        echo "FATAL: cpuset holds $i cpus, need $n" >&2
        return 1
    }
    export OMP_PLACES="$out" OMP_NUM_THREADS="$n" OMP_PROC_BIND=close
    export OPENBLAS_NUM_THREADS="$n" MKL_NUM_THREADS="$n" BLIS_NUM_THREADS="$n"
}

setup_lane_root() {
    local root="$1"
    mkdir -p "$root/tmp" || return 1
    export BUILD_ROOT="$root" TMPDIR="$root/tmp"
    export DACE_default_build_folder="$root/dacecache" DACE_cache=name
}

WARM_THREADS="$LANE_NCORES"
[ "$WARM_THREADS" -gt 72 ] && WARM_THREADS=72

# Preflight. srun starts tasks with SIGCHLD blocked; sdfg.compile() shells out to cmake, which
# reaps its configure helpers through SIGCHLD and otherwise spins in select() until the walltime.
# dace's unblock shim lives in codegen/compiler.py -- present on `extended`, ABSENT on `FaCe`.
# Only the combination is fatal, so the guard tests exactly that combination.
"$PY" -c "
import inspect, signal, sys
import dace, dace.codegen.compiler as c
blocked = signal.SIGCHLD in signal.pthread_sigmask(signal.SIG_BLOCK, [])
shim = 'SIGCHLD' in inspect.getsource(c)
print('DACE_PATH=' + dace.__file__)
print('SIGCHLD_BLOCKED=%d DACE_SIGCHLD_SHIM=%d' % (blocked, shim))
sys.exit(1 if (blocked and not shim) else 0)
"
if [ $? -ne 0 ]; then
    echo "ABORT: this dace has no SIGCHLD unblock shim and SIGCHLD is blocked -- every" >&2
    echo "  sdfg.compile() would hang in cmake configure. Point venv-meas at a dace that" >&2
    echo "  carries dace/codegen/compiler.py _build_subprocess_sigmask, or run this lane" >&2
    echo "  outside an srun step." >&2
    exit 1
fi

rc=0
run() { # record the first-worst exit in $rc; a red point must not cost the lane its siblings
    local r
    echo "+ $*"
    "$@"
    r=$?
    [ "$r" -eq 0 ] || rc="$r"
    return 0
}

CLOUDSC_DRIVER="$MEAS/samples/cloudsc/run_cloudsc_perf.py"
VELOCITY_DRIVER="$MEAS/samples/velocity_tendencies/run_velocity_perf.py"
CLOUDSC_H5="$MEAS/samples/cloudsc/data/input.h5"
NPZ="${VELOCITY_NPZ:-$BR2/velocity_r02b05_nproma32.npz}"
NPZ_NOLOOPEXCH="${VELOCITY_NPZ_NOLOOPEXCH:-$BR2/velocity_r02b05_nproma30720.npz}"
npz_for() {
    case "$1" in
        noloopexch) echo "$NPZ_NOLOOPEXCH" ;;
        *) echo "$NPZ" ;;
    esac
}

if [ "$MODE" = meas ]; then
    : "${CSV:?meas mode needs CSV=<path>}"
    [ "${DACE_FORTRAN_NO_REBUILD:-0}" = 1 ] || {
        echo "ABORT: meas mode requires DACE_FORTRAN_NO_REBUILD=1" >&2
        exit 1
    }
fi

case "$VARIANT" in
    cloudsc-klon | cloudsc-nblks)
        m="${VARIANT#cloudsc-}"
        # A frozen clone cannot download a deck, and a missing one only WARNs before switching to
        # synthetic inputs -- which would silently measure a different problem.
        [ -f "$CLOUDSC_H5" ] || {
            echo "FATAL: no cloudsc deck at $CLOUDSC_H5" >&2
            exit 1
        }
        for lane in dace-gcc dace-llvm; do
            b="${lane#dace-}"
            setup_lane_root "$BR2/cloudsc_${m}_${lane}" || {
                rc=1
                continue
            }
            if [ "$MODE" = warm ]; then
                set_omp_lane "$WARM_THREADS" || { rc=1; continue; }
                run "$PY" "$CLOUDSC_DRIVER" --mode "$m" --backend "$b" --build-only
            else
                for t in $THREADS; do
                    if [ "$t" -gt "$LANE_NCORES" ]; then
                        echo "skip threads=$t (> $LANE_NCORES cpus in this cpuset)"
                        continue
                    fi
                    set_omp_lane "$t" || { rc=1; continue; }
                    run "$PY" "$CLOUDSC_DRIVER" --mode "$m" --backend "$b" --csv "$CSV"
                done
            fi
        done
        ;;
    velocity-gcc | velocity-llvm)
        b="${VARIANT#velocity-}"
        [ -f "$NPZ" ] && [ -f "$NPZ_NOLOOPEXCH" ] || {
            echo "FATAL: no velocity npz at $NPZ or $NPZ_NOLOOPEXCH" >&2
            exit 1
        }
        setup_lane_root "$BR2/velocity_tendencies_dace-${b}" || exit 1
        if [ "$MODE" = warm ]; then
            set_omp_lane "$WARM_THREADS" || exit 1
            for v in $VELOCITY_VARIANTS; do
                run "$PY" "$VELOCITY_DRIVER" --variant "$v" --backend "$b" \
                    --npz "$(npz_for "$v")" --build-only
            done
        else
            # threads outer / TU variant inner, matching samples/.../run_velocity.sbatch.
            for t in $THREADS; do
                if [ "$t" -gt "$LANE_NCORES" ]; then
                    echo "skip threads=$t (> $LANE_NCORES cpus in this cpuset)"
                    continue
                fi
                set_omp_lane "$t" || { rc=1; continue; }
                for v in $VELOCITY_VARIANTS; do
                    run "$PY" "$VELOCITY_DRIVER" --variant "$v" --backend "$b" \
                        --npz "$(npz_for "$v")" --csv "$CSV"
                done
            done
        fi
        ;;
    *)
        echo "unknown variant: $VARIANT" >&2
        exit 2
        ;;
esac

echo "LANE_${VARIANT}_${MODE}_EXIT=$rc"
exit "$rc"
