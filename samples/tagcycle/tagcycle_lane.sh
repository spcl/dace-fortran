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
# Data layout is a sweep dimension inside a velocity rank, not a property of the TU variant.
VELOCITY_LAYOUTS="${VELOCITY_LAYOUTS:-nproma32 flat}"

# variant -> the lane roots under $BR2 that hold its .sdfgz artifacts
lane_roots() {
    case "$1" in
        cloudsc-klon) echo "cloudsc_klon_dace-gcc cloudsc_klon_dace-llvm" ;;
        cloudsc-nblks) echo "cloudsc_nblks_dace-gcc cloudsc_nblks_dace-llvm" ;;
        velocity-loopexch | velocity-noloopexch) echo "velocity_tendencies_dace-gcc velocity_tendencies_dace-llvm" ;;
        *) return 1 ;;
    esac
}

# Both velocity TU variants keep their artifacts in the per-backend root (that is where the
# baseline arglists live), so a velocity variant selects its own by filename prefix.
lane_prefix() {
    case "$1" in
        velocity-loopexch) echo "velocity_loopexch_" ;;
        velocity-noloopexch) echo "velocity_noloopexch_" ;;
        *) echo "" ;;
    esac
}

if [ "$MODE" = artifacts ]; then
    tag="${EXPECTED_TAG:?EXPECTED_TAG must be set}"
    roots="$(lane_roots "$VARIANT")" || { echo "unknown variant: $VARIANT" >&2; exit 2; }
    pfx="$(lane_prefix "$VARIANT")"
    for r in $roots; do
        for f in "$BR2/$r/$pfx"*_"$tag".sdfgz; do
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

# $2 partitions one root into private build/scratch dirs: DACE_cache=name keys the build folder on
# the SDFG name, and both velocity TU variants build an SDFG named velocity_tendencies -- without a
# suffix the two concurrent velocity ranks would compile into the same dacecache/velocity_tendencies.
setup_lane_root() {
    local root="$1" sfx="${2:-}"
    local cache="dacecache${sfx:+_$sfx}" tmp="tmp${sfx:+_$sfx}"
    mkdir -p "$root/$tmp" || return 1
    export BUILD_ROOT="$root" TMPDIR="$root/$tmp"
    export DACE_default_build_folder="$root/$cache" DACE_cache=name
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

warm_build() {
    local log="$1" r
    shift
    echo "+ $*"
    "$@" > "$log" 2>&1
    r=$?
    cat "$log"
    if grep -q '^phase A done:' "$log"; then
        [ "$r" -eq 0 ] || echo "MARKER OK despite rc=$r"
        return 0
    fi
    echo "ERR: no phase A marker in $log (rc=$r)"
    rc=1
    return 1
}

CLOUDSC_DRIVER="$MEAS/samples/cloudsc/run_cloudsc_perf.py"
VELOCITY_SRC="$MEAS/samples/velocity_tendencies"
VELOCITY_DRIVER="$VELOCITY_SRC/run_velocity_perf.py"
CLOUDSC_H5="$MEAS/samples/cloudsc/data/input.h5"
VELOCITY_DATA_DIR="${VELOCITY_DATA_DIR:-$BR2/data_r02b06}"
VELOCITY_TIMESTEP="${VELOCITY_TIMESTEP:-1}"
NPZ_NPROMA32="${VELOCITY_NPZ_NPROMA32:-$BR2/velocity_r02b06_nproma32.npz}"
NPZ_FLAT="${VELOCITY_NPZ_FLAT:-$BR2/velocity_r02b06_nproma491520.npz}"
npz_for() {
    case "$1" in
        flat) echo "$NPZ_FLAT" ;;
        *) echo "$NPZ_NPROMA32" ;;
    esac
}
nproma_for() {
    case "$1" in
        flat) echo 491520 ;;
        *) echo 32 ;;
    esac
}

# openacc-cpu is the raw Fortran baseline (driver_velocity.f90 + dump_data.py's stream-binary
# dump), not the npz the DaCe lanes above read -- same shape labels, different file format.
VELOCITY_DUMP32="${VELOCITY_DUMP32:-$BR2/velocity_dump_r02b06_nproma32}"
VELOCITY_DUMP_FLAT="${VELOCITY_DUMP_FLAT:-$BR2/velocity_dump_r02b06_nproma491520}"
dump_for() {
    case "$1" in
        flat) echo "$VELOCITY_DUMP_FLAT" ;;
        *) echo "$VELOCITY_DUMP32" ;;
    esac
}
# The driver hardcodes the dataset label of the deck it was written against.
VELOCITY_INPUTS="${VELOCITY_INPUTS:-r02b06}"

# Warm only: the flock keeps the two concurrent velocity ranks from both spending ~4.5 min (and
# tens of GB) converting the same deck.
ensure_npz() {
    local n lock
    n="$(npz_for "$1")"
    [ -f "$n" ] && return 0
    lock="$n.lock"
    mkdir -p "$(dirname "$n")" || return 1
    (
        flock 9
        [ -f "$n" ] && exit 0
        echo "+ convert_data.py --nproma $(nproma_for "$1") --out $n"
        "$PY" "$VELOCITY_SRC/convert_data.py" --data-dir "$VELOCITY_DATA_DIR" \
            --timestep "$VELOCITY_TIMESTEP" --nproma "$(nproma_for "$1")" --out "$n.partial" \
            && mv -f "$n.partial" "$n"
    ) 9> "$lock"
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
                warm_build "$BUILD_ROOT/warm_build.log" \
                    "$PY" "$CLOUDSC_DRIVER" --mode "$m" --backend "$b" --build-only
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
    velocity-loopexch | velocity-noloopexch)
        v="${VARIANT#velocity-}"
        if [ "$MODE" = warm ]; then
            for l in $VELOCITY_LAYOUTS; do
                ensure_npz "$l" || rc=1
            done
        fi
        for l in $VELOCITY_LAYOUTS; do
            [ -f "$(npz_for "$l")" ] || {
                echo "FATAL: no velocity npz at $(npz_for "$l") for layout $l" >&2
                exit 1
            }
        done
        for b in gcc llvm; do
            setup_lane_root "$BR2/velocity_tendencies_dace-${b}" "$v" || {
                rc=1
                continue
            }
            if [ "$MODE" = warm ]; then
                set_omp_lane "$WARM_THREADS" || { rc=1; continue; }
                warm_build "$BUILD_ROOT/warm_build_${v}.log" \
                    "$PY" "$VELOCITY_DRIVER" --variant "$v" --backend "$b" --build-only
            else
                # threads outer / layout inner, matching samples/.../run_velocity.sbatch.
                for t in $THREADS; do
                    if [ "$t" -gt "$LANE_NCORES" ]; then
                        echo "skip threads=$t (> $LANE_NCORES cpus in this cpuset)"
                        continue
                    fi
                    set_omp_lane "$t" || { rc=1; continue; }
                    for l in $VELOCITY_LAYOUTS; do
                        run "$PY" "$VELOCITY_DRIVER" --variant "$v" --backend "$b" \
                            --npz "$(npz_for "$l")" --csv "$CSV"
                    done
                done
            fi
        done
        ;;
    velocity-openacc)
        # shellcheck disable=SC1091
        . /capstor/scratch/cscs/ybudanaz/aarch64/spack/share/spack/setup-env.sh
        spack load nvhpc
        NVFORTRAN="$(command -v nvfortran || true)"
        [ -n "$NVFORTRAN" ] || {
            echo "FATAL: no nvfortran on PATH after spack load nvhpc" >&2
            exit 1
        }
        REPS="${REPS:-50}"
        WARMUP="${WARMUP:-2}"
        # driver_velocity's automatic transients are hundreds of MB at nproma=32, far more at
        # nproma=491520 (baselines.sh hits the same wall).
        ulimit -s unlimited 2> /dev/null || echo "warning: could not raise the stack limit" >&2
        setup_lane_root "$BR2/velocity_tendencies_openacc" || exit 1
        DRIVER="$BUILD_ROOT/driver_velocity"
        if [ "$MODE" = warm ]; then
            for l in $VELOCITY_LAYOUTS; do
                d="$(dump_for "$l")"
                [ -f "$d/manifest.txt" ] || run "$PY" "$VELOCITY_SRC/dump_data.py" --data-dir "$VELOCITY_DATA_DIR" \
                    --timestep "$VELOCITY_TIMESTEP" --nproma "$(nproma_for "$l")" --out "$d"
            done
            run "$NVFORTRAN" -O3 -acc=multicore -c "$VELOCITY_SRC/velocity_advection_acc.f90" \
                -o "$BUILD_ROOT/velocity_advection_acc.o" -module "$BUILD_ROOT"
            run "$NVFORTRAN" -O3 -acc=multicore -c "$VELOCITY_SRC/driver_velocity.f90" \
                -o "$BUILD_ROOT/driver_velocity.o" -module "$BUILD_ROOT"
            run "$NVFORTRAN" -O3 -acc=multicore "$BUILD_ROOT/velocity_advection_acc.o" \
                "$BUILD_ROOT/driver_velocity.o" -o "$DRIVER"
        else
            for l in $VELOCITY_LAYOUTS; do
                d="$(dump_for "$l")"
                [ -f "$d/manifest.txt" ] || {
                    echo "FATAL: no velocity dump at $d for layout $l (warm mode builds it)" >&2
                    rc=1
                }
            done
            [ -x "$DRIVER" ] || {
                echo "FATAL: no driver binary at $DRIVER (warm mode builds it)" >&2
                exit 1
            }
            [ "$rc" -eq 0 ] || exit "$rc"
            for t in $THREADS; do
                if [ "$t" -gt "$LANE_NCORES" ]; then
                    echo "skip threads=$t (> $LANE_NCORES cpus in this cpuset)"
                    continue
                fi
                set_omp_lane "$t" || { rc=1; continue; }
                export ACC_NUM_CORES="$t"
                for l in $VELOCITY_LAYOUTS; do
                    d="$(dump_for "$l")"
                    log="$BUILD_ROOT/last_run_${l}_${t}.log"
                    if ! "$DRIVER" "$d" openacc-cpu "$t" "$REPS" "$WARMUP" > "$log" 2>&1; then
                        echo "FATAL: driver_velocity failed (layout $l threads=$t); tail of $log:" >&2
                        tail -5 "$log" >&2
                        rc=1
                        continue
                    fi
                    [ -f "$CSV" ] || echo "kernel,mode,nproma,nblks_e,threads,rep,ms,inputs,lane" > "$CSV"
                    # The ACC twin IS the loop-exchange source, so its mode label stands; only the
                    # dataset label needs correcting, and nproma/nblks_e already carry the layout.
                    grep '^velocity_tendencies,' "$log" | sed "s/,r02b05,/,${VELOCITY_INPUTS},/" \
                        | tee -a "$CSV"
                done
            done
            unset ACC_NUM_CORES
        fi
        ;;
    velocity-openmp)
        . /capstor/scratch/cscs/ybudanaz/aarch64/spack/share/spack/setup-env.sh
        spack load gcc@16.1.0 +graphite
        spack load nvhpc
        OMP_NVHPC_ROOT="$(spack location -i nvhpc 2> /dev/null || true)"
        if [ -n "$OMP_NVHPC_ROOT" ]; then
            OMP_CUDA_HOME="$(ls -d "$OMP_NVHPC_ROOT"/Linux_aarch64/*/cuda 2> /dev/null | head -1)"
            if [ -n "$OMP_CUDA_HOME" ]; then
                export CUDA_HOME="$OMP_CUDA_HOME"
                export PATH="$CUDA_HOME/bin:$PATH"
                export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            fi
        fi
        OMP_GFORTRAN="$(command -v gfortran || true)"
        OMP_FLANG="$(command -v flang-22 || command -v flang || true)"
        OMP_NVFORTRAN="$(command -v nvfortran || true)"
        REPS="${REPS:-50}"
        WARMUP="${WARMUP:-2}"
        ulimit -s unlimited 2> /dev/null || echo "warning: could not raise the stack limit" >&2
        setup_lane_root "$BR2/velocity_tendencies_openmp" || exit 1
        if [ "$MODE" = warm ]; then
            for l in $VELOCITY_LAYOUTS; do
                d="$(dump_for "$l")"
                [ -f "$d/manifest.txt" ] || run "$PY" "$VELOCITY_SRC/dump_data.py" --data-dir "$VELOCITY_DATA_DIR" \
                    --timestep "$VELOCITY_TIMESTEP" --nproma "$(nproma_for "$l")" --out "$d"
            done
        else
            for l in $VELOCITY_LAYOUTS; do
                [ -f "$(dump_for "$l")/manifest.txt" ] || {
                    echo "FATAL: no velocity dump at $(dump_for "$l") for layout $l (warm mode builds it)" >&2
                    rc=1
                }
            done
            [ "$rc" -eq 0 ] || exit "$rc"
        fi
        for oc in gcc flang nvhpc; do
            case "$oc" in
                gcc) ofc="$OMP_GFORTRAN" oflags=(-O3 -march=native -fopenmp) ;;
                flang) ofc="$OMP_FLANG" oflags=(-O3 -fopenmp) ;;
                nvhpc) ofc="$OMP_NVFORTRAN" oflags=(-O3 -mp) ;;
            esac
            olane="original-openmp-$oc"
            odir="$BUILD_ROOT/$oc"
            if [ -z "$ofc" ]; then
                [ "$MODE" = warm ] && echo "SKIP $olane: no compiler on PATH" >&2
                continue
            fi
            ODRIVER="$odir/driver_velocity"
            if [ "$MODE" = warm ]; then
                mkdir -p "$odir" || { rc=1; continue; }
                run "$ofc" "${oflags[@]}" -c "$VELOCITY_SRC/velocity_advection_acc.f90" \
                    -o "$odir/velocity_advection_acc.o" -module "$odir"
                run "$ofc" "${oflags[@]}" -c "$VELOCITY_SRC/driver_velocity.f90" \
                    -o "$odir/driver_velocity.o" -module "$odir"
                run "$ofc" "${oflags[@]}" "$odir/velocity_advection_acc.o" "$odir/driver_velocity.o" -o "$ODRIVER"
                continue
            fi
            [ -x "$ODRIVER" ] || {
                echo "FATAL: no driver binary at $ODRIVER (warm mode builds it)" >&2
                rc=1
                continue
            }
            for t in $THREADS; do
                if [ "$t" -gt "$LANE_NCORES" ]; then
                    echo "skip threads=$t (> $LANE_NCORES cpus in this cpuset)"
                    continue
                fi
                set_omp_lane "$t" || { rc=1; continue; }
                for l in $VELOCITY_LAYOUTS; do
                    d="$(dump_for "$l")"
                    log="$odir/last_run_${l}_${t}.log"
                    if ! "$ODRIVER" "$d" "$olane" "$t" "$REPS" "$WARMUP" > "$log" 2>&1; then
                        echo "FATAL: driver_velocity failed ($olane layout $l threads=$t); tail of $log:" >&2
                        tail -5 "$log" >&2
                        rc=1
                        continue
                    fi
                    [ -f "$CSV" ] || echo "kernel,mode,nproma,nblks_e,threads,rep,ms,inputs,lane" > "$CSV"
                    grep '^velocity_tendencies,' "$log" | sed "s/,r02b05,/,${VELOCITY_INPUTS},/" \
                        | tee -a "$CSV"
                done
            done
        done
        ;;
    *)
        echo "unknown variant: $VARIANT" >&2
        exit 2
        ;;
esac

echo "LANE_${VARIANT}_${MODE}_EXIT=$rc"
exit "$rc"
