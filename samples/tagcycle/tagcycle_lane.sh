#!/usr/bin/env bash
# One rank = one VARIANT of the tag cycle. Three modes:
#   warm      phase A only (--build-only); builds into $BR2, never into the repo
#   meas      the timed sweep; runs under DACE_FORTRAN_NO_REBUILD, so a cache miss dies loudly
#   artifacts print the .sdfgz this variant owns at $EXPECTED_TAG, one per line
# Never calls srun itself -- the caller decides whether this runs inside a step or in the batch body.
set -uo pipefail

MODE="${1:?usage: tagcycle_lane.sh <warm|meas|artifacts> <variant>}"
VARIANT="${2:?usage: tagcycle_lane.sh <warm|meas|artifacts> <variant>}"

# Always invoked by path out of the repo (never spooled), so BASH_SOURCE is the real file.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-$REPO/samples/_work}"
# MEAS/BR2 keep their pinned-mirror names: the tree under test (now the repo) and the output root.
MEAS="${MEAS:-$REPO}"
BR2="${BR2:-$WORK_ROOT/meas}"
# Data layout is a sweep dimension inside a velocity rank, not a property of the TU variant.
# CPU default = both blocked decks (r02b05 and r02b06 at nproma=32), which is what the paper's
# velocity panels need; `flat` (nproma=491520) is a GPU layout and is opt-in here.
VELOCITY_LAYOUTS="${VELOCITY_LAYOUTS:-r02b05 r02b06}"
REPS="${REPS:-50}"
WARMUP="${WARMUP:-2}"

# variant -> the lane roots under $BR2 that hold its .sdfgz artifacts
lane_roots() {
    case "$1" in
        cloudsc-klon) echo "cloudsc_klon_dace-gcc cloudsc_klon_dace-llvm" ;;
        cloudsc-nblks) echo "cloudsc_nblks_dace-gcc cloudsc_nblks_dace-llvm" ;;
        cloudsc-sweep) echo "cloudsc_sweep_dace-gcc cloudsc_sweep_dace-llvm" ;;
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
ulimit -c 0 # a lane abort used to leave ~10 GB of core dumps in the tree
# common.sh sources samples/env.sh and turns on -e; the lane keeps going past a red point instead.
# shellcheck disable=SC1091
. "$MEAS/samples/common.sh"
set +e

TAG_NOW="$(git -C "$MEAS" describe --always --dirty)"
[ "$TAG_NOW" = "${EXPECTED_TAG:?EXPECTED_TAG must be set}" ] || {
    echo "ABORT: repo describes as $TAG_NOW, expected $EXPECTED_TAG" >&2
    exit 1
}

# A view argument silently reinterprets a non-contiguous array; never on in a measurement.
export DACE_compiler_allow_view_arguments=false
probe_compilers

# common.sh sourced alloc_pool.sh, so the preload (if any) is already exported and every $PY the
# lane launches inherits it. The driver-emitted CSV rows carry the 9 original columns; the
# Fortran/OpenACC drivers cannot know the allocator, so the lane stamps it on.
ALLOC="${MEAS_ALLOC_ACTIVE:-system}"
echo "lane $VARIANT: alloc=$ALLOC${MEAS_ALLOC_SO:+ preload=$MEAS_ALLOC_SO}"

# Refuse to MEASURE a mixed-OpenMP-runtime process. libgomp and libomp in one process each see
# only their own threads, oversubscribe the cpuset and cost ~34x -- and the run still completes
# with plausible-looking rows, so it has to be an abort, not a warning. Runs in this environment
# (after alloc_pool), so the mapping it checks is the one the timed run will get.
omp_preflight() {
    local backend="$1" so="$2"
    # warm mode is what builds the .so in the first place; nothing to check yet.
    [ "$MODE" = meas ] || return 0
    "$PY" "$MEAS/samples/omp_preflight.py" --so "$so" --expect "$backend" --label "$VARIANT/dace-$backend"
}

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
    echo "  sdfg.compile() would hang in cmake configure. Point \$PYTHON at a dace that" >&2
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

# `spack` is a shell FUNCTION from setup-env.sh; samples/env.sh normally defines it already.
# SPACK_ROOT (which env.sh also exports) is the only path input -- nothing here is hardcoded.
load_spack() {
    command -v spack > /dev/null 2>&1 && return 0
    local s="${SPACK_SETUP_ENV:-${SPACK_ROOT:+$SPACK_ROOT/share/spack/setup-env.sh}}"
    [ -n "$s" ] && [ -f "$s" ] || {
        echo "FATAL: no spack on PATH -- set SPACK_ROOT or SPACK_SETUP_ENV in samples/env.sh" >&2
        return 1
    }
    # shellcheck disable=SC1090
    . "$s"
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
VELOCITY_DATA_DIR_R05="${VELOCITY_DATA_DIR_R05:-$VELOCITY_SRC/data_r02b05}"
VELOCITY_TIMESTEP="${VELOCITY_TIMESTEP:-1}"
NPZ_R02B05="${VELOCITY_NPZ_R02B05:-$BR2/velocity_r02b05_nproma32.npz}"
NPZ_NPROMA32="${VELOCITY_NPZ_NPROMA32:-$BR2/velocity_r02b06_nproma32.npz}"
NPZ_FLAT="${VELOCITY_NPZ_FLAT:-$BR2/velocity_r02b06_nproma491520.npz}"
# Layout -> deck.  `nproma32` is kept as an alias of `r02b06` so an older VELOCITY_LAYOUTS
# export keeps meaning what it meant.
npz_for() {
    case "$1" in
        r02b05) echo "$NPZ_R02B05" ;;
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
# The raw ICON dump each layout is converted from; r02b05 ships in the repo, r02b06 is staged
# into $BR2 by prepare_r02b06.sh (see the warm job).
data_dir_for() {
    case "$1" in
        r02b05) echo "$VELOCITY_DATA_DIR_R05" ;;
        *) echo "$VELOCITY_DATA_DIR" ;;
    esac
}
# CSV `inputs` label; the DaCe driver derives it from the npz name, the Fortran drivers do not.
inputs_for() {
    case "$1" in
        r02b05) echo r02b05 ;;
        *) echo r02b06 ;;
    esac
}

# openacc-cpu is the raw Fortran baseline (driver_velocity.f90 + dump_data.py's stream-binary
# dump), not the npz the DaCe lanes above read -- same shape labels, different file format.
VELOCITY_DUMP05="${VELOCITY_DUMP05:-$BR2/velocity_dump_r02b05_nproma32}"
VELOCITY_DUMP32="${VELOCITY_DUMP32:-$BR2/velocity_dump_r02b06_nproma32}"
VELOCITY_DUMP_FLAT="${VELOCITY_DUMP_FLAT:-$BR2/velocity_dump_r02b06_nproma491520}"
dump_for() {
    case "$1" in
        r02b05) echo "$VELOCITY_DUMP05" ;;
        flat) echo "$VELOCITY_DUMP_FLAT" ;;
        *) echo "$VELOCITY_DUMP32" ;;
    esac
}

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
        "$PY" "$VELOCITY_SRC/convert_data.py" --data-dir "$(data_dir_for "$1")" \
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
                omp_preflight "$b" "$BUILD_ROOT/dacecache/cloudscouter/build/libcloudscouter.so" || {
                    echo "ABORT: OpenMP purity preflight failed for $lane -- refusing to measure" >&2
                    rc=1
                    continue
                }
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
    cloudsc-sweep)
        # Figure-A problem-size sweep for the DaCe lanes: one SDFG per (NGPTOT, backend), because
        # KLON/NBLOCKS are specialized into the graph.  Phase A is the expensive half (~13 min a
        # build), so the warm job shards CLOUDSC_SWEEP_SIZES/CLOUDSC_SWEEP_BACKENDS across ranks.
        # shellcheck disable=SC1091
        . "$MEAS/samples/cloudsc/sweep_sizes.sh"
        [ -f "$CLOUDSC_H5" ] || {
            echo "FATAL: no cloudsc deck at $CLOUDSC_H5" >&2
            exit 1
        }
        cloudsc_print_size_table || echo "WARNING: continuing with a cliff-flagged NPROMA" >&2
        for b in ${CLOUDSC_SWEEP_BACKENDS:-gcc llvm}; do
            for s in $CLOUDSC_SWEEP_SIZES; do
                for n in $CLOUDSC_SWEEP_NPROMAS; do
                    setup_lane_root "$BR2/cloudsc_sweep_dace-${b}" "n${n}_s${s}" || {
                        rc=1
                        continue
                    }
                    if [ "$MODE" = warm ]; then
                        set_omp_lane "$WARM_THREADS" || { rc=1; continue; }
                        warm_build "$BUILD_ROOT/warm_build_n${n}_s${s}.log" \
                            "$PY" "$CLOUDSC_DRIVER" --mode nblks --ngptot "$s" --nproma "$n" \
                            --backend "$b" --build-only
                        continue
                    fi
                    omp_preflight "$b" \
                        "$BUILD_ROOT/dacecache_n${n}_s${s}/cloudscouter/build/libcloudscouter.so" || {
                        echo "ABORT: OpenMP purity preflight failed for dace-$b (ngptot $s) -- refusing" >&2
                        rc=1
                        continue
                    }
                    for t in $THREADS; do
                        if [ "$t" -gt "$LANE_NCORES" ]; then
                            echo "skip threads=$t (> $LANE_NCORES cpus in this cpuset)"
                            continue
                        fi
                        set_omp_lane "$t" || { rc=1; continue; }
                        run "$PY" "$CLOUDSC_DRIVER" --mode nblks --ngptot "$s" --nproma "$n" \
                            --backend "$b" --reps "$REPS" --warmup "$WARMUP" --csv "$CSV"
                    done
                done
            done
        done
        ;;
    cloudsc-sweep-fortran | cloudsc-sweep-c)
        # The two non-DaCe sweep lanes are just baselines.sh in sweep mode: it already owns the
        # dwarf build recipes, the h5 deck wiring and the 10-column CSV contract, and neither lane
        # has a phase A, so warm only has to produce the binary (a 4096-column smoke run does that
        # and proves it runs).  BUILD_ROOT_BASE is shared warm->meas so the binary survives.
        case "$VARIANT" in
            cloudsc-sweep-fortran) blane=original-openmp ;;
            *) blane=c-openmp ;;
        esac
        [ -f "$CLOUDSC_H5" ] || {
            echo "FATAL: no cloudsc deck at $CLOUDSC_H5" >&2
            exit 1
        }
        unset BUILD_ROOT
        export BUILD_ROOT_BASE="$BR2/cloudsc_sweep_${blane}"
        mkdir -p "$BUILD_ROOT_BASE" || rc=1
        if [ "$MODE" = warm ]; then
            sizes="${CLOUDSC_WARM_SMOKE_SIZE:-4096}" reps=1 warm=0 threads="$WARM_THREADS"
        else
            sizes="$CLOUDSC_SWEEP_SIZES" reps="$REPS" warm="$WARMUP" threads="$THREADS"
        fi
        # baselines.sh runs under `set -e`; keep its failure from killing this lane's siblings.
        (
            export BASELINE_LANES="$blane" CLOUDSC_SWEEP=1 CLOUDSC_SWEEP_SIZES="$sizes"
            export REPS="$reps" WARMUP="$warm" THREADS="$threads" CSV="${CSV:-/dev/null}"
            bash "$MEAS/samples/cloudsc/baselines.sh"
        )
        r=$?
        [ "$r" -eq 0 ] || rc="$r"
        # warm has no phase A marker of its own; say so in the same shape the warm job greps.
        [ "$MODE" = warm ] && echo "phase A done: cloudsc_sweep_${blane} (binary built)"
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
                omp_preflight "$b" \
                    "$BUILD_ROOT/dacecache_${v}/velocity_tendencies/build/libvelocity_tendencies.so" || {
                    echo "ABORT: OpenMP purity preflight failed for dace-$b ($v) -- refusing to measure" >&2
                    rc=1
                    continue
                }
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
        load_spack || exit 1
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
                [ -f "$d/manifest.txt" ] || run "$PY" "$VELOCITY_SRC/dump_data.py" --data-dir "$(data_dir_for "$l")" \
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
                    [ -f "$CSV" ] || echo "kernel,mode,nproma,nblks_e,threads,rep,ms,inputs,lane,alloc" > "$CSV"
                    # The ACC twin IS the loop-exchange source, so its mode label stands; only the
                    # dataset label needs correcting, and nproma/nblks_e already carry the layout.
                    grep '^velocity_tendencies,' "$log" | sed "s/,r02b05,/,$(inputs_for "$l"),/" \
                        | sed "s/\$/,${ALLOC}/" | tee -a "$CSV"
                done
            done
            unset ACC_NUM_CORES
        fi
        ;;
    velocity-openmp)
        load_spack || exit 1
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
                [ -f "$d/manifest.txt" ] || run "$PY" "$VELOCITY_SRC/dump_data.py" --data-dir "$(data_dir_for "$l")" \
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
                # module-dir spelling differs: gfortran -J, flang -module-dir, nvfortran -module
                gcc) ofc="$OMP_GFORTRAN" oflags=(-O3 -march=native -fopenmp) omod=-J ;;
                flang) ofc="$OMP_FLANG" oflags=(-O3 -fopenmp) omod=-module-dir ;;
                nvhpc) ofc="$OMP_NVFORTRAN" oflags=(-O3 -mp) omod=-module ;;
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
                    -o "$odir/velocity_advection_acc.o" "$omod" "$odir"
                run "$ofc" "${oflags[@]}" -c "$VELOCITY_SRC/driver_velocity.f90" \
                    -o "$odir/driver_velocity.o" "$omod" "$odir"
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
                    [ -f "$CSV" ] || echo "kernel,mode,nproma,nblks_e,threads,rep,ms,inputs,lane,alloc" > "$CSV"
                    grep '^velocity_tendencies,' "$log" | sed "s/,r02b05,/,$(inputs_for "$l"),/" \
                        | sed "s/\$/,${ALLOC}/" | tee -a "$CSV"
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
