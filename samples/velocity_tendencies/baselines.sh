#!/usr/bin/env bash
# ICON velocity_tendencies Fortran baseline lanes: the original kernel, its own compiler's
# parallelism, no DaCe.  Source is velocity_advection_acc.f90 -- the OpenACC/OpenMP-annotated twin
# of the e2e TU (see its header and scripts/annotate_velocity_acc.py) -- driven by
# driver_velocity.f90, which reads the raw dump from dump_data.py and prints the sample's 9-column
# CSV rows itself.  Structure, CSV contract and loud-SKIP policy mirror samples/cloudsc/baselines.sh;
# unlike cloudsc there is no HDF5 anywhere, so no per-compiler HDF5 prefix is needed.
#
# Rep semantics: ONE driver process per lane point, WARMUP untimed + REPS timed calls inside it
# (run_velocity_perf.py does exactly that, so the rows are comparable).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$REPO/samples/common.sh"
probe_compilers
probe_topology

REPS="${REPS:-50}"
WARMUP="${WARMUP:-2}"
NPROMA="${VELOCITY_NPROMA:-32}"
CSV="${CSV:-velocity_baselines_${SLURM_JOB_ID:-local}.csv}"
BASELINE_LANES="${BASELINE_LANES:-gfortran-autopar openacc-cpu openacc}"
CSV_HEADER="kernel,mode,nproma,nblks_e,threads,rep,ms,inputs,lane,alloc"
# driver_velocity emits the 9 original columns; the allocator is a property of the process the
# harness launched it in, so the alloc column is appended here rather than in the Fortran driver.
ALLOC="${MEAS_ALLOC_ACTIVE:-system}"
DATA_DIR="${VELOCITY_DATA_DIR:-$HERE/data_r02b05}"
R06_DATA_DIR="${VELOCITY_R06_DATA_DIR:-$HERE/data_r02b06}"
R06_TIMESTEP="${VELOCITY_TIMESTEP:-1}"

# Never inherit a dacecache root from the sbatch's dace lanes: these lanes share nothing with them.
unset BUILD_ROOT
setup_build_root "velocity_baselines"
DUMP="${VELOCITY_DUMP_DIR:-$BUILD_ROOT/dump}"

# The kernel's automatic transients are a few hundred MB of stack at nproma=32.
ulimit -s unlimited 2> /dev/null || echo "warning: could not raise the stack limit" >&2

if [ ! -f "$DUMP/manifest.txt" ]; then
    OUTPUT_DIR="$DATA_DIR" bash "$HERE/download_data.sh" || true
    "$PY" "$HERE/dump_data.py" --data-dir "$DATA_DIR" --nproma "$NPROMA" --out "$DUMP" || true
fi
if [ ! -f "$DUMP/manifest.txt" ]; then
    echo "SKIP all baselines: no input dump at $DUMP (download_data.sh + dump_data.py could not build one)" >&2
    exit 0
fi

# build_driver <dir> <fc> <flags...>: modules first, the TU defines every module the driver USEs.
build_driver() (
    local dir="$1" fc="$2"
    shift 2
    mkdir -p "$dir" && cd "$dir"
    "$fc" "$@" -c "$HERE/velocity_advection_acc.f90" -o velocity_advection_acc.o
    "$fc" "$@" -c "$HERE/driver_velocity.f90" -o driver_velocity.o
    "$fc" "$@" velocity_advection_acc.o driver_velocity.o -o driver_velocity
)

emit_header() {
    [ -f "$CSV" ] || echo "$CSV_HEADER" > "$CSV"
    echo "$CSV_HEADER"
}

r06_dump_for() {
    case "$1" in
        flat) echo "$BUILD_ROOT/dump_r02b06_nproma491520" ;;
        *) echo "$BUILD_ROOT/dump_r02b06_nproma32" ;;
    esac
}
r06_nproma_for() {
    case "$1" in
        flat) echo 491520 ;;
        *) echo 32 ;;
    esac
}
ensure_r06_dump() {
    local d
    d="$(r06_dump_for "$1")"
    [ -f "$d/manifest.txt" ] && return 0
    "$PY" "$HERE/dump_data.py" --data-dir "$R06_DATA_DIR" --timestep "$R06_TIMESTEP" \
        --nproma "$(r06_nproma_for "$1")" --out "$d"
}

run_lane() {
    local exe="$1" dump_dir="$2" lane="$3" threads="$4" inputs="${5:-}" log="$BUILD_ROOT/last_run.log" rows
    if ! "$exe" "$dump_dir" "$lane" "$threads" "$REPS" "$WARMUP" > "$log" 2>&1; then
        echo "FATAL: $exe failed (lane $lane, threads $threads); tail of $log:" >&2
        tail -5 "$log" >&2
        return 1
    fi
    if ! grep -q '^velocity_tendencies,' "$log"; then
        echo "FATAL: no CSV rows from $exe (lane $lane, threads $threads)" >&2
        return 1
    fi
    rows="$(grep '^velocity_tendencies,' "$log")"
    [ -z "$inputs" ] || rows="$(printf '%s\n' "$rows" | sed "s/,r02b05,/,${inputs},/")"
    printf '%s\n' "$rows" | sed "s/\$/,${ALLOC}/" | tee -a "$CSV"
}

emit_header
for lane in $BASELINE_LANES; do
    case "$lane" in
        gfortran-autopar)
            if [ "$HAVE_GFORTRAN" != 1 ]; then
                echo "SKIP $lane: no gfortran on PATH" >&2
                continue
            fi
            for t in $THREADS; do
                if [ "$t" -gt "$TOPO_NCORES" ]; then
                    echo "skip threads=$t (> $TOPO_NCORES physical cores)"
                    continue
                fi
                # thread count is a compile-time constant for autopar: one build per t
                build_driver "$BUILD_ROOT/$lane-$t" "$GFORTRAN" -O3 -march=native -ftree-parallelize-loops="$t"
                set_omp_env "$t"
                run_lane "$BUILD_ROOT/$lane-$t/driver_velocity" "$DUMP" "$lane" "$t"
            done
            ;;
        openacc-cpu)
            if [ "$HAVE_NVFORTRAN" != 1 ]; then
                echo "SKIP $lane: no nvfortran on PATH" >&2
                continue
            fi
            # one build: -acc=multicore takes its core count at runtime (ACC_NUM_CORES)
            build_driver "$BUILD_ROOT/$lane" "$NVFORTRAN" -O3 -acc=multicore
            for t in $THREADS; do
                if [ "$t" -gt "$TOPO_NCORES" ]; then
                    echo "skip threads=$t (> $TOPO_NCORES physical cores)"
                    continue
                fi
                set_omp_env "$t"
                export ACC_NUM_CORES="$t"
                run_lane "$BUILD_ROOT/$lane/driver_velocity" "$DUMP" "$lane" "$t"
            done
            unset ACC_NUM_CORES
            ;;
        openacc)
            if [ "$HAVE_NVFORTRAN" != 1 ]; then
                echo "SKIP $lane: no nvfortran on PATH" >&2
                continue
            fi
            if ! command -v nvidia-smi > /dev/null || ! nvidia-smi -L 2> /dev/null | grep -q '^GPU'; then
                echo "SKIP $lane: no GPU visible (nvidia-smi -L lists none)" >&2
                continue
            fi
            # No mem:managed and no -gpu=deepcopy: driver_velocity.f90 stages every array with
            # explicit ENTER DATA (OpenACC manual deep copy), so the kernel's PRESENT clauses are
            # present-hits on real device memory, and a miss is a hard runtime error.
            build_driver "$BUILD_ROOT/$lane" "$NVFORTRAN" -O3 -acc=gpu
            set_omp_env 1
            # threads column 0: no CPU thread sweep applies (same convention as cloudsc gpu_baselines.sh)
            run_lane "$BUILD_ROOT/$lane/driver_velocity" "$DUMP" "$lane" 0
            ;;
        original-openmp-gcc | original-openmp-flang | original-openmp-nvhpc)
            oc="${lane#original-openmp-}"
            case "$oc" in
                gcc) ofc="$GFORTRAN" ohave="$HAVE_GFORTRAN" oflags=(-O3 -march=native -fopenmp) ;;
                flang) ofc="$FLANG" ohave="$HAVE_FLANG" oflags=(-O3 -fopenmp) ;;
                nvhpc) ofc="$NVFORTRAN" ohave="$HAVE_NVFORTRAN" oflags=(-O3 -mp) ;;
            esac
            if [ "$ohave" != 1 ]; then
                echo "SKIP $lane: no $oc Fortran compiler on PATH" >&2
                continue
            fi
            build_driver "$BUILD_ROOT/$lane" "$ofc" "${oflags[@]}"
            for t in $THREADS; do
                if [ "$t" -gt "$TOPO_NCORES" ]; then
                    echo "skip threads=$t (> $TOPO_NCORES physical cores)"
                    continue
                fi
                set_omp_env "$t"
                for ol in nproma32 flat; do
                    if ! ensure_r06_dump "$ol"; then
                        echo "SKIP $lane layout $ol: no R02B06 dump at $(r06_dump_for "$ol")" >&2
                        continue
                    fi
                    run_lane "$BUILD_ROOT/$lane/driver_velocity" "$(r06_dump_for "$ol")" "$lane" "$t" r02b06
                done
            done
            ;;
        *)
            echo "unknown baseline lane: $lane" >&2
            exit 1
            ;;
    esac
done
echo "done: $CSV"
