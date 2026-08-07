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
CSV_HEADER="kernel,mode,nproma,nblks_e,threads,rep,ms,inputs,lane"
DATA_DIR="${VELOCITY_DATA_DIR:-$HERE/data_r02b05}"

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

# The driver prints CSV rows on stdout and diagnostics on stderr; both land in the log.
run_lane() {
    local exe="$1" lane="$2" threads="$3" log="$BUILD_ROOT/last_run.log"
    if ! "$exe" "$DUMP" "$lane" "$threads" "$REPS" "$WARMUP" > "$log" 2>&1; then
        echo "FATAL: $exe failed (lane $lane, threads $threads); tail of $log:" >&2
        tail -5 "$log" >&2
        return 1
    fi
    if ! grep -q '^velocity_tendencies,' "$log"; then
        echo "FATAL: no CSV rows from $exe (lane $lane, threads $threads)" >&2
        return 1
    fi
    grep '^velocity_tendencies,' "$log" | tee -a "$CSV"
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
                run_lane "$BUILD_ROOT/$lane-$t/driver_velocity" "$lane" "$t"
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
                run_lane "$BUILD_ROOT/$lane/driver_velocity" "$lane" "$t"
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
            run_lane "$BUILD_ROOT/$lane/driver_velocity" "$lane" 0
            ;;
        *)
            echo "unknown baseline lane: $lane" >&2
            exit 1
            ;;
    esac
done
echo "done: $CSV"
