#!/usr/bin/env bash
# CloudSC CPU baseline lanes, all built from the vendored dwarf-p-cloudsc tree
# (tests/cloudsc/variants/dwarf-p-cloudsc, Apache-2.0): its dwarf_cloudsc.F90 driver reads the
# same input.h5 deck the DaCe lanes use and prints a per-run "TOTAL" Time(msec) line
# (common/module/timer_mod.F90) that we parse, so no timing harness is invented.  The e2e TU
# (tests/cloudsc/full/cloudsc.F90) is NOT used for any lane: it has no standalone driver and
# carries no OpenMP pragmas (grep '!$' finds none), so the dwarf tree is the only self-timed
# original-Fortran path.  Rep semantics: one dwarf process per rep, WARMUP untimed + REPS timed
# (closest process-level match to the drivers' 2 warmup + 50 timed calls).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$REPO/samples/common.sh"
probe_compilers
probe_topology

DWARF="$REPO/tests/cloudsc/variants/dwarf-p-cloudsc/src"
NGPTOT="${CLOUDSC_NGPTOT:-65536}"
REPS="${REPS:-50}"
WARMUP="${WARMUP:-2}"
CSV="${CSV:-cloudsc_baselines_${SLURM_JOB_ID:-local}.csv}"
BASELINE_LANES="${BASELINE_LANES:-gfortran-serial gfortran-autopar original-openmp openacc-cpu flang-serial}"
CSV_HEADER="kernel,mode,klon,nblocks,threads,rep,ms,inputs,lane,alloc"
# The allocator is a property of the process the harness launched, not of the timed binary,
# so the alloc column is stamped here from what alloc_pool.sh actually managed to preload.
ALLOC="${MEAS_ALLOC_ACTIVE:-system}"
# Size table + mimalloc-cliff verdicts, shared with the DaCe lanes and the sweep sbatch jobs.
# shellcheck disable=SC1091
source "$HERE/sweep_sizes.sh"

# CLOUDSC_SWEEP=1 turns the two fixed modes into the figure-A problem-size sweep.
CLOUDSC_SWEEP="${CLOUDSC_SWEEP:-0}"

# One "<mode> <nproma> <ngptot>" line per timed point.  Default keeps the historical pair
# (mirrors run_cloudsc_perf.py MODE_DEFAULTS, so the rows stay comparable); the sweep walks
# sizes at the pinned NPROMA instead, because klon-mode at these sizes neither scales nor
# stays off the allocator cliff (KLON=65536 puts a transient row at exactly 512 KiB).
baseline_points() {
    local s n
    if [ "$CLOUDSC_SWEEP" = 1 ]; then
        for s in $CLOUDSC_SWEEP_SIZES; do
            for n in $CLOUDSC_SWEEP_NPROMAS; do
                printf 'nblks %s %s\n' "$n" "$s"
            done
        done
    else
        printf 'klon %s %s\n' "$NGPTOT" "$NGPTOT"
        printf 'nblks 32 %s\n' "$NGPTOT"
    fi
}

if [ "$HAVE_GCC" != 1 ]; then
    echo "SKIP all baselines: gcc missing (needed for the mycpu.c shim)" >&2
    exit 0
fi

bash "$HERE/download_data.sh" || true
if [ ! -f "$HERE/data/input.h5" ] || [ ! -f "$HERE/data/reference.h5" ]; then
    echo "SKIP all baselines: dwarf h5 deck missing and download_data.sh could not fetch it" >&2
    exit 0
fi

# Never inherit a dacecache root from the sbatch's dace lanes: these lanes share nothing with them
# (velocity_tendencies/baselines.sh does the same).
unset BUILD_ROOT
setup_build_root "cloudsc_baselines"
RUNDIR="$BUILD_ROOT/run"
mkdir -p "$RUNDIR"
ln -sf "$HERE/data/input.h5" "$RUNDIR/input.h5"
ln -sf "$HERE/data/reference.h5" "$RUNDIR/reference.h5"

# Module compile order derived from the USE statements (the vendored tree is src/-only: the
# upstream ecbuild CMake files cannot drive a build here).  MPI/serialbox/FIELD_API/CUF files
# excluded; only HAVE_HDF5 is defined.
COMMON_SRCS="parkind1 yomphyder abor1 hdf5_file_mod file_io_mod yomcst yoethf yoecldp yoephli
             cloudsc_mpi_mod timer_mod expand_mod validate_mod cloudsc_global_state_mod"
FORTRAN_DIR="$DWARF/cloudsc_fortran"
FORTRAN_SRCS="cloudsc cloudsc_driver_mod dwarf_cloudsc"
# openacc-cpu builds the same scc target as the gpu openacc lane (gpu_baselines.sh):
# dwarf-cloudsc-gpu-scc of cloudsc_gpu/CMakeLists.txt, -DCLOUDSC_GPU_SCC selects it.
ACC_SCC_DIR="$DWARF/cloudsc_gpu"
ACC_SCC_SRCS="cloudsc_gpu_scc_mod cloudsc_driver_gpu_scc_mod dwarf_cloudsc_gpu"

# build_dwarf <dir> <fc> <link_libs> <flags...>: mycpu.c shim stands in for upstream
# common/module/mycpu.c, which is not vendored (only the .cu twin is).  FORTRAN_DIR/FORTRAN_SRCS
# select the driver set (cloudsc_fortran by default, cloudsc_gpu for the acc lane).
build_dwarf() (
    local dir="$1" fc="$2" libs="$3" f objs="mycpu.o"
    shift 3
    mkdir -p "$dir" && cd "$dir"
    printf '#define _GNU_SOURCE\n#include <sched.h>\nint mycpu(void) { return sched_getcpu(); }\n' > mycpu.c
    "$GCC" -O2 -c mycpu.c
    for f in $COMMON_SRCS; do
        "$fc" "$@" -DHAVE_HDF5 -I"$DWARF/common/include" -c "$DWARF/common/module/$f.F90" -o "$f.o"
        objs="$objs $f.o"
    done
    for f in $FORTRAN_SRCS; do
        "$fc" "$@" -DHAVE_HDF5 -I"$DWARF/common/include" -c "$FORTRAN_DIR/$f.F90" -o "$f.o"
        objs="$objs $f.o"
    done
    # shellcheck disable=SC2086
    "$fc" "$@" $objs $libs -o dwarf-cloudsc
)

# Time(msec) is 3rd-from-last field of the rank-0 TOTAL line once ':' separators go; the
# Fortran (timer_mod) and CUDA (cloudsc_driver.cu) formats agree on that position.
total_ms() { awk '/TOTAL[[:space:]]*$/ { gsub(":", " "); print $(NF - 3); exit }'; }

emit_header() {
    [ -f "$CSV" ] || echo "$CSV_HEADER" > "$CSV"
    echo "$CSV_HEADER"
}

run_lane() {
    local exe="$1" lane="$2" threads="$3" numomp="$4" log="$RUNDIR/last_run.log" mode nproma size nblocks rep ms
    while read -r mode nproma size; do
        nblocks=$(((size + nproma - 1) / nproma))
        for rep in $(seq "$((-WARMUP))" "$((REPS - 1))"); do
            if ! (cd "$RUNDIR" && "$exe" "$numomp" "$size" "$nproma" > "$log" 2>&1); then
                echo "FATAL: $exe failed (lane $lane, mode $mode, ngptot $size); tail of $log:" >&2
                tail -5 "$log" >&2
                return 1
            fi
            ms="$(total_ms < "$log")"
            if [ -z "$ms" ]; then
                echo "FATAL: no TOTAL line from $exe (lane $lane, mode $mode, ngptot $size)" >&2
                return 1
            fi
            if [ "$rep" -ge 0 ]; then
                echo "cloudsc,$mode,$nproma,$nblocks,$threads,$rep,$ms,h5,$lane,$ALLOC" | tee -a "$CSV"
            fi
        done
    done < <(baseline_points)
}

# The C rewrite times every rep INSIDE one process (CLOUDSC_REPS/CLOUDSC_WARMUP, one
# ` REP <i> <ms>` line each), so this lane pays the h5 load + column expansion once per sweep
# point instead of once per rep.  That matters: at NGPTOT=262144 the expansion dominates a
# per-rep-process lane by an order of magnitude.  Timers are host timers around the block
# loop only -- no I/O and no allocation inside the bracket.
run_lane_c() {
    local exe="$1" lane="$2" threads="$3" numomp="$4" log="$RUNDIR/last_run_c.log" mode nproma size nblocks rep ms
    while read -r mode nproma size; do
        nblocks=$(((size + nproma - 1) / nproma))
        if ! (cd "$RUNDIR" && CLOUDSC_REPS="$REPS" CLOUDSC_WARMUP="$WARMUP" \
            "$exe" "$numomp" "$size" "$nproma" > "$log" 2>&1); then
            echo "FATAL: $exe failed (lane $lane, ngptot $size); tail of $log:" >&2
            tail -5 "$log" >&2
            return 1
        fi
        rep=0
        while read -r ms; do
            echo "cloudsc,$mode,$nproma,$nblocks,$threads,$rep,$ms,h5,$lane,$ALLOC" | tee -a "$CSV"
            rep=$((rep + 1))
        done < <(awk '/^ REP /{ print $3 }' "$log")
        if [ "$rep" -ne "$REPS" ]; then
            echo "FATAL: $exe emitted $rep REP lines, expected $REPS (lane $lane, ngptot $size)" >&2
            return 1
        fi
    done < <(baseline_points)
}

# The C rewrite is the dwarf's cloudsc_cuda kernel compiled for the host: cuda_shim/cuda.h
# neutralises `__global__` and supplies the thread-local block/column indices, and
# c_openmp/cloudsc_driver_omp.cpp replaces the CUDA driver with an OpenMP block loop.  The
# other three vendored translation units (load_state, cloudsc_validate, mycpu) are already
# pure host code and compile unchanged.  -w: the vendored sources are warning-noisy and not
# ours to clean.
build_dwarf_c() (
    local dir="$1" h5inc="$2" h5lib="$3"
    local cu="$DWARF/cloudsc_cuda/cloudsc"
    mkdir -p "$dir" && cd "$dir"
    # shellcheck disable=SC2046 -- hdf5_ldflags emits two words (-L and -Wl,-rpath) on purpose
    "$CXX" -O3 -march=native -std=c++17 -fopenmp -DHAVE_HDF5 -w \
        -I"$HERE/c_openmp/cuda_shim" -I"$cu" -I"$h5inc" \
        -x c++ "$cu/load_state.cu" "$cu/cloudsc_validate.cu" "$cu/mycpu.cu" "$cu/cloudsc_c.cu" \
        "$HERE/c_openmp/cloudsc_driver_omp.cpp" \
        $(hdf5_ldflags "$h5lib") -lhdf5 -lm -o dwarf-cloudsc-c-omp
)

# g++ + the HDF5 C library (not the Fortran one the other lanes need): sets CXX/C_H5INC/C_H5LIB.
c_openmp_ok() {
    CXX="${CXX:-$(command -v g++ || true)}"
    if [ -z "$CXX" ]; then
        echo "SKIP $1: no g++ on PATH" >&2
        return 1
    fi
    if [ -n "${HDF5_C_ROOT:-}" ]; then
        C_H5INC="$HDF5_C_ROOT/include" C_H5LIB="$(hdf5_libdir "$HDF5_C_ROOT")"
    elif [ "$HAVE_H5CC" = 1 ]; then
        C_H5INC="$("$H5CC" -show | tr ' ' '\n' | sed -n 's/^-I//p' | head -1)"
        C_H5LIB="$("$H5CC" -show | tr ' ' '\n' | sed -n 's/^-L//p' | head -1)"
    else
        echo "SKIP $1: no h5cc and no HDF5_C_ROOT (the C rewrite reads the deck via the HDF5 C API)" >&2
        return 1
    fi
    [ -n "$C_H5INC" ] && [ -n "$C_H5LIB" ]
}

# Sets GF_HDF5_LIBS. spack's h5fc emits `-L<dir> -lhdf5_fortran -lhdf5` with NO rpath, so the
# binary links and then dies at startup on libhdf5_fortran.so.310. Stamp the rpath instead of
# exporting LD_LIBRARY_PATH: that would also override the flang lane's DT_RUNPATH and silently
# swap a gfortran-built libhdf5_fortran under it.
gfortran_ok() {
    local lib
    if [ "$HAVE_H5FC" != 1 ] || ! "$H5FC" -show | grep -qw gfortran; then
        echo "SKIP $1: no h5fc wrapping gfortran (dwarf reader needs the HDF5 Fortran library)" >&2
        return 1
    fi
    if [ -n "${HDF5_GFORTRAN_ROOT:-}" ]; then
        lib="$(hdf5_libdir "$HDF5_GFORTRAN_ROOT")"
    else
        lib="$("$H5FC" -show | tr ' ' '\n' | sed -n 's/^-L//p' | head -1)"
    fi
    GF_HDF5_LIBS="${lib:+$(hdf5_ldflags "$lib")}"
}

# hdf5.mod is compiler-specific; flang cannot read the gfortran-built one h5fc points at.
# HDF5_FLANG_ROOT (a flang-built HDF5 prefix) overrides; the probe compile decides.
flang_hdf5() {
    local inc lib
    if [ -n "${HDF5_FLANG_ROOT:-}" ]; then
        inc="$HDF5_FLANG_ROOT/include" lib="$(hdf5_libdir "$HDF5_FLANG_ROOT")"
    elif [ "$HAVE_H5FC" = 1 ]; then
        inc="$("$H5FC" -show | tr ' ' '\n' | sed -n 's/^-I//p' | head -1)"
        lib="$("$H5FC" -show | tr ' ' '\n' | sed -n 's/^-L//p' | head -1)"
    else
        return 1
    fi
    [ -n "$inc" ] || return 1
    printf 'program p\nuse hdf5\nend program p\n' > "$BUILD_ROOT/h5probe.f90"
    "$FLANG" -c -I"$inc" "$BUILD_ROOT/h5probe.f90" -o "$BUILD_ROOT/h5probe.o" 2>/dev/null || return 1
    FLANG_HDF5_FLAGS="-I$inc"
    FLANG_HDF5_LIBS="${lib:+$(hdf5_ldflags "$lib") }-lhdf5_fortran -lhdf5"
}

# Same gate as the gpu openacc lane (gpu_baselines.sh): nvfortran cannot read a gfortran-built
# hdf5.mod, so this lane needs an nvfortran-built HDF5 prefix; the probe compile decides.
nvfortran_hdf5() {
    local inc
    [ -n "${HDF5_NVFORTRAN_ROOT:-}" ] || return 1
    inc="$HDF5_NVFORTRAN_ROOT/include"
    printf 'program p\nuse hdf5\nend program p\n' > "$BUILD_ROOT/h5probe.f90"
    "$NVFORTRAN" -c -I"$inc" "$BUILD_ROOT/h5probe.f90" -o "$BUILD_ROOT/h5probe.o" 2>/dev/null || return 1
    NVF_HDF5_FLAGS="-I$inc"
    NVF_HDF5_LIBS="$(hdf5_ldflags "$(hdf5_libdir "$HDF5_NVFORTRAN_ROOT")") -lhdf5_fortran -lhdf5"
}

emit_header
for lane in $BASELINE_LANES; do
    case "$lane" in
        gfortran-serial)
            gfortran_ok "$lane" || continue
            build_dwarf "$BUILD_ROOT/$lane" "$H5FC" "$GF_HDF5_LIBS" -O3 -march=native
            set_omp_env 1
            run_lane "$BUILD_ROOT/$lane/dwarf-cloudsc" "$lane" 1 1
            ;;
        gfortran-autopar)
            gfortran_ok "$lane" || continue
            for t in $THREADS; do
                if [ "$t" -gt "$TOPO_NCORES" ]; then
                    echo "skip threads=$t (> $TOPO_NCORES physical cores)"
                    continue
                fi
                # thread count is a compile-time constant for autopar: one build per t
                build_dwarf "$BUILD_ROOT/$lane-$t" "$H5FC" "$GF_HDF5_LIBS" -O3 -march=native -ftree-parallelize-loops="$t"
                set_omp_env "$t"
                run_lane "$BUILD_ROOT/$lane-$t/dwarf-cloudsc" "$lane" "$t" 1
            done
            ;;
        original-openmp)
            gfortran_ok "$lane" || continue
            build_dwarf "$BUILD_ROOT/$lane" "$H5FC" "$GF_HDF5_LIBS" -O3 -march=native -fopenmp
            # the dwarf block loop is schedule(runtime): pin it, or timings drift with libgomp defaults
            export OMP_SCHEDULE=static
            for t in $THREADS; do
                if [ "$t" -gt "$TOPO_NCORES" ]; then
                    echo "skip threads=$t (> $TOPO_NCORES physical cores)"
                    continue
                fi
                set_omp_env "$t"
                run_lane "$BUILD_ROOT/$lane/dwarf-cloudsc" "$lane" "$t" "$t"
            done
            unset OMP_SCHEDULE
            ;;
        openacc-cpu)
            if [ "$HAVE_NVFORTRAN" != 1 ]; then
                echo "SKIP $lane: no nvfortran on PATH" >&2
                continue
            fi
            if ! nvfortran_hdf5; then
                echo "SKIP $lane: no nvfortran-compatible HDF5 Fortran (set HDF5_NVFORTRAN_ROOT)" >&2
                continue
            fi
            # one build: -acc=multicore takes its core count at runtime (ACC_NUM_CORES)
            FORTRAN_DIR="$ACC_SCC_DIR" FORTRAN_SRCS="$ACC_SCC_SRCS" \
                build_dwarf "$BUILD_ROOT/$lane" "$NVFORTRAN" "$NVF_HDF5_LIBS" \
                -O3 -acc=multicore -DCLOUDSC_GPU_SCC "$NVF_HDF5_FLAGS"
            for t in $THREADS; do
                if [ "$t" -gt "$TOPO_NCORES" ]; then
                    echo "skip threads=$t (> $TOPO_NCORES physical cores)"
                    continue
                fi
                set_omp_env "$t"
                export ACC_NUM_CORES="$t"
                run_lane "$BUILD_ROOT/$lane/dwarf-cloudsc" "$lane" "$t" 1
            done
            unset ACC_NUM_CORES
            ;;
        c-openmp)
            # The C rewrite, same thread handling as original-openmp: one build, OMP_NUM_THREADS
            # and the argv NUMOMP both carry the thread count, schedule pinned to static.
            c_openmp_ok "$lane" || continue
            build_dwarf_c "$BUILD_ROOT/$lane" "$C_H5INC" "$C_H5LIB"
            export OMP_SCHEDULE=static
            for t in $THREADS; do
                if [ "$t" -gt "$TOPO_NCORES" ]; then
                    echo "skip threads=$t (> $TOPO_NCORES physical cores)"
                    continue
                fi
                set_omp_env "$t"
                run_lane_c "$BUILD_ROOT/$lane/dwarf-cloudsc-c-omp" "$lane" "$t" "$t"
            done
            unset OMP_SCHEDULE
            ;;
        flang-serial)
            # flang has no -ftree-parallelize-loops equivalent, so serial is the only flang lane
            if [ "$HAVE_FLANG" != 1 ]; then
                echo "SKIP $lane: no LLVM flang on PATH" >&2
                continue
            fi
            if ! flang_hdf5; then
                echo "SKIP $lane: no flang-compatible HDF5 Fortran (set HDF5_FLANG_ROOT)" >&2
                continue
            fi
            build_dwarf "$BUILD_ROOT/$lane" "$FLANG" "$FLANG_HDF5_LIBS" -O3 "$FLANG_HDF5_FLAGS"
            set_omp_env 1
            run_lane "$BUILD_ROOT/$lane/dwarf-cloudsc" "$lane" 1 1
            ;;
        *)
            echo "unknown baseline lane: $lane" >&2
            exit 1
            ;;
    esac
done
echo "done: $CSV"
