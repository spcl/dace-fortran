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
BASELINE_LANES="${BASELINE_LANES:-gfortran-serial gfortran-autopar original-openmp flang-serial}"
CSV_HEADER="kernel,mode,klon,nblocks,threads,rep,ms,inputs,lane"
# mode:NPROMA pairs mirror run_cloudsc_perf.py MODE_DEFAULTS (same NGPTOT split, comparable rows).
MODES="klon:65536 nblks:32"

if [ "$HAVE_GCC" != 1 ]; then
    echo "SKIP all baselines: gcc missing (needed for the mycpu.c shim)" >&2
    exit 0
fi

bash "$HERE/download_data.sh" || true
if [ ! -f "$HERE/data/input.h5" ] || [ ! -f "$HERE/data/reference.h5" ]; then
    echo "SKIP all baselines: dwarf h5 deck missing and download_data.sh could not fetch it" >&2
    exit 0
fi

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
FORTRAN_SRCS="cloudsc cloudsc_driver_mod dwarf_cloudsc"

# build_dwarf <dir> <fc> <link_libs> <flags...>: mycpu.c shim stands in for upstream
# common/module/mycpu.c, which is not vendored (only the .cu twin is).
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
        "$fc" "$@" -DHAVE_HDF5 -I"$DWARF/common/include" -c "$DWARF/cloudsc_fortran/$f.F90" -o "$f.o"
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
    local exe="$1" lane="$2" threads="$3" numomp="$4" log="$RUNDIR/last_run.log" spec mode nproma nblocks rep ms
    for spec in $MODES; do
        mode="${spec%%:*}" nproma="${spec##*:}"
        nblocks=$(((NGPTOT + nproma - 1) / nproma))
        for rep in $(seq "$((-WARMUP))" "$((REPS - 1))"); do
            if ! (cd "$RUNDIR" && "$exe" "$numomp" "$NGPTOT" "$nproma" > "$log" 2>&1); then
                echo "FATAL: $exe failed (lane $lane, mode $mode); tail of $log:" >&2
                tail -5 "$log" >&2
                return 1
            fi
            ms="$(total_ms < "$log")"
            if [ -z "$ms" ]; then
                echo "FATAL: no TOTAL line from $exe (lane $lane, mode $mode)" >&2
                return 1
            fi
            if [ "$rep" -ge 0 ]; then
                echo "cloudsc,$mode,$nproma,$nblocks,$threads,$rep,$ms,h5,$lane" | tee -a "$CSV"
            fi
        done
    done
}

gfortran_ok() {
    if [ "$HAVE_H5FC" != 1 ] || ! "$H5FC" -show | grep -qw gfortran; then
        echo "SKIP $1: no h5fc wrapping gfortran (dwarf reader needs the HDF5 Fortran library)" >&2
        return 1
    fi
}

# hdf5.mod is compiler-specific; flang cannot read the gfortran-built one h5fc points at.
# HDF5_FLANG_ROOT (a flang-built HDF5 prefix) overrides; the probe compile decides.
flang_hdf5() {
    local inc lib
    if [ -n "${HDF5_FLANG_ROOT:-}" ]; then
        inc="$HDF5_FLANG_ROOT/include" lib="$HDF5_FLANG_ROOT/lib"
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
    FLANG_HDF5_LIBS="${lib:+-L$lib }-lhdf5_fortran -lhdf5"
}

emit_header
for lane in $BASELINE_LANES; do
    case "$lane" in
        gfortran-serial)
            gfortran_ok "$lane" || continue
            build_dwarf "$BUILD_ROOT/$lane" "$H5FC" "" -O3 -march=native
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
                build_dwarf "$BUILD_ROOT/$lane-$t" "$H5FC" "" -O3 -march=native -ftree-parallelize-loops="$t"
                set_omp_env "$t"
                run_lane "$BUILD_ROOT/$lane-$t/dwarf-cloudsc" "$lane" "$t" 1
            done
            ;;
        original-openmp)
            gfortran_ok "$lane" || continue
            build_dwarf "$BUILD_ROOT/$lane" "$H5FC" "" -O3 -march=native -fopenmp
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
        flang-serial)
            # flang has no -ftree-parallelize-loops equivalent, so serial is the only flang lane
            if [ "$HAVE_FLANG" != 1 ]; then
                echo "SKIP $lane: no flang-new-21/flang on PATH" >&2
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
