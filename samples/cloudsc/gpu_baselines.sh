#!/usr/bin/env bash
# CloudSC GPU baseline lanes from the vendored dwarf-p-cloudsc tree: cuda (src/cloudsc_cuda,
# nvcc) and openacc (src/cloudsc_gpu scc variant, nvfortran).  Build-gated skeleton under the
# CPU-first policy: each lane builds and runs only where its toolchain exists; a missing
# toolchain is one loud SKIP line and exit 0.  threads column is 0 for GPU lanes -- no CPU thread
# sweep applies.
#
# Rep semantics differ by lane.  The CUDA driver runs the rep loop itself, host-timed as
# sync/t0/kernel/sync/t1 per rep, and prints one " REP <rep> <ms>" line per timed rep: ONE process
# per lane, REPS rows, kernel only.  The OpenACC dwarf driver still times a whole-run TOTAL, so
# that lane keeps baselines.sh's one-process-per-rep convention.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$REPO/samples/common.sh"
probe_compilers

DWARF="$REPO/tests/cloudsc/variants/dwarf-p-cloudsc/src"
NGPTOT="${CLOUDSC_NGPTOT:-65536}"
NPROMA="${CLOUDSC_GPU_NPROMA:-128}"
REPS="${REPS:-50}"
WARMUP="${WARMUP:-2}"
CSV="${CSV:-cloudsc_gpu_baselines_${SLURM_JOB_ID:-local}.csv}"
GPU_LANES="${GPU_LANES:-cuda openacc}"
CSV_HEADER="kernel,mode,klon,nblocks,threads,rep,ms,inputs,lane,alloc"
# The allocator is a property of the process the harness launched, not of the timed binary,
# so the alloc column is stamped here from what alloc_pool.sh actually managed to preload.
ALLOC="${MEAS_ALLOC_ACTIVE:-system}"
NBLOCKS=$(((NGPTOT + NPROMA - 1) / NPROMA))

bash "$HERE/download_data.sh" || true
if [ ! -f "$HERE/data/input.h5" ] || [ ! -f "$HERE/data/reference.h5" ]; then
    echo "SKIP all GPU baselines: dwarf h5 deck missing and download_data.sh could not fetch it" >&2
    exit 0
fi

setup_build_root "cloudsc_gpu_baselines"
RUNDIR="$BUILD_ROOT/run"
mkdir -p "$RUNDIR"
ln -sf "$HERE/data/input.h5" "$RUNDIR/input.h5"
ln -sf "$HERE/data/reference.h5" "$RUNDIR/reference.h5"

# Same TOTAL parse and rep semantics as baselines.sh (one process per rep).
total_ms() { awk '/TOTAL[[:space:]]*$/ { gsub(":", " "); print $(NF - 3); exit }'; }

emit_header() {
    [ -f "$CSV" ] || echo "$CSV_HEADER" > "$CSV"
    echo "$CSV_HEADER"
}

run_lane() {
    local exe="$1" lane="$2" log="$RUNDIR/last_run.log" rep ms
    for rep in $(seq "$((-WARMUP))" "$((REPS - 1))"); do
        if ! (cd "$RUNDIR" && "$exe" 1 "$NGPTOT" "$NPROMA" > "$log" 2>&1); then
            echo "FATAL: $exe failed (lane $lane); tail of $log:" >&2
            tail -5 "$log" >&2
            return 1
        fi
        ms="$(total_ms < "$log")"
        if [ -z "$ms" ]; then
            echo "FATAL: no TOTAL line from $exe (lane $lane)" >&2
            return 1
        fi
        if [ "$rep" -ge 0 ]; then
            echo "cloudsc,nblks,$NPROMA,$NBLOCKS,0,$rep,$ms,h5,$lane,$ALLOC" | tee -a "$CSV"
        fi
    done
}

run_lane_reps() {
    local exe="$1" lane="$2" log="$RUNDIR/last_run.log" rows
    if ! (cd "$RUNDIR" && CLOUDSC_REPS="$REPS" CLOUDSC_WARMUP="$WARMUP" \
        "$exe" 1 "$NGPTOT" "$NPROMA" > "$log" 2>&1); then
        echo "FATAL: $exe failed (lane $lane); tail of $log:" >&2
        tail -5 "$log" >&2
        return 1
    fi
    rows="$(awk -v k="cloudsc,nblks,$NPROMA,$NBLOCKS,0" -v l="$lane" -v a="$ALLOC" \
        '$1 == "REP" { print k "," $2 "," $3 ",h5," l "," a }' "$log")"
    if [ -z "$rows" ]; then
        echo "FATAL: no REP lines from $exe (lane $lane) -- driver too old for per-rep timing" >&2
        return 1
    fi
    printf '%s\n' "$rows" | tee -a "$CSV"
}

build_cuda() (
    cd "$BUILD_ROOT"
    local inc lib
    inc="$("$H5CC" -show | tr ' ' '\n' | sed -n 's/^-I//p' | head -1)"
    lib="$("$H5CC" -show | tr ' ' '\n' | sed -n 's/^-L//p' | head -1)"
    # -rdc=true mirrors upstream CUDA_SEPARABLE_COMPILATION ON (cloudsc_cuda/CMakeLists.txt);
    # source list = the dwarf-cloudsc-gpu-cuda-c target there (base scc variant).
    # --allow-unsupported-compiler: the host compiler is gcc 16.1.0, past nvcc's supported-version
    # wall, which is a hard error rather than a warning. --expt-relaxed-constexpr: the dwarf's
    # __device__ code calls constexpr host functions.
    # -Xcompiler -fopenmp: cloudsc_driver.cu times with omp_get_wtime; the flag must also
    # reach the link step or libgomp stays unresolved.
    # -L alone only satisfies the LINKER: spack's HDF5 is outside the loader's default search
    # path, so without a matching rpath the binary builds and then dies at exec with
    # "libhdf5.so.310: cannot open shared object file". nvcc needs -Xlinker to forward it.
    "$NVCC" -O3 -arch=native -rdc=true -DHAVE_HDF5 \
        --allow-unsupported-compiler --expt-relaxed-constexpr -Xcompiler -fopenmp \
        -I"$DWARF/cloudsc_cuda/cloudsc" ${inc:+-I"$inc"} \
        "$DWARF/cloudsc_cuda/cloudsc/load_state.cu" \
        "$DWARF/cloudsc_cuda/cloudsc/cloudsc_validate.cu" \
        "$DWARF/cloudsc_cuda/cloudsc/mycpu.cu" \
        "$DWARF/cloudsc_cuda/cloudsc/cloudsc_driver.cu" \
        "$DWARF/cloudsc_cuda/cloudsc/cloudsc_c.cu" \
        "$DWARF/cloudsc_cuda/dwarf_cloudsc.cpp" \
        ${lib:+-L"$lib" -Xlinker -rpath -Xlinker "$lib"} -lhdf5 -o dwarf-cloudsc-cuda
)

# hdf5.mod is compiler-specific: nvfortran cannot read a gfortran-built one, so this lane
# needs HDF5_NVFORTRAN_ROOT (an nvfortran-built HDF5 prefix); the probe compile decides.
nvfortran_hdf5() {
    local inc
    [ -n "${HDF5_NVFORTRAN_ROOT:-}" ] || return 1
    inc="$HDF5_NVFORTRAN_ROOT/include"
    printf 'program p\nuse hdf5\nend program p\n' > "$BUILD_ROOT/h5probe.f90"
    "$NVFORTRAN" -c -I"$inc" "$BUILD_ROOT/h5probe.f90" -o "$BUILD_ROOT/h5probe.o" 2>/dev/null || return 1
    NVF_HDF5_FLAGS="-I$inc"
    NVF_HDF5_LIBS="$(hdf5_ldflags "$(hdf5_libdir "$HDF5_NVFORTRAN_ROOT")") -lhdf5_fortran -lhdf5"
}

# Same module order as baselines.sh COMMON_SRCS; ACC sources = the dwarf-cloudsc-gpu-scc
# target of cloudsc_gpu/CMakeLists.txt (base scc variant, -DCLOUDSC_GPU_SCC selects it in
# dwarf_cloudsc_gpu.F90).
build_openacc() (
    local f objs="mycpu.o"
    mkdir -p "$BUILD_ROOT/openacc" && cd "$BUILD_ROOT/openacc"
    printf '#define _GNU_SOURCE\n#include <sched.h>\nint mycpu(void) { return sched_getcpu(); }\n' > mycpu.c
    "${NVC:-$GCC}" -O2 -c mycpu.c
    for f in parkind1 yomphyder abor1 hdf5_file_mod file_io_mod yomcst yoethf yoecldp yoephli \
        cloudsc_mpi_mod timer_mod expand_mod validate_mod cloudsc_global_state_mod; do
        "$NVFORTRAN" -O3 -acc=gpu -DHAVE_HDF5 -DCLOUDSC_GPU_SCC "$NVF_HDF5_FLAGS" \
            -I"$DWARF/common/include" -c "$DWARF/common/module/$f.F90" -o "$f.o"
        objs="$objs $f.o"
    done
    for f in cloudsc_gpu_scc_mod cloudsc_driver_gpu_scc_mod dwarf_cloudsc_gpu; do
        "$NVFORTRAN" -O3 -acc=gpu -DHAVE_HDF5 -DCLOUDSC_GPU_SCC "$NVF_HDF5_FLAGS" \
            -I"$DWARF/common/include" -c "$DWARF/cloudsc_gpu/$f.F90" -o "$f.o"
        objs="$objs $f.o"
    done
    # shellcheck disable=SC2086
    "$NVFORTRAN" -O3 -acc=gpu $objs $NVF_HDF5_LIBS -o dwarf-cloudsc-gpu-scc
)

emit_header
for lane in $GPU_LANES; do
    case "$lane" in
        cuda)
            if [ "$HAVE_NVCC" != 1 ] || [ "$HAVE_H5CC" != 1 ]; then
                echo "SKIP cuda: needs nvcc + h5cc (HDF5 C library for the dwarf reader)" >&2
                continue
            fi
            build_cuda
            run_lane_reps "$BUILD_ROOT/dwarf-cloudsc-cuda" cuda
            ;;
        openacc)
            if [ "$HAVE_NVFORTRAN" != 1 ] || [ -z "${NVC:-}${GCC:-}" ]; then
                echo "SKIP openacc: needs nvfortran plus nvc/gcc (mycpu.c shim)" >&2
                continue
            fi
            if ! nvfortran_hdf5; then
                echo "SKIP openacc: no nvfortran-compatible HDF5 Fortran (set HDF5_NVFORTRAN_ROOT)" >&2
                continue
            fi
            build_openacc
            # upstream test env for the base scc variant (cloudsc_gpu/CMakeLists.txt)
            NVCOMPILER_ACC_CUDA_HEAPSIZE=64M run_lane "$BUILD_ROOT/openacc/dwarf-cloudsc-gpu-scc" openacc
            ;;
        *)
            echo "unknown GPU baseline lane: $lane" >&2
            exit 1
            ;;
    esac
done
echo "done: $CSV"
