#!/usr/bin/env bash
# Single source of truth for the CloudSC problem-size sweep (figure A of the paper spec).
# SOURCE this; running it directly prints the table and exits 0, which is the dry-run check:
#   bash samples/cloudsc/sweep_sizes.sh
#
# Sizes mirror f2dace-artifact's cloudsc figure jobs (cloudsc/original/job_cpu_c.sh and
# job_cpu_fortran.sh): total columns 4096..262144 in powers of two, one NPROMA per run,
# NGPBLKS derived.  The artifact swept NPROMA in {16,64,128} for its original-code lanes;
# this harness pins NPROMA=32 across every lane instead, because
#   (a) it is the decomposition the DaCe `nblks` mode already measures, so the sweep and the
#       canonical 65536 point share one SDFG cache and one set of numbers, and
#   (b) 32 is the only value in the artifact's neighbourhood that keeps every per-block DaCe
#       transient below 256 KiB, i.e. clear of the mimalloc 512 KiB-1 MiB size-class cliff.
# CLOUDSC_SWEEP_NPROMAS can widen it back to the artifact set; the table prints the verdict.
#
# The cliff: mimalloc serves <=512 KiB from thread-local pages and >=512 KiB from the segment
# allocator.  A per-block transient that lands in [512 KiB, 1 MiB) is allocated and freed on
# the hot path of `#pragma omp parallel for` over blocks, and pays the slow path every block
# iteration.  Only the DaCe lanes heap-allocate their transients; the Fortran and C lanes use
# automatic (stack) arrays, so the flag is advisory for them and load-bearing for dace-*.

CLOUDSC_KLEV="${CLOUDSC_KLEV:-137}"
CLOUDSC_NCLV="${CLOUDSC_NCLV:-5}"
CLOUDSC_SWEEP_SIZES="${CLOUDSC_SWEEP_SIZES:-4096 8192 16384 32768 65536 131072 262144}"
CLOUDSC_SWEEP_NPROMAS="${CLOUDSC_SWEEP_NPROMAS:-32}"
# The efficiency panel's canonical point: full thread grid lives here, nowhere else.
CLOUDSC_CANONICAL_SIZE="${CLOUDSC_CANONICAL_SIZE:-65536}"

CLOUDSC_CLIFF_LO=524288  # 512 KiB
CLOUDSC_CLIFF_HI=1048576 # 1 MiB
CLOUDSC_SAFE_MAX=262144  # 256 KiB: "prefer decompositions that stay under this"

# Blocks needed to cover <size> columns at <nproma>, rounding up like the dwarf driver.
cloudsc_nblocks() { echo $((($1 + $2 - 1) / $2)); }

# Bytes of the LARGEST per-block transient class at <nproma>: the NCLV*(KLEV+1) arrays
# (zpfplsx and friends) dominate; the NCLV*KLEV and KLEV classes are strictly smaller.
cloudsc_transient_bytes() { echo $((CLOUDSC_NCLV * (CLOUDSC_KLEV + 1) * $1 * 8)); }

# SAFE / OVER256K / CLIFF for <nproma>; CLIFF means "do not measure DaCe here".
cloudsc_cliff_verdict() {
    local b
    b="$(cloudsc_transient_bytes "$1")"
    if [ "$b" -ge "$CLOUDSC_CLIFF_LO" ] && [ "$b" -lt "$CLOUDSC_CLIFF_HI" ]; then
        echo CLIFF
    elif [ "$b" -gt "$CLOUDSC_SAFE_MAX" ]; then
        echo OVER256K
    else
        echo SAFE
    fi
}

# Every (size, nproma) point of the sweep, one "size nproma nblocks bytes verdict" line.
cloudsc_sweep_points() {
    local s n
    for s in $CLOUDSC_SWEEP_SIZES; do
        for n in $CLOUDSC_SWEEP_NPROMAS; do
            printf '%s %s %s %s %s\n' "$s" "$n" "$(cloudsc_nblocks "$s" "$n")" \
                "$(cloudsc_transient_bytes "$n")" "$(cloudsc_cliff_verdict "$n")"
        done
    done
}

cloudsc_print_size_table() {
    local s n nb by vd bad=0
    echo "CloudSC sweep: KLEV=$CLOUDSC_KLEV NCLV=$CLOUDSC_NCLV canonical=$CLOUDSC_CANONICAL_SIZE"
    echo "mimalloc cliff band = [$CLOUDSC_CLIFF_LO, $CLOUDSC_CLIFF_HI) bytes; safe <= $CLOUDSC_SAFE_MAX"
    printf '%10s %8s %9s %14s  %s\n' NGPTOT NPROMA NGPBLKS "max-transient" verdict
    while read -r s n nb by vd; do
        printf '%10s %8s %9s %11s KiB  %s\n' "$s" "$n" "$nb" "$((by / 1024))" "$vd"
        [ "$vd" = CLIFF ] && bad=1
        [ $((s % n)) -eq 0 ] || echo "    WARNING: NPROMA=$n does not divide NGPTOT=$s (ragged last block)"
    done < <(cloudsc_sweep_points)
    if [ "$bad" = 1 ]; then
        echo "CLOUDSC_SIZE_TABLE_EXIT=1  (a listed NPROMA lands DaCe transients in the mimalloc cliff)"
        return 1
    fi
    echo "CLOUDSC_SIZE_TABLE_EXIT=0"
    return 0
}

# Executed rather than sourced -> print and exit with the verdict.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    cloudsc_print_size_table
    exit $?
fi
