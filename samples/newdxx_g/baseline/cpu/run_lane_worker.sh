#!/usr/bin/env bash
# One lane of the QE us_exx CPU experiment, confined to one exclusive socket by the caller's
# `srun --exact --cpus-per-task=<n> --cpu-bind=cores` (never numactl).  Caller exports REPO,
# CSV (per-rank) and BUILD_ROOT_BASE (per-lane); TOPO_CPUSET_ONLY=1 confines probe_topology to
# this process's affinity mask instead of the whole node.
#
# Timing is steady-state, not cold-start: one process per (deck, set, deck_rep, threads) runs the
# driver's own bench mode (WARMUP discarded + REPS timed calls in-process), and the driver verifies
# EVERY timed call against the dumped pw.x ground truth, so a lane that drifts numerically cannot
# contribute rows.
#
# DECK_REPS is the deck-replication sweep (see samples/qe_deck_replicate.py and the kernel README).
# Each value R is passed to the driver as DECK_REP=R, which grows the problem R-fold: addusxx_g
# along the G-vector axis, newdxx_g along the atom axis.  It is recorded in the `deck_rep` CSV
# column -- NOT the pre-existing `rep` column, which is the timing loop's repetition INDEX (0..REPS-1)
# and means something else entirely.  `inputs` is also tagged set<n>_rep<R> for R>1 so figures built
# on the shared schema (which derives problem_size from the nnr/ngmt columns, and those are the
# DECK's static sizes either way) cannot silently merge different R into one bar.
set -euo pipefail

# The replication negative control deliberately produces WRONG results; it belongs in the
# verification path (DECK_REP_NOOFFSET=1 against verify_*), never in a timing CSV.
if [ -n "${DECK_REP_NOOFFSET:-}" ]; then
    echo "FATAL: DECK_REP_NOOFFSET is set. That is the replication negative control, which fails" >&2
    echo "  verification by design; it must never emit measurement rows. Unset it to measure." >&2
    exit 1
fi

LANE="$1"
: "${REPO:?REPO must be exported}"
: "${CSV:?CSV must be exported}"
: "${BUILD_ROOT_BASE:?BUILD_ROOT_BASE must be exported}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL="$(cd "$HERE/../.." && basename "$PWD")"
KDIR="$(cd "$HERE/../.." && pwd)"

# shellcheck disable=SC1091
source "$REPO/samples/common.sh"
export TOPO_CPUSET_ONLY=1
probe_topology
probe_compilers
echo "lane $LANE ($KERNEL): $(hostname) cpuset=$TOPO_NCORES cpus"

MATS="${MATS:-BaO_nat002 BaTiO3_nat005}"
REPS="${REPS:-20}"
WARMUP="${WARMUP:-3}"
DECK_REPS="${DECK_REPS:-1 64}"
DATA_ROOT="${DATA_ROOT:-$KDIR/data}"
ALLOC="${MEAS_ALLOC_ACTIVE:-system}"
# deck_rep is APPENDED: f2dace_style.load_runs takes the two size columns positionally as
# cols[2]/cols[3], so nothing may be inserted before `threads`.
CSV_HEADER="kernel,mode,nnr,ngmt,threads,rep,ms,inputs,lane,alloc,deck_rep"

unset BUILD_ROOT
setup_build_root "${KERNEL}_${LANE}"
lock_build_root

DRIVER="$REPO/samples/qe_cpu_perf.py"
case "$LANE" in
    original-openmp | flang-openmp)
        BIN="$BUILD_ROOT/$(basename "$(ls "$HERE"/verify_*.f90 | head -1)" .f90)"
        LANE="$LANE" BUILD_DIR="$BUILD_ROOT/lib" BIN_PATH="$BIN" bash "$HERE/build.sh"
        ;;
    dace-gcc | dace-llvm)
        # The DaCe lanes are the SDFG's own C++ compiled by this lane's C++ compiler -- no
        # Fortran TU and no BLAS link (optimize() reports no top-level BLAS library node in
        # either kernel), so the ONLY OpenMP runtime in the process is the one -fopenmp pulled
        # in, and omp_preflight asserts it is this lane's and not the other family's.
        BACKEND="${LANE#dace-}"
        SO="$BUILD_ROOT/dacecache/${KERNEL}_cpu_${BACKEND}/build/lib${KERNEL}_cpu_${BACKEND}.so"
        # *.sdfg is gitignored, so the optimized SDFG is a build artifact like any other.
        OPT_SDFG="$KDIR/outputs/${KERNEL}_opt.sdfg"
        [ -f "$OPT_SDFG" ] || (cd "$KDIR" && "$PY" "optimize_sdfg_${KERNEL}.py" \
            --input "outputs/${KERNEL}_original.sdfg" --output "$OPT_SDFG")
        set_omp_env "$TOPO_NCORES"
        "$PY" "$DRIVER" --kernel "$KERNEL" --lane "$LANE" --build-only
        "$PY" "$REPO/samples/omp_preflight.py" --so "$SO" --expect "$BACKEND" --label "$LANE"
        ;;
    *)
        echo "FATAL: unknown lane $LANE (original-openmp | flang-openmp | dace-gcc | dace-llvm)" >&2
        exit 1
        ;;
esac

meta_val() { awk -v k="$2" '$1 == k { print $2; exit }' "$1"; }

[ -f "$CSV" ] || echo "$CSV_HEADER" > "$CSV"
for mat in $MATS; do
    deck="$DATA_ROOT/$mat"
    static="$deck/adxndx_static_meta.txt"
    if [ ! -f "$static" ]; then
        echo "SKIP $mat: no deck at $deck (run download_data.sh)" >&2
        continue
    fi
    nnr="$(meta_val "$static" nnr)"
    ngmt="$(meta_val "$static" ngmt)"
    # Set ids come from the deck itself: the per-set files are <pfx>_<n>_*, the shared static
    # tables are adxndx_static_* and must not be mistaken for a set.
    sets="$(ls "$deck"/*_[0-9]_meta.txt 2> /dev/null | sed -E 's/.*_([0-9]+)_meta\.txt$/\1/' | sort -u | tr '\n' ' ')"
    if [ -z "$sets" ]; then
        echo "SKIP $mat: deck has no <pfx>_<n>_meta.txt sets" >&2
        continue
    fi
    echo "deck $mat: nnr=$nnr ngmt=$ngmt sets=[$sets] deck_reps=[$DECK_REPS]"
    for iset in $sets; do
      for R in $DECK_REPS; do
        inputs="set$iset"
        [ "$R" -eq 1 ] || inputs="set${iset}_rep${R}"
        for t in $THREADS; do
            if [ "$t" -gt "$TOPO_NCORES" ]; then
                echo "skip threads=$t (> $TOPO_NCORES cpus in this cpuset)"
                continue
            fi
            set_omp_env "$t"
            log="$BUILD_ROOT/run_${mat}_${iset}_r${R}_${t}.log"
            rc=0
            if [ -n "${BACKEND:-}" ]; then
                # The DaCe driver verifies every timed call itself and appends the same 11 columns.
                "$PY" "$DRIVER" --kernel "$KERNEL" --lane "$LANE" --deck "$deck" --sets "$iset" \
                    --reps "$REPS" --warmup "$WARMUP" --rep "$R" --csv "$CSV" > "$log" 2>&1 || rc=$?
            else
                DECK_REP="$R" "$BIN" "$deck" "$iset" "$REPS" "$WARMUP" > "$log" 2>&1 || rc=$?
            fi
            if [ "$rc" -ne 0 ] || ! grep -q "VERIFY: PASS" "$log"; then
                echo "FATAL: $KERNEL $mat set $iset deck_rep=$R at $t threads did not verify (rc=$rc); tail:" >&2
                tail -10 "$log" >&2
                exit 1
            fi
            if [ -n "${BACKEND:-}" ]; then
                n="$(awk '$1 == "wrote" { print $2 }' "$log")"
            else
                # The driver echoes what it actually replicated; a silently ignored DECK_REP would
                # otherwise label R=1 timings as R=64.
                if ! grep -q "^replicate $R " "$log"; then
                    echo "FATAL: driver did not report 'replicate $R' -- DECK_REP was not honoured" >&2
                    exit 1
                fi
                n=0
                while read -r secs; do
                    awk -v k="$KERNEL" -v m="$mat" -v a="$nnr" -v b="$ngmt" -v th="$t" -v r="$n" \
                        -v s="$secs" -v i="$inputs" -v l="$LANE" -v al="$ALLOC" -v dr="$R" \
                        'BEGIN { printf "%s,%s,%s,%s,%s,%s,%.6f,%s,%s,%s,%s\n", k, m, a, b, th, r, s * 1000.0, i, l, al, dr }' \
                        >> "$CSV"
                    n=$((n + 1))
                done < <(awk '$1 == "kernel_time_s" { print $2 }' "$log")
            fi
            if [ "$n" -ne "$REPS" ]; then
                echo "FATAL: $mat set $iset deck_rep=$R at $t threads emitted $n timed calls, expected $REPS" >&2
                exit 1
            fi
            echo "  $mat $inputs threads=$t deck_rep=$R: $n reps"
        done
      done
    done
done
echo "done: $CSV ($(($(wc -l < "$CSV") - 1)) rows)"
