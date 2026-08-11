#!/bin/bash
#SBATCH -J tagcycle-warm
#SBATCH -p normal
#SBATCH -A g34
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH -t 01:30:00
#SBATCH -o /capstor/scratch/cscs/ybudanaz/aarch64/probe/tagcycle_warm-%j.out
#SBATCH -e /capstor/scratch/cscs/ybudanaz/aarch64/probe/tagcycle_warm-%j.out
# Step 3 of the tag cycle: phase A for the 4 measurement variants at $TAG (vexx deliberately NOT
# in the set), then the arglist diff against the 094035d baseline, then the refreeze.
# Needs the clone WRITABLE (build_bridge takes an flock on a file inside it, even on a no-op).
#
# WARM_PARALLEL=1 (default) runs the 4 variants as 4 socket-pinned srun steps at once. Sequential
# (WARM_PARALLEL=0) is correct too but measured ~105 min of phase A at 094035d, i.e. over this
# walltime -- raise -t to 03:00:00 if you take that route.
#   sbatch --export=ALL,TAG=<tag> tagcycle_warm.sbatch
set -uo pipefail

PROBE=/capstor/scratch/cscs/ybudanaz/aarch64/probe
MEAS=/capstor/scratch/cscs/ybudanaz/aarch64/dace-fortran-meas
BR2=/capstor/scratch/cscs/ybudanaz/aarch64/dace-fortran-samples-meas
LOG="$BR2/logs"
VARIANTS="${VARIANTS:-cloudsc-klon cloudsc-nblks velocity-loopexch velocity-noloopexch}"
BASE_TAG="${BASE_TAG:-094035d}"
BASE_INDEX="$BR2/arglists_${BASE_TAG}.index"
WARM_PARALLEL="${WARM_PARALLEL:-1}"
DUMP_PYTHON="${DUMP_PYTHON:-/capstor/scratch/cscs/ybudanaz/aarch64/venv-meas/bin/python}"
: "${TAG:?TAG must be exported into this job (--export=ALL,TAG=<tag>)}"

mkdir -p "$LOG"
cd "$MEAS" || exit 1
ulimit -c 0
# shellcheck disable=SC1091
. samples/env.sh > /dev/null 2>&1

# Phase A BUILDS -- the whole point. NO_REBUILD belongs to the measurement job only.
unset DACE_FORTRAN_NO_REBUILD
export EXPECTED_TAG="$TAG" MEAS BR2
export DACE_compiler_allow_view_arguments=false

TAG_NOW="$(git -C "$MEAS" describe --always --dirty)"
echo "WARM host=$(hostname) tag=$TAG_NOW expected=$TAG parallel=$WARM_PARALLEL start=$(date -u +%FT%TZ)"
[ "$TAG_NOW" = "$TAG" ] || {
    echo "ABORT: clone tag is $TAG_NOW, expected $TAG" >&2
    exit 1
}
[ -w "$MEAS/dace_fortran" ] || {
    echo "ABORT: $MEAS is frozen -- phase A needs it writable (build_bridge flock)" >&2
    exit 1
}

# Do this OUTSIDE any srun: if the bridge is stale it rebuilds here, where cmake can reap its
# children. Inside a step the same rebuild hangs on the blocked SIGCHLD until walltime.
"$PYTHON" -c "import dace_fortran.build_bridge as b; raise SystemExit(1 if b.needs_build() else 0)"
if [ $? -eq 0 ]; then
    echo "BRIDGE_FRESH=1"
else
    echo "BRIDGE_FRESH=0"
    echo "ABORT: bridge still stale after tagcycle_bridge.sbatch" >&2
    exit 1
fi

# --- phase A ------------------------------------------------------------------
declare -A WARM_RC
warm_one() { # warm_one <variant>
    bash "$PROBE/tagcycle_lane.sh" warm "$1" > "$LOG/warm_${1}_${TAG}.log" 2>&1
}

if [ "$WARM_PARALLEL" = 1 ]; then
    declare -A PID
    for v in $VARIANTS; do
        # NEVER numactl inside a step: this node exposes 36 NUMA nodes and --cpunodebind against
        # an srun cpuset returns sched_setaffinity EINVAL (killed 4390808/4390809).
        srun --exact --ntasks=1 --cpus-per-task=72 --cpu-bind=cores --mem-bind=local \
            bash "$PROBE/tagcycle_lane.sh" warm "$v" > "$LOG/warm_${v}_${TAG}.log" 2>&1 &
        PID[$v]=$!
    done
    for v in $VARIANTS; do
        wait "${PID[$v]}"
        WARM_RC[$v]=$?
    done
else
    for v in $VARIANTS; do
        warm_one "$v"
        WARM_RC[$v]=$?
    done
fi

warm_all=0
for v in $VARIANTS; do
    tail -15 "$LOG/warm_${v}_${TAG}.log"
    echo "WARM_${v}_EXIT=${WARM_RC[$v]}"
    [ "${WARM_RC[$v]}" -eq 0 ] || warm_all=1
done

# --- arglist collection + diff vs the baseline --------------------------------
# A new required SDFG argument (or a respelled one) is invisible until the timed call fails, so
# the signature is compared before any measurement runs.
NEW_INDEX="$BR2/arglists_${TAG}.index"
: > "$NEW_INDEX"
diff_all=0
for v in $VARIANTS; do
    mapfile -t arts < <(EXPECTED_TAG="$TAG" bash "$PROBE/tagcycle_lane.sh" artifacts "$v")
    if [ "${#arts[@]}" -eq 0 ]; then
        echo "ARGLIST_DIFF_${v}=DIVERGED (no .sdfgz at $TAG)"
        diff_all=1
        continue
    fi
    "$DUMP_PYTHON" "$PROBE/dump_arglists.py" "$BR2" "${arts[@]}" >> "$NEW_INDEX"
    verdict=OK
    for a in "${arts[@]}"; do
        new="${a%.sdfgz}.arglist.txt"
        base="${new%_${TAG}.arglist.txt}_${BASE_TAG}.arglist.txt"
        stem="$LOG/arglist_$(basename "${new%.arglist.txt}")"
        if [ ! -f "$new" ]; then
            echo "  MISSING dump: $new"
            verdict=DIVERGED
        elif [ ! -f "$base" ]; then
            echo "  NO BASELINE: $base"
            verdict=DIVERGED
        else
            diff -u <(tail -n +4 "$base") <(tail -n +4 "$new") > "$stem.diff" || true
            if ! diff -u <(tail -n +4 "$base" | cut -f2) <(tail -n +4 "$new" | cut -f2) > "$stem.names.diff"; then
                echo "  SIGNATURE CHANGED: $(basename "$new") vs $(basename "$base")"
                head -40 "$stem.names.diff"
                verdict=DIVERGED
            fi
        fi
    done
    echo "ARGLIST_DIFF_${v}=$verdict"
    [ "$verdict" = OK ] || diff_all=1
done
echo "--- $NEW_INDEX ---"
cat "$NEW_INDEX"
echo "--- baseline $BASE_INDEX ---"
cat "$BASE_INDEX" 2> /dev/null

# --- refreeze -----------------------------------------------------------------
overall=0
[ "$warm_all" -eq 0 ] && [ "$diff_all" -eq 0 ] || overall=1
if [ "$overall" -eq 0 ]; then
    chmod -R a-w "$MEAS"
    echo "REFROZEN=1"
else
    echo "REFROZEN=0"
    echo "refreeze by hand once triaged:  chmod -R a-w $MEAS"
fi
echo "WARM_ALL_EXIT=$overall"
echo "WARM_DONE end=$(date -u +%FT%TZ)"
exit "$overall"
