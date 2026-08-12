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
#SBATCH -o samples/_work/logs/tagcycle_warm-%j.out
#SBATCH -e samples/_work/logs/tagcycle_warm-%j.out
# Step 3 of the tag cycle: phase A for the 4 measurement variants at $TAG (vexx deliberately NOT
# in the set), then the arglist diff against the 094035d baseline.
#
# SUBMIT FROM THE REPO ROOT: slurm resolves the -o/-e paths above against the submit dir.
#
# There is no refreeze any more.  The pinned mirror clone is gone -- jobs run out of the working
# repo, which must never be chmod'ed read-only.  What the freeze bought is now two other guards:
# the EXPECTED_TAG check below (git describe --dirty, so a moved OR modified tree refuses to run)
# and DACE_FORTRAN_NO_REBUILD=1 in every measurement job (a phase A cache miss dies, never builds).
#
# WARM_PARALLEL=1 (default) runs the 4 variants as 4 socket-pinned srun steps at once. Sequential
# (WARM_PARALLEL=0) is correct too but measured ~105 min of phase A at 094035d, i.e. over this
# walltime -- raise -t to 03:00:00 if you take that route.
#   sbatch --export=ALL,TAG=<tag> samples/tagcycle/tagcycle_warm.sbatch
set -uo pipefail

# Under sbatch, BASH_SOURCE is the slurm spool copy -- fall back to the submit dir (the repo root).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
[ -d "$REPO/samples/tagcycle" ] || REPO="${SLURM_SUBMIT_DIR:-$PWD}"
WORK_ROOT="${WORK_ROOT:-$REPO/samples/_work}"
TAGDIR="$REPO/samples/tagcycle"
# MEAS/BR2 keep their pinned-mirror names: the tree under test (now the repo) and the output root.
MEAS="${MEAS:-$REPO}"
BR2="${BR2:-$WORK_ROOT/meas}"
LOG="$BR2/logs"
VARIANTS="${VARIANTS:-cloudsc-klon cloudsc-nblks velocity-loopexch velocity-noloopexch}"
BASE_TAG="${BASE_TAG:-094035d}"
BASE_INDEX="$BR2/arglists_${BASE_TAG}.index"
WARM_PARALLEL="${WARM_PARALLEL:-1}"
: "${TAG:?TAG must be exported into this job (--export=ALL,TAG=<tag>)}"

mkdir -p "$LOG"
cd "$MEAS" || exit 1
ulimit -c 0
# shellcheck disable=SC1091
. samples/env.sh > /dev/null 2>&1

# Phase A BUILDS -- the whole point. NO_REBUILD belongs to the measurement job only.
unset DACE_FORTRAN_NO_REBUILD
export EXPECTED_TAG="$TAG" REPO WORK_ROOT MEAS BR2
export DACE_compiler_allow_view_arguments=false
# dump_arglists.py loads .sdfgz, so it needs the dace-carrying interpreter samples/env.sh exports.
# python3.11 before python3: the system python3 on these nodes is 3.6, which no sample runs on.
DUMP_PYTHON="${DUMP_PYTHON:-${PYTHON:-$(command -v python3.11 || command -v python3 || true)}}"
[ -x "$DUMP_PYTHON" ] || {
    echo "ABORT: no usable interpreter at '$DUMP_PYTHON' -- set PYTHON in samples/env.sh" >&2
    exit 1
}

TAG_NOW="$(git -C "$MEAS" describe --always --dirty)"
echo "WARM host=$(hostname) tag=$TAG_NOW expected=$TAG parallel=$WARM_PARALLEL start=$(date -u +%FT%TZ)"
[ "$TAG_NOW" = "$TAG" ] || {
    echo "ABORT: repo describes as $TAG_NOW, expected $TAG (check out the tag, clean the tree)" >&2
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
    bash "$TAGDIR/tagcycle_lane.sh" warm "$1" > "$LOG/warm_${1}_${TAG}.log" 2>&1
}

if [ "$WARM_PARALLEL" = 1 ]; then
    declare -A PID
    for v in $VARIANTS; do
        # NEVER numactl inside a step: this node exposes 36 NUMA nodes and --cpunodebind against
        # an srun cpuset returns sched_setaffinity EINVAL (killed 4390808/4390809).
        srun --exact --ntasks=1 --cpus-per-task=72 --cpu-bind=cores --mem-bind=local \
            bash "$TAGDIR/tagcycle_lane.sh" warm "$v" > "$LOG/warm_${v}_${TAG}.log" 2>&1 &
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
    mapfile -t arts < <(EXPECTED_TAG="$TAG" bash "$TAGDIR/tagcycle_lane.sh" artifacts "$v")
    if [ "${#arts[@]}" -eq 0 ]; then
        echo "ARGLIST_DIFF_${v}=DIVERGED (no .sdfgz at $TAG)"
        diff_all=1
        continue
    fi
    "$DUMP_PYTHON" "$TAGDIR/dump_arglists.py" "$BR2" "${arts[@]}" >> "$NEW_INDEX"
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

# --- verdict ------------------------------------------------------------------
# Re-assert the tag: a build that raced an edit to the working tree would otherwise leave caches
# keyed to $TAG that no longer describe the code in them.
overall=0
[ "$warm_all" -eq 0 ] && [ "$diff_all" -eq 0 ] || overall=1
TAG_END="$(git -C "$MEAS" describe --always --dirty)"
if [ "$TAG_END" = "$TAG" ]; then
    echo "TAG_STABLE=1"
else
    echo "TAG_STABLE=0 (tree moved to $TAG_END during phase A -- the caches are not at $TAG)"
    overall=1
fi
echo "WARM_ALL_EXIT=$overall"
echo "WARM_DONE end=$(date -u +%FT%TZ)"
exit "$overall"
