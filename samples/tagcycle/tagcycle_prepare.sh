#!/usr/bin/env bash
# Step 1 of the tag cycle, LOGIN SIDE: check that this repo is a clean tree at the tag you are
# about to measure, create the gitignored work root, and print the sbatch lines for steps 2-4.
# Submits nothing, and -- unlike the pinned-mirror version this replaces -- checks out nothing,
# copies no scripts to a sibling directory and freezes no tree.
#   usage: tagcycle_prepare.sh [<tag-or-sha>]      (no argument = whatever HEAD describes as)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-$REPO/samples/_work}"

[ -d "$REPO/.git" ] || {
    echo "FATAL: $REPO is not a git work tree" >&2
    exit 1
}

TAG_DESC="$(git -C "$REPO" describe --always --dirty)"
SHA="$(git -C "$REPO" rev-parse HEAD)"

# The .sdfgz cache key and every EXPECTED_TAG check embed this string, so a modified tracked file
# would silently key a cache to a tag that does not describe the code inside it.
case "$TAG_DESC" in
    *-dirty)
        echo "FATAL: repo describes as $TAG_DESC -- a tracked file is modified." >&2
        echo "  Commit or stash before starting a cycle; the jobs will refuse to run." >&2
        exit 1
        ;;
esac

# The mirror used to be moved to the requested tag by this script. It no longer moves anything:
# you check out the tag yourself, and an argument here is only an assertion that you did.
TAG="${1:-$TAG_DESC}"
if [ "$TAG" != "$TAG_DESC" ]; then
    echo "FATAL: asked for $TAG but the repo describes as $TAG_DESC" >&2
    echo "  git -C $REPO checkout $TAG   then re-run this script" >&2
    exit 1
fi

# slurm resolves the #SBATCH -o/-e paths against the SUBMIT DIRECTORY and will not create it, so
# the whole cycle is submitted from the repo root and the log dir is made here.
mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/meas/runs" "$WORK_ROOT/meas/logs" "$WORK_ROOT/dev"

echo "REPO=$REPO"
echo "WORK_ROOT=$WORK_ROOT"
echo "TAG_DESC=$TAG_DESC"
echo "HEAD=$SHA"
echo "PREPARE_EXIT=0"

# Untracked leftovers git status will not clear, but which do eat quota.
cores="$(find "$REPO" -maxdepth 1 -name 'core_*' | wc -l)"
[ "$cores" -eq 0 ] || echo "NOTE: $cores untracked core dumps in $REPO (~10 GB) -- rm at will"

cat <<EOF

--- steps 2-4, run these yourself FROM $REPO (nothing was submitted) ---------
cd $REPO
# 2. bridge rebuild at $TAG_DESC (own job, no srun anywhere: cmake configure deadlocks
#    under srun's blocked SIGCHLD, and build_bridge's cmake has no unblock shim)
B=\$(sbatch --parsable --export=ALL,TAG=$TAG_DESC samples/tagcycle/tagcycle_bridge.sbatch)
echo "bridge \$B"
# verdict:  grep -E 'BRIDGE_BUILD_EXIT=' samples/_work/logs/tagcycle_bridge-\$B.out  -> must be =0

# 3. warms, all normal partition, all parallel behind the bridge. Every build writes under
#    BUILD_ROOT/GPUROOT inside $WORK_ROOT, never into the repo tree itself.
W1=\$(sbatch --parsable --dependency=afterok:\$B --export=ALL,TAG=$TAG_DESC,CHUNK=small samples/tagcycle/cpu_warm.sbatch)
W2=\$(sbatch --parsable --dependency=afterok:\$B --export=ALL,TAG=$TAG_DESC,CHUNK=large samples/tagcycle/cpu_warm.sbatch)
W3=\$(sbatch --parsable --dependency=afterok:\$B --export=ALL,TAG=$TAG_DESC samples/tagcycle/cpu_velocity_warm.sbatch)
W4=\$(sbatch --parsable --dependency=afterok:\$B --export=ALL,TAG=$TAG_DESC samples/tagcycle/gpu_warm.sbatch)
W5=\$(sbatch --parsable --dependency=afterok:\$B --export=ALL,TAG=$TAG_DESC samples/tagcycle/tagcycle_warm.sbatch)
echo "warms \$W1 \$W2 \$W3 \$W4 \$W5"
# verdict:  grep -E '_EXIT=|ARGLIST_DIFF_|TAG_STABLE=' on each samples/_work/logs/*-\$W*.out
#           all *_EXIT=0, all ARGLIST_DIFF_*=OK, TAG_STABLE=1 (tagcycle_warm only)
#
# DO NOT touch the working tree between here and the last meas job: every job re-asserts
# 'git describe --always --dirty' == $TAG_DESC and aborts if you moved or dirtied it.

# 4. meas, all debug partition, ONE AT A TIME: afterok on the warm that feeds each job,
#    afterany on the previous meas job so only one measures at any moment.
M1=\$(sbatch --parsable --dependency=afterok:\$W5 --export=ALL,TAG=$TAG_DESC samples/tagcycle/meas_4rank.sbatch)
M2=\$(sbatch --parsable --dependency=afterok:\$W1:\$W2,afterany:\$M1 --export=ALL,TAG=$TAG_DESC,THREADS=72,REPS=50 samples/tagcycle/cpu_sweep_cloudsc.sbatch)
M3=\$(sbatch --parsable --dependency=afterok:\$W1:\$W2,afterany:\$M2 --export=ALL,TAG=$TAG_DESC,THREADS=1,REPS=10 samples/tagcycle/cpu_sweep_cloudsc.sbatch)
M4=\$(sbatch --parsable --dependency=afterok:\$W1:\$W5,afterany:\$M3 --export=ALL,TAG=$TAG_DESC samples/tagcycle/cpu_cloudsc_efficiency.sbatch)
M5=\$(sbatch --parsable --dependency=afterok:\$W3,afterany:\$M4 --export=ALL,TAG=$TAG_DESC samples/tagcycle/cpu_velocity_r02b06.sbatch)
M6=\$(sbatch --parsable --dependency=afterok:\$W4,afterany:\$M5 --export=ALL,TAG=$TAG_DESC samples/tagcycle/gpu_meas_cloudsc.sbatch)
M7=\$(sbatch --parsable --dependency=afterok:\$W4,afterany:\$M6 --export=ALL,TAG=$TAG_DESC samples/tagcycle/gpu_meas_velocity.sbatch)
echo "meas \$M1 \$M2 \$M3 \$M4 \$M5 \$M6 \$M7"
# verdict:  grep -E '_EXIT=' on each meas .out -> every stage =0
-----------------------------------------------------------------------------
EOF
