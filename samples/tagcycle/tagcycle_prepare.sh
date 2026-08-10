#!/usr/bin/env bash
# Step 1 of the tag cycle, LOGIN SIDE: unfreeze the measurement clone and move it to <tag>.
# Submits nothing -- it prints the sbatch lines for steps 2-4 for you to run yourself.
#   usage: tagcycle_prepare.sh <tag-or-sha>
set -euo pipefail

PROBE=/capstor/scratch/cscs/ybudanaz/aarch64/probe
MEAS=/capstor/scratch/cscs/ybudanaz/aarch64/dace-fortran-meas
BR2=/capstor/scratch/cscs/ybudanaz/aarch64/dace-fortran-samples-meas

TAG="${1:-}"
[ -n "$TAG" ] || {
    echo "usage: $0 <tag-or-sha>" >&2
    exit 2
}
[ -d "$MEAS/.git" ] || {
    echo "FATAL: no git clone at $MEAS" >&2
    exit 1
}

echo "UNFREEZE $MEAS"
chmod -R u+w "$MEAS"

git -C "$MEAS" fetch --prune --tags origin
git -C "$MEAS" checkout --detach --force "$TAG"

TAG_DESC="$(git -C "$MEAS" describe --always --dirty)"
SHA="$(git -C "$MEAS" rev-parse HEAD)"
case "$TAG_DESC" in
    *-dirty)
        echo "FATAL: clone describes as $TAG_DESC -- a tracked file is modified. The .sdfgz cache" >&2
        echo "  key and every EXPECTED_TAG check embed this string; clean it before continuing." >&2
        exit 1
        ;;
esac

echo "TAG_DESC=$TAG_DESC"
echo "HEAD=$SHA"
echo "PREPARE_EXIT=0"

# Untracked leftovers checkout will not touch, but which do eat quota.
cores="$(find "$MEAS" -maxdepth 1 -name 'core_*' | wc -l)"
[ "$cores" -eq 0 ] || echo "NOTE: $cores untracked core dumps in $MEAS (~10 GB) -- rm at will"

cat <<EOF

--- steps 2-4, run these yourself (nothing was submitted) --------------------
# 2. bridge rebuild at $TAG_DESC (own job, no srun anywhere: cmake configure deadlocks
#    under srun's blocked SIGCHLD, and build_bridge's cmake has no unblock shim)
B=\$(sbatch --parsable --export=ALL,TAG=$TAG_DESC $PROBE/tagcycle_bridge.sbatch)
echo "bridge \$B"
# verdict:  grep -E 'BRIDGE_BUILD_EXIT=' $PROBE/tagcycle_bridge-\$B.out   -> must be =0

# 3. pre-warm the 4 variants + arglist diff vs $BR2/arglists_094035d.index
W=\$(sbatch --parsable --dependency=afterok:\$B --export=ALL,TAG=$TAG_DESC $PROBE/tagcycle_warm.sbatch)
echo "warm \$W"
# verdict:  grep -E 'WARM_.*_EXIT=|ARGLIST_DIFF_|REFROZEN=' $PROBE/tagcycle_warm-\$W.out
#           all WARM_*_EXIT=0, all ARGLIST_DIFF_*=OK, REFROZEN=1

# 4. THE measurement: 1 node, 4 socket-pinned ranks, one variant each
M=\$(sbatch --parsable --dependency=afterok:\$W --export=ALL,TAG=$TAG_DESC $PROBE/meas_4rank.sbatch)
echo "meas \$M"
# verdict:  grep -E 'MEAS_.*_EXIT=' $PROBE/meas_4rank-\$M.out
-----------------------------------------------------------------------------
EOF
