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

# cpu_job_common.sh and meas_4rank exec $PROBE/tagcycle_lane.sh at run time, so a stale
# probe copy would run old lane code against the new tag. Sync everything from the tag.
echo "SYNC $PROBE <- $MEAS/samples/tagcycle @ $TAG_DESC"
cp -f "$MEAS"/samples/tagcycle/*.sbatch "$MEAS"/samples/tagcycle/*.sh \
    "$MEAS"/samples/tagcycle/dump_arglists.py "$PROBE"/
echo "SYNC_EXIT=0"

cat <<EOF

--- steps 2-4, run these yourself (nothing was submitted) --------------------
# 2. bridge rebuild at $TAG_DESC (own job, no srun anywhere: cmake configure deadlocks
#    under srun's blocked SIGCHLD, and build_bridge's cmake has no unblock shim)
B=\$(sbatch --parsable --export=ALL,TAG=$TAG_DESC $PROBE/tagcycle_bridge.sbatch)
echo "bridge \$B"
# verdict:  grep -E 'BRIDGE_BUILD_EXIT=' $PROBE/tagcycle_bridge-\$B.out   -> must be =0

# 3. warms, all normal partition, all parallel behind the bridge. tagcycle_warm is the
#    only one that refreezes the clone; every build writes under BUILD_ROOT/GPUROOT
#    outside it, so the refreeze racing the other warms is harmless.
W1=\$(sbatch --parsable --dependency=afterok:\$B --export=ALL,TAG=$TAG_DESC,CHUNK=small $PROBE/cpu_warm.sbatch)
W2=\$(sbatch --parsable --dependency=afterok:\$B --export=ALL,TAG=$TAG_DESC,CHUNK=large $PROBE/cpu_warm.sbatch)
W3=\$(sbatch --parsable --dependency=afterok:\$B --export=ALL,TAG=$TAG_DESC $PROBE/cpu_velocity_warm.sbatch)
W4=\$(sbatch --parsable --dependency=afterok:\$B --export=ALL,TAG=$TAG_DESC $PROBE/gpu_warm.sbatch)
W5=\$(sbatch --parsable --dependency=afterok:\$B --export=ALL,TAG=$TAG_DESC $PROBE/tagcycle_warm.sbatch)
echo "warms \$W1 \$W2 \$W3 \$W4 \$W5"
# verdict:  grep -E '_EXIT=|ARGLIST_DIFF_|REFROZEN=' on each -\$W*.out
#           all *_EXIT=0, all ARGLIST_DIFF_*=OK, REFROZEN=1 (tagcycle_warm only)

# 4. meas, all debug partition, ONE AT A TIME: afterok on the warm that feeds each job,
#    afterany on the previous meas job so only one measures at any moment.
M1=\$(sbatch --parsable --dependency=afterok:\$W5 --export=ALL,TAG=$TAG_DESC $PROBE/meas_4rank.sbatch)
M2=\$(sbatch --parsable --dependency=afterok:\$W1:\$W2,afterany:\$M1 --export=ALL,TAG=$TAG_DESC,THREADS=72,REPS=50 $PROBE/cpu_sweep_cloudsc.sbatch)
M3=\$(sbatch --parsable --dependency=afterok:\$W1:\$W2,afterany:\$M2 --export=ALL,TAG=$TAG_DESC,THREADS=1,REPS=10 $PROBE/cpu_sweep_cloudsc.sbatch)
M4=\$(sbatch --parsable --dependency=afterok:\$W1:\$W5,afterany:\$M3 --export=ALL,TAG=$TAG_DESC $PROBE/cpu_cloudsc_efficiency.sbatch)
M5=\$(sbatch --parsable --dependency=afterok:\$W3,afterany:\$M4 --export=ALL,TAG=$TAG_DESC $PROBE/cpu_velocity_r02b06.sbatch)
M6=\$(sbatch --parsable --dependency=afterok:\$W4,afterany:\$M5 --export=ALL,TAG=$TAG_DESC $PROBE/gpu_meas_cloudsc.sbatch)
M7=\$(sbatch --parsable --dependency=afterok:\$W4,afterany:\$M6 --export=ALL,TAG=$TAG_DESC $PROBE/gpu_meas_velocity.sbatch)
echo "meas \$M1 \$M2 \$M3 \$M4 \$M5 \$M6 \$M7"
# verdict:  grep -E '_EXIT=' on each meas .out -> every stage =0
-----------------------------------------------------------------------------
EOF
