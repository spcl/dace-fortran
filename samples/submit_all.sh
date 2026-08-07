#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# env knobs (THREADS, LANES, CLOUDSC_*, VEXX_*) reach jobs via sbatch --export=ALL default
JOBS=(
    cloudsc/run_cloudsc_klon.sbatch
    cloudsc/run_cloudsc_nblks.sbatch
    vexx/run_vexx_bands.sbatch
    vexx/run_vexx_grid.sbatch
    velocity_tendencies/run_velocity.sbatch
)

for job in "${JOBS[@]}"; do
    script="$HERE/$job"
    if [[ ! -f "$script" ]]; then
        echo "missing sbatch file: $script" >&2
        exit 1
    fi
    jid="$(sbatch --parsable "$@" "$script")"
    echo "$jid $(basename "$job" .sbatch)"
done
