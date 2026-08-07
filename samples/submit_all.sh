#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# env knobs (THREADS, LANES, CLOUDSC_*, VEXX_*) reach jobs via sbatch --export=ALL default; an
# exported LANES wins for every job, <sbatch>:<LANES> below is only the default. baselines.sh
# covers both NPROMA modes, so one cloudsc job asks for it; gpu lanes stay opt-in.
JOBS=(
    "cloudsc/run_cloudsc_klon.sbatch:dace-gcc dace-llvm"
    "cloudsc/run_cloudsc_nblks.sbatch:dace-gcc dace-llvm baselines"
    "vexx/run_vexx_bands.sbatch:dace-gcc dace-llvm"
    "vexx/run_vexx_grid.sbatch:dace-gcc dace-llvm"
    "velocity_tendencies/run_velocity.sbatch:dace-gcc dace-llvm"
)

for entry in "${JOBS[@]}"; do
    job="${entry%%:*}"
    script="$HERE/$job"
    if [[ ! -f "$script" ]]; then
        echo "missing sbatch file: $script" >&2
        exit 1
    fi
    if [[ -n "${LANES:-}" ]]; then
        jid="$(sbatch --parsable "$@" "$script")"
    else
        jid="$(sbatch --parsable --export="ALL,LANES=${entry#*:}" "$@" "$script")"
    fi
    echo "$jid $(basename "$job" .sbatch)"
done
