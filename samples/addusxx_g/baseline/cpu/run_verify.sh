#!/bin/bash
# Correctness gate: verify the standalone addusxx_g kernel against every
# dumped set of one material deck at the requested thread counts.  No timing.
#   usage: ./run_verify.sh [dumpdir] ["thread list"]
#   env:   MAT=BaO_nat002|BaTiO3_nat005   material deck under ../../data
#          (an explicit [dumpdir] argument overrides MAT)
set -u
cd "$(dirname "$0")"
MAT=${MAT:-BaO_nat002}
DUMP=${1:-../../data/$MAT}
THREADS=${2:-"32"}
if ! compgen -G "$DUMP/adx_*_meta.txt" > /dev/null 2>&1; then
  echo "ERROR: no deck at '$DUMP' (no adx_*_meta.txt)." >&2
  echo "       available under ../../data: $(ls ../../data 2>/dev/null | tr '\n' ' ')" >&2
  echo "       use MAT=<name> or pass a dumpdir; ../../download_data.sh fetches decks." >&2
  exit 1
fi
# large-stack guard, mirrored from the vexx baseline (gfortran temporaries)
ulimit -s unlimited 2>/dev/null || echo "warning: could not raise stack limit" >&2
export OMP_STACKSIZE=${OMP_STACKSIZE:-512M}
export OMP_PLACES=${OMP_PLACES:-cores}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}

overall=0
nchecked=0
for nt in $THREADS; do
  export OMP_NUM_THREADS=$nt
  echo; echo "################ OMP_NUM_THREADS=$nt ################"
  for set in 0 1 2 3; do
    [ -f "$DUMP/adx_${set}_meta.txt" ] || { echo "set $set: no dump, skip"; continue; }
    echo "---- set $set ----"
    if ! ./verify_addusxx "$DUMP" "$set"; then
      echo "set $set @ $nt threads: FAILED"
      overall=1
    fi
    nchecked=$(( nchecked + 1 ))
  done
done
# a gate that checked nothing must not pass
[ $nchecked -eq 0 ] && { echo; echo "ERROR: 0 verifications ran"; exit 1; }
echo; [ $overall -eq 0 ] && echo "ALL $nchecked VERIFICATIONS PASSED" || echo "SOME VERIFICATIONS FAILED"
exit $overall
