#!/bin/bash
# Correctness gate: verify the standalone vexx_bp_k kernel against every
# dumped slot of one material deck, in both operator flavours (full US+PAW,
# and norm-conserving), at the requested thread counts.  No timing.
#   usage: ./run_verify.sh [dumpdir] ["thread list"]
#   env:   MAT=BaTiO3_nat005|BaO_nat002   material deck under ../../data
#          (an explicit [dumpdir] argument overrides MAT)
set -u
cd "$(dirname "$0")"
MAT=${MAT:-BaTiO3_nat005}
DUMP=${1:-../../data/$MAT}
THREADS=${2:-"32"}
if ! compgen -G "$DUMP/vexx_*_meta.txt" > /dev/null 2>&1; then
  echo "ERROR: no deck at '$DUMP' (no vexx_*_meta.txt)." >&2
  echo "       available under ../../data: $(ls ../../data 2>/dev/null | tr '\n' ' ')" >&2
  echo "       use MAT=<name> (e.g. MAT=BaO_nat002) or pass a dumpdir; ../../download_data.sh fetches decks." >&2
  exit 1
fi
# the kernel + gfortran temporaries need a big stack (segfaults otherwise)
ulimit -s unlimited 2>/dev/null || echo "warning: could not raise stack limit" >&2
export OMP_STACKSIZE=${OMP_STACKSIZE:-512M}
export OMP_PLACES=${OMP_PLACES:-cores}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}

overall=0
nchecked=0
for nt in $THREADS; do
  export OMP_NUM_THREADS=$nt
  echo; echo "################ OMP_NUM_THREADS=$nt ################"
  for slot in 0 1 2 3; do
    [ -f "$DUMP/vexx_${slot}_meta.txt" ] || { echo "slot $slot: no dump, skip"; continue; }
    for mode in full nc; do
      echo "---- slot $slot / $mode ----"
      if ! ./verify_vexx "$DUMP" "$slot" "$mode"; then
        echo "slot $slot $mode @ $nt threads: FAILED"
        overall=1
      fi
      nchecked=$(( nchecked + 1 ))
    done
  done
done
# a gate that checked nothing must not pass
[ $nchecked -eq 0 ] && { echo; echo "ERROR: 0 verifications ran"; exit 1; }
echo; [ $overall -eq 0 ] && echo "ALL $nchecked VERIFICATIONS PASSED" || echo "SOME VERIFICATIONS FAILED"
exit $overall
