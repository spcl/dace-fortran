#!/bin/bash
# Correctness gate: verify the standalone vexx_bp_k kernel against every
# dumped slot, in both operator flavours (full US+PAW, and norm-conserving),
# at the requested thread counts.  Verification only -- no timing.
#   usage: ./run_verify.sh [dumpdir] ["thread list"]
set -u
cd "$(dirname "$0")"
DUMP=${1:-../../data}
THREADS=${2:-"32"}
# the kernel + gfortran temporaries need a big stack (segfaults otherwise)
ulimit -s unlimited 2>/dev/null || echo "warning: could not raise stack limit" >&2
export OMP_STACKSIZE=${OMP_STACKSIZE:-512M}
export OMP_PLACES=${OMP_PLACES:-cores}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}

overall=0
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
    done
  done
done
echo; [ $overall -eq 0 ] && echo "ALL VERIFICATIONS PASSED" || echo "SOME VERIFICATIONS FAILED"
exit $overall
