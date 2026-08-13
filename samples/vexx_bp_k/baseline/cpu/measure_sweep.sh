#!/bin/bash
# Node-partition sweep of the verified CPU baseline kernel across
# (MPI ranks x OMP_NUM_THREADS).  Each rank is an INDEPENDENT kernel instance
# (negrp==1 kernel, COMM_SELF, no inter-rank communication) loading the same
# dump slot on its own disjoint core set -- the sweep measures how per-kernel
# latency behaves under node partitioning / co-location.  Every rank verifies
# against the dumped ground truth before its time counts (VERIFY gate).
#
# WARMUP/REPS live HERE, not in the driver: verify_vexx performs exactly one
# verified kernel call per invocation, so each repetition is a fresh lane
# launch (fresh processes: FFT planning and the coulomb-cache build happen
# inside every timed call).  That makes every rep a like-for-like
# "first application" measurement, uniform across lanes and reps; the first
# WARMUP invocations are still PASS-gated but their rows are discarded.
#
#   usage:  ./measure_sweep.sh            run the sweep
#           ./measure_sweep.sh --list     print the lanes + count, do not run
#   env:    TOTAL=128            physical cores on the node
#           MIN_CORES=32         skip lanes using fewer total cores
#           NPS="1 2 4 8 16 32"  MPI rank counts to try
#           NTS="1 2 4 8 16 32 64 128"   OMP thread counts to try
#           (lane runs iff MIN_CORES <= np*nt <= TOTAL)
#           WARMUP=1 REPS=10     discarded / recorded invocations per lane
#           MAT=BaTiO3_nat005|BaO_nat002   material deck under ../../data
#           SLOT=0 MODE=full DUMP=../../data/$MAT CSV=measure_sweep.csv
#           LAUNCHER=mpirun|srun   (srun for Cray/SLURM: srun -n np -c nt -l)
#           MPIRUN_EXTRA=...
set -u
cd "$(dirname "$0")"
MAT=${MAT:-BaTiO3_nat005}
DUMP=${DUMP:-../../data/$MAT}
SLOT=${SLOT:-0}
MODE=${MODE:-full}
TOTAL=${TOTAL:-128}
MIN_CORES=${MIN_CORES:-32}
NPS=${NPS:-"1 2 4 8 16 32"}
NTS=${NTS:-"1 2 4 8 16 32 64 128"}
WARMUP=${WARMUP:-1}
REPS=${REPS:-10}
CSV=${CSV:-measure_sweep.csv}
LAUNCHER=${LAUNCHER:-mpirun}
MPIRUN_EXTRA=${MPIRUN_EXTRA:-}
[ "$(id -u)" = "0" ] && MPIRUN_EXTRA="$MPIRUN_EXTRA --allow-run-as-root"

ulimit -s unlimited 2>/dev/null || echo "warning: could not raise stack limit" >&2
export OMP_STACKSIZE=${OMP_STACKSIZE:-512M}
export OMP_PLACES=${OMP_PLACES:-cores}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export UCX_TLS=${UCX_TLS:-self,sysv,cma}
# nested libraries must not spawn their own thread pools inside the kernel's
# OMP region (see build.sh toolchain note)
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

# --list: enumerate the lanes the current env selects, then exit
if [ "${1:-}" = "--list" ] || [ "${1:-}" = "-l" ]; then
  echo "lanes for TOTAL=$TOTAL MIN_CORES=$MIN_CORES NPS=\"$NPS\" NTS=\"$NTS\" (deck $DUMP, slot $SLOT / $MODE):"
  count=0
  printf "%-6s %-9s %s\n" "np" "threads" "cores"
  for np in $NPS; do
    for nt in $NTS; do
      cores=$(( np * nt ))
      [ $cores -lt $MIN_CORES ] && continue
      [ $cores -gt $TOTAL ] && continue
      printf "%-6s %-9s %s\n" "$np" "$nt" "$cores"
      count=$(( count + 1 ))
    done
  done
  echo "total: $count lanes"
  exit 0
fi

[ -x ./verify_vexx ] || { echo "run ./build.sh first" >&2; exit 1; }
[ -f "$DUMP/vexx_${SLOT}_meta.txt" ] || { echo "no dump slot $SLOT in $DUMP" >&2; exit 1; }

echo "np,threads,cores,rep,rank,kernel_s,verified" > "$CSV"
overall=0
for np in $NPS; do
  for nt in $NTS; do
    cores=$(( np * nt ))
    [ $cores -lt $MIN_CORES ] && continue
    [ $cores -gt $TOTAL ] && continue
    echo
    echo "==== np=$np x threads=$nt  ($cores cores)  slot $SLOT / $MODE  warmup=$WARMUP reps=$REPS ===="
    if [ "$LAUNCHER" = "srun" ]; then
      launch=(srun -n "$np" -c "$nt" --cpu-bind=cores -l)
    else
      launch=(mpirun $MPIRUN_EXTRA -np "$np" --map-by "slot:PE=$nt" --bind-to core --tag-output)
    fi
    lane_ok=1
    lane_times=""
    for inv in $(seq 1 $(( WARMUP + REPS ))); do
      out=$(OMP_NUM_THREADS=$nt "${launch[@]}" ./verify_vexx "$DUMP" "$SLOT" "$MODE" 2>&1)
      rc=$?
      npass=$(echo "$out" | grep -c "VERIFY: PASS")
      if [ $rc -ne 0 ] || [ "$npass" -ne "$np" ]; then
        echo "LANE FAILED at invocation $inv (rc=$rc, $npass/$np ranks verified) -- lane discarded"
        echo "$out" | grep -E "VERIFY: FAIL|FATAL|Error|error stop" | head -5
        lane_ok=0
        overall=1
        break
      fi
      if [ "$inv" -le "$WARMUP" ]; then
        echo "  warmup $inv/$WARMUP done (discarded)"
        continue
      fi
      rep=$(( inv - WARMUP ))
      # Per-rank rows.  Launcher prefixes vary too much for strict regexes
      # (OpenMPI --tag-output "[job,rank]<stdout>:" where job may contain
      # dots; srun -l "rank:" possibly padded), so: the TIME is always the
      # LAST field of a kernel_time_s line; the rank is taken from the prefix
      # when one parses, else falls back to the line's order in the output.
      rows=$(echo "$out" | grep "kernel_time_s" | sed -E \
          -e 's/^[[:space:]]*\[[^]]*,([0-9]+)\][^:]*:[[:space:]]*/\1 /' \
          -e 's/^[[:space:]]*([0-9]+):[[:space:]]*/\1 /' | \
        awk '$NF ~ /^[0-9]+([.][0-9]*)?$/ {
               rank = ($1 ~ /^[0-9]+$/ && $2 == "kernel_time_s") ? $1 : NR-1
               printf "%d %s\n", rank, $NF }')
      if [ -z "$rows" ]; then
        echo "LANE FAILED at rep $rep: no kernel_time_s parsed from launcher output"
        echo "$out" | grep "kernel_time_s" | head -3
        lane_ok=0
        overall=1
        break
      fi
      echo "$rows" | awk -v np=$np -v nt=$nt -v c=$cores -v r=$rep \
          '{printf "%d,%d,%d,%d,%d,%s,PASS\n", np, nt, c, r, $1, $2}' >> "$CSV"
      lane_times="$lane_times $(echo "$rows" | awk '{print $2}')"
      echo "  rep $rep/$REPS: $(echo "$rows" | awk '{printf "%s ", $2}')"
    done
    if [ "$lane_ok" = "1" ]; then
      echo "$lane_times" | tr " " "\n" | awk 'NF{v=$1; s+=v; if(mn==""||v<mn)mn=v; if(v>mx)mx=v; k++}
         END{if (k==0) {print "lane summary: NO SAMPLES"; exit 1}
             printf "lane summary: samples=%d (ranks x reps)  kernel_s min=%.3f max=%.3f mean=%.3f\n", k, mn, mx, s/k}'
    fi
  done
done
echo
echo "==== $CSV ===="
awk -F, '{printf "%-4s %-8s %-6s %-4s %-5s %-10s %s\n", $1, $2, $3, $4, $5, $6, $7}' "$CSV"
exit $overall
