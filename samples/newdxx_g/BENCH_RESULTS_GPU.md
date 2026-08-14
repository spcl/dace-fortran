# newdxx_g — full GPU offload of the SDFG (2026-08-15)

Full-GPU offload of `outputs/newdxx_g_original.sdfg` via `offload_newdxx.py`
(clone of `samples/addusxx_g/offload_addusxx.py`; method:
`samples/velocity_tendencies/offload_velocity.py`).  Output:
`outputs/newdxx_g_gpu.sdfg` — the block dot product stays a BLAS `Dot`
LIBRARY NODE and the phase dot product a `Reduce` node, both running on the
GPU (expansion at compile time).

Harness: `bench_newdxx_gpu.py` — calls the compiled SDFG directly from Python
against the QE dump decks (`data/<MAT>`, `ndx_<set>_*`), same correctness gate
as `verify_newdxx.f90` on the inout `deexx`
(`max|out-ref| <= 1e-11 * max(1, max|ref|)`); every TIMED call is
re-verified.  **Timing setup: 3 warmup calls + 20 timed reps per dump set**
(`--warmup`/`--reps`; defaults in the argparse block of
`bench_newdxx_gpu.py`).  The copy states are Timer-instrumented: reported are
the FULL call and **execution-only = full − the H2D copy-in state**.

Env: container qe2dace, RTX 3090 (sm_86), CUDA 13.2, DaCe 2.0.0a5,
`PYTHONHASHSEED=0`, `max_concurrent_streams=-1`, default block size 32.

## daint GH200 (sm_90), compute-only — job 4479163, 2026-08-14

Current methodology, and the one to quote.  `bench_newdxx_gpu.py` no longer times a whole
call: `samples/qe_gpu_timing.py` gates the SDFG's copy states behind the symbols
`qe_copy_in` / `qe_compute` / `qe_copy_out`, and each rep runs
copy-in (untimed) -> **COMPUTE (timed)** -> copy-out (untimed) -> verify.  `__DACE_NO_SYNC=1`
is pinned before `import dace`, so the graph's only sync is a trailing `_qe_sync` state and a
phase call returns exactly when its own GPU work has drained.  **Nothing is subtracted** and
the copies are a separate line item.  Run via `samples/run_qe_gpu.sbatch` (1 node, 2 ranks,
1 GPU each; cache warmed on the login node, `cuda_arch=90`).

| deck | set | COMPUTE ms | H2D ms | D2H ms | verify (rel) |
|---|---|---|---|---|---|
| BaO_nat002 | 0 | **35.61** | 1.86 | 1.05 | 2.2e-19 PASS |
| BaO_nat002 | 1 | **35.58** | 1.83 | 1.02 | 6.9e-18 PASS |
| BaTiO3_nat005 | 0 | **118.09** | 3.09 | 1.08 | 3.4e-21 PASS |
| BaTiO3_nat005 | 1 | **118.10** | 3.09 | 1.08 | 1.7e-18 PASS |

20 reps/set, spread min-max < 0.3%.  Phase gating needs one extra repair on this kernel:
`gpu_auxvc` is the graph's only `AllocationLifetime.Scope` device transient, and DaCe emits its
`cudaMalloc` at first use (inside the compute phase) but its `cudaFree` at the end of the SDFG
scope (outside every phase), so skipping compute freed an uninitialized pointer.
`_hoist_scope_gpu_transients` gives it Persistent lifetime and the `dfftt_nl_d0` bound —
`auxvc` and `nl` share the G-vector axis, replication included.

## Results on RTX 3090 (older harness: FULL-CALL timing, copies included)

Kept for provenance.  The `execution` column here is `full - H2D copy-in`, a subtraction the
current harness does not do; it is not comparable to the COMPUTE column above.

| variant | BaO full | BaO **execution** | BaTiO3 full | BaTiO3 **execution** | correctness (rel) |
|---|---|---|---|---|---|
| GPU offload (shipped `newdxx_g_gpu.sdfg`) | 85.1 | **69.7** | 260.9 | **222.5** | ≤7e-18 PASS |

Measured H2D copy-in state (median): 15.4 ms on BaO / 38.4 ms on BaTiO3
(qgm-dominated, same uploads as addusxx_g).  D2H copy-out (deexx only, <1 KB):
~0.01 ms.  No launch-collapse ablation exists: unlike addusxx_g, the collapse
fixed point is a NO-OP here (the CPU pipeline already left few launches), so
naive and shipped variants coincide.

Reference points (BENCH_RESULTS.md):

| reference | BaO_nat002 | BaTiO3_nat005 |
|---|---|---|
| DaCe CPU binding, base build (1T; thread-insensitive) | 352 | 938 |
| DaCe CPU binding, opt build, 32T (scales 1.44x/2.34x) | 244 | 402 |

Execution-only vs the CPU numbers: 5.1x / 4.2x against the serial base build,
3.5x / 1.8x against the best 32-thread CPU build.  Numerics are
machine-precision against the pw.x dumps (rel 3e-21 … 7e-18 across decks/sets).

## Why the speedup is smaller than addusxx_g's

addusxx_g was launch-bound and the collapse fixed that (9-11x).  newdxx_g is
KERNEL-PARALLELISM-bound: per (iblock, na) there are only two kernels, but the
projector kernel is a map over the projector index — **nh(nt) = 8–19 threads
per launch** — and each thread runs the jh loop with two sequential
256-element aux1 passes plus a sequential 256-element `Dot` per jh.  The GPU
runs a handful of threads while the block dot products — the natural 256-wide
parallel axis — execute serially inside each thread.  The structure faithfully
mirrors what LoopToMap chose on the CPU (parallel over the independent deexx
index); recasting the per-(ih, jh) dots as block-wide parallel reductions (or
the whole update as a small qgm^T-GEMV per block) is the next transformation.

## Kernel structure of the shipped SDFG (4 static launch sites)

Once per call: the eigqts kernel (with the in-kernel 3-element `Reduce`,
Sequential + identity=0) and the `auxvc(g) = vc(nl(g))` gather kernel over the
full G sphere (indirection in-kernel).  Per (iblock, na | tvanp): the block
structure-factor kernel (`aux2(s) = conj(auxvc)*eigqts*eigts1/2/3` via mill
indirection, 256 threads) and the projector kernel (map over the deexx index;
jh loop, aux1 passes, in-kernel `Dot` with impl=pure, `deexx += fact*dot` and
the gamma G=0 correction states inside).  ~295 launches/call on BaO, ~1080 on
BaTiO3 — launch overhead is NOT the bottleneck here.

## Repro

    cd /workspace/dace-fortran/samples/newdxx_g
    PYTHONHASHSEED=0 python3 offload_newdxx.py             # -> outputs/newdxx_g_gpu.sdfg
    PYTHONHASHSEED=0 python3 bench_newdxx_gpu.py --deck data/BaO_nat002
    PYTHONHASHSEED=0 python3 bench_newdxx_gpu.py --deck data/BaTiO3_nat005

See `OFFLOADING_TO_GPU.md` for the process findings, including the THREE DaCe
2.0.0a5 issues this kernel hit (the two addusxx_g ones plus a new nested-arg
const-qualifier bug worked around at compile time in the bench script).
