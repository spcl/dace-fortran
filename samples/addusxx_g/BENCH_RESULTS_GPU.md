# addusxx_g — full GPU offload of the SDFG (2026-08-15)

Full-GPU offload of `outputs/addusxx_g_original.sdfg` via
`offload_addusxx.py` (method: `samples/velocity_tendencies/offload_velocity.py`;
parallel structure: the hand-written CUDA offloads in `baseline/ref`).
Output: `outputs/addusxx_g_gpu.sdfg`, Reduce nodes kept as
LIBRARY NODES (expansion happens at compile time, on the GPU).

Harness: `bench_addusxx_gpu.py` — calls the compiled SDFG directly from Python
against the QE dump decks (`data/<MAT>`), same correctness gate as
`verify_addusxx.f90` (`max|out-ref| <= 1e-11 * max(1, max|ref|)`); every TIMED
call is also re-verified.  **Timing setup: 3 warmup calls + 20 timed reps per
dump set** (`--warmup`/`--reps`; defaults live in the argparse block of
`bench_addusxx_gpu.py`).  The copy states are Timer-instrumented: reported are
the FULL call (whole SDFG) and **execution-only = full − the H2D copy-in
state** (the number that excludes host-to-device copies; it still contains the
host launch loop, kernel drain, and the ~1 MB rhoc D2H).

Env: container qe2dace, RTX 3090 (sm_86), CUDA 13.2, DaCe 2.0.0a5,
`PYTHONHASHSEED=0`, `max_concurrent_streams=-1` (default stream),
default block size 32.

## Results (median ms/call, warmup 3 + 20 reps, both dump sets agree to <1%)

| variant | BaO full | BaO **execution** | BaTiO3 full | BaTiO3 **execution** | correctness (rel) |
|---|---|---|---|---|---|
| GPU offload, full collapse (shipped `addusxx_g_gpu.sdfg`) | 24.8 | **9.8** | 64.5 | **25.5** | 7e-18 / 2e-19 PASS |
| GPU offload, no collapse (velocity-naive ablation; 5 reps, set 0) | 106.9 | 91.6 | 319.5 | 280.5 | PASS |

Measured H2D copy-in state (median): 15.0 ms on BaO (~71 MB, qgm-dominated),
39.0 ms on BaTiO3 (~180 MB) — pageable-PCIe ~4.6 GB/s.  D2H copy-out (rhoc
only): 0.3–7 ms as reported (drain-inclusive on the collapsed variant).

Reference points:

| reference | BaO_nat002 | BaTiO3_nat005 |
|---|---|---|
| DaCe CPU binding, base build, any threads (BENCH_RESULTS.md) | 350 | 923–937 |
| DaCe CPU binding, opt build 32T (anti-scales) | 1240 | 3569 |
| QE native CPU kernel, np32 aggregate (BENCH_RESULTS.md) | — | ~120 |

On full-call terms the SDFG is 14x / 14.5x faster than the best CPU DaCe
build; the execution-only column is the copies-excluded number for a caller
that keeps the static tables device-resident.  Numerics are machine-precision
against the pw.x dumps (GPU FMA contraction shifts the last bits, rel ~7e-18
at worst).

Historical (before the copy-out fix, when the copy-out state downloaded ALL 13
mirrors, full-call only): full collapse 40.1 / 104.8 ms, jh-fold-only
intermediate 51.7 / 143.7 ms, no collapse 123.6 / 361.6 ms.

## What the offload script does (`offload_addusxx.py`)

1. **CPU optimize() pipeline** (imported from `optimize_sdfg_addusxx_g.py`,
   incl. the scal_* gate copy-in repair): Fortran loops -> maps.
2. **eigts lbound rebase (SDFG level)** — bug 4 of
   dace-fortran-fixes-needed-6c99810.md, fixed on the memlet subsets
   (`mill_at - 1  ->  mill_at - 1 + (d0+1)//2`) instead of the CPU flow's
   post-codegen .cpp regex, so it survives CUDA codegen.
3. **Scatter loop -> map (manual)** — the sequential `rhoc(nl(ig)) += aux2(ig)`
   loop is a host tasklet driven by an interstate indirection read; rebuilt as a
   map whose tasklet does the `dfftt_nl` indirection in-kernel through dynamic
   volume-1 memlets.  No WCR needed: `dfftt%nl` is injective, and separate
   (na, iblock) instances are serialized by the host loop nest.
4. **Launch collapse** — fixed point of stock `MoveLoopIntoMap` /
   `StateFusionExtended` / `FullMapFusion`, plus two scoped manual steps:
   range-normalizing the [1:256] zero-fill maps to the `Min(256, ngm-offset)`
   block range, and `_merge_accumulate_into_block_kernel` (FullMapFusion
   refuses the read-modify-write aux1/aux2 boundary, so the per-ih
   `aux2 += aux1*conj(becphi)` tasklet and the aux1 zero fill are folded into
   the jh kernel manually, with the loop-invariant `loopend = nh[nt]` bound
   assignment hoisted to the loop entry).  Net effect: the jh AND ih
   accumulation loops both run INSIDE the block kernel — the structure of
   `kernel_addusxx_baseline` in `baseline/ref/usxx_kernels.cu`.
5. **Schedules** — top-of-state maps/library nodes on host-level SDFGs ->
   `GPU_Device`; anything inside a scope or a kernel-side nested SDFG ->
   `Sequential` (device-aware version of the velocity phase).
6. **Manual data movement** — `_mirror_kernelside_nontransients_to_gpu`: every
   kernel-touched non-transient gets a persistent `gpu_<name>` mirror, one
   copy-in state before the start block uploads all 13, and the copy-out state
   after the end block downloads ONLY what kernels write (rhoc) — downloading
   the read-only mirrors cost 15/40 ms per call before this restriction;
   kernel-side access nodes/memlets retargeted, host-side kept on the
   CPU array; arrays read by HOST interstate edges (gates, loop bounds:
   okvan/gamma_only/nh/ityp/ofsbeta/upf_tvanp/nij_type/dfftt_ngm) excluded.
   Transients: host-level arrays -> `GPU_Global`, in-kernel arrays ->
   `Register`, scalars -> `Register`.  Then gpu_ prefix renames, NSDFG
   connector reconciliation, storage propagation (all per velocity).
7. **Reductions stay Reduce library nodes on the GPU** — the one Reduce (the
   3-element `(xk-xkq).tau` phase dot product) sits inside the eigqts kernel:
   `ScheduleType.Sequential` + `identity=0`, so the 'auto' expansion lowers it
   to the deterministic sequential accumulator in device code at compile time.
   (velocity tasklet-ized reductions instead; a host-launched Reduce would get
   `GPU_Device` -> ExpandReduceGPUAuto.)

Two generic repairs found necessary on this DaCe version (2.0.0a5):
`_fix_nsdfg_symbol_scoping` (MoveLoopIntoMap loses interstate-ASSIGNED loop
bounds like `loopend_65` — neither declared in the nested SDFG nor in its
symbol mapping -> codegen KeyError) and `_dedup_block_labels` (state fusion can
land two same-named blocks in one region -> validation error).

## Data movement per call

13 mirrored arrays uploaded per call: becphi_c, becpsi_c, dfftt_nl, eigts1/2/3,
ijtoh, mill, qgm, rhoc (inout), tau, xk, xkq; only rhoc is copied back.
Mirrors are `AllocationLifetime.Persistent` (allocated once in `__dace_init`;
a per-call cudaMalloc of qgm would dominate).  qgm dominates the copy-in:
67 MB (BaO) / 175 MB (BaTiO3), measured 15 / 39 ms per call at pageable-PCIe
bandwidth.  This SDFG re-uploads everything per call — the full-call column is
the honest whole-kernel-offload cost; the execution column is the
copies-excluded number for a caller that keeps the static tables
device-resident.

## Kernel structure of the shipped SDFG (5 static launch sites)

Per (nt, iblock, na) host iteration: aux2 zero-fill; the merged block kernel
(256 threads, per g-element `for ih { aux1 = 0; for jh { aux1 += qgm*becpsi };
aux2 += aux1*conj(becphi) }`); the structure-factor kernel (eigqts*eigts1/2/3
via mill indirection); the rhoc scatter kernel.  Plus eigqts (with the in-kernel Reduce)
once per call.  ~584 launches/call on BaO, ~2160 on BaTiO3.

## Known limits / next steps

* The iblock loop is still a HOST loop: kernels are 256 threads (8 blocks of
  32) each, so the GPU runs far below occupancy; the remaining time is roughly
  launch/drain overhead + qgm upload.  A gang-over-iblock
  structure (one kernel per species, `iblock` as the grid dimension,
  `GPU_ThreadBlock` inner maps) is the natural next transformation, but needs
  a `LoopToMap` that accepts the rhoc scatter's injective-indirection writes
  (or a WCR-atomic form).
* FullMapFusion still refuses the aux2-zero/merged/SF/scatter chain (RMW
  boundaries) — 4 launches per (nt, iblock, na) instead of 1.
* `apply_gpu_transformations()` was not used: it has no notion of the host
  interstate-read gates, the indirection scatter, or the mirror naming contract
  this sample needs (same rationale as velocity stage 4).

Repro:

    cd /workspace/dace-fortran/samples/addusxx_g
    PYTHONHASHSEED=0 python3 offload_addusxx.py            # -> outputs/addusxx_g_gpu.sdfg
    PYTHONHASHSEED=0 python3 bench_addusxx_gpu.py --deck data/BaO_nat002
    PYTHONHASHSEED=0 python3 bench_addusxx_gpu.py --deck data/BaTiO3_nat005
