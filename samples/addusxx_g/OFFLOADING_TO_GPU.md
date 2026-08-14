# Offloading addusxx_g to the GPU — process findings

Companion to `BENCH_RESULTS_GPU.md` (numbers there, process here).  Scope:
taking the raw builder SDFG `outputs/addusxx_g_original.sdfg` to a fully
GPU-resident `outputs/addusxx_g_gpu.sdfg` with `offload_addusxx.py`, verified
at machine precision against pw.x dumps and benchmarked on an RTX 3090.

Method template: `samples/velocity_tendencies/offload_velocity.py` (schedules
first, then a manual mirror-based data-movement transformation, transient
promotion, `gpu_` prefix renames, connector reconciliation, storage
propagation).  Target parallel structure: the hand-written full offloads in
`baseline/ref/usxx_kernels.cu` (per-g-element threads, sequential na/ih/jh
accumulation in-thread).

## What was finally done (the shipped pipeline)

1. **CPU `optimize()` pipeline first** (imported from
   `optimize_sdfg_addusxx_g.py`, which brings the len1-staging monkeypatch and
   the scal_* gate copy-in repair).  The raw SDFG has 19 Fortran loops and no
   usable maps; LoopToMap converts 6 loops — the per-block element loops and
   the eigqts atom loop — leaving the nt/iblock/na/ih/jh nest as host control
   flow.  `run_pipeline` **calls** `dace_fortran.pipelines.optimize` (the same
   stage list CloudSC runs); it used to inline a stage-by-stage copy of it.
   Only the two sample-specific bookends stay local: the gate copy-in repair
   before, the writer-less-transient audit and the BLAS implementation pick
   after.

2. **eigts lbound rebase, at the SDFG level.**  Bug 4 of
   `dace-fortran-fixes-needed-6c99810.md`: QE declares `eigts1(-nr:nr, nat)`,
   the frontend lowers 1-based, so kernel memlets read
   `eigts[mill_at - 1, na - 1]` — out of bounds for `mill <= 0`.  The CPU flow
   patches the generated `.cpp` with a regex; that does not carry to CUDA
   codegen, so here the memlet subsets themselves get the
   `+ (d0+1)//2` first-dim shift (`pystr_to_symbolic` so `//` stays
   `int_floor`).  Only single-index dim-0 subsets are shifted; full-range
   boundary memlets are left alone.  This makes the saved SDFG self-contained
   and correct for ANY code generator.

3. **Scatter loop → map (manual).**  The `rhoc(nl(ig)) += aux2(ig)` update
   survives the CPU pipeline as a sequential loop whose body is a bare host
   tasklet, with the `dfftt_nl` indirection read on an interstate edge
   (`dfftt_nl_at5 = dfftt_nl[...]`).  A host tasklet cannot touch device
   arrays, so this is rebuilt as a map whose tasklet does the indirection
   in-kernel: scalar `_nl` read from `dfftt_nl`, then a read-modify-write of
   `rhoc` through full-range dynamic volume-1 memlets
   (`_rout[_nl-1] = _rin[_nl-1] + _val`).  **No WCR/atomics needed**: QE's
   `dfftt%nl` is an injective G-vector→FFT-grid map, so indices within one map
   instance are distinct, and distinct (na, iblock) instances are serialized
   by the host loop nest.  The defensive `Max(0, ...)` in the loop bound is
   stripped to the sibling maps' `Min(256, ngm-offset)` form so the scatter
   kernel stays range-compatible for fusion.

4. **Launch collapse** — the single most important performance step (measured:
   ~9–11x on execution time).  A fixed point of stock `MoveLoopIntoMap` /
   `StateFusionExtended` / `FullMapFusion` plus two scoped manual steps:

   * `_normalize_block_map_ranges`: the aux1/aux2 zero-fill maps are constant
     `[1:256]` while the compute maps are `[1:Min(256, ngm-offset)]`; identical
     ranges are a precondition for every fusion.  Shrinking the zero fill is
     safe — the tail elements are never read.
   * `_merge_accumulate_into_block_kernel`: **FullMapFusion refuses any fusion
     here** because aux1/aux2 are read-modify-write at the map boundary (in
     AND out edges), which its intermediate-node analysis rejects.  Per
     element the chain is plainly `aux1 = 0; for jh: aux1 += qgm*becpsi;
     aux2 += aux1*conj(becphi)`, so the consumer tasklet is appended as a
     final state INSIDE the jh kernel's nested SDFG (map parameter renamed),
     the zero fill becomes an init state there, and the now-empty zero-fill
     state is deleted.  The jh-bound assignment `loopend = nh[nt]` sitting on
     the deleted state's outgoing edge is loop-INVARIANT and is hoisted to the
     loop's entry edge — without that hoist the loop body is never a single
     state and MoveLoopIntoMap cannot fire.
   * With the body reduced to one single-map state, `MoveLoopIntoMap` folds
     the **ih** loop into the kernel in the next round (it folded **jh** in
     round 1).

   Final kernel: per g-element
   `for ih { aux1 = 0; for jh { aux1 += qgm*becpsi }; aux2 += aux1*conj(becphi) }`
   — exactly `kernel_addusxx_baseline` from `baseline/ref`.  Static launch
   sites drop 7 → 5; dynamic launches per call drop ~35K → ~600 (BaO) /
   ~110K → ~2.2K (BaTiO3): 4 kernels per (nt, iblock, na) plus eigqts.

5. **Cache unblocking** (`_unblock_cache_blocking_loop`) — the second big
   launch step, and the one that finally fills the GPU.  QE's
   `DO iblock = 1, numblock` with `blocksize = 256` is *CPU cache blocking*,
   not parallelism: iteration `iblock` touches
   `g = 256*(iblock-1)+1 .. 256*(iblock-1)+realblocksize` and nothing else, and
   the chunks tile `1..ngms` exactly.  So the loop region is deleted, `iblock`
   is pinned to 1 (which turns every `offset + ig` subset back into a plain
   global `ig`), the four block maps are re-ranged `1:ngms`, and the aux1/aux2
   scratch is widened from 256 to ngms so each G-vector keeps its own element.
   The loop is recognised by the maps it drives — every block map ends at
   `Min(256, ngms - 256*iblock + 256)`, which yields both the loop variable and
   the unblocked trip count — never by name.
   * **Race freedom is the SAME argument as item 3, only wider.**  The scatter
     needs `dfftt%nl` injective over the whole G sphere rather than within one
     block; it is (a G-vector → FFT-grid permutation), so no WCR is needed.
   * **Persistent → Scope lifetime** for aux1/aux2: their new extent is a
     data-dependent symbol (`dfftt%ngm`, staged from an argument), which the
     init-time allocation of a Persistent transient cannot see.  Scope is what
     the sibling ngms-sized transients in this kernel family already use.
   * **Guard:** widening refuses on a kernel-side TRANSIENT — that is
     per-thread scratch that `_promote_transient_arrays_to_gpu` sends to
     Register, where a data-dependent extent is a device VLA.  This is exactly
     why the same step does NOT port to newdxx_g (see its own notes).
   Dynamic launches per call go from `1 + 4·numblock·nat` to `1 + 4·nat`:
   585 → 9 (BaO_nat002, numblock 73) and 2161 → 21 (BaTiO3_nat005,
   numblock 108).  Each kernel now gets ngms threads instead of 256 — the
   generated `.cu` launches `dim3(int_ceil(scal_dfftt_ngm, 32))` blocks.
   Static launch sites are unchanged at 5; only host control flow was removed
   (the emitted host code has just the `nt` and `na` loops left).
   Numerics are BIT-IDENTICAL before and after (`max|diff|` unchanged to every
   digit on all four dump sets) — no accumulation was reassociated.

6. **Schedules, device-aware.**  Velocity's rule (top-of-state map/libnode →
   `GPU_Device`, in-scope → `Sequential`) extended with an `in_kernel` flag
   threaded through the nested-SDFG walk, because `scope_dict()` restarts per
   NestedSDFG — a node at the top of an NSDFG state inside a GPU map has no
   visible scope but must NOT become a kernel launcher.

7. **Reductions stay Reduce LIBRARY nodes and run on the GPU** (velocity
   tasklet-ized them instead).  The one Reduce in this graph — the 3-element
   `(xk-xkq)·tau` phase dot product — sits inside the eigqts kernel, so it
   gets `ScheduleType.Sequential` + `identity=0`; the `'auto'` expansion then
   lowers it to the deterministic sequential accumulator in device code at
   compile time.  The saved SDFG still contains the unexpanded library node.
   A host-level Reduce would get `GPU_Device` → `ExpandReduceGPUAuto`.

8. **Manual data movement (the mirror transformation).**  Every kernel-touched
   non-transient gets a transient `gpu_<name>` sibling with
   `AllocationLifetime.Persistent` (a per-call cudaMalloc of the 67–175 MB qgm
   mirror would dominate); one copy-in state before the start block uploads
   all 13 (becphi_c, becpsi_c, dfftt_nl, eigts1/2/3, ijtoh, mill, qgm, rhoc,
   tau, xk, xkq); the copy-out state downloads **only what kernels write** —
   rhoc.  Kernel-side AccessNodes/memlets are retargeted to the mirror,
   host-side ones keep the CPU array.  Arrays read by HOST-level interstate
   edges (okvan/gamma_only gates, nh/ityp/ofsbeta/upf_tvanp/nij_type/
   dfftt_ngm loop bounds and conditions) are excluded — mirroring one hands
   host code a device pointer.  Device-level interstate reads (the mill/ijtoh
   indirections INSIDE kernels) must NOT be excluded — they execute in the
   kernel and need the GPU copy.  Transient storage: host-level arrays →
   `GPU_Global`, in-kernel arrays → `Register`, scalars → `Register`.

## The two DaCe 2.0.0a5 issues hit (generic, not kernel-specific)

### 1. `MoveLoopIntoMap` loses interstate-ASSIGNED symbols

**Symptom:** CUDA codegen crashes with `KeyError: 'loopend_65'` in
`SDFG.arglist` (`cpu.py:_generate_NestedSDFG`), after `MoveLoopIntoMap` moved
the jh loop into the kernel.

**Root cause:** `MoveLoopIntoMap.apply()` forwards the new nested SDFG's free
symbols only when they appear in `sdfg.symbols`.  Symbols created by
interstate-edge ASSIGNMENTS (here the loop bound `loopend_65 = nh[nt-1]`) are
not registered there, so the moved-in loop's bound ends up neither declared in
the nested SDFG's symbol table nor present in its `symbol_mapping`.
Validation does not catch it; codegen dies later.

**Workaround** (`_fix_nsdfg_symbol_scoping`): after all loop restructuring,
walk every nested SDFG and for each free symbol missing from the node's
`symbol_mapping` add an identity mapping, and for each missing from
`sdfg.symbols` declare it — resolving the dtype from the parent SDFG chain
rather than re-minting blindly (symbol identity is name-based in DaCe; a
re-minted default-dtype symbol aliases silently).

### 2. State fusion can leave duplicate block labels in one region

**Symptom:** `sdfg.validate()` fails with
`Found multiple blocks with the same name in loop_ih_60`.

**Root cause:** the Fortran frontend reuses block labels across regions
(`single_state_body` appears in many loop bodies) — legal, since uniqueness is
only required among a region's DIRECT children.  `StateFusionExtended` (and
loop restructuring generally) can migrate two same-named blocks into the SAME
region, which validation then rejects.

**Workaround** (`_dedup_block_labels`): deterministic rename
(`<label>_dedup<i>`, region node order) across all regions before the final
validate.

Both workarounds are self-contained phases in `offload_addusxx.py` and are
candidates for the dace-fortran bug-repro collection / upstream fixes.

## Other findings worth keeping

* **The launch storm dominates a naive offload.**  Velocity-style schedule
  assignment alone leaves one 256-thread kernel per (jh, ih, na, iblock) —
  ~110K launches/call on BaTiO3, 280 ms of execution.  Folding jh+ih into the
  kernel cuts execution to 25.5 ms.  Launch count, not kernel speed, is the
  first-order knob for deep Fortran loop nests.
* **Copy-out must be write-set only.**  Mirroring velocity's copy-back-all
  design cost 15/40 ms per call downloading read-only mirrors (qgm!).
  Restricting the copy-out state to kernel-written arrays (rhoc) halved the
  full-call time.
* **The D2H copy-out timer is drain-inclusive.**  The copy-out is queued
  behind all kernels on the same stream, so a Timer around that state reads
  kernel-drain + download.  Benchmarks must subtract only the H2D copy-in
  (execution := full − copy-in); subtracting copy-out silently deletes real
  kernel time.
* **Python-direct testing works and is much lighter than the Fortran binding
  route** — the offload keeps the host array ABI (copies live inside the
  SDFG), so `bench_addusxx_gpu.py` feeds numpy arrays straight in.  Gotchas:
  DaCe rejects numpy VIEWS (F-order reshapes need owning copies); the eigts
  dumps are sized on the DENSE grid (nr=54 → d0=109), not the FFT-t grid;
  dead-ABI dfftt_* arrays can be dummied with all dims = 1.
* **Numerics were identical across every structural variant** (naive,
  jh-folded, fully merged): rel 7e-18 / 2e-19 vs the pw.x dumps — the
  restructuring never reassociated the accumulation order per element.
* **Cache blocking is not parallelism, and undoing it beats scheduling it.**
  The headroom this section used to name (host-side iblock loop → 256-thread
  kernels) was closed by item 5, but NOT the way it predicted.  A
  gang-over-iblock structure would have needed a LoopToMap that accepts the
  scatter's injective-indirection writes; simply *deleting* the blocking loop
  and re-ranging the maps over the whole sphere needs no such transformation,
  and gives one thread per G-vector — the `baseline/ref` structure — directly.
  Read the blocking loop as an artifact of the CPU source, not as a level of
  the parallel decomposition.
* **Remaining headroom:** FullMapFusion still refuses the
  aux2zero/merged/SF/scatter chain (read-modify-write boundaries on aux2), so
  it is 4 launches per (nt, na) rather than 1.  With the `numblock` factor gone
  that is 9-21 launches per call, no longer the first-order cost.

## Bottom line

`outputs/addusxx_g_gpu.sdfg`: verified PASS on both decks (4/4 dump sets),
numerics bit-identical to the pre-unblocking graph.

The RTX 3090 numbers this section used to quote (full-call 24.8 / 64.5 ms,
execution-only 9.8 / 25.5 ms) predate item 5.  Post-unblocking, a **login-node**
GH200 A/B on the same process gives compute-only 12.41 → 1.29 ms (BaO_nat002)
and 32.84 → 1.37 ms (BaTiO3_nat005); the login node is shared, so those are
valid as a RATIO only.  `BENCH_RESULTS_GPU.md` still carries the job-measured
pre-unblocking table (job 4479163: 6.55 / 19.47 ms compute-only) and needs a
re-run on a compute node before its numbers can be quoted.

Repro: `PYTHONHASHSEED=0 python3 offload_addusxx.py` then
`PYTHONHASHSEED=0 python3 bench_addusxx_gpu.py --deck data/<MAT>`.
