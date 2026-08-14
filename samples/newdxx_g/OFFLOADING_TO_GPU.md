# Offloading newdxx_g to the GPU — process findings

Companion to `BENCH_RESULTS_GPU.md` (numbers there, process here).  Scope:
taking the raw builder SDFG `outputs/newdxx_g_original.sdfg` to a fully
GPU-resident `outputs/newdxx_g_gpu.sdfg` with `offload_newdxx.py`, verified at
machine precision against pw.x dumps and benchmarked on an RTX 3090.  The
script is a clone of `samples/addusxx_g/offload_addusxx.py` (read that
sample's `OFFLOADING_TO_GPU.md` first — the method, the mirror-based manual
data movement, and two of the DaCe workarounds are identical); this document
covers what newdxx_g did DIFFERENTLY.

## What was finally done

1. **CPU `optimize()` pipeline first** (imported from
   `optimize_sdfg_newdxx_g.py`: len1 monkeypatch + gate copy-in repair).
   LoopToMap converts 7 loops — crucially INCLUDING the projector loop, whose
   iterations write disjoint `deexx(ikb0+i)` elements, and the full-sphere
   `auxvc(g) = vc(nl(g))` gather (an indirect READ parallelizes; the addusxx
   scatter could not).
2. **eigts lbound rebase** — identical to addusxx (memlet-subset
   `+ (d0+1)//2`, 3 sites in the block structure-factor kernel).
3. **No scatter conversion needed** — `_assert_no_host_indirection_loops`
   verifies the addusxx-style host indirection loop is absent (the deexx
   updates are in-kernel).  `_wrap_bare_device_tasklets` guards the other
   host-tasklet hazard (a bare tasklet writing a GPU-bound transient); in the
   shipped build it finds nothing — the ``fact = omega`` staging came out as a
   Register scalar, which a host tasklet may legally write and kernels take by
   value.
4. **Launch collapse: a structural NO-OP.**  The fixed point (MoveLoopIntoMap
   / range-normalize / StateFusion / FullMapFusion / acc-merge) runs with all
   its guards and applies nothing: the jh loop's body is multi-state (the
   gamma/complex arms + the Dot + the G=0 correction), so MoveLoopIntoMap
   cannot fire, and there is no zero-fill/accumulate map pair to merge.  The
   guards existing is the point — the same script structure ports between the
   two kernels and degrades to a no-op where the pipeline already did the job.
5. **Reductions stay LIBRARY NODES on the GPU — including a BLAS node.**  The
   per-(ih, jh) block dot product is a `Dot` library node inside the projector
   kernel: `_configure_library_nodes` gives every in-kernel library node
   Sequential schedule and FORCES `implementation='pure'` for non-Reduce nodes
   — the default BLAS dispatch must not link a host BLAS into device code.
   The 3-element phase `Reduce` gets Sequential + identity=0 as in addusxx.
6. **Manual data movement with DUAL RESIDENCY.**  15 arrays mirrored
   (upload), copy-out restricted to what kernels write (`deexx`, <1 KB).  New
   here: `nh` is read by HOST interstate edges (the projector-count loop
   bounds `loopend = nh[nt-1]`) AND flows into the projector kernel as
   dataflow.  The velocity/addusxx rule excluded every host-interstate-read
   name from mirroring — that would strand the kernel side on a CPU pointer.
   The newdxx mirror only excludes names with NO kernel-side AccessNode:
   ``nh`` gets a ``gpu_nh`` mirror, per-node retargeting moves only the
   kernel-side AccessNodes to it (interstate reads reference the NAME, not an
   AccessNode, so host reads keep the CPU array), and connector reconciliation
   rewrites the DEVICE-level interstate reads inside the nested SDFG
   (`loopend_100 = nh[nt-1]`, evaluated in-kernel) to `gpu_nh`.

## DaCe 2.0.0a5 issues

Issues 1 and 2 from addusxx_g (`_fix_nsdfg_symbol_scoping`,
`_dedup_block_labels`) are carried over unchanged.  newdxx_g hit a THIRD:

### 3. Nested device-function arguments lose ``const`` for arrays sized by interstate-assigned symbols

**Symptom:** nvcc fails with `argument of type "const dace::complex128 *" is
incompatible with parameter of type "dace::complex128 *"` at the call of the
block structure-factor kernel's nested device function: the kernel parameter
`gpu_auxvc` is (correctly) const, the nested function's is not.

**Root cause:** `emit_memlet_reference` (cpp.py) resolves a nested argument's
ctype from `declared_arrays` — the allocation-time, NON-const ctype — instead
of the kernel-scope `defined_vars` registration whenever the descriptor's
shape contains a symbol that is not a program argument.  `auxvc(ngms)` is
sized by `ngms`, an interstate-ASSIGNED symbol (`ngms = dfftt_ngm[0]`), so it
takes that path; `eigqts(nat)` and friends are sized by argument symbols and
keep their const kernel-scope type.  (A first hypothesis — the duplicated
`_in_auxvc_0/_in_auxvc_1` tasklet connectors — was disproved by
`_dedup_tasklet_inputs`, which is kept anyway as graph hygiene: one connector
per distinct (data, subset) input.)

**Workaround** (`compile_with_cuda_const_fix` in `bench_newdxx_gpu.py`): on
this specific compile failure, strip pointee-const from pointer parameters in
the generated `.cu` (`const T * __restrict__` -> `T * __restrict__`) and
rebuild with ninja.  A generated-file fix, same class as the repo's
established `.cpp` patches: no writes are introduced, `__restrict__` stays,
and every caller/callee pair becomes compatible.  A proper upstream fix would
prefer the kernel-scope `defined_vars` ctype when available.

## Other findings worth keeping

* **The bottleneck moved: launches -> kernel parallelism.**  addusxx_g needed
  the launch collapse (9-11x); newdxx_g launches only ~300-1100 kernels/call
  but the projector kernel exposes just nh = 8-19 threads, each serially
  running two 256-element aux1 passes plus a 256-element Dot per jh.
  Execution is 69.7 / 222.5 ms (BaO / BaTiO3) — 3.5x / 1.8x over the best
  32-thread CPU build.  The next transformation is recasting the per-(ih, jh)
  dots as block-wide parallel reductions (or the whole projector update as a
  small qgm^T-GEMV per block); the Dot library node is exactly the seam where
  a parallel expansion would slot in.
* **The same offload skeleton ported in one session.**  Everything
  kernel-specific sits in four scoped phases (eigts rebase, structural guard,
  tasklet wrap, library-node config); schedules, mirroring, promotion,
  renames, and the workaround phases transferred verbatim.
* **Numerics:** rel 3.4e-21 / 1.7e-18 (BaO sets) and 2.2e-19 / 6.9e-18
  (BaTiO3 sets) against the pw.x dumps; the in-kernel pure-sequential Dot and
  Reduce expansions preserve the CPU accumulation order per element.

## Bottom line

`outputs/newdxx_g_gpu.sdfg`: verified PASS on both decks, full-call
85.1 / 260.9 ms and execution-only 69.7 / 222.5 ms (BaO_nat002 /
BaTiO3_nat005), vs 352 / 938 ms serial and 244 / 402 ms 32-thread CPU DaCe
builds.  Repro: `PYTHONHASHSEED=0 python3 offload_newdxx.py` then
`PYTHONHASHSEED=0 python3 bench_newdxx_gpu.py --deck data/<MAT>`.
