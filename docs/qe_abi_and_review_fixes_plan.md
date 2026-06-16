# Fix plan: QE module-state ABI + code-review findings

Status as of 2026-06-16. Kernel correctness work is committed (dace-fortran
`49aebbd` complex-view/reductions/Gate H, `efcb34e` global code-block subscript
design; d-face `0dcae82ac` re/im). QE `vexx_bp_k_gpu` COMPILES + VALIDATES;
Graupel numerical PASSES. The items below are (A) the QE module-state ABI needed
for the QE numerical RUN, and (B) eight code-review findings on the committed
diff.

---

## Part A — QE module-state ABI

The QE numerical RUN currently needs 107 extra non-transient array args (+ free
shape symbols) — the whole QE module-level state, not just `deexx`. Probe:
`/home/primrose/Work/tmp/cc/qe_cat.py` (neutralises
`_diagnose_unresolved_free_symbols`). Split:

| bucket | count | examples | disposition |
|---|---|---|---|
| `intent=inout` array | 49 | coulomb_fac, exxbuff, x_occupation, becpsi_k, igk_exx | kernel-allocated → transient (FIX#1) OR external → FIX#2 |
| `intent=inout` scalar | 33 | exxalfa, many_fft, inter_egrp_comm | module-global scalars (len-1 arrays) |
| `intent=in` array | 12 | becphi_c, becphi_r, becpsi_r | inlined-sub dummies bound to OPTIONAL args (absent on no-op) |
| `intent=''` array | 12 | vcut_*, dfftt_nl, at | module-globals / derived members |

### FIX #1 — kernel-allocated module globals → transients
`deexx` (entry `IF(okvan) ALLOCATE(deexx(nkb,ialloc))`, line 1431) is ALREADY a
transient. `coulomb_fac`/`coulomb_done` (line 1354/1356,
`IF(.NOT.ALLOCATED(coulomb_fac)) ALLOCATE(coulomb_fac(ngm,nqs,nks))`) leak as
args because they are ALLOCATEd inside an **inlined subroutine**
(`g2_convolution_all`), not the entry. The allocation-lifetime / writable-init
mechanism only catches entry-level allocates + baked-constant/kernel-written-
with-known-init globals:
- `dace_fortran/builder/descriptors.py:390-396` (`_global_is_baked_constant` → transient)
- `dace_fortran/builder/__init__.py:764, 884` (writable-init transients: module globals the kernel writes with a known init)
- bridge alloc-site binding `bindAllocSite` (dispatch.cpp ~1425) marks entry-level allocatables.

**Fix:** recognize a module-global `ALLOCATE` (including the
`IF(.NOT.ALLOCATED) ALLOCATE` guard) anywhere in the inlined body, not just the
entry scope, and classify that global as a kernel-owned transient (default Scope
lifetime, NOT Persistent). Steps: (1) in the bridge alloc-site walk, stop
requiring the allocate to be in the entry func; collect allocates from inlined
callee bodies too (their declares are already spliced in). (2) When a global
appears both as a module-global (intent inout, writable-ABI) AND has a
kernel-side allocate, the allocate wins → transient. (3) The `.NOT.ALLOCATED`
guard already lowers to an `<name>_allocated` tracker symbol — make sure the
transient path sets it. Repro to drive: build QE, confirm `coulomb_fac`/
`coulomb_done` drop off the arg list (probe qe_cat.py).
Risk: shape symbols (`ngm`,`nqs`,`nks`) become transient-extent symbols computed
mid-stream — must be assigned before the allocate state (the existing
`<buf>_d<i>` site-binding handles this for entry allocates; extend to inlined).

### FIX #2 — external USE'd globals → zero-transient-if-dead / caller-bound
`exxbuff`/`exxbuff_d`/`x_occupation`/`x_occupation_d`/`gt`/`index_xk`/`index_xkq`/
`igk_exx`/`ibands`/... (`USE exx_base, ONLY: ...`, line 1368) are allocated/
initialised ELSEWHERE in real QE; the kernel does not allocate them. On the
no-op path (okvan/okpaw/noncolin=F, negrp=1, nqs=0) they are read only on DEAD
branches.
**Fix (two parts, per-global by reachability on the LIVE path):**
1. Dead-on-live-path → **zero-transient** (uninitialised-read-safe): mark the
   external global a transient seeded to zero, so the no-op identity test runs
   without the caller supplying it. Reuse the writable-init-transient path
   (__init__.py:884) but seed 0 for an allocatable with no compile-time init.
   Gate this on "not referenced on any live (non-dead-branch) statement".
2. Live-external → **caller-bound** + emit a BINDING STUB. This ties into the
   README "generate the binding from the SDFG" goal: a generator that, from the
   built SDFG's non-transient arg list + free symbols, emits a Fortran/C binding
   skeleton (or a Python harness) that the user fills with the real module state.
   For vexx the live set on the no-op path should be just psi/hpsi + dims.
**Also:** the 12 `intent=in` OPTIONAL-bound arrays (`becphi_c`/`becphi_r`/...) —
on the no-op path PRESENT()=false, so the branch reading them is dead → elide to
transient (same dead-read-safe rule).

### deexx section-alias (subset of FIX#1's bucket — separate but related)
The inlined-sub `deexx` DUMMIES (`paw_newdxx_deexx` etc., assumed-shape
`REAL/COMPLEX(:)` bound to `deexx(:,ii)`) leak as args. Minimal reproducers
ALIAS correctly, so it is a QE-specific inlining subtlety (multi-callee /
optional args / dead okpaw). `asAssumedShapeAlias` (trace_utils.cpp:567) refuses
rank-mismatch + doesn't traverse a designate memref; the section-alias detection
(extract_vars.cpp:2519/2549) peels convert/box_addr/copy_in but NOT embox/rebox.
Candidate: add embox/rebox to that peel loop IF the assumed-shape box binding has
one. Dump the entry POST-INLINE IR for the deexx arg (qe_full.hlfir line
16245/16315 paw_newdxx, 22013 entry deexx alloca) to confirm.

---

## Part B — code-review findings (on `8ce09d1..efcb34e` + d-face `0dcae82ac`)

### Correctness
**B1 — complex-component alias as a READ in another statement is unhandled.**
`emit_cfg.py:183` routes only assigns whose TARGET is a cc-alias to
`emit_complex_component_assign`. `result(i) = qg(1,i) + qg(2,i)` (qg a non-target
read) falls to `emit_tasklet`, which builds a 2-D float subscript
`qg[(c)-1,(i)-1]` against the 1-D complex View → rank mismatch / no re/im
extraction. Matches the "component READ in expr" TODO already in memory.
**Fix:** in `emit_tasklet`'s read-wiring, detect a `complex_component_aliases`
read and lower it as a staged `.real()/.imag()` read of the complex View element
(intermediate float scalar), like `emit_complex_component_assign` does for the
RHS. Add a test `result = qg(1,i) + qg(2,i)`.

**B2 — section-reduce drops the section STRIDE.** `emit_library.py emit_reduce`
section-view path parses only `lo:hi` from `renderDesignateSubsetStrings`, which
emits `lo-1:hi:st` for a non-unit-stride section. View shape becomes the full
span (not span/st) and stride = parent element stride (not parent_stride*st), so
`MINVAL(a(1:9:2))` in a condition reduces 9 contiguous elements instead of 5
strided. **Fix:** split on `:` into up to 3 parts; extent = `ceil((hi-lo)/st)`,
view stride = `parent_stride * st`, keep the 3-part subset on the linking memlet.
Add a strided-section reduce-in-condition test.

**B3 — cc-alias RHS self-read `re.sub(\bqg\b → _cur)` is index-blind.**
`emit_tasklet.py` (emit_complex_component_assign) replaces EVERY `qg` in the rhs
with the write-element's `_cur`. `qg(1,ig) = qg(1,ig) + qg(2,ig-1)` then uses the
real part of element `ig` for the cross-element `qg(2,ig-1)` read → silently
wrong. **Fix:** only fold the qg read whose index == the write index into `_cur`;
wire any other qg read as a separate masked `.real()/.imag()` read (shares B1's
read-lowering). Add a cross-element self-read test.

### Behavior-scope / global risk
**B4 — memset View-write branch applies to ALL Views, not just cc-aliases**
(`emit_library.py:212`). The `isinstance(tgt_desc, View)` check routes
`bounds_remap_view`/`view_alias` memsets through the new fresh-write+writeback
path. The broad regression (incl. bounds_remap_view + memset) PASSED, so it is
likely equivalent — but the comment over-scopes it. **Fix:** either confirm + fix
the comment to say "any View target", or gate the new path on the cc-alias set if
the old `acc` path was intentional for other Views.

**B5 — the global `re`/`im` cppunparse rename is now near-dead** (d-face
`cppunparse.py` `_renamed_funcs`). The emitter switched to `.real()/.imag()`
methods, so `re()/im()` are only used by Tier-1 synthetic tests — yet the rename
rewrites ANY `re(...)`/`im(...)` Call across all dace codegen (a kernel with an
`im(x)` function breaks). **Fix (debloat):** drop the `re`/`im` `_renamed_funcs`
entries + the `dace::math::re/im` helpers in `math.h` + the Tier-1 re/im tests in
`tests/complex_component_alias_test.py`, since production uses `.real()/.imag()`.
This removes a global-collision risk for zero production benefit.

### Defensive / cleanup
**B6 — `cc_alias_view_spec` float-dtype fallback** (`access.py:173`). When the
complex source VarInfo is missing, dtype defaults to the float alias dtype →
silently recreates the invalid float-View-of-complex the feature exists to avoid.
**Fix:** raise a clear NotImplementedError/RuntimeError when `src_v is None`
(the source must be a registered complex array).

**B7 — redundant per-op handlers in `buildExprWithSubscripts`**
(`control_flow.cpp`). The general `kForceSubscripts` fall-through (set before
delegating to `buildExpr`) already spells every op AND keeps leaf subscripts, so
the power (`math.fpowi/powf/powi`), bin-ops, unary-intrinsics, and negf handlers
above it are now dead-equivalent (~200 lines), and the comment claims it "retires"
them. **Fix (debloat / altitude):** delete those per-op handlers; keep ONLY the
const-extent inline-unroll reduction table (the `kCondReductionScalars`
map-check + the load-of-designate subscript handler + the convert/no_reassoc
peels). Re-run the condition/elemental suite to confirm byte-identical condition
strings.

**B8 — verify DO-WHILE reduction re-evaluation** (`dispatch.cpp:140`
`materialiseCondReductions` pushes the reduce into the scf.while BEFORE-region).
For `DO WHILE (MAXVAL(a)>0)` whose body mutates `a`, the reduce must re-run each
iteration. `condition_reduction_test.py::test_do_while_maxval_condition` exists —
**confirm it mutates `a` in the body** (not just decrements a scalar); if not, add
a body-mutating-`a` variant and verify the loop terminates correctly.

---

## Suggested order
A-FIX#1 (concrete, unblocks coulomb_fac/done) → A-FIX#2 part 1 (zero-transient
dead globals → QE no-op RUN) → B1/B2/B3 (real correctness) → B5/B7 (debloat,
matches the roadmap) → B6/B4/B8 (defensive/verify) → A-FIX#2 part 2 (binding
generator, ties to the README e2e example).
