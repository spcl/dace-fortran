# Allocatable buffer tracking: a unifying design

Design note for the bridge's handling of `ALLOCATABLE` arrays across
arbitrary `ALLOCATE` / `DEALLOCATE` / conditional-allocate patterns,
including the currently-`xfail` combinations (conditional + realloc;
realloc chain inside an `IF`).

## 1. What the Fortran standard enforces (the semantics we must model)

An `ALLOCATABLE` declared at routine (or `BLOCK`) scope has:

* **One name, one current buffer.** At any execution point `a` refers to
  at most one allocated buffer. `ALLOCATE`(a) when `a` is already
  allocated is an **error** (you must `DEALLOCATE` first; or use
  allocatable assignment `a = expr`, which auto-reallocates).
* **Allocation status persists across control flow within the scope.**
  `ALLOCATE` inside an `IF` branch that executes leaves `a` allocated
  *after* the `IF`; it is **not** auto-deallocated at the branch end.
* **Referencing an unallocated allocatable is prohibited** (undefined;
  compilers error / trap / read garbage). So:
  - `IF (c) ALLOCATE(a(n)) ELSE ALLOCATE(a(m))` then use `a` after the
    `IF` -> **legal**: `a` is allocated on every path.
  - `IF (c) THEN; ALLOCATE(a(n)); ENDIF` then use `a` after the `IF` ->
    **legal only when `c` is true**; the programmer guarantees it (a
    matching `IF (c)` guard, or `ALLOCATED(a)` test). The standard does
    not auto-allocate.
* **End of scope** deallocates any still-allocated allocatable.

Implication for the bridge: we never have to *prove* allocation; we model
"the current buffer at each point", trusting the program is standard-
conforming. Where Fortran would leave `a` unallocated on some path (the
single-branch case with `c` false), the bridge may *over-allocate* (give
the DaCe transient a concrete symbolic shape and allocate it anyway) --
harmless, because a conforming program never reads `a` on that path.

## 2. The unifying abstraction: buffer reaching-definitions

Model each `ALLOCATE` site as a **buffer definition** and each
`DEALLOCATE` as a **kill**. Over the routine's CFG compute, for every
point, the set of `ALLOCATE` sites whose buffer **reaches** it (a path
exists from the site to the point with no intervening `DEALLOCATE` or
re-`ALLOCATE` of `a`).

Two `ALLOCATE` sites belong to the **same DaCe transient** iff their
buffers can reach a common point as alternatives -- i.e. they are the two
arms of an `IF` that both stay live to the join. Sites whose buffers are
never simultaneously-reaching are **distinct transients** (sequential
re-allocation: one dies before the next is born).

Formally: build a *merge relation* over sites -- `s ~ t` if `s` and `t`
are both in the reaching set at some join / use -- and take its
equivalence classes (union-find). **Each class = one DaCe transient.**

This single rule reproduces every case:

| pattern | sites | reaching at the post-IF use | classes (transients) |
|---|---|---|---|
| `IF/ELSE` both alloc | s0(then,n), s1(else,m) | {s0, s1} | **{s0,s1}** -> 1 buf, branch extent |
| `IF/ELSEIF/ELSE` | s0,s1,s2,s3 (each branch) | all four | **{s0..s3}** -> 1 buf, branch extent |
| single-branch | s0(then,n) | {s0} (or {} if `c` false) | **{s0}** -> 1 buf, concrete `n` |
| sequential `A;dealloc;A` | s0(n), s1(m) | s0 dies before s1 | **{s0},{s1}** -> 2 bufs |
| chain x4 | s0..s3 | each dies before next | **{s0},{s1},{s2},{s3}** -> 4 bufs |
| conditional + realloc | s0(then,n),s1(else,m),s2(realloc,k) | {s0,s1} at IF join; {s2} after dealloc+realloc | **{s0,s1},{s2}** -> 2 bufs |
| realloc-chain inside `IF` then-branch | s0(n),s1(n2) in then, s2(m) in else | {s1,s2} at join (s0 died in then) | **{s0},{s1,s2}** -> 2 bufs |

## 3. Per-class shape (the PHI)

For a class with a single site, or all sites the same extent: **concrete
shape** = `shapeFromAllocSite(site)`.

For a class whose sites have differing extents (a real conditional): the
shape is a **branch/path-dependent extent symbol** `<buf>_d<i>` (a PHI of
the per-site extents). Each site assigns `<buf>_d<i> = <that site's
extent>` on its own path; the assignments merge at the join (DaCe binds
the symbol from whichever path ran). This is exactly the mechanism now
shipping for the pure-conditional case -- generalised to any merged
class, named by the class's buffer rather than always `a_d0`.

## 4. Buffer naming and alias routing

Order classes by first-definition in IR walk order. Class 0 keeps the
base name `a`; classes 1.. get `a_alloc1`, `a_alloc2`, ... (the existing
`allocAliasName`). The bridge's existing **alias map** already routes
reads/writes to the "current" buffer as it walks the IR -- the only
change is that at each `ALLOCATE` site we `setAllocAlias(a, <class
buffer>)` (the site's *class*, not a per-site index). At an `IF` join the
two branches set the **same** merged-class buffer, so post-join reads
route correctly with no extra join handling.

## 5. Algorithm (structured dataflow over `scf`/`fir.if`)

The IR is well structured (allocate/deallocate sit in `scf.if` regions),
so a recursive walk suffices -- no general iterative dataflow needed:

```
walk(region, reaching) -> reaching_out:          # reaching: set<site>
  for op in region (in order):
    if op is ALLOCATE(a) site s:  reaching = {s}
    elif op is DEALLOCATE(a):     reaching = {}
    elif op is a USE of a:        if |reaching|>1: merge_all(reaching)
    elif op is scf.if/fir.if:
        r_then = walk(thenRegion, copy(reaching))
        r_else = walk(elseRegion, copy(reaching))   # else empty -> reaching
        reaching = r_then ∪ r_else
        if r_then and r_else both nonempty: merge_all(r_then ∪ r_else)
  return reaching
```
`merge_all(S)` unions the sites of `S` in a union-find. After the walk,
the union-find classes are the transients. (A nested `IF` recurses; a
realloc inside a branch updates `reaching` within that branch's walk, so
the chain-inside-`IF` falls out correctly.)

## 6. Implementation plan

* **`extract_vars.cpp`**
  - New `groupAllocSites(name, module)` running the §5 walk -> returns the
    list of classes (each a vector of `fir.allocmem`), in first-def order.
    Factor the pairwise-exclusivity check already in
    `allocSitesInExclusiveBranches` into the merge step.
  - Replace the current `condAlloc` branch + the `a_allocK` versioning
    loop with: one VarInfo per class. Class buffer name via
    `allocAliasName`. Shape per §3 (extent symbol -> register `<buf>_d<i>`
    symbols; else concrete via `shapeFromAllocSite`).
* **`ast/dispatch.cpp` `bindAllocSite`**
  - Compute the same classes; map this site -> its class buffer name;
    `setAllocAlias(a, <class buffer>)`. If the class is an extent-symbol
    class, emit `<buf>_d<i> = traceExtentExpr(allocmem.shape[i])`.
  - Keep emitting `<a>_allocated = 1` (status tracker unchanged).
* Both sides call the **same** `groupAllocSites` so naming/shape/alias
  agree (expose it in `extract_vars.h`, like `collectAllocSites`).

**Risk containment:** the grouping is identical to today for the two
cases already shipping -- a single all-exclusive group reproduces the
pure-conditional transient; all-singleton groups reproduce sequential
versioning (group index == site index). Only genuinely-mixed routines
change behaviour, and those are the `xfail` today. Regression surface is
therefore the existing allocatable/realloc tests (must stay green) plus
the new mixed-case tests.

## 7. Verification

Extend `tests/conditional_alloc_test.py`:
* conditional + sequential realloc (the current `xfail`) -> un-`xfail`.
* realloc chain (alloc/dealloc x3) inside one `IF` branch, other branch a
  single alloc; use after the `IF`.
* nested: conditional inside a sequential epoch and vice-versa.
Each verified vs an f2py reference on several inputs (both branch
selectors, several sizes). Then the full `-m "not mpi"` sweep must stay at
0 failed; flip the `xfail` only when its f2py comparison passes.

## Out of scope / non-goals
* `MOVE_ALLOC` and allocatable assignment auto-realloc (`a = expr`) -- a
  separate lowering.
* Buffer-reuse / aliasing optimisation (giving merged classes the same
  storage) -- a DaCe-core transformation, not a frontend concern.
