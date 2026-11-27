# Changelog

All notable changes to dace-fortran. `helpers/release_patch.sh` adds a section per release.

## 2.0.0a2

Requires `dace-extended>=1.0.5`.

- optimize: a loop is versioned at run time only where LoopToMap refused it for a loop-invariant branch
  (ParallelizeUnderConstraint after LoopToMap).
- Parallelization gate: a loop versioned at run time counts as parallelized.
- Tests: four exact duplicates dropped; three structure-and-value pairs folded into one test each; the QE h_psi
  numerical check is skipped on this line.
- Toolchain: LLVM 22 is the core version and runs every CI lane; LLVM 23 is supported and its fast lane gates CI.
  LLVM 21 is no longer supported.
- Bridge: copies of one inlined OPTIONAL dummy bound through a present test stay separate variables (ICON solve_nh:
  `z_dwdz_dd`'s memset wrote `exner`).
- Bridge: an automatic array's bound read from an array element is a value symbol seeded at entry.
- Builder: a local no statement assigns is defined as 0 at entry; a presence flag the bridge adds (`x_allocated`,
  `x_present`) is not such a local.
- `ALLOCATE(x, SOURCE=y)`: a scalar copy is lowered, the source's literal lower bounds are kept (QE h_psi), and a form
  the pass cannot model stays a runtime call the bridge refuses where the kernel makes it.
- Bindings: a free symbol with no source is refused only if it sizes an array, and a record member pointing back to its
  container is walked once (ICON ocean `solve_free_sfc` recursed without end).

## 2.0.0a1

The 2.x line targets DaCe extended: install `dace-fortran[extended]` (needs `dace-extended>=1.0.2`, Python >= 3.12).
Upstream DaCe lacks the extended-only APIs the frontend uses, such as `specialize_symbols`, so the `[dace]` extra is
removed from 2.x. Users of upstream DaCe pin `dace-fortran<2`, the 1.x line.
