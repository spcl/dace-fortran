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

## 2.0.0a1

The 2.x line targets DaCe extended: install `dace-fortran[extended]` (needs `dace-extended>=1.0.2`, Python >= 3.12).
Upstream DaCe lacks the extended-only APIs the frontend uses, such as `specialize_symbols`, so the `[dace]` extra is
removed from 2.x. Users of upstream DaCe pin `dace-fortran<2`, the 1.x line.
