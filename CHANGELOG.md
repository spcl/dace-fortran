# Changelog

All notable changes to dace-fortran. `helpers/release_patch.sh` adds a section per release.

## 2.0.0a1

The 2.x line targets DaCe extended: install `dace-fortran[extended]` (needs `dace-extended>=2.0.0a1`, Python >= 3.12).
Upstream DaCe lacks the extended-only APIs the frontend uses, such as `specialize_symbols`, so the `[dace]` extra is
removed from 2.x. Users of upstream DaCe pin `dace-fortran<2`, the 1.x line.
