# Third-party fixture license notice

`lulesh.f90` and `lulesh_comp_kernels.f90` in this directory are **vendored
third-party sources** and are **NOT** covered by the dace-fortran BSD license.

| | |
|---|---|
| Upstream | https://github.com/ludgerpaehler/LULESH-Fortran |
| Original work | Fortran LULESH — Crown Copyright 2014 AWE (a Fortran port of LLNL LULESH, LLNL-CODE-461231) |
| License | **GNU General Public License v3 or later** |

They are included **only as test fixtures** for the dace-fortran inliner
(`tests/lulesh/test_lulesh_inliner.py`): the inliner ingests them and the
`CalcElemVolumeDerivative` kernel drives an end-to-end numerical-correctness
check. The full driver (`lulesh.f90`) is an incomplete upstream work-in-progress
and is exercised only by the inliner's whole-program merge — never executed.

## Modifications

Per GPL §5 (marking changed files), the dace-fortran authors made the following
changes, solely to bring the upstream (which targets a patched flang) to
standards-conforming, parseable Fortran:

- **lulesh.f90** — replaced three C-style `DO (i=lo,hi)` loop headers with
  `DO i=lo,hi`; removed a duplicate `plane,row,col` declaration.
- **lulesh_comp_kernels.f90** — made `m_nodeElemCornerList` rank-1; rewrote
  `AllocateNodeElemIndexes` and the two force-gather consumers
  (`IntegrateStressForElems`, `CalcFBHourglassForceForElems`) to the canonical
  LULESH node→element corner-list algorithm (the upstream left these explicitly
  marked broken).

Each file carries the same summary in a header comment.

> Note: including GPL-v3 fixtures alongside BSD-licensed code is a deliberate,
> repository-owner decision recorded here for transparency; the fixtures are
> test-only and are not linked into any distributed dace-fortran artifact.
