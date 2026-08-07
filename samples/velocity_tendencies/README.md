# velocity_tendencies sample

Run ICON `mo_velocity_advection::velocity_tendencies` (built by dace-fortran from the two
committed TUs in `tests/icon/atmosphere/`, `__LOOP_EXCHANGE` on/off) on REAL R02B05 grid
data, CPU thread sweep.

## Data

ETH PolyBox tarball (see `download_data.sh`; ~9 GB `tar.xz`, ~12 GB ASCII extracted).
Icosahedral R02B05: 81920 cells, 122880 edges, 40962 verts; recorded blocking
nproma=81920 (nblks_c=1, nblks_e=2, nblks_v=1), nlev=90.  Post-extract sed patch
(inlined from `icon-artifacts/velocity/patch_out_is_associated.sh`) is MANDATORY.

## Serde format (ASCII, struct-aware)

Per array: `# rank / N / # size / s0.. / # lbound / l0.. / # entries / values` (Fortran
order).  Allocatables prefixed `# alloc 0|1`, pointers `# assoc` + `# missing`; struct
files concatenate `# <member>` records in declaration order.  Parser: vendored
`third_party/serde_velocity.hpp` (VelocityTendenciesPipeline), wrapped by the
`velocity_data` nanobind module (`load` / `reblock` / `selftest`).

## Re-blocking (nproma 81920 -> 32)

Global 0-based id `g = (blk-1)*np_old + (idx-1)`; new `idx = g%32+1`, `blk = g/32+1`;
`nblks = ceil(valid/32)` -> cells 2560, edges 3840, verts 1281 blocks.  Data arrays remap
dim0/dim2 by the same g (tail pad 0); connectivity `(idx, blk)` pairs also recompute
VALUES into the target space (pad replicates a valid element, gathers stay in-bounds);
refin-ctrl start/end arrays recompute values (order is invariant under re-chunking) and
valid counts derive from them (hard-fail cross-check vs the true R02B05 counts).

## Run

```
bash download_data.sh                                   # symlink/skip/curl+patch
python convert_data.py --nproma 32 --out velocity.npz   # load + reblock + npz (cluster: needs RAM)
python convert_data.py --selftest                       # no data needed
sbatch -p <partition> -A <account> run_velocity.sbatch  # sweep THREADS x both TU variants
```

`download_data.sh` never downloads when local data exists: set `LOCAL_DATA_DIR` (or
`ICON_VELOCITY_DATA_DIR`) to an existing dataset dir, e.g. an icon-artifacts checkout
like `~/Work/icon-artifacts/velocity/data_r02b05`, and it symlinks instead.

npz keys are the harness `mesh_buffers` flat names (`tests/icon/ocean/_ocean_e2e.py`);
refin-ctrl arrays are embedded in the harness 33-slot / lbound -16 window.
`run_velocity_perf.py` (phase A sdfgz cache keyed by variant/backend/git-describe, phase B
timed reps, CSV `kernel,mode,nproma,nblks_e,threads,rep,ms,inputs,lane` -- `inputs` is always
`r02b05`) is cluster-only: do not run it on the dev box. Knobs: `--reps` (50), `--warmup` (2).

Lanes: `LANES` env, space list from `dace-gcc` (default) and `dace-llvm` (`--backend llvm`,
clang++ as the DaCe CPU compiler -- see `samples/README.md` lane matrix). No Fortran perf
baseline lanes exist here: no standalone driver exists for this kernel outside the harness.
