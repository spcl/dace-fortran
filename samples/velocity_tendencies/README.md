# velocity_tendencies sample

Run ICON `mo_velocity_advection::velocity_tendencies` (built by dace-fortran from the two
committed TUs in `tests/icon/atmosphere/`, `__LOOP_EXCHANGE` on/off) on REAL ICON grid
data, CPU thread sweep.

## Data

Two decks, both ETH PolyBox tarballs.

`download_data.sh` -> `data_r02b05`.  The name is historical and wrong about the grid: what
the tarball actually holds (and what its own filename, `nproma20480_data_files.tar.xz`, says)
is **20480 cells, 30720 edges, 10242 verts** -- R02B04 scale, not R02B05, and not the 81920
cells this file used to claim.  Recorded blocking nproma=20480 (nblks_c=1, nblks_e=2,
nblks_v=1), nlev=90; 13 recorded timesteps, ~40 GB extracted.  Post-extract sed patch
(inlined from `icon-artifacts/velocity/patch_out_is_associated.sh`) is MANDATORY.

`prepare_r02b06.sh` -> `data_r02b06` plus both `.npz` decks.  The real R02B06 grid:
**327680 cells, 491520 edges, 163842 verts**, nlev=90 -- 16x the cells of the deck above, one
timestep, ~40 GB extracted from a 3.1 GB `tar.xz`.  Recorded blocking nproma=48.  This
tarball already ships the serde patch applied, and the script detects that and skips it.

## Serde format (ASCII, struct-aware)

Per array: `# rank / N / # size / s0.. / # lbound / l0.. / # entries / values` (Fortran
order).  Allocatables prefixed `# alloc 0|1`, pointers `# assoc` + `# missing`; struct
files concatenate `# <member>` records in declaration order.  Parser: vendored
`third_party/serde_velocity.hpp` (VelocityTendenciesPipeline), wrapped by the
`velocity_data` nanobind module (`load` / `reblock` / `selftest`).

## Re-blocking (recorded nproma -> target nproma)

Global 0-based id `g = (blk-1)*np_old + (idx-1)`; new `idx = g%np+1`, `blk = g/np+1`;
`nblks = ceil(valid/np)`.  R02B04-scale deck at nproma=32 -> cells 640, edges 960, verts 321
blocks; R02B06 at nproma=32 -> 10240 / 15360 / 5121, and at nproma=491520 -> 1 / 1 / 1.  Data arrays remap
dim0/dim2 by the same g (tail pad 0); connectivity `(idx, blk)` pairs also recompute
VALUES into the target space (pad replicates a valid element, gathers stay in-bounds);
refin-ctrl start/end arrays recompute values (order is invariant under re-chunking) and
valid counts derive from them (hard-fail cross-check vs the true recorded counts).

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
timed reps, CSV `kernel,mode,nproma,nblks_e,threads,rep,ms,inputs,lane` -- `mode` is the TU
variant, `nproma`+`nblks_e` the data layout, `inputs` the dataset read off the deck's filename)
is cluster-only: do not run it on the dev box. Knobs: `--reps` (50), `--warmup` (2),
`--verify <ref_dir>` (one untimed run checked against a reference dump, gate `1e-10` relative).

Lanes: `LANES` env, space list from `dace-gcc` (default), `dace-llvm` (`--backend llvm`, clang++ as
the DaCe CPU compiler) and `baselines` (see `samples/README.md` lane matrix).

## Fortran baseline lanes

`baselines.sh` (`LANES=baselines`, or run it directly) times the ORIGINAL kernel with its own
compiler's parallelism: `gfortran-autopar` (`-O3 -march=native -ftree-parallelize-loops=$t`, one
build per thread count), `openacc-cpu` (nvfortran `-acc=multicore`, one build, `ACC_NUM_CORES`
sweep) and `openacc` (nvfortran `-acc=gpu -gpu=mem:managed`, single config, threads column 0,
only when `nvidia-smi -L` lists a GPU). Subset via `BASELINE_LANES`.

Sources: `velocity_advection_acc.f90` (the `!$ACC`/`!$OMP`-annotated twin of the e2e TU,
regenerate with `scripts/annotate_velocity_acc.py`) and `driver_velocity.f90`, which reads
`dump_data.py`'s raw dump (`manifest.txt` + `scalars.txt` + one Fortran-order `<name>.bin` per
array), rebuilds the derived types with the recorded lower bounds and prints the CSV rows itself.

```
python dump_data.py --out <dir> [--reference]   # raw dump; --reference adds a DaCe-computed oracle
bash baselines.sh                               # VELOCITY_DUMP_DIR / BASELINE_LANES / REPS / THREADS
<build>/driver_velocity <dir> <lane> <threads> <reps> <warmup> [verify|dumpref [ref_dir]]
```

Numerics gate: build one serial reference (`gfortran -O1`), run it with `dumpref <ref_dir>`, then
run each parallel lane with `verify <ref_dir>`. It fails above `1e-10` relative -- not zero,
because the lanes build with `-march=native` and FMA contraction plus autopar reassociation move
the last digits (measured worst gap: `9e-16`). `dump_data.py --reference` writes the same layout
from the DaCe kernel; `run_velocity_perf.py --verify <ref_dir>` is the mirror gate for the DaCe
lane (measured against the serial Fortran reference on this deck: bit-exact, both TU variants).

The OpenACC lanes need the derived types on the device before the kernel's `PRESENT(...)` data
region; the driver does that with a shallow `!$ACC ENTER DATA COPYIN`, which is why the GPU lane
builds `-gpu=mem:managed` (nvfortran cannot deep-copy the `POINTER` components).

Only the `loopexch` TU is annotated; the `noloopexch` variant has no baseline lane.
