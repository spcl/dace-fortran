# samples

Reproducible CPU scaling experiments on production kernels, driven through the same build +
`pipelines.optimize` path the e2e tests use. Perf only -- numerics stay in the e2e lane
(`tests/e2e/`).

- `cloudsc/`: CLOUDSC cloud-microphysics kernel, KLON/NBLOCKS sweep (below).
- `vexx/`: QE `exx_bp::vexx_bp_k_gpu` kernel, bands/grid sweep (`vexx/README.md`).
- `velocity_tendencies/`: ICON `mo_velocity_advection::velocity_tendencies`, loop-exchange
  sweep, cluster-only (`velocity_tendencies/README.md`).

All drivers and baseline scripts write the same 9-column CSV,
`kernel,mode,<p1>,<p2>,threads,rep,ms,inputs,lane`: position 2 is always `mode`, 3-4 are the
kernel's own shape ints, 8 is `inputs`, 9 is `lane` -- one plotting script consumes everything.

## lane matrix

One submission produces comparable CSVs across compiler/baseline lanes, selected per job with
the env `LANES` (space list; default `dace-gcc`):

| lane            | kernels       | source                                | compiler + flags                        |
|-----------------|---------------|---------------------------------------|-----------------------------------------|
| dace-gcc        | all three     | DaCe pipeline (`--backend gcc`)       | g++, `PERF_CPU_ARGS`                    |
| dace-llvm       | all three     | DaCe pipeline (`--backend llvm`)      | clang++-21/clang++, same flags          |
| gfortran-serial | cloudsc       | dwarf tree (`baselines.sh`)           | gfortran via h5fc, `-O3 -march=native`  |
| gfortran-autopar| cloudsc       | dwarf tree (`baselines.sh`)           | same + `-ftree-parallelize-loops=$t`    |
| original-openmp | cloudsc       | dwarf tree (`baselines.sh`)           | same + `-fopenmp`, `OMP_NUM_THREADS=$t` |
| flang-serial    | cloudsc       | dwarf tree (`baselines.sh`)           | flang-new-21 `-O3` (flang: no autopar)  |
| cuda            | cloudsc (GPU) | dwarf cloudsc_cuda (`gpu_baselines.sh`)| nvcc `-O3 -arch=native -rdc=true`      |
| openacc         | cloudsc (GPU) | dwarf cloudsc_gpu scc (`gpu_baselines.sh`)| nvfortran `-O3 -acc=gpu`            |

`--backend llvm` swaps only the DaCe CPU compiler (`DACE_compiler_cpu_executable`); the flag
string stays identical, and the Phase-A cache tag includes the backend so lanes never share
compiled artifacts. cloudsc baselines run from the vendored dwarf-p-cloudsc tree because its
driver self-times (a parseable `TOTAL` msec line) and reads the same `input.h5` deck; the e2e
TU has no standalone driver and no OpenMP pragmas. vexx and velocity_tendencies get no Fortran
perf baselines: no standalone driver exists for those kernels outside the harness. Baseline
CSV rows use `mode`/shape columns for their own blocking (`klon`/`nblks` NPROMA) and
`threads=0` for GPU lanes (no CPU thread sweep applies).

Toolchain requirements per lane (probed by `common.sh probe_compilers`, never assumed;
missing toolchain = loud SKIP): dace-llvm needs clang++-21 or clang++; gfortran lanes need
h5fc (HDF5 Fortran, the dwarf reader dependency); flang-serial needs flang-new-21 plus a
flang-built HDF5 (`HDF5_FLANG_ROOT` -- `hdf5.mod` is compiler-specific); cuda needs nvcc +
h5cc; openacc needs nvfortran + `HDF5_NVFORTRAN_ROOT`.

Submit examples:

    sbatch -p <partition> -A <account> samples/cloudsc/run_cloudsc_klon.sbatch                 # dace-gcc only
    LANES="dace-gcc dace-llvm" sbatch --export=ALL -p <p> -A <a> samples/vexx/run_vexx_grid.sbatch
    LANES="dace-gcc dace-llvm baselines gpu" sbatch --export=ALL -p <p> -A <a> \
        samples/cloudsc/run_cloudsc_nblks.sbatch                                               # full matrix
    bash samples/cloudsc/baselines.sh                                                          # baselines, no slurm

## cloudsc

One CLOUDSC outer call (`cloudscouter`, `tests/cloudsc/full/cloudsc.F90`) runs all `NBLOCKS`
blocks; arrays are `(KLON, KLEV=137, NBLOCKS)`. Two ways to spend the same NGPTOT=65536 points:

| mode  | KLON  | NBLOCKS | parallelism from        |
|-------|-------|---------|-------------------------|
| klon  | 65536 | 1       | inner JL (column) loops |
| nblks | 32    | 2048    | IBL block loop          |

Sizes reach `_registries.py` via `CLOUDSC_KLON` / `CLOUDSC_NBLOCKS` env vars (read at import
time; unset = old test defaults 1/4, tests unchanged).

### run

    sbatch -p <partition> -A <account> samples/cloudsc/run_cloudsc_klon.sbatch
    sbatch -p <partition> -A <account> samples/cloudsc/run_cloudsc_nblks.sbatch

Headers are site-neutral: partition/account always on the command line.

Each job: Phase A once per lane (build SDFG, specialize species counts, optimize with scalar
fission, compile; cached as `.sdfgz` keyed by KLON/NBLOCKS/backend/git-describe under
`~/.cache/dace-fortran-samples/`), then the timed sweep. Output: one CSV row per rep,
`kernel,mode,klon,nblocks,threads,rep,ms,inputs,lane`, to stdout and
`cloudsc_<mode>_<jobid>.csv`; `baselines.sh`/`gpu_baselines.sh` append the same schema to
`cloudsc_baselines_<jobid>.csv`/`cloudsc_gpu_baselines_<jobid>.csv` (their rows cover both
NPROMA modes, so request the `baselines`/`gpu` lanes from only one job of the pair).

### threads + binding

Sweep `THREADS="1 4 8 16 32 64"` (env-overridable). Per point: `OMP_PROC_BIND=close` and an
EXPLICIT `OMP_PLACES` list of physical-core CPU ids (no hyperthread siblings), NUMA-node-major
so close threads fill one NUMA region first. Topology (sockets, NUMA nodes, core list) is
probed from `lscpu` at job runtime, never hardcoded -- works on 1- or 2-socket nodes, any NPS
mode. Knobs: `--reps` (50), `--warmup` (2) -- 52 calls per sweep point -- seed 42.

### cloudsc inputs: the dwarf-p-cloudsc h5 deck (default)

`run_cloudsc_perf.py --inputs h5` (the default) drives the kernel with the ECMWF
dwarf-p-cloudsc input deck instead of the synthetic registry generator; `--inputs synthetic`
keeps the old behavior, and a missing deck falls back to synthetic with a warning. The CSV
`inputs` column records which one produced each row.

**Provenance / attribution.** The input deck (`input.h5`, md5
`b3282794694f035e8371a8aec1b93a1f`) and reference results (`reference.h5`, md5
`92d401388e74fbe5340baad19cff106a`) come from dwarf-p-cloudsc, the IFS cloud-microphysics
parameterization mini-app maintained by ECMWF (European Centre for Medium-Range Weather
Forecasts), <https://github.com/ecmwf-ifs/dwarf-p-cloudsc>, commit `f078f8f`, path
`config-files/*.h5`, licensed under Apache License 2.0. Apache-2.0 permits redistribution
with attribution; this section is that attribution.

**Download, not vendored.** `samples/cloudsc/download_data.sh` fetches both files into
`samples/cloudsc/data/` (gitignored): skip-if-md5-verified, then a local-cache/checkout
probe (`DWARF_CLOUDSC_DATA_DIR` first -- e.g. a rescued copy under
`~/Documents/dwarf-p-cloudsc-data` -- then a `dwarf-p-cloudsc` checkout next to this repo),
then `curl` from the pinned commit -- never a branch -- with the md5 pin enforced in every
mode. The sbatch jobs call it in their setup phase
(idempotent). Vendoring the 5.2 MB `input.h5` is blocked by the pre-commit
`dace-fortran-large-files` hook (500 KiB limit); to vendor anyway, add this one line under
that hook's entry in `.pre-commit-config.yaml`:
`exclude: '^samples/cloudsc/data/'`.

**Expansion semantics.** The deck holds KLON=100 source columns at KLEV=137. Exactly like
the dwarf's own expansion, columns are replicated cyclically to fill the run's gridpoints:
gridpoint `g = ib*KLON + il` (0-based) reads source column `g mod 100`. Tendency components
the deck does not ship (o3/u/v, the whole `tendency_loc` block) are zero-filled, matching
the dwarf's zero-initialized state. Constants and physics scalars (PTSPHY=3600 s, the
LAER*/LCLD* switches, NAECL* indices, NSHAPEP/Q) come from the deck verbatim.

**Constants cross-check.** `dwarf_inputs.py` cross-checks every deck constant against the
synthetic registry (`tests/cloudsc/full/_registries.py::_PHYSICAL_CONSTANTS`). The universal
yomcst/yoethf constants (flat datasets) must agree to literal-rounding precision (rtol 1e-6)
or the load hard-fails. The YRECLDP_* cloud-scheme tunables are the dwarf's namelist
configuration and genuinely differ from the registry's synthetic values for ~40 names (e.g.
`RCL_APB1` 71.8 vs 7.14e11, `RCL_KKAac` 1350 vs 67): the deck wins and the full divergence
table is printed to stderr on every load -- the synthetic and h5 regimes are NOT the same
physics configuration, so their timings are compared via the CSV `inputs` column, never
mixed. Kept from the port rather than the deck (scalar-RBETA signature): KFLDX=1,
NBETA=NBLOCKS, RBETA/RBETAP1 scalar 0.5 (the deck ships 101-entry beta tables; NSSOPT=1
never reads them).
