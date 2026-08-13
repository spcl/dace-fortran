# CloudSC CPU: reproducing the 6-lane comparison

Six lanes, in three toolchain-paired groups. Each pair builds **identical sources** and differs
only in compiler, so a pair isolates the toolchain and the groups isolate the implementation.

| Lane | Implementation | Compiler | Built by |
|------|----------------|----------|----------|
| `dace-gcc` | Original Fortran -> DaCe -> C++ | g++ | `run_cloudsc_perf.py` |
| `dace-llvm` | Original Fortran -> DaCe -> C++ | clang++ | `run_cloudsc_perf.py` |
| `original-openmp` | Original Fortran (dwarf-p-cloudsc) | gfortran | `baselines.sh` |
| `flang-openmp` | Original Fortran (dwarf-p-cloudsc) | flang | `baselines.sh` |
| `c-openmp` | Hand-written C rewrite | g++ | `baselines.sh` |
| `c-openmp-clang` | Hand-written C rewrite | clang++ | `baselines.sh` |

## Run it

```bash
cd <repo root>          # submit from the repo root: the -o/-e paths are repo-relative
sbatch -p normal -A g34 --time=08:00:00 samples/cloudsc/run_cloudsc_nblks.sbatch   # KLON=32
sbatch -p normal -A g34 --time=08:00:00 samples/cloudsc/run_cloudsc_klon.sbatch    # KLON=65536
```

Six lanes over the full thread sweep do not fit the 30-minute `debug` cap; `normal` allows up to
24 h. Both modes hold NGPTOT=65536 so they stay directly comparable.

**Request `baselines` from only one of the two jobs.** `baselines.sh` runs both klon and nblks
modes itself, so asking both jobs for it duplicates every baseline row. The default `LANES` on
each script includes it, so drop it from the second submission:

```bash
sbatch -p normal -A g34 --time=08:00:00 \
    --export=ALL,LANES="dace-gcc dace-llvm" samples/cloudsc/run_cloudsc_klon.sbatch
```

## Knobs

| Variable | Default | Effect |
|----------|---------|--------|
| `LANES` | `dace-gcc dace-llvm baselines` | Which lanes the sbatch runs |
| `BASELINE_LANES` | `original-openmp flang-openmp c-openmp c-openmp-clang` | Which lanes `baselines.sh` runs |
| `THREADS` | `1 2 4 8 16 32 64 72` (`common.sh`) | Thread sweep; points above the physical core count are skipped |
| `REPS` / `WARMUP` | `50` / `2` | Timed and untimed repetitions |
| `CLOUDSC_NGPTOT` | `65536` | Problem size |
| `CLOUDSC_SWEEP` | `0` | `1` walks the figure-A problem-size sweep instead of the two fixed modes |
| `MEAS_ALLOC` | `mimalloc` | Allocator; `system` opts out. Stamped into the CSV `alloc` column |

The older diagnostic lanes are still available: `BASELINE_LANES="gfortran-serial gfortran-autopar
openacc-cpu flang-serial"` brings them back. `flang-serial` exists because flang has no
`-ftree-parallelize-loops` equivalent, so it cannot mirror `gfortran-autopar`; `flang-openmp` is the
lane that *does* pair with `original-openmp`.

A lane whose compiler or HDF5 flavour is missing prints `SKIP <lane>: ...` and the run continues,
so a partial toolchain yields fewer columns rather than a failed job. Check the log for `SKIP` if a
column is missing from the figure.

## Allocator

mimalloc is the default for every native lane and is preloaded by `samples/alloc_pool.sh`
(`mimalloc@3.3.2` in spack). Every CSV row carries an `alloc` column, so a mimalloc run can never
be silently mixed with a system-malloc run. Do not compare rows across different `alloc` values.

## Outputs and plotting

Each job writes `cloudsc_<mode>_<jobid>.csv` (DaCe lanes) and `cloudsc_baselines_<jobid>.csv`
(baseline lanes) with the shared header:

```
kernel,mode,klon,nblocks,threads,rep,ms,inputs,lane,alloc
```

```bash
python samples/figures/plot_cloudsc_cpu.py
```

By default the plot discovers every `cloudsc*.csv` under `$WORK_ROOT` and prints the list it used.
Pin the inputs explicitly when you want one specific run:

```bash
CLOUDSC_CSVS=/path/a.csv,/path/b.csv python samples/figures/plot_cloudsc_cpu.py
```

Input paths used to be hardcoded job ids, which meant a new run silently plotted the old numbers
instead of failing -- hence the discovery step and the printed source list.
