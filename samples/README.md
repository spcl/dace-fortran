# samples

CPU scaling experiments on cloudsc, vexx and velocity_tendencies, driven through the same
build + `pipelines.optimize` path the e2e tests use. Perf only -- numerics stay in `tests/e2e/`.

## reproduce

1. Install with the pinned interpreter (`~/.pyenv/versions/py13/bin/python`, override with
   `PYTHON=`): `pip install -e '.[samples]'`. DaCe comes in as a dependency, from `FaCe`.
2. Fetch data. The jobs do it themselves and skip when the files are there. By hand:
   `bash samples/cloudsc/download_data.sh`, `bash samples/velocity_tendencies/download_data.sh`.
3. Build the velocity python bindings (`convert_data.py` also builds them on first import):
   `python samples/velocity_tendencies/velocity_data_build.py`.
4. Submit the full CPU matrix: `./samples/submit_all.sh -p <partition> -A <account>`. Its per-job
   lane lists are defaults; an exported `LANES` overrides every job. GPU lanes are opt-in: add
   `gpu` to `LANES` on the cloudsc jobs.
5. Collect: one CSV per job, `kernel,mode,<p1>,<p2>,threads,rep,ms,inputs,lane`.

## lanes

| lane             | builds                                        | toolchain gate                    |
|------------------|-----------------------------------------------|-----------------------------------|
| dace-gcc         | DaCe pipeline, g++                            | -                                 |
| dace-llvm        | DaCe pipeline, clang++-21                     | clang++                           |
| gfortran-serial  | dwarf tree, `-O3 -march=native`               | h5fc wrapping gfortran            |
| gfortran-autopar | same + `-ftree-parallelize-loops=$t`          | h5fc wrapping gfortran            |
| original-openmp  | same + `-fopenmp`                             | h5fc wrapping gfortran            |
| openacc-cpu      | dwarf scc variant, nvfortran `-acc=multicore` | nvfortran + `HDF5_NVFORTRAN_ROOT` |
| flang-serial     | dwarf tree, flang `-O3` (flang: no autopar)   | flang + `HDF5_FLANG_ROOT`         |
| cuda (GPU)       | dwarf `cloudsc_cuda`, nvcc                    | nvcc + h5cc                       |
| openacc (GPU)    | dwarf scc variant, nvfortran `-acc=gpu`       | nvfortran + `HDF5_NVFORTRAN_ROOT` |

A missing toolchain is a loud SKIP on stderr, never a failure. Every lane below `dace-llvm` is
cloudsc-only, from `cloudsc/baselines.sh` (`BASELINE_LANES`) or `cloudsc/gpu_baselines.sh`
(`GPU_LANES`). The serial and autopar lanes compile without `-fopenmp`/`-fopenacc`, so the
`!$omp` and `!$acc` lines in those sources are inert comments -- no source is ever stripped.

vexx and velocity_tendencies have no Fortran baseline lanes yet; velocity autopar and OpenACC
(CPU and GPU) wait on a port of the `!$ACC` directives from
`tests/icon/full/icon-model/src/atm_dyn_iconam/mo_velocity_advection.f90` onto the inlined TU.

Detail sits next to the code: `samples/cloudsc/dwarf_inputs.py` (deck expansion, constants
cross-check), `samples/vexx/README.md`, `samples/velocity_tendencies/README.md`.

## attribution

`input.h5` (md5 `b3282794694f035e8371a8aec1b93a1f`) and `reference.h5` (md5
`92d401388e74fbe5340baad19cff106a`) come from ECMWF dwarf-p-cloudsc,
<https://github.com/ecmwf-ifs/dwarf-p-cloudsc>, commit `f078f8f`, Apache-2.0.
Downloaded by `download_data.sh`, never vendored.
