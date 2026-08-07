# samples

CPU scaling experiments on cloudsc, vexx and velocity_tendencies, driven through the same
build + `pipelines.optimize` path the e2e tests use. Perf only -- numerics stay in `tests/e2e/`.

## prerequisites

One toolchain per lane (table below); anything missing is skipped, so a partial install still
produces the lanes it can build. With spack:

```
spack install newlib target=nvptx-none &&
spack install nvptx-tools &&
spack install -j$(nproc) gcc@16.1.0 +graphite +nvptx +binutils languages=c,c++,fortran %gcc@14.2.0 ^cuda &&
spack compiler find $(spack location -i gcc@16.1.0+nvptx) &&
spack install -j$(nproc) llvm@22.1.7 +polly +cuda cuda_arch=80,90 +flang +libomptarget +mlir +lld targets=nvptx,aarch64 %gcc@16.1.0 && # aarch, nvidia
spack install -j$(nproc) llvm@22.1.5 +polly +cuda cuda_arch=80,90 +flang +libomptarget +mlir +lld targets=nvptx,x86 %gcc@16.1.0 && # x86, nvidia
spack install -j$(nproc) llvm@22.1.7 +polly +flang +libomptarget +mlir +lld targets=x86,amdgpu %gcc@16.1.0  && # x86, amd
spack install -j$(nproc) nvhpc +mpi +blas +lapack &&
spack install -j$(nproc) openblas +fortran threads=openmp %gcc@16.1.0
```

Libraries. The Fortran baseline lanes read the dwarf `input.h5` deck through the HDF5 Fortran
API and `hdf5.mod` is not portable across compilers, so they need one HDF5 build per Fortran
compiler; the vexx FFT library nodes link `libfftw3` (single-threaded tasklets, so no `+openmp`):

```
spack compiler find $(spack location -i llvm@22.1.7) &&
spack compiler find $(spack location -i nvhpc) &&
spack install -j$(nproc) hdf5 +fortran +hl %gcc@16.1.0 &&
spack install -j$(nproc) hdf5 +fortran +hl %llvm@22.1.7 &&
spack install -j$(nproc) hdf5 +fortran +hl %nvhpc &&
spack install -j$(nproc) fftw %gcc@16.1.0
```

Python is not taken from spack by default: the drivers run on the pinned interpreter
(`~/.pyenv/versions/py13/bin/python`). A spack `python@3.13` works if `PYTHON=` points at it.
The velocity bindings configure needs cmake >= 3.18 (system package or `spack install cmake`).

## reproduce

1. Install with the pinned interpreter (`~/.pyenv/versions/py13/bin/python`, override with
   `PYTHON=`): `pip install -e '.[samples]'`. DaCe comes in as a dependency, from `FaCe`.
2. Fetch data. The jobs do it themselves and skip when the files are there. By hand:
   `bash samples/cloudsc/download_data.sh`, `bash samples/velocity_tendencies/download_data.sh`.
3. Build the velocity python bindings (`convert_data.py` also builds them on first import):
   `python samples/velocity_tendencies/velocity_data_build.py`.
4. Put the toolchain on PATH: copy `samples/env.spack.example` to `samples/env.sh` (or export
   `SAMPLES_ENV=<file>`); `common.sh` sources it in every job before probing compilers.
5. Submit the full CPU matrix: `./samples/submit_all.sh -p <partition> -A <account>`. Its per-job
   lane lists are defaults; an exported `LANES` overrides every job. GPU lanes are opt-in: add
   `gpu` to `LANES` on the cloudsc jobs.
6. Collect: one CSV per job, `kernel,mode,<p1>,<p2>,threads,rep,ms,inputs,lane`.

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
cloudsc-only (`cloudsc/baselines.sh` `BASELINE_LANES`, `cloudsc/gpu_baselines.sh` `GPU_LANES`)
except `gfortran-autopar`, `openacc-cpu` and `openacc`: velocity_tendencies runs those three too,
from `velocity_tendencies/baselines.sh` (`LANES=baselines`, subset via `BASELINE_LANES`), building
`velocity_advection_acc.f90` -- the directive-annotated twin of the e2e TU, see
`scripts/annotate_velocity_acc.py` -- against its own `driver_velocity.f90` on a raw dump from
`dump_data.py`, no HDF5. The serial and autopar lanes compile without `-fopenmp`/`-fopenacc`, so
the `!$omp` and `!$acc` lines in those sources are inert comments -- no source is ever stripped.

vexx has no Fortran baseline lanes yet.

Detail sits next to the code: `samples/cloudsc/dwarf_inputs.py` (deck expansion, constants
cross-check), `samples/vexx/README.md`, `samples/velocity_tendencies/README.md`.

## attribution

`input.h5` (md5 `b3282794694f035e8371a8aec1b93a1f`) and `reference.h5` (md5
`92d401388e74fbe5340baad19cff106a`) come from ECMWF dwarf-p-cloudsc,
<https://github.com/ecmwf-ifs/dwarf-p-cloudsc>, commit `f078f8f`, Apache-2.0.
Downloaded by `download_data.sh`, never vendored.
