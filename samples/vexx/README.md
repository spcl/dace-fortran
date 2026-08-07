# vexx

CPU thread-sweep for QE `exx_bp::vexx_bp_k_gpu` (`tests/qe/exx_bp/ast_v1_vexx_bp_k_gpu.f90`),
built and bound exactly like the e2e lane (`tests/e2e/test_vexx.py`). Perf only.

## the no-op trap

The committed init (`init_vexx_bp_k_gpu_state_c`) exists for the CORRECTNESS lane: it sets
`nqs=0` (kills the `vexxmain` q-loop) and `nibands=0` (kills the band loops), so the kernel is an
identity copy of `hpsi`. A naive perf run on it measures nothing. This sample calls
`init_vexx_bp_k_gpu_perf_state_c` (same caller file) instead: `nqs>=1`, `nibands(1)=m`,
all-pairs band list, every array the active path touches allocated with in-range contents.
The driver aborts if the first call leaves `hpsi` unchanged.

## modes

Two SHAPE modes, same kernel, roughly comparable `m * nnr` (within ~2x):

| mode  | m  | nnr           | work sits in                  |
|-------|----|---------------|-------------------------------|
| bands | 64 | 48^3 = 110592 | band loop (many bands)        |
| grid  | 4  | 128^3 = 2097152 | ir grid loops (huge grid)   |

All dims are SYNTHETIC -- no production QE sizes are documented in-tree. Only the mode CONTRAST
is the experiment, not the absolute numbers. Note: the exchange is all-pairs, so partner-loop
arithmetic scales `m^2 * nnr`; bands mode does more of it than grid mode at matched `m * nnr`.

## knobs

Env: `VEXX_M`, `VEXX_NNR`, `VEXX_N` (default 4096), `VEXX_LDA` (default `VEXX_N`),
`VEXX_NQS` (default 1), `VEXX_JBLOCK` (default 1, chunks the partner-band loop -- secondary
knob, larger values batch the Rho FFT views). Flags: `--reps` (50), `--warmup` (2), `--seed`
(42), `--fft` (below). Threads: `THREADS="1 4 8 16 32 64"` via `../common.sh`, physical-core
pinning as in the cloudsc sample.

## FFT finding

The QE `fwfft`/`invfft` call sites are NOT external stubs on the DaCe side: the bridge
recognises them (`dispatch.cpp` qeFftCalleeTag) and lowers each to a `dace.libraries.fft`
FFT/IFFT library node over the flat grid buffer -- 1-D, descriptor dims and `howmany` ignored
("multi-D semantics is a follow-up gap"). The empty `fwfft_y`/`invfft_y` stubs in the caller
only serve gfortran reference lanes. Consequences:

* Default `--fft fftw3` expands the nodes to FFTW3 tasklets: links `libfftw3` (the dependency
  the repo's `fftw` pytest marker documents). The tasklets are single-threaded, so the sweep
  has a serial FFT floor in both modes.
* The IFFT nodes carry QE's `1/N` inverse normalisation as a lib-node factor; the FFTW3
  expansion rejects non-1 factors, so `fftw3` mode drops it. Values are therefore NOT QE
  physics (they already were not, given the flat-1-D lowering); work shape is unchanged.
* `--fft pure` keeps the library default: an O(nnr^2) DFT map -- thread-scalable but only
  feasible for tiny grids.

## run

    sbatch -p <partition> -A <account> samples/vexx/run_vexx_bands.sbatch
    sbatch -p <partition> -A <account> samples/vexx/run_vexx_grid.sbatch

Phase A once per lane (SDFG build + `pipelines.optimize` + binding link; cached as the linked
`.so` keyed by mode/fft/backend/git-describe under `~/.cache/dace-fortran-samples/` -- dims
stay symbolic, so one `.so` serves all dims). Phase B: one CSV row per rep,
`kernel,mode,m,nnr,threads,rep,ms,inputs,lane` (`inputs` is always `synthetic` -- all vexx
dims are synthetic, see above), to stdout and `vexx_<mode>_<jobid>.csv`. The timed call
includes the binding's module-global marshalling (`exxbuff` etc.) -- constant per-rep
overhead. Wipe the cache dir to force a rebuild.

Lanes: `LANES` env, space list from `dace-gcc` (default) and `dace-llvm` (`--backend llvm`,
clang++ as the DaCe CPU compiler -- see `samples/README.md` lane matrix). No Fortran perf
baseline lanes exist for vexx: no standalone driver exists for this kernel outside the
harness (the QE checkpoint only runs it inside the full code).
