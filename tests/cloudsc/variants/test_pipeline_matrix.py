"""The 3 Fortran CloudSC variant SDFGs (full-CPU, gpu_scc k-caching, multistep) driven through
DaCe's canon_cpu / canon_gpu / parallelize pipelines (``tests.corpus.cloudsc.pipelines``), on legacy
and experimental_readable CPU codegen, compared to the f2py reference AT EVERY PHASE BOUNDARY.

The pipelines are grouped into phases (``start`` + 3-4 optimization phases; see ``pipelines.py``); at
each boundary ``run_pipeline`` validates the SDFG and calls ``numeric_check(transformed, phase_name)``,
which runs the partially-transformed SDFG on the seeded physical inputs and compares every output +
state array to the un-transformed f2py Fortran reference. A divergence is pinned to the exact phase
that introduced it. External-TU split stays at its default (off).

Heavy + py13 only (needs flang + dace_fortran + f2py). Each phase compiles+runs the SDFG, so a full
combo is ~5 compiles; run a few at a time under a memory cap. Checkpointed: a phase whose ``.sdfgz``
is already under ``CLOUDSC_E2E_DUMP`` passed its check on an earlier run and is resumed past rather
than redone -- point that variable at a fresh directory to force everything from scratch.
"""
import copy
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

import dace
from dace.config import Config

from _util import build_sdfg, f2py_compile, have_flang
from cloudsc.full._harness import f2py_argnames, lower_keys, sdfg_call_args
from cloudsc.full._registries import CLOUDSC_F90FLAGS, program_outputs, parameters as CLOUDSC_PARAMS
from cloudsc.full._registries import get_inputs_physical, get_outputs
from cloudsc.variants._harness import (SCALAR_TYPES, assert_species_parameters_baked, extract_variant_tu,
                                       mismatch_report)

# dotted ``tests.corpus...`` would collide with this repo's own top-level ``tests`` package
# (both define ``tests/__init__.py``; whichever binds sys.modules['tests'] first wins) -- load by path.
#
# Resolved against the IMPORTED dace, not this file's position on disk. ``parents[4]`` hardcoded the
# sibling-checkout layout and so always loaded /home/primrose/Work/dace, whatever PYTHONPATH selected:
# running against a worktree then paired one tree's library with another tree's phase plan, offload
# gate and checkpoint signature, silently measuring a combination that exists in no checkout.
_PIPELINES_PATH = Path(dace.__file__).resolve().parents[1] / 'tests' / 'corpus' / 'cloudsc' / 'pipelines.py'
if not _PIPELINES_PATH.is_file():
    # The dace on PYTHONPATH is an installed package without its tests/ tree (CI), so the cloudsc
    # pipeline corpus this replication matrix drives is genuinely absent -- skip like a missing
    # toolchain, rather than erroring at collection.  Point PYTHONPATH at a dace checkout to run it.
    pytest.skip(
        f'{_PIPELINES_PATH} not found -- the dace on PYTHONPATH ({Path(dace.__file__).resolve().parents[1]}) is an '
        'installed package without its tests/ tree; point PYTHONPATH at a dace checkout',
        allow_module_level=True)
_spec = importlib.util.spec_from_file_location('cloudsc_dace_pipelines', _PIPELINES_PATH)
_pipelines = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pipelines)
dump_root, run_pipeline, uniquely_named = _pipelines.dump_root, _pipelines.run_pipeline, _pipelines.uniquely_named

#: Opt-in GPU offload for the GPU-capable variants. Off by default: the offloaded graph is
#: device-scheduled and no longer host-runnable, so its final phase trades the numeric check for a
#: validate + CUDA-codegen check. Run it as its own sweep, not alongside the CPU baseline.
OFFLOAD = os.environ.get('CLOUDSC_OFFLOAD', '') == '1'

#: Opt-in timing leg: recompiles the FINAL graph at -O3 and times it. Off inside a correctness sweep
#: -- it changes the build regime, and a correctness leg must not silently measure a different binary
#: than the one it validated.
BENCH = os.environ.get('CLOUDSC_BENCH', '') == '1'
BENCH_TSV = os.environ.get('CLOUDSC_BENCH_TSV', '')
BENCH_REPS = int(os.environ.get('CLOUDSC_BENCH_REPS', '5'))

HERE = Path(__file__).resolve().parent
FULL_DIR = HERE.parent / 'full'

#: SDFG-vs-Fortran per-phase tolerance, PER PRECISION REGIME (see CASES[...]['tol']).
#: - force_double_precision cases (gpu_scc / multistep, all fp64): tight 1e-11 -- a hair of slack over
#:   1e-12 for the reassociating Loop2X / Loop2Map / fusion phases.
#: - native full_cpu: keeps the kernel's reduced-precision REAL(JPRM)=fp32 exp/pow helpers, so the SDFG
#:   (fp64 libm) vs gfortran (fp32 libm) legitimately differ ~1e-6 -- checked loose.
_TOL_FP64 = 1e-11
_TOL_MIXED = 1e-6


def _full_cpu_tu(_out_dir: Path) -> str:
    # full CPU cloudsc.F90 is already a single self-contained TU (cpp pre-expanded); build as-is.
    return (FULL_DIR / 'cloudsc.F90').read_text()


def _variant_tu(wrapper: str, name: str):
    return lambda out_dir: extract_variant_tu(HERE / wrapper, out_dir, name)


#: name -> (build_tu(out_dir)->str, state-carry arrays, whether species PARAMETERs must be baked).
CASES = {
    'gpu_scc':
    dict(build_tu=_variant_tu('cloudsc_outer_scc_k_caching.F90', 'gpu_scc'),
         state=('tendency_loc_T', 'tendency_loc_a', 'tendency_loc_q', 'tendency_loc_cld'),
         baked=True,
         tol=_TOL_FP64),
    'multistep':
    dict(build_tu=_variant_tu('cloudsc_outer_multistep.F90', 'multistep'),
         state=('PT', 'PQ', 'PA', 'PCLV'),
         baked=True,
         tol=_TOL_FP64),
    'full_cpu':
    dict(build_tu=_full_cpu_tu, state=(), baked=False, tol=_TOL_MIXED),
}
VARIANTS = ('parallelize', 'canon_cpu', 'canon_gpu')
CODEGENS = ('legacy', 'experimental_readable')

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


@pytest.fixture(scope='module', params=list(CASES))
def case(request, tmp_path_factory):
    """Build the base SDFG + f2py reference once per source; tests deepcopy the SDFG per combo."""
    name = request.param
    spec = CASES[name]
    tu_text = spec['build_tu'](tmp_path_factory.mktemp(f'{name}_tu'))
    f2py_ref = f2py_compile(tu_text,
                            tmp_path_factory.mktemp(f'{name}_ref'),
                            f'{name}_ref',
                            extra_f90flags=CLOUDSC_F90FLAGS,
                            only=('cloudscouter', ))
    sdfg = build_sdfg(tu_text, tmp_path_factory.mktemp(f'{name}_sdfg'), name=name, entry='cloudscouter').build()
    if spec['baked']:
        assert_species_parameters_baked(sdfg)

    rng = np.random.default_rng(42)
    inputs = lower_keys(get_inputs_physical(rng))
    # Pristine initial buffers -- every phase's SDFG run must start from these. f2py runs on COPIES
    # below; copying the reference AFTER the f2py call would seed the SDFG with the integrated result.
    outputs_init = lower_keys(get_outputs(rng))
    state_names = tuple(s.lower() for s in spec['state'])
    state_init = {k: np.array(inputs[k], copy=True, order='F') for k in state_names}

    outputs_ref = {k: v.copy(order='F') for k, v in outputs_init.items()}
    state_ref = {k: v.copy(order='F') for k, v in state_init.items()}
    accepted = f2py_argnames(f2py_ref.cloudscouter)
    f2py_ref.cloudscouter(**{k: v for k, v in {**inputs, **outputs_ref, **state_ref}.items() if k in accepted})

    # Specialize the species-index PARAMETERs (nclv, ncldqi, ...) to their literal values so the
    # `specialize` config-prop step folds species-index guards before simplify. ONLY these --
    # grid-shape dims (klev/klon/nblocks) stay symbolic (baking them makes grid loops constant-trip,
    # which ShortLoopUnroll then unrolls into a blown-up graph). Species are usually already baked by
    # the frontend, so this is typically a no-op (constants empty).
    free = {str(s).lower() for s in sdfg.free_symbols}
    constants = {
        k: int(CLOUDSC_PARAMS[k.upper()])
        for k in ('nclv', 'ncldql', 'ncldqi', 'ncldqr', 'ncldqs', 'ncldqv') if k in free and k.upper() in CLOUDSC_PARAMS
    }

    return name, sdfg, inputs, outputs_init, state_init, outputs_ref, state_ref, spec['state'], constants, spec['tol']


@pytest.mark.parametrize('codegen', CODEGENS, ids=('old', 'new'))
@pytest.mark.parametrize('variant', VARIANTS)
def test_pipeline_matrix(case, variant, codegen, _strict_fp_cpu_args):
    name, base_sdfg, inputs, outputs_init, state_init, outputs_ref, state_ref, state_arrays, constants, tol = case
    sdfg = uniquely_named(copy.deepcopy(base_sdfg), f'{name}_{variant}_{codegen}')
    compare_names = list(program_outputs) + list(state_arrays)

    def call_kwargs(transformed, phase_name='final'):
        """Argument dict for one run of ``transformed``, on FRESH output/state buffers.

        Shared by the numeric check and the timing leg so both drive the SDFG the same way: a second,
        hand-rolled builder is how a perf number ends up measuring a different call than the one that
        was validated."""
        out = {k: v.copy(order='F') for k, v in outputs_init.items()}
        st = {k: v.copy(order='F') for k, v in state_init.items()}
        scalars = {k: v for k, v in inputs.items() if isinstance(v, SCALAR_TYPES)}
        scalars.setdefault('irank', np.int32(0))
        scalars.setdefault('numproc', np.int32(1))
        kwargs = {k: v for k, v in inputs.items() if not isinstance(v, SCALAR_TYPES)}
        kwargs.update(out)
        kwargs.update(st)
        kwargs.update(sdfg_call_args(transformed, scalars))
        arglist = transformed.arglist()
        kwargs = {k: v for k, v in kwargs.items() if k in arglist}
        missing = sorted(set(arglist) - set(kwargs))
        assert not missing, f"{name}/{variant}/{codegen}/{phase_name}: SDFG arguments not covered: {missing}"
        return kwargs, out, st

    def numeric_check(transformed, phase_name):
        """Run the phase's transformed SDFG on the seeded inputs (fresh buffers) and compare every
        output + state array to the un-transformed f2py reference -- so a divergence pins to this
        exact phase."""
        Config.set('compiler', 'cpu', 'implementation', value=codegen)
        kwargs, out, st = call_kwargs(transformed, phase_name)
        transformed(**kwargs)
        report = mismatch_report({**out, **st}, {**outputs_ref, **state_ref}, compare_names, rtol=tol, atol=tol)
        assert not report, (f"{name}/{variant}/{codegen}/{phase_name} numerical mismatch (tol={tol:.0e}):\n" +
                            "\n".join(report))

    # resume=True: a checkpoint on disk means that phase already passed its numeric check, and
    # run_pipeline now loads checkpoints strictly -- one that lost anything on the way through JSON
    # raises and is skipped, so a resume can only start from a graph that read back intact.
    # CLOUDSC_E2E_DUMP scopes the checkpoints per run; point it somewhere fresh to force a full redo.
    final = run_pipeline(sdfg,
                         variant,
                         dump_root(),
                         constants=constants,
                         tag=f'{name}_{variant}_{codegen}',
                         numeric_check=numeric_check,
                         resume=True,
                         offload=OFFLOAD and variant in _pipelines.OFFLOAD_VARIANTS)

    if BENCH:
        # Same graph, -O3 instead of the numeric leg's -O0. FP semantics are unchanged between the
        # legs (both -fno-fast-math -ffp-contract=off), so this measures the optimizer, not a flag
        # swap. CUDA args stay at their configured default; STRICT_FP_CUDA_ARGS applies only inside
        # check_offload_phase, which has already returned by now.
        Config.set('compiler', 'cpu', 'implementation', value=codegen)
        with dace.config.set_temporary('compiler', 'cpu', 'args', value=_pipelines.PERF_CPU_ARGS), \
             dace.config.set_temporary('compiler', 'build_type', value='Release'):
            kwargs, _out, _st = call_kwargs(final)
            stats = _pipelines.benchmark_candidate(final, kwargs, reps=BENCH_REPS, tag=f'{name}_{variant}_{codegen}')
        row = (f'{name}\t{variant}\t{codegen}\t{"offload" if OFFLOAD else "cpu"}\t'
               f'{stats["median"]:.6f}\t{stats["min"]:.6f}\t{stats["max"]:.6f}\t{int(stats["reps"])}')
        print(f'BENCH\t{row}')
        if BENCH_TSV:
            with open(BENCH_TSV, 'a') as fh:
                fh.write(row + '\n')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
