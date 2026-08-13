# VelocityTendenciesPipeline

End-to-end DaCe pipeline for the ICON velocity-tendencies dynamical-core
kernel. Six stages take an f2dace-generated SDFG through structural
canonicalisation (1, 2), schedule preparation (3), GPU offloading (4),
neighbour/blk index compression (5), and layout-permutation sweep (6).

## Layout

```
baseline_inputs/   self-contained Fortran AST (velocity_modified.f90)
baseline/          f2dace output + the 4 specialised SDFG variants
codegen/stageN/    per-stage SDFG outputs (1..6)
generate_baselines.py        AoS->SoA + symbol resolution + 4-way specialise
utils/passes/                 pipeline passes (DaCe-side or velocity-only)
utils/stages/stage{1..6}.py   per-stage drivers (--optimize / --compile)
tools/                        F90 -> SDFG, baseline regen, submission gen
```

## Stage summary

| Stage | What | Driver |
|-------|------|--------|
| 0     | f2dace: F90 -> SDFG | `tools/sdfg_from_velocity_f90.py` |
| 0a    | AoS -> SoA + 4 specialised variants | `generate_baselines.py` |
| 1     | Maximally parallel form (LoopToMap, MoveIfIntoMap, ...) | `utils.stages.stage1` |
| 2     | Lift trivial ifs, offset Fortran 1..N+1 -> 0..N, simplify exprs | `utils.stages.stage2` |
| 3     | Set transient lifetimes, int64->int32, lift transients to root | `utils.stages.stage3` |
| 4     | GPU offload, persistent transients, body timer, async reductions | `utils.stages.stage4` |
| 5     | Neighbour/blk compression (fold + uint16/uint8 variants) | `utils.stages.stage5` |
| 6     | Layout-permutation sweep (per-pattern config) | `utils.stages.stage6` |

## Quick start

```bash
# 0. Toolchain. nvcc, nvc++ and nvfortran all come from one nvhpc, so the DaCe binary and the
#    OpenACC Fortran reference are built by the same compiler family.
. ./env_nvhpc.sh

# 1. Regenerate baselines from F90 (needs DaCe on the f2dace/staging branch).
bash tools/regenerate_baselines.sh

# 2. Build a stage 5 GPU binary directly:
python -m utils.stages.stage5 --optimize --compile

# 3. Or generate sbatch scripts for a sweep (daint.alps + beverin):
python tools/generate_submissions.py \
    --config tools/submissions/example.json \
    --out-dir submissions/
sbatch submissions/run_stage6_default_sweep_daint.sh
```

## Engines: split-4 DaCe vs original OpenACC Fortran

`main.cpp` drives two engines over the same deserialised structs, selected with `--engine`:

| Engine | What runs | Entry points |
|--------|-----------|--------------|
| `dace` (default) | The 4 specialised SDFGs behind one caller-facing interface | `velocity_dace_init` / `_run` / `_finalize` (`src/velocity_split_dispatch.cpp`) |
| `acc` | The original single-TU OpenACC Fortran `velocity_tendencies` | `velocity_acc_setup` / `_run` / `_teardown` (`src/velocity_acc_bridge.f90`) |
| `both` | Each in turn from the same dumped t0 state; got/want under `<output-dir>/<engine>/` | — |

The split into 4 variants is an implementation detail of the pipeline: `velocity_dace_run` does the
`(lvn_only, istep)` dispatch, so neither the runner nor ICON branches on it. `velocity_dace_init`
brings up all 4 handles at once, which is legal because `unify_variant_signatures` makes the 4
signatures byte-identical.

`VELOCITY_CALL_ARGS` expands to 18 `serde::ARRAY_META_DICT_AT` lookups (the extents of the 3 bare
`z_*` arrays; everything else comes from struct fields), so the wrapper needs serde. serde's 53
namespace-scope functions therefore carry `inline` -- **not** `static`: `ARRAY_META_DICT()` holds a
function-local `static` map, and internal linkage would give every translation unit its own empty
copy, turning a link error into an `std::out_of_range` at run time. `inline` merges the definition
and shares the map.

The ACC engine is opt-in at build time, because the two ACC translation units define the same
module and only one can be linked:

```bash
. ./env_nvhpc.sh
VT_WITH_ACC=1 VT_ACC_TU=loopexch python -m utils.stages.stage5 --compile   # or noloopexch
sbatch -p debug -A g34 samples/velocity_tendencies/opt_pipeline/run_acc_vs_dace.sbatch
```

`run_acc_vs_dace.sbatch` runs correctness at `reps=1` (both engines diffed against the same `want`
via `utils/compare_got_and_want.py`) and then performance at `reps=50`.

Both engines are timed under the same floating-point contract: nvfortran gets
`-gpu=cc90,fastmath,flushz,fma` against the DaCe side's `--use_fast_math --ftz=true --fmad=true
--prec-div=false --prec-sqrt=false`. Override with `VT_ACC_FFLAGS`.

The standalone is **big-nproma only**, by design: `data_r02b05` has `z_kin_hor_e` at
`(20480, 90, 2)` -- nproma=20480 over 2 blocks -- which is the flat `no_nproma` layout the 4
variants were specialised for. Small-nproma velocity is an ICON-side concern, not a standalone one.

`src/velocity_acc_mirror.f90` is generated — rerun `tools/gen_acc_mirror.py` after regenerating
`include/velocity_tendencies_no_nproma.h`, and `tools/gen_acc_mirror.py --check` fails if it is
stale. The bridge asserts every C++ `sizeof` against its Fortran mirror before first use.

Configuration this comparison assumes, taken from the recorded dump scalars (`l_vert_nested=0`,
`ddt_vn_cor_associated=0`, `lextra_diffu=1`): `p_diag%vn_ie_ubc` and `p_diag%ddt_vn_cor_pc` are
never written, which is why the DaCe frontend folded both out of the generated signature. The
bridge allocates them zeroed and checks at teardown that they stayed zero.

## Regenerating baselines from F90

`tools/regenerate_baselines.sh` is the single entry point for rebuilding
every baseline + per-stage SDFG from the Fortran AST snapshot in
`baseline_inputs/velocity_modified.f90`. The phases are:

| Phase | Output | Tool |
|-------|--------|------|
| 0 | `baseline/velocity_no_nproma.sdfgz`               | `tools/sdfg_from_velocity_f90.py` |
| 1 | `baseline/velocity_no_nproma_if_prop_lvn_only_{0,1}_istep_{1,2}.sdfgz` | `generate_baselines.py` |
| 2 | `codegen/stage1/<variant>.sdfgz`                  | `python -m utils.stages.stage1 --optimize` |
| 3 | `codegen/stage2/<variant>.sdfgz`                  | `python -m utils.stages.stage2 --optimize` |
| 4 | `codegen/stage3/<variant>.sdfgz`                  | `python -m utils.stages.stage3 --optimize` |
| 5 | `codegen/stage4/<variant>.sdfgz`                  | `python -m utils.stages.stage4 --optimize` |
| 6 | `codegen/stage5/<variant>.sdfgz`                  | `python -m utils.stages.stage5 --optimize` |

### Phase 0: f2dace (F90 -> SDFG)

`baseline_inputs/velocity_modified.f90` is a stage-1 (preprocessed)
self-contained Fortran AST: every `mo_*` dependency reachable from
`mo_velocity_advection.velocity_tendencies` is inlined,
`mo_exception`/`mo_real_timer` are pre-stubbed. So phase 0 skips
`create_preprocessed_ast` and calls `create_singular_sdfg_from_ast`
directly:

```python
ParseConfig(sources=[velocity_modified.f90],
            entry_points=[("mo_velocity_advection", "velocity_tendencies")])
-> create_internal_ast()
SDFGConfig({"velocity_tendencies": ("mo_velocity_advection", "velocity_tendencies")},
           normalize_offsets=True, multiple_sdfgs=False)
-> create_sdfg_from_internal_ast() -> {"velocity_tendencies": SDFG}
```

This requires a DaCe checkout that ships the Fortran frontend (typically
branch `f2dace/staging`). Phases 1-6 run on the day-to-day branch
(`yakup/dev`); the script itself does not switch branches -- the caller
is responsible.

### Env overrides (regenerate_baselines.sh)

| Variable | Default | Purpose |
|----------|---------|---------|
| `PIPELINE_DIR` | repo root | path to this checkout |
| `PYTHON`       | `python` | python interpreter |
| `SKIP_F2DACE`  | `0`     | set to `1` to skip phase 0 |
| `ONLY_PHASE`   | `all`    | run only phase `0..6` |
| `STAGE_FLAGS`  | `--optimize` | flags forwarded to each `utils.stages.stage*` |

### Skipping phase 0

If the f2dace branch isn't conveniently checked out, run phase 0
elsewhere and drop `velocity_no_nproma.sdfgz` into `baseline/`, then:

```bash
SKIP_F2DACE=1 bash tools/regenerate_baselines.sh
```

## Generating submission scripts

`tools/generate_submissions.py` reads a JSON experiment config and emits
one `run_<experiment>_<platform>.sh` per cell under `--out-dir`. The
input schema is documented in the docstring of that file; an example
lives at `tools/submissions/example.json`.

Each emitted script is a fully-self-contained sbatch:

1. Sources `activate.sh` (override via `ACTIVATE_PATH=<path>`).
2. Loads platform spack modules (`gcc`, `cuda`/`rocm`, ...).
3. Exports platform env (`GENCODE_NUMBER=90a`, `BEVERIN=1`, ...).
4. `cd $PIPELINE_DIR` and runs `python -m utils.stages.stage<N>` per
   config in the experiment's `configs` list.
5. Copies emitted CSVs from `codegen/stage<N>/<config>/` into the
   per-platform results tree.

Supported platforms (auto-merged from the JSON):

| Platform | Partition | GPU arch | Notes |
|----------|-----------|----------|-------|
| daint    | normal    | sm_90a (GH200) | needs `--account` |
| beverin  | mi300     | gfx942 (MI300A) | exports `BEVERIN=1` |

Add a new platform by appending to the `platforms` block of your JSON
config; the generator never hardcodes a platform name.

### Example

```json
{
  "defaults": {"stage": 6, "stage_flags": "--optimize --compile",
               "configs": ["unpermuted", "nlev_first", "index_only"],
               "time": "06:00:00", "account": "g177-1",
               "pipeline_dir": "$HOME/Work/VelocityTendenciesPipeline"},
  "platforms": {
    "daint":   {"partition": "normal",  "cpus_per_task": 288, "gpus_per_task": 1,
                "extra_setup": ["spack load gcc/76jw6nu", "spack load cuda@12.9"],
                "env": {"GENCODE_NUMBER": "90a", "OMP_NUM_THREADS": "288"}},
    "beverin": {"partition": "mi300",   "cpus_per_task": 192, "gpus_per_task": 1,
                "extra_setup": ["spack load gcc/ktd4slj"],
                "env": {"BEVERIN": "1", "ARCH": "gfx942",
                        "_TBLOCK_DIMS": "32,4,1", "OMP_NUM_THREADS": "96"}}
  },
  "experiments": [
    {"name": "stage6_default_sweep", "stage": 6,
     "configs": ["unpermuted", "nlev_first", "index_only"]},
    {"name": "stage6_full_sweep",    "stage": 6, "configs": ["full_sweep"]}
  ]
}
```

```bash
python tools/generate_submissions.py \
    --config tools/submissions/example.json \
    --out-dir submissions/

# Optional: subset of platforms.
python tools/generate_submissions.py \
    --config tools/submissions/example.json \
    --out-dir submissions/ \
    --platforms daint
```

## Stage 4 vs stage 5 (compression)

Stage 4 owns every "make this run on the GPU at all" concern: GPU
offload (`OffloadVelocityToGPU` = `ToGPU` + `prune` + `KernelLaunchRestructure`),
SDFG-lifetime transients promoted to `Persistent` storage, host-side
chrono timer wrapped around the GPU body, async scalar-return reductions
(`reduce_max_async_host_gpu`) with deferred sync.

Stage 5 is **compression-only**: fold single-valued `*_blk` arrays to
the constant Fortran `1` (no runtime branch -- always-applied because
test grids are single-block), and emit `uint16`/`uint8` compressed
variants for the neighbour-index / multi-valued-blk arrays guarded by
runtime `nproma * nblks_* < 65536` / `nblks_* < 256` checks. Numerics
match stage 4 within FMA-fusion band (~7e-15 relative).

Either stage can be the build target; stage 6 then permutes layouts
on top.

## Multi-variant linking

`utils/compile_if_propagated_sdfgs.py` links the 4 specialised variants
into a single executable via per-variant `ld -r` followed by
`objcopy --localize-symbols=<file>` on dynamically detected colliders
(`__dace_init_cuda`, `__dace_runkernel_*`, etc.). The 4 public
per-SDFG entry points (`__dace_init_<sdfg>`, `__program_<sdfg>`, ...)
stay global, so `main.cpp` can dispatch by name.
