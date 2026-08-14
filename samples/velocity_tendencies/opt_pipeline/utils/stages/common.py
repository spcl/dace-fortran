"""Shared helpers for the staged pipeline (stage 1 onwards).

Mirrors the ``--optimize`` / ``--compile`` structure from
``icon-artifacts/velocity/utils/stages/common.py`` but adapted to the new
repo's layout:

- Stage 0 is the baseline produced by ``generate_baselines.py`` -- the 4
  variant SDFGs live in ``baseline/<name>.sdfgz``.
- Stage N (N >= 1) reads from ``codegen/stage<N-1>/<name>.sdfgz`` (or
  baseline for N==1) and writes to ``codegen/stage<N>/<name>.sdfgz``.
- Compilation delegates to ``utils.compile_if_propagated_sdfgs`` with the
  per-stage build folder namespaced under ``codegen/stage<N>/<name>/``.
"""

import os
from pathlib import Path
from typing import Dict

import dace

from utils.compile_if_propagated_sdfgs import compile_if_propagated_sdfgs


_VARIANT_TEMPLATE = "velocity_no_nproma_if_prop_lvn_only_{lvn}_istep_{istep}"
VARIANT_NAMES = tuple(
    _VARIANT_TEMPLATE.format(lvn=lvn, istep=istep)
    for lvn in (0, 1)
    for istep in (1, 2)
)

BASELINE_DIR = "baseline"
CODEGEN_DIR = "codegen"

# The split-4 dispatch wrapper. One init/run/finalize interface in front of all 4 variants, so the
# runner here and ICON later never branch on (lvn_only, istep) themselves. It is a real translation
# unit: serde's 53 namespace-scope functions carry `inline`, so the ARRAY_META_DICT_AT lookups
# VELOCITY_CALL_ARGS expands to can appear in a second TU without a duplicate-symbol link error,
# and the metadata map stays shared rather than being duplicated per TU.
DISPATCH_SOURCES = ["src/velocity_split_dispatch.cpp"]

# Zeroed body-timer globals for the pre-GPU stages; stage 4's _insert_body_timer defines them for real.
BODY_TIMER_STUB = "src/velocity_body_timer_stub.cpp"

# The original single-TU OpenACC Fortran velocity_tendencies plus the serde bridge, in module
# dependency order (nvfortran compiles these serially, so the order is load-bearing). Enabled with
# VT_WITH_ACC=1; it is what the DaCe variants are timed against.
ACC_SOURCES = [
    "../velocity_advection_acc.f90",
    "src/velocity_acc_mirror.f90",
    "src/velocity_acc_bridge.f90",
]

ACC_NOLOOPEXCH_SOURCES = [
    "../velocity_advection_noloopexch_acc.f90",
    "src/velocity_acc_mirror.f90",
    "src/velocity_acc_bridge.f90",
]


def engine_sources() -> tuple:
    """(extra sources, extra defines) for the dispatch wrapper and the optional ACC reference.

    The two ACC translation units define the same module and subroutine, so exactly one can be
    linked; VT_ACC_TU selects which.
    """
    sources = list(DISPATCH_SOURCES)
    defines: list = []
    if os.getenv("VT_WITH_ACC", "0").lower() in ("1", "true", "yes"):
        # DaCe is always loop-exchange (generate_baselines.py DEFAULT_TU, and the ICON lib's
        # harvested -D set), so the Fortran reference must be too -- on CPU and GPU alike.
        tu = os.getenv("VT_ACC_TU", "loopexch")
        if tu not in ("loopexch", "noloopexch"):
            raise SystemExit(f"VT_ACC_TU must be loopexch or noloopexch (got {tu!r})")
        sources += ACC_SOURCES if tu == "loopexch" else ACC_NOLOOPEXCH_SOURCES
        defines.append("VT_WITH_ACC")
    return sources, defines


def sdfg_names() -> list:
    return list(VARIANT_NAMES)


def stage_input(
    name: str,
    stage: int,
    baseline_dir: str = BASELINE_DIR,
    codegen_dir: str = CODEGEN_DIR,
) -> str:
    """Stage ``N`` reads from stage ``N-1``'s output; stage 1 reads baseline."""
    if stage <= 1:
        return f"{baseline_dir}/{name}.sdfgz"
    return f"{codegen_dir}/stage{stage - 1}/{name}.sdfgz"


def stage_output(name: str, stage: int, codegen_dir: str = CODEGEN_DIR) -> str:
    return f"{codegen_dir}/stage{stage}/{name}.sdfgz"


def compile_action(
    stage: int,
    sdfgs: Dict[str, dace.SDFG],
    main_name: str = "main.cpp",
    gpu: bool = False,
    release: bool = True,
    debuginfo: bool = True,
    output: str = None,
    extra_sources=None,
    extra_include_dirs=None,
    extra_defines=None,
):
    """Codegen + compile + link the staged SDFGs into ``velocity_stage<N>``."""
    repo = Path(__file__).resolve().parent.parent.parent
    codegen_root = repo / CODEGEN_DIR / f"stage{stage}"
    for name, sdfg in sdfgs.items():
        sdfg.build_folder = str(codegen_root / name)

    if output is None:
        output = f"velocity_stage{stage}"

    compile_if_propagated_sdfgs(
        list(sdfgs.values()),
        gpu=gpu,
        release=release,
        generate_code=True,
        lib=False,
        main_name=main_name,
        stage=stage,
        debuginfo=debuginfo,
        output=output,
        extra_sources=extra_sources,
        extra_include_dirs=extra_include_dirs,
        extra_defines=extra_defines,
    )
