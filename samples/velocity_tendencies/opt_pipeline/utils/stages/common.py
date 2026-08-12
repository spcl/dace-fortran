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
    )
