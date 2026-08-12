"""Stage 6: GPU layout-permutation sweep.

Reads ``codegen/stage5/<name>.sdfgz`` and, for each requested
``PermuteConfig``, deepcopies the SDFG, applies the permutation, and saves
``codegen/stage6/<config>/<name>.sdfgz``. Compile mode (``--compile``)
runs a per-config codegen + build, producing
``velocity_stage6_<config>`` binaries.

GPU-only: the permutation transform is built around core-dialect semantics, and
in this pipeline only the post-stage5 GPU SDFGs satisfy it. CPU SDFGs
(top-level map schedule != GPU_Device) are skipped with a notice.

Usage::

    python -m utils.stages.stage6                        # default configs, optimize+compile
    python -m utils.stages.stage6 --optimize
    python -m utils.stages.stage6 --compile
    python -m utils.stages.stage6 --config nlev_first    # single named config
"""

import argparse
import copy
from pathlib import Path
from typing import Dict, List

import dace

from utils.passes.fuse_full_and_endpoint import fuse_full_and_endpoint
from utils.passes.permute_configs import extended_configs_from_candidates
from utils.passes.permute_layout import PermuteConfig, permute_layout
from utils.stages import common


STAGE_ID = 6


def _sdfg_uses_gpu(sdfg: dace.SDFG) -> bool:
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            if node.map.schedule == dace.ScheduleType.GPU_Device:
                return True
    return False


def _array_dim_map(sdfg: dace.SDFG) -> Dict[str, int]:
    return {name: len(arr.shape) for name, arr in sdfg.arrays.items()}


def _select_configs(name_filter: str | None, json_path: Path,
                    sdfg_arrays: Dict[str, int]) -> List[PermuteConfig]:
    configs = extended_configs_from_candidates(json_path, sdfg_arrays=sdfg_arrays)
    if name_filter is None:
        return configs
    selected = [c for c in configs if c.name == name_filter]
    if not selected:
        known = ", ".join(c.name for c in configs)
        raise SystemExit(f"Stage #{STAGE_ID}: no config named '{name_filter}' (known: {known})")
    return selected


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    default_json = (
        Path("/home/primrose/Work/SC26-Layout-AD")
        / "Experiments" / "E6_VelocityTendencies"
        / "access_analysis" / "layout_candidates.json"
    )

    argp = argparse.ArgumentParser()
    argp.add_argument("--optimize", action=argparse.BooleanOptionalAction, default=False)
    argp.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    argp.add_argument("--config", type=str, default=None,
                      help="run only the named config (default: every config from JSON)")
    argp.add_argument("--candidates", type=Path, default=default_json,
                      help="path to layout_candidates.json")
    argp.add_argument("--debug", dest="release", action="store_false", default=True,
                      help="build with -O0 + DACE_VELOCITY_DEBUG (default: release)")
    args = argp.parse_args()
    if not args.optimize and not args.compile:
        args.optimize = args.compile = True

    names = common.sdfg_names()

    if args.optimize:
        for name in names:
            infile = common.stage_input(name, STAGE_ID)
            print(f"Stage #{STAGE_ID}: loading {name} from {infile}")
            sdfg = dace.SDFG.from_file(infile)
            sdfg.name = name

            if not _sdfg_uses_gpu(sdfg):
                print(f"  [skip] {name}: no GPU_Device map; permutations are GPU-only")
                continue

            configs = _select_configs(args.config, args.candidates, _array_dim_map(sdfg))
            for cfg in configs:
                permuted = copy.deepcopy(sdfg)
                permuted.name = name
                count = permute_layout(permuted, cfg, shuffle_map=True)
                print(f"  [{cfg.name}] permuted {count} array(s)")
                # Adjacent (full-range copy, endpoint-constant-write) state
                # pairs (e.g. z_w_con_c written for lev 0..89 then zeroed
                # at lev 90) collapse into a single map with an internal
                # ConditionalBlock dispatch. Pure structural rewrite; runs
                # in-place even when no permutation was applied.
                # n_fused = fuse_full_and_endpoint(permuted)  # disabled while debugging
                # if n_fused:
                #     print(f"  [{cfg.name}] fused {n_fused} (full, endpoint-zero) state pair(s)")

                outfile = (repo_root / common.CODEGEN_DIR / f"stage{STAGE_ID}"
                           / cfg.name / f"{name}.sdfgz")
                outfile.parent.mkdir(parents=True, exist_ok=True)
                permuted.save(str(outfile), compress=True)
                print(f"  [{cfg.name}] saved {outfile.relative_to(repo_root)}")

    if args.compile:
        # One compile_action per (config, variant) — the runner produces
        # ``velocity_stage6_<config>`` binaries side by side. Pass
        # ``sdfg_arrays=None`` so the per-array dim filter is skipped:
        # at this point we only need the config NAMES to look up the
        # already-saved stage6 SDFGs on disk; the actual permutations
        # were applied in the --optimize phase.
        configs = _select_configs(args.config, args.candidates, sdfg_arrays=None)
        for cfg in configs:
            sdfgs: Dict[str, dace.SDFG] = {}
            for name in names:
                stage6_path = (repo_root / common.CODEGEN_DIR / f"stage{STAGE_ID}"
                               / cfg.name / f"{name}.sdfgz")
                if not stage6_path.exists():
                    print(f"  [skip-compile] {cfg.name}/{name}: no SDFG (run --optimize first)")
                    continue
                sdfgs[name] = dace.SDFG.from_file(str(stage6_path))
            if not sdfgs:
                continue
            # Async reductions land in stage 4 and are still referenced by
            # stage 6 binaries -- pass the library sources or the link
            # fails on ``reduce_max_async_host_gpu`` / ``reduce_gpu_init``
            # / ``reduce_gpu_finalize`` / ``reduce_max_store_gpu``.
            common.compile_action(
                STAGE_ID,
                sdfgs,
                gpu=True,
                release=args.release,
                output=f"velocity_stage{STAGE_ID}_{cfg.name}",
                extra_sources=["src/reductions.cpp", "src/reductions_kernel.cu"],
                extra_include_dirs=["include"],
            )


if __name__ == "__main__":
    main()
