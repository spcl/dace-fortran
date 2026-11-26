# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""ICON velocity tendencies through the CPU pipeline: ``ConfigPropagation`` bakes the namelist configuration into one
external SDFG per combination, and each variant is parallelized and checked bit for bit against the gfortran build of
the same source.
"""

import json
import os
from pathlib import Path

import dace
import pytest
from dace.sdfg.state import ConditionalBlock

from dace_fortran.build import build_sdfg
from dace_fortran.config_propagation import ConfigPropagation
from tests.icon.atmosphere._icon_mesh import velocity_patch
from tests.icon.ocean._ocean_e2e import _REFIN_CTRL_EXTENT, run_kernel_e2e

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]
VELOCITY_SOURCE = ROOT / "tests" / "icon" / "atmosphere" / "velocity_advection_inlined_single_tu.f90"
ENTRY = "mo_velocity_advection::velocity_tendencies"
DOMAINS = {"lvn_only": [0, 1], "istep": [1, 2], "ldeepatmo": [0], "lextra_diffu": [1]}
CONFIG_NAMES = tuple(DOMAINS)
#: Maps after parallelization per (lvn_only, istep); the unspecialized kernel has 39. Fewer maps is what
#: pruning the configuration branches buys, so a change here is a change in what the pipeline removes.
MAPS = {(0, 1): 34, (0, 2): 26, (1, 1): 27, (1, 2): 19}


def branch_conditions(sdfg: dace.SDFG) -> list[str]:
    return [
        cond.as_string
        for block in sdfg.all_control_flow_blocks(recursive=True)
        if isinstance(block, ConditionalBlock)
        for cond, _ in block.branches
        if cond is not None
    ]


def test_config_propagation_prunes_every_config_branch(tmp_path):
    sdfg = build_sdfg(VELOCITY_SOURCE.read_text(), entry=ENTRY, name="velocity_tendencies", out_dir=str(tmp_path))
    before = len(branch_conditions(sdfg))
    out = tmp_path / "variants"
    root = ConfigPropagation(DOMAINS, str(out)).apply_pass(sdfg, {})

    dispatch = [b for b in root.nodes() if isinstance(b, ConditionalBlock)]
    # One branch per variant, then the generic fallback for values outside the domains
    assert len(dispatch) == 1 and len(dispatch[0].branches) == len(MAPS) + 1
    variants = sorted(
        f"velocity_tendencies_lvn_only_{lvn}_istep_{step}_ldeepatmo_0_lextra_diffu_1.sdfgz" for lvn, step in MAPS
    )
    assert sorted(p.name for p in out.iterdir()) == sorted([*variants, "velocity_tendencies_generic.sdfgz"])
    for name in variants:
        variant = dace.SDFG.from_file(str(out / name))
        conditions = branch_conditions(variant)
        assert not [c for c in conditions if any(n in c for n in CONFIG_NAMES)], name
        assert len(conditions) < before, name


@pytest.mark.parametrize("lvn_only, istep", sorted(MAPS))
def test_cpu_pipeline_matches_gfortran(lvn_only, istep, monkeypatch):
    # The harness child imports this hook by name
    paths = [str(ROOT), os.environ.get("PYTHONPATH", "")]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(p for p in paths if p))
    monkeypatch.setenv("OCEAN_E2E_SDFG_HOOK", "tests.e2e.velocity_config_hook:run")
    monkeypatch.setenv(
        "VT_CONFIG", json.dumps({"lvn_only": lvn_only, "istep": istep, "ldeepatmo": 0, "lextra_diffu": 1})
    )
    # A random 8 x 8 block patch with locality-biased connectivity, 7 full levels
    patch = velocity_patch(refin_extent=_REFIN_CTRL_EXTENT)
    # nflatlev is a DO lower bound, so its BSS 0 would start jk at 0, out of bounds
    patch["module_seeds"] |= {"nflatlev": 1, "nrdmax": 1, "lextra_diffu": 1}
    result = run_kernel_e2e(
        VELOCITY_SOURCE,
        ENTRY,
        seed=1001,
        scalar_overrides={"lvn_only": lvn_only, "istep": istep, "ldeepatmo": 0},
        **patch,
    )
    assert result["passed"], result["output"][-3500:]
    assert f"VT_PIPELINE maps={MAPS[lvn_only, istep]} loops=0" in result["output"]
    assert result["n_changed"] > 0, "no output changed: the kernel did no work"
    assert result["max_diff"] == 0.0, f"differs from gfortran, max |d| = {result['max_diff']:.3e}"
