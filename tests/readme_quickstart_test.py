# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Runs the README quick-start blocks verbatim, so the documented snippets cannot rot.

The optimize blocks carry their own numerical checks (pre- vs post-optimization bit for bit, and against
gfortran); a failing assert inside a block fails its test. They are ``e2e`` tests, the quick start is ``long``: each
block builds and compiles a whole kernel SDFG (graupel, CloudSC).
"""

import os
import re
from pathlib import Path

import dace
import pytest

REPO = Path(__file__).resolve().parent.parent


def _run_block(tag: str) -> dict[str, object]:
    """Execute the README's ``<!-- tag:begin -->`` python block from the repository root; return its namespace."""
    block = re.compile(rf"<!-- {tag}:begin -->\s*```python\n(.*?)```\s*<!-- {tag}:end -->", re.DOTALL)
    match = block.search((REPO / "README.md").read_text())
    assert match is not None, f"README.md lost its {tag}:begin/end block"
    cwd, cpu_args = os.getcwd(), dace.Config.get("compiler", "cpu", "args")
    os.chdir(REPO)
    try:
        namespace: dict[str, object] = {}
        exec(compile(match.group(1), f"README.md:{tag}", "exec"), namespace)
    finally:
        os.chdir(cwd)
        dace.Config.set("compiler", "cpu", "args", value=cpu_args)  # the optimize blocks set their own
    assert isinstance(namespace["sdfg"], dace.SDFG)
    return namespace


@pytest.mark.long
def test_readme_quickstart_runs() -> None:
    assert callable(_run_block("quickstart")["graupel_run"])


@pytest.mark.e2e
def test_readme_graupel_optimize_matches_gfortran() -> None:
    _run_block("optimize-graupel")


@pytest.mark.e2e
def test_readme_cloudsc_optimize_matches_gfortran() -> None:
    _run_block("optimize-cloudsc")


if __name__ == "__main__":
    test_readme_quickstart_runs()
    test_readme_graupel_optimize_matches_gfortran()
    test_readme_cloudsc_optimize_matches_gfortran()
