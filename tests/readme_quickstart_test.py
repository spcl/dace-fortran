# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Runs the README quick-start blocks verbatim, so the documented snippets cannot rot.

Marker ``long``: each block builds and compiles a whole kernel SDFG (graupel, CloudSC).
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
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        namespace: dict[str, object] = {}
        exec(compile(match.group(1), f"README.md:{tag}", "exec"), namespace)
    finally:
        os.chdir(cwd)
    assert isinstance(namespace["sdfg"], dace.SDFG)
    return namespace


@pytest.mark.long
def test_readme_quickstart_runs() -> None:
    assert callable(_run_block("quickstart")["graupel_run"])


@pytest.mark.long
def test_readme_graupel_optimize_runs() -> None:
    assert callable(_run_block("optimize-graupel")["graupel_run"])


@pytest.mark.long
def test_readme_cloudsc_optimize_runs() -> None:
    assert callable(_run_block("optimize-cloudsc")["cloudsc"])


if __name__ == "__main__":
    test_readme_quickstart_runs()
    test_readme_graupel_optimize_runs()
    test_readme_cloudsc_optimize_runs()
