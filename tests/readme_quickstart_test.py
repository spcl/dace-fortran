# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Runs the README quick-start block verbatim, so the documented snippet cannot rot.

Marker ``long``: the block builds and compiles the whole AES graupel SDFG.
"""

import os
import re
from pathlib import Path

import dace
import pytest

REPO = Path(__file__).resolve().parent.parent
_BLOCK = re.compile(r"<!-- quickstart:begin -->\s*```python\n(.*?)```\s*<!-- quickstart:end -->", re.DOTALL)


def _quickstart_code() -> str:
    match = _BLOCK.search((REPO / "README.md").read_text())
    assert match is not None, "README.md lost its quickstart:begin/end block"
    return match.group(1)


@pytest.mark.long
def test_readme_quickstart_runs() -> None:
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        namespace: dict[str, object] = {}
        exec(compile(_quickstart_code(), "README.md:quickstart", "exec"), namespace)
    finally:
        os.chdir(cwd)
    assert isinstance(namespace["sdfg"], dace.SDFG)
    assert callable(namespace["graupel_run"])


if __name__ == "__main__":
    test_readme_quickstart_runs()
