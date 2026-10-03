# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Isolate the build-system integration tests from the bridge's build-dir variable."""

import pytest


@pytest.fixture(autouse=True)
def isolate_build_dir_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """``DACE_FORTRAN_BUILD_DIR`` names the bridge's CMake build tree for ``build_bridge``, and ``dace_fortran.m4`` / ``dace_fortran.mk`` read the same name as the preprocess OUTPUT directory.  A developer environment that sets it for the bridge would redirect the generated ``*.preprocessed.f90`` away from the project tree the tests inspect, so the tests run without it."""
    monkeypatch.delenv("DACE_FORTRAN_BUILD_DIR", raising=False)
