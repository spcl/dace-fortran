# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared setup for the e2e lane: every test here is ``e2e``-marked and builds at -O3."""
import pytest

import dace

# -march=native so the vectorizer sees the real ISA; -fno-math-errno + -fno-trapping-math so libm
# calls (sin/cos) vectorize through libmvec. Deliberately NOT -ffast-math: that licenses FP
# reassociation, which breaks the bit-exact comparison against gfortran. -ffp-contract=off pins
# FMA formation off for the same reason.
E2E_CPU_ARGS = ('-fPIC -O3 -march=native -fno-fast-math -ffp-contract=off -fno-math-errno '
                '-fno-trapping-math -Wno-unused-parameter -Wno-unused-label')


@pytest.fixture
def e2e_cpu_args():
    """Build the SDFG with the paper's flags; restore whatever was set before."""
    prev = dace.Config.get('compiler', 'cpu', 'args')
    dace.Config.set('compiler', 'cpu', 'args', value=E2E_CPU_ARGS)
    yield E2E_CPU_ARGS
    dace.Config.set('compiler', 'cpu', 'args', value=prev)
