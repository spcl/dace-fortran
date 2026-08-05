# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared setup for the e2e lane: every test here is ``e2e``-marked and builds at -O3."""
import pytest

import dace

from _util import BITEXACT_CPU_ARGS


@pytest.fixture
def e2e_cpu_args():
    """Build the SDFG with the paper's flags; restore whatever was set before."""
    prev = dace.Config.get('compiler', 'cpu', 'args')
    dace.Config.set('compiler', 'cpu', 'args', value=BITEXACT_CPU_ARGS)
    yield BITEXACT_CPU_ARGS
    dace.Config.set('compiler', 'cpu', 'args', value=prev)
