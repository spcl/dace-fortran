# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The CPU velocity-tendencies pipeline, as an e2e harness hook (``OCEAN_E2E_SDFG_HOOK``).

The harness runs it on the freshly built SDFG in its child process: bake the configuration in ``VT_CONFIG``, drop
the branches it decides, parallelize, and re-snapshot the Fortran signature the binding is generated from.
"""

import json
import os

from dace.sdfg import SDFG

from dace_fortran.bindings.frozen_signature import refreeze
from dace_fortran.config_propagation import specialize_variant
from dace_fortran.pipelines import num_loops, num_maps, optimize


def run(sdfg: SDFG) -> None:
    values = json.loads(os.environ["VT_CONFIG"])
    specialize_variant(sdfg, values)
    optimize(sdfg)
    print(f"VT_PIPELINE maps={num_maps(sdfg)} loops={num_loops(sdfg)}", flush=True)
    refreeze(sdfg)
