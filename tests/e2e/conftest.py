# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared setup for the e2e lane: every test here is ``e2e``-marked and builds at -O3."""

import os

import dace

# ONE translation unit at a time. These kernels are single, very large TUs (CloudSC ~25k lines),
# and each concurrent compile carries its own multi-GB peak -- together they exhaust this box and
# take the session down, so the lane is worth more serial than parallel. Set the env var too, not
# just the config: a DACE_* variable OUTRANKS Config.set, so a caller that exported build jobs
# would otherwise silently win. setdefault leaves a deliberate override alone.
os.environ.setdefault("DACE_compiler_build_jobs", "1")
dace.Config.set("compiler", "build_jobs", value=1)
