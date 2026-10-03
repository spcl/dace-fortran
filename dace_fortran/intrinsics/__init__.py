# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fortran intrinsics that lower to DaCe library nodes (matmul / transpose / dot_product, count / merge / ...).

Elementwise intrinsics and reductions are rendered by the bridge itself.
"""

from dace_fortran.intrinsics.base import LibNodeIntrinsic
from dace_fortran.intrinsics.linalg import LINALG, STANDARD


def libnode_spec(name: str) -> LibNodeIntrinsic | None:
    """Return the ``LibNodeIntrinsic`` for ``name`` or ``None``.  Looks up both
    the linalg registry (matmul/transpose/dot_product) and the standard registry (count/merge/...)."""
    spec = LINALG.get(name)
    if spec is not None:
        return spec
    return STANDARD.get(name)
