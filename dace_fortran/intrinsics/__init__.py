# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fortran intrinsic -> DaCe lowering registry for the HLFIR frontend.

Emitter code should only talk to this module, never import the per-family
registries directly -- adding an intrinsic means editing one file under this
package.  Families: elementwise.py (sin/cos/exp/...), reduction.py
(sum/product/minval/maxval), linalg.py (matmul/transpose/dot_product).
"""

from dace_fortran.intrinsics.elementwise import ELEMENTWISE_INTRINSICS
from dace_fortran.intrinsics.reduction import REDUCTIONS
from dace_fortran.intrinsics.base import LibNodeIntrinsic
from dace_fortran.intrinsics.linalg import LINALG, STANDARD


def is_elementwise(name: str) -> bool:
    """True if ``name`` is an elementwise Fortran intrinsic."""
    return name in ELEMENTWISE_INTRINSICS


def is_reduction(name: str) -> bool:
    """True if ``name`` is a reduction Fortran intrinsic."""
    return name in REDUCTIONS


def is_libnode(name: str) -> bool:
    """True if ``name`` lowers to a DaCe library node (linalg or standard)."""
    return name in LINALG or name in STANDARD


def libnode_spec(name: str) -> LibNodeIntrinsic | None:
    """Return the ``LibNodeIntrinsic`` for ``name`` or ``None``.  Looks up both
    the linalg registry (matmul/transpose/dot_product) and the standard registry (count/merge/...)."""
    spec = LINALG.get(name)
    if spec is not None:
        return spec
    return STANDARD.get(name)
