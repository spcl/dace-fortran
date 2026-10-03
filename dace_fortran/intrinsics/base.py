# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The record type of the library-node intrinsic registries in ``linalg.py``."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LibNodeIntrinsic:
    """Intrinsic that becomes a direct DaCe library-node emission
    (``blas.Matmul``, ``linalg.Transpose``, ``blas.Dot``, ``fft.FFT``) --
    populated by ``linalg.py``, consumed by ``builder/emit_library.py``."""

    name: str
    module: str  # e.g. "blas", "standard", "fft"
    node_cls: str  # e.g. "Matmul", "Transpose", "Dot"
