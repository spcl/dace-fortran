"""Data-class shapes shared by every intrinsic sub-registry.

Each registry file populates a dict of these dataclasses so the public
helpers in ``__init__.py`` stay family-agnostic.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ElementwiseIntrinsic:
    """Fortran intrinsic lowered to a per-element scalar call inside an
    ``hlfir.elemental`` body; name is used verbatim in tasklet code, resolved
    via ``_ALLOWED_MODULES`` (dace/dtypes.py) to ``dace/runtime/include/dace/math.h``."""

    name: str
    arity: int


@dataclass(frozen=True)
class ReductionIntrinsic:
    """Whole-array reduction that becomes a ``standard.Reduce`` library
    node via ``state.add_reduce(wcr, axes, identity)``  --  populated by
    ``reduction.py``, consumed by ``builder/emit_library.py``."""

    name: str
    wcr: str
    identity: str


@dataclass(frozen=True)
class LibNodeIntrinsic:
    """Intrinsic that becomes a direct DaCe library-node emission
    (``blas.Matmul``, ``linalg.Transpose``, ``blas.Dot``, ``fft.FFT``) --
    populated by ``linalg.py``, consumed by ``builder/emit_library.py``."""

    name: str
    module: str  # e.g. "blas", "standard", "fft"
    node_cls: str  # e.g. "Matmul", "Transpose", "Dot"
