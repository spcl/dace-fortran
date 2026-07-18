"""Fortran intrinsic -> DaCe lowering registry for the HLFIR frontend.

Emitter code should only talk to this module, never import the per-family
registries directly -- adding an intrinsic means editing one file under this
package.  Families: elementwise.py (sin/cos/exp/...), reduction.py
(sum/product/minval/maxval), linalg.py (matmul/transpose/dot_product),
direct.py (SIZE/LBOUND/... stub).
"""

from dace_fortran.intrinsics.elementwise import ELEMENTWISE_INTRINSICS
from dace_fortran.intrinsics.reduction import REDUCTIONS
from dace_fortran.intrinsics.linalg import LINALG, STANDARD
from dace_fortran.intrinsics.direct import DIRECT_INTRINSICS


def is_elementwise(name: str) -> bool:
    """True if ``name`` is an elementwise Fortran intrinsic."""
    return name in ELEMENTWISE_INTRINSICS


def is_reduction(name: str) -> bool:
    """True if ``name`` is a reduction Fortran intrinsic."""
    return name in REDUCTIONS


def is_libnode(name: str) -> bool:
    """True if ``name`` lowers to a DaCe library node (linalg or standard)."""
    return name in LINALG or name in STANDARD


def is_intrinsic(name: str) -> bool:
    """True if ``name`` is a known Fortran intrinsic in any family."""
    return (is_elementwise(name) or is_reduction(name) or is_libnode(name) or name in DIRECT_INTRINSICS)


def render_call(name: str, args: list[str]) -> str:
    """Return ``name(arg0, arg1, ...)`` verbatim.  Only validates elementwise
    arity today; reduction/libnode callers get their own render helpers later."""
    spec = ELEMENTWISE_INTRINSICS.get(name)
    if spec is not None:
        assert len(args) == spec.arity, (f"{name} expects {spec.arity} arg(s), got {len(args)}")
    return f"{name}({', '.join(args)})"


def reduction_spec(name: str):
    """Return the ``ReductionIntrinsic`` for ``name`` or ``None``."""
    return REDUCTIONS.get(name)


def libnode_spec(name: str):
    """Return the ``LibNodeIntrinsic`` for ``name`` or ``None``.  Looks up both
    the linalg registry (matmul/transpose/dot_product) and the standard registry (count/merge/...)."""
    spec = LINALG.get(name)
    if spec is not None:
        return spec
    return STANDARD.get(name)
