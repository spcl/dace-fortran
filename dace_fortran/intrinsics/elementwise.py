"""Elementwise Fortran intrinsics.

Each entry lowers to a bare-name scalar call in the tasklet body (``_out =
sin(_in_a)``); DaCe's codegen maps it through ``_ALLOWED_MODULES``
(dace/dtypes.py) to ``dace::math::...``, so no ``math.`` prefix or language
switch is needed.  The SDFG emitter consults ``is_elementwise`` to keep the
name bare instead of rewriting it to an ``_in_sin`` connector.
"""

from dace_fortran.intrinsics.base import ElementwiseIntrinsic


def _one(name: str, arity: int = 1) -> tuple[str, ElementwiseIntrinsic]:
    return name, ElementwiseIntrinsic(name=name, arity=arity)


ELEMENTWISE_INTRINSICS: dict[str, ElementwiseIntrinsic] = dict([
    # Transcendentals
    _one('sin'),
    _one('cos'),
    _one('tan'),
    _one('asin'),
    _one('acos'),
    _one('atan'),
    _one('sinh'),
    _one('cosh'),
    _one('tanh'),
    _one('exp'),
    _one('log'),
    _one('log10'),
    _one('sqrt'),
    # Rounding / sign
    _one('abs'),
    _one('floor'),
    _one('ceil'),
    # Special functions
    _one('erf'),
    _one('erfc'),
    # Two-arg
    _one('min', arity=2),
    _one('max', arity=2),
    _one('pow', arity=2),
    _one('atan2', arity=2),
])
