"""Whole-array scalar reductions -> DaCe ``standard.Reduce``.

``sum``/``product``/``minval``/``maxval`` each lower through Flang into a
dedicated HLFIR op whose result is a scalar or (with ``dim=``) a reduced-rank
array.  The bridge's extract_ast emits ``kind="reduce"`` carrying the
parameters below; hlfir_to_sdfg calls ``state.add_reduce(wcr, axes, identity)``.
"""

from dace_fortran.intrinsics.base import ReductionIntrinsic

REDUCTIONS: dict[str, ReductionIntrinsic] = {
    'sum':
    ReductionIntrinsic(name='sum', wcr='lambda a, b: a + b', identity='0'),
    'product':
    ReductionIntrinsic(name='product', wcr='lambda a, b: a * b', identity='1'),
    'minval':
    ReductionIntrinsic(
        name='minval',
        wcr='lambda a, b: min(a, b)',
        # +inf identity so the first real element always wins (resolved via _ALLOWED_MODULES).
        identity='math.inf'),
    'maxval':
    ReductionIntrinsic(name='maxval', wcr='lambda a, b: max(a, b)', identity='-math.inf'),
}
