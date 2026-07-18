"""Auto-resolve the bridge's synthetic Fortran extent symbols at call time.

Bindings-emitted callers always pass correct values; direct ``sdfg(...)``
calls (test suite) need ``<arr>_d<i>``/``offset_<arr>_d<i>`` filled in -- from
the passed array's shape when available, else a don't-care default.  SDFG
signature itself is unchanged.
"""
import re

import dace

#: ``<arr>_d<i>`` / ``offset_<arr>_d<i>`` synthetic-extent symbol name; greedy
#: ``.+`` matches the rightmost ``_d<i>`` since an array name may itself contain ``_d``.
_DIM_SYMBOL_RE = re.compile(r'^(?P<off>offset_)?(?P<arr>.+)_d(?P<idx>\d+)$')


class _AutoDimSDFG(dace.SDFG):
    """``SDFG`` that fills missing synthetic Fortran extent symbols from
    the passed array arguments (or a don't-care default) before the
    real call."""

    def __call__(self, *args, **kwargs):
        for sym in (str(s) for s in self.free_symbols):
            if sym in kwargs:
                continue
            m = _DIM_SYMBOL_RE.match(sym)
            if m is None:
                continue
            is_offset = m.group('off') is not None
            actual = kwargs.get(m.group('arr'))
            shape = getattr(actual, 'shape', None)
            idx = int(m.group('idx'))
            if not is_offset and shape is not None and idx < len(shape):
                kwargs[sym] = int(shape[idx])  # always the correct extent
            elif is_offset:
                # Defaults to Fortran's 1-based lower bound (access lowers to
                # ``arr[idx - offset]``); non-default bounds (e.g. ICON's
                # ``end_block(min_rl:)``) are passed explicitly by the bindings
                # emitter via ``lbound``.  Previously defaulted to 0 -- an
                # off-by-one read of every such array.
                kwargs[sym] = 1
            else:
                kwargs[sym] = 1  # unused extent: don't care
        return super().__call__(*args, **kwargs)

    def to_json(self, *args, **kwargs):
        """Serialise as plain ``SDFG`` so strict-type ``from_json`` (e.g.
        ``distributed_compile`` reloading a gzipped ``program.sdfgz`` per rank)
        accepts the dump; the auto-fill ``__call__`` wrapper isn't persisted
        state, so nothing is lost."""
        d = super().to_json(*args, **kwargs)
        d['type'] = dace.SDFG.__name__
        return d


def install_auto_dim_symbols(sdfg: dace.SDFG) -> dace.SDFG:
    """Rebind ``sdfg`` so direct calls auto-resolve synthetic extents."""
    sdfg.__class__ = _AutoDimSDFG
    return sdfg
