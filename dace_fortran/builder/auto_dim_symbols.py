"""Auto-resolve the bridge's synthetic Fortran extent symbols at call time.

Every Fortran array dimension is a *required* SDFG input under the
current DaCe arglist methodology -- the bridge synthesises one
``<arr>_d<i>`` symbol per extent (and ``offset_<arr>_d<i>`` per lower
bound).  Real callers go through the emitted Fortran bindings, which
always pass the correct ``size(...)``.  Direct ``sdfg(...)`` calls
(the bridge test suite) would otherwise have to spell out every
synthetic symbol by hand.

This installs a thin ``SDFG`` subclass whose ``__call__`` fills any
required extent symbol the caller left out:

* the symbol names an array that *was* passed -> its real extent
  (``passed_array.shape[i]``) so the value is always correct;
* otherwise the descriptor is unused (a never-``ALLOCATE``-d
  allocatable, a pointer repointed before any read, ...) -- the value
  is irrelevant, so a harmless ``1`` (``0`` for an offset) is supplied.

The SDFG signature is left untouched: every dimension symbol stays a
program input, exactly as the methodology requires.
"""
import re

import dace

#: ``<arr>_d<i>`` / ``offset_<arr>_d<i>`` synthetic-extent symbol name.
#: The greedy ``.+`` splits at the *rightmost* ``_d<i>`` -- the
#: synthetic suffix is always last and an array name may itself
#: contain ``_d`` (e.g. ``grid_d``).
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
            else:
                kwargs[sym] = 0 if is_offset else 1  # unused: don't care
        return super().__call__(*args, **kwargs)


def install_auto_dim_symbols(sdfg: dace.SDFG) -> dace.SDFG:
    """Rebind ``sdfg`` so direct calls auto-resolve synthetic extents.

    :param sdfg: the freshly built kernel SDFG.
    :returns: the same instance, retyped to :class:`_AutoDimSDFG`.
    """
    sdfg.__class__ = _AutoDimSDFG
    return sdfg
