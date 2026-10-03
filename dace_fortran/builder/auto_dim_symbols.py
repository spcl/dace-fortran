# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Auto-resolve the bridge's synthetic Fortran extent symbols at call time.

Bindings-emitted callers always pass correct values; direct ``sdfg(...)``
calls (test suite) need ``<arr>_d<i>``/``offset_<arr>_d<i>`` filled in -- from
the passed array's shape; an extent no argument supplies raises (never defaulted).  SDFG
signature itself is unchanged.
"""

from __future__ import annotations
import re
from typing import Protocol, runtime_checkable

import dace
from typing import Any


@runtime_checkable
class HasShape(Protocol):
    """Any array-like call argument (numpy, cupy, torch, ...)."""

    @property
    def shape(self) -> tuple[int, ...]:
        ...


#: ``<arr>_d<i>`` / ``offset_<arr>_d<i>`` synthetic-extent symbol name; greedy
#: ``.+`` matches the rightmost ``_d<i>`` since an array name may itself contain ``_d``.
_DIM_SYMBOL_RE = re.compile(r'^(?P<off>offset_)?(?P<arr>.+)_d(?P<idx>\d+)$')


class AutoDimSDFG(dace.SDFG):
    """``SDFG`` that fills missing synthetic Fortran extent symbols from
    the passed array arguments before the real call; an extent no
    argument supplies raises instead of defaulting.

    The annotated attributes are the sidecars ``SDFGBuilder.build`` stashes for the binding emitter."""

    _fortran_offset_values: dict[str, int]
    _fortran_interface_raw: Any
    _flatten_plan_raw: Any

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        for sym in (str(s) for s in self.free_symbols):
            if sym in kwargs:
                continue
            m = _DIM_SYMBOL_RE.match(sym)
            if m is None:
                continue
            is_offset = m.group('off') is not None
            actual = kwargs.get(m.group('arr'))
            shape = actual.shape if isinstance(actual, HasShape) else None
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
                raise ValueError(f"extent symbol {sym!r} is unbound: no argument {m.group('arr')!r} supplies its extent "
                                 f"(pass {sym}=<extent> explicitly); refusing to default it")
        return super().__call__(*args, **kwargs)

    def to_json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Serialise as plain ``SDFG`` so strict-type ``from_json`` (e.g.
        ``distributed_compile`` reloading a gzipped ``program.sdfgz`` per rank)
        accepts the dump; the auto-fill ``__call__`` wrapper isn't persisted
        state, so nothing is lost."""
        d = super().to_json(*args, **kwargs)
        d['type'] = dace.SDFG.__name__
        return d
