# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bind the OpenMP thread count to a symbol for thread-strided ``CPU_Persistent`` maps.

A persistent map strides its work over the team (``for i in tid:N:__omp_num_threads``), and the code
generator defines ``__omp_num_threads`` inside the parallel region. Outside it, the same name is a free
symbol: a per-thread buffer, or the map's own range, would need the caller to pass the thread count in,
which would put the OpenMP configuration into the kernel's call signature.

:class:`BindOmpThreadCount` instead defines it at SDFG entry. A new first state holds a C++ tasklet with side
effects that reads ``omp_get_max_threads()`` into a transient scalar, and the interstate edge leaving that
state assigns the scalar to the symbol.
"""
from typing import Optional

import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible

#: The name the code generator already gives the team size inside a ``CPU_Persistent`` scope.
OMP_NUM_THREADS_SYMBOL = '__omp_num_threads'


@explicit_cf_compatible
class BindOmpThreadCount(ppl.Pass):  # type: ignore[type-var]  # dace types the decorator as taking an instance
    """Define the free symbol ``symbol`` as ``omp_get_max_threads()`` at the entry of the SDFG.

    Does nothing when the SDFG does not use the symbol, or already defines it on an interstate edge.
    """

    def __init__(self, symbol: str = OMP_NUM_THREADS_SYMBOL):
        super().__init__()
        self.symbol = symbol

    def modifies(self) -> ppl.Modifies:
        """Adds a state, an interstate edge, a scalar and a symbol."""
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """One-shot: the symbol is bound once and is no longer free."""
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[str]:
        """Bind ``self.symbol``. Returns the name of the scalar holding the thread count, or ``None`` if unchanged."""
        if self.symbol not in {str(s) for s in sdfg.free_symbols}:
            return None

        scalar, _ = sdfg.add_scalar('__omp_max_threads_value', dace.int32, transient=True, find_new_name=True)
        first = sdfg.add_state_before(sdfg.start_block,
                                      label='bind_omp_thread_count',
                                      is_start_block=True,
                                      assignments={self.symbol: scalar})
        tasklet = first.add_tasklet('omp_max_threads', {}, {'__out'},
                                    '__out = omp_get_max_threads();',
                                    language=dtypes.Language.CPP,
                                    code_global='#include <omp.h>',
                                    side_effects=True)
        first.add_edge(tasklet, '__out', first.add_write(scalar), None, dace.Memlet(f'{scalar}[0]'))

        sdfg.symbols[self.symbol] = dace.int32
        return scalar
