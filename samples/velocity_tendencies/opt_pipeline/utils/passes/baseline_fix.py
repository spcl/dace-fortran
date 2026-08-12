"""Baseline-load sanitiser for main-DaCe (yakup/dev) stages.

The f2dace-branch ``StructToContainerGroups`` + our own baseline passes leave
the config symbols ``istep`` / ``lvn_only`` registered as both a ``Scalar`` in
``sdfg.arrays`` and a symbol in ``sdfg.symbols``. FaCe tolerated this;
yakup/dev's SDFG validator rejects duplicate names across ``symbols`` /
``arrays`` / subarrays / rdistarrays.

Rather than mutate the f2dace-side baseline pipeline (where we still run
``StructToContainerGroups``), we strip the stale ``Scalar`` entries when the
baseline is loaded into a main-DaCe stage. Idempotent: calling it on an
already-clean SDFG is a no-op.
"""

import dace


_DEAD_SYMBOL_NAMES = ("istep", "lvn_only")


def fix_baseline(sdfg: dace.SDFG) -> int:
    """Drop ``istep`` / ``lvn_only`` from ``sdfg.arrays`` when they also live
    in ``sdfg.symbols``. Returns the number of descriptors removed."""
    removed = 0
    for name in _DEAD_SYMBOL_NAMES:
        if name in sdfg.arrays and name in sdfg.symbols:
            try:
                sdfg.remove_data(name, validate=False)
            except Exception:
                del sdfg.arrays[name]
            removed += 1
    return removed
