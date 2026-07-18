"""Drop the Fortran offset / dimension symbols of genuine scalars.

The HLFIR bridge synthesises ``<name>_d<i>``/``offset_<name>_d<i>`` symbols
per array dimension/lower bound.  When ``<name>`` is (or becomes) a true
``Scalar``, those symbols are spurious and a stray free one raises
``KeyError: Missing program argument`` at call time -- so they're removed.
Genuine array dimension symbols are untouched (every extent stays a required
SDFG input).  Keeps any symbol still referenced by some array's shape/offset;
recurses into nested SDFGs.
"""
import re

import dace
from dace import properties
from dace.transformation import pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class RemoveScalarFortranShapeSymbols(ppl.Pass):
    """Remove the bridge's ``<scalar>_d<i>``/``offset_<scalar>_d<i>`` symbols
    (a ``Scalar`` has neither shape nor offset)."""

    recursive = properties.Property(dtype=bool, default=True, desc="Recurse into nested SDFGs.")

    def __init__(self, recursive: bool = True):
        super().__init__()
        self.recursive = recursive

    def modifies(self) -> ppl.Modifies:
        """This pass only mutates the SDFG symbol table."""
        return ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """One-shot: nothing this pass produces re-triggers it."""
        return False

    def _rewrite(self, sdfg: dace.SDFG) -> set:
        # Symbols referenced by some array's shape/offset are kept even if the name pattern matches.
        referenced: set = set()
        for desc in sdfg.arrays.values():
            for s in getattr(desc, 'shape', ()):
                referenced.update(str(x) for x in dace.symbolic.symlist(s).values())
            for s in getattr(desc, 'offset', ()):
                referenced.update(str(x) for x in dace.symbolic.symlist(s).values())

        removed: set = set()
        scalars = [n for n, d in sdfg.arrays.items() if isinstance(d, dace.data.Scalar)]
        for s in scalars:
            pat = re.compile(rf'^(offset_)?{re.escape(s)}_d\d+$')
            for sym in list(sdfg.symbols):
                if sym in referenced:
                    continue
                if pat.match(sym):
                    sdfg.symbols.pop(sym, None)
                    removed.add(sym)

        if self.recursive:
            for state in sdfg.all_states():
                for node in state.nodes():
                    if isinstance(node, dace.nodes.NestedSDFG):
                        self._rewrite(node.sdfg)
        return removed

    def apply_pass(self, sdfg: dace.SDFG, _: dict):
        """Returns the set of removed symbol names, or ``None`` if none."""
        removed = self._rewrite(sdfg)
        return removed or None
