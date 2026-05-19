"""Drop the Fortran offset / dimension symbols of genuine scalars.

The HLFIR bridge synthesises one ``<name>_d<i>`` symbol per array
dimension and one ``offset_<name>_d<i>`` symbol per lower bound.
When the data behind ``<name>`` is (or becomes, e.g. after
``ConvertLengthOneArraysToScalars``) a true ``Scalar`` those symbols
are spurious -- a scalar has no shape and no offset -- and under the
current DaCe core a stray free ``<name>_d0`` turns into a
``KeyError: Missing program argument`` at call time.  They are simply
removed.

Genuine array dimension symbols are deliberately *not* touched here:
every array extent stays a required SDFG input (the new arglist
methodology), with correct values supplied at the call boundary --
the bindings emitter for real callers, the test call shim for direct
``sdfg(...)`` invocations.  This pass only strips the symbols of
things that are not arrays at all.

Keeps any symbol still referenced by some array's shape / offset, and
recurses into nested SDFGs.
"""
import re

import dace
from dace import properties
from dace.transformation import pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class RemoveScalarFortranShapeSymbols(ppl.Pass):
    """Remove the bridge's ``<scalar>_d<i>`` / ``offset_<scalar>_d<i>``
    symbols (a ``Scalar`` has neither shape nor offset).

    :param recursive: Also clean nested SDFGs.
    """

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
        # Symbols still legitimately referenced by some array's shape /
        # offset must be kept even if the name pattern matches.
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
        """:returns: the set of removed symbol names, or ``None`` if none."""
        removed = self._rewrite(sdfg)
        return removed or None
