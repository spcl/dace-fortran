"""Drop the Fortran offset / dimension symbols of scalar or dead data.

The HLFIR bridge synthesises one ``<name>_d<i>`` symbol per array
dimension and one ``offset_<name>_d<i>`` symbol per lower bound.
Under the current DaCe core a free symbol is a *required* SDFG
argument, so a stray ``<name>_d0`` turns into a
``KeyError: Missing program argument`` at call time.  Two cases make
such a symbol meaningless:

* the data behind ``<name>`` is (or becomes, e.g. after
  ``ConvertLengthOneArraysToScalars``) a true ``Scalar`` -- a scalar
  has no shape and no offset; the symbols are simply removed.
* ``<name>`` is an array the SDFG never actually accesses (no memlet
  references it): a never-``ALLOCATE``-d allocatable only fed to
  ``ALLOCATED()``, a pointer dummy repointed to an internal target
  before any read, etc.  Its extent is dead, so the free synthetic
  dim/offset symbols are ``sdfg.specialize``-d to ``1`` -- concretising
  the unused descriptor and dropping the symbols from the signature.

Both keep any symbol still referenced by an *accessed* array, and
recurse into nested SDFGs.
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
        return ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
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

        # Dead arrays: a descriptor the SDFG never reads/writes (no
        # memlet references it) -- e.g. a never-ALLOCATEd allocatable
        # only fed to ALLOCATED(), or a pointer dummy repointed before
        # use.  Its extent is irrelevant; specialise its free synthetic
        # dim/offset symbols to 1 so the unused descriptor is concrete
        # and the symbols leave the signature.  Guarded so a symbol
        # shared with an *accessed* array is never touched.
        accessed: set = set()
        for state in sdfg.all_states():
            for edge in state.edges():
                mem = edge.data
                if mem is not None and mem.data is not None:
                    accessed.add(mem.data)
        live_refs: set = set()
        for name, desc in sdfg.arrays.items():
            if name not in accessed:
                continue
            for s in getattr(desc, 'shape', ()):
                live_refs.update(str(x) for x in dace.symbolic.symlist(s).values())
            for s in getattr(desc, 'offset', ()):
                live_refs.update(str(x) for x in dace.symbolic.symlist(s).values())
        for name, desc in list(sdfg.arrays.items()):
            if name in accessed or isinstance(desc, dace.data.Scalar):
                continue
            pat = re.compile(rf'^(offset_)?{re.escape(name)}_d\d+$')
            dead = [s for s in list(sdfg.symbols) if pat.match(s) and s not in live_refs]
            if dead:
                sdfg.specialize({s: 1 for s in dead})
                removed.update(dead)

        if self.recursive:
            for state in sdfg.all_states():
                for node in state.nodes():
                    if isinstance(node, dace.nodes.NestedSDFG):
                        self._rewrite(node.sdfg)
        return removed

    def apply_pass(self, sdfg: dace.SDFG, _: dict):
        removed = self._rewrite(sdfg)
        return removed or None
