"""Flatten plan -- single source of truth for AoS->SoA unpacking.

Produced by ``hlfir-flatten-structs`` (an MLIR module attribute),
consumed by the binding emitter.  One ``FlattenRecipe`` per unpacked
outer storage path, carrying both the forward (outer->flat) and
inverse (flat->outer) element expressions so the emitter needs no
knowledge of which flattening scheme fired.

Index convention: ``$i1``, ``$i2``, ... placeholders in recipe
expressions stand for the N loop indices the copy nest declares;
:func:`substitute_indices` fills in concrete names.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class FlattenRecipe:
    """One recorded unpacking.  Three emitter shapes, by flag combo:
    ``aliasable=True`` zero-copy ``c_f_pointer`` alias;
    ``aos_alloc=False, aliasable=False`` explicit allocate + deep
    ``do``-loop copy; ``aos_alloc=True`` padding-to-max pack/unpack for
    an AoS dummy with allocatable/pointer array members (Phase 5c-B,
    see ``aos_alloc`` below).

    flat_names: SDFG-visible flat names, argument order (plain member
        one name; complex-split two: re/im; aos_alloc one).
    read_exprs: parallel to flat_names -- Fortran expr for that flat's
        element at ($i1, $i2, ...) from the outer source.
    write_expr: Fortran expr reconstructing the outer element from the
        flats.  Empty when outer is read-only, the recipe is aliased,
        or aos_alloc (bespoke pack-out code instead).
    rank: number of loop indices used; 0 for scalar unpacks.  For
        aos_alloc this is outer_rank + 1.
    shape_exprs: per-rank extent expr, length == rank (typically
        ``size(<outer>, dim=1)``, ...).  For aos_alloc the inner dim
        is the cap symbol verbatim, not a size() call.
    aliasable: True iff pure element identity + matching storage
        layout -- emitter skips allocate/copy, aliases via
        c_f_pointer.  Mutually exclusive with aos_alloc.
    scratch_dtype: SDFG element dtype for flat scratch buffers; all
        flats of one recipe share a dtype.
    aos_alloc: Phase 5c-B (AoS + allocatable/pointer array member at
        the SDFG boundary).  True switches to the padding-to-max
        pack/unpack path::

            cap = max_i(merge(size(A(i)%w), 0, allocated(A(i)%w)))
            allocate(A_w(N, cap)); A_w = 0
            do i = 1, N; if (allocated(A(i)%w)) A_w(i, 1:size(A(i)%w)) = A(i)%w
            <SDFG call>
            do i = 1, N; if (allocated(A(i)%w)) A(i)%w = A_w(i, 1:size(A(i)%w))   ! intent(out)/(inout)
            deallocate(A_w)

        Companion buffer is always ``A_<member>(N, cap)``, one flat
        per recipe (complex-split doesn't combine with aos_alloc).
        Mixed structs (one allocatable + one plain member) split
        across two recipes: aos_alloc=True for the allocatable one,
        aliasable=True for the rest.
    cap_symbol: SDFG runtime symbol carrying the cap.  Empty unless
        aos_alloc=True, else ``cap_<base>_<member>``.
        ``_build_symbol_assigns`` skips it (pack-in computes it directly).
    source_logical_kind: N when the source member is Fortran
        LOGICAL(KIND=N) (1/2/4/8), else 0.  SDFG storage stays bool (1
        byte) regardless; drives a boundary bridge (wrapper-local
        pointer declared at the source kind + a bool scratch +
        per-element conversion) so a default LOGICAL slot isn't
        clobbered by a 1-byte SDFG write (root cause of the "free():
        invalid next size" glibc diagnostic in the ICON
        velocity_tendencies e2e).
    """
    flat_names: Tuple[str, ...]
    read_exprs: Tuple[str, ...]
    write_expr: str = ''
    rank: int = 0
    shape_exprs: Tuple[str, ...] = field(default_factory=tuple)
    aliasable: bool = False
    scratch_dtype: str = 'float64'
    aos_alloc: bool = False
    cap_symbol: str = ''
    source_logical_kind: int = 0

    # ----- JSON I/O ---------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict (tuple fields become lists)."""
        d = asdict(self)
        d['flat_names'] = list(self.flat_names)
        d['read_exprs'] = list(self.read_exprs)
        d['shape_exprs'] = list(self.shape_exprs)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'FlattenRecipe':
        """Rebuild from a :meth:`to_dict` mapping (lists back to tuples)."""
        d = dict(d)
        d['flat_names'] = tuple(d.get('flat_names', []))
        d['read_exprs'] = tuple(d.get('read_exprs', []))
        d['shape_exprs'] = tuple(d.get('shape_exprs', []))
        return cls(**d)


@dataclass(frozen=True)
class FlattenEntry:
    """One outer dummy / storage path that was unpacked.

    outer_expr: Fortran expr the user passes (``st`` or ``st%a%b%c``),
        threaded verbatim into generated code.
    outer_type: Fortran type of outer_expr, used in auto-gen comments.
    writeback_intent: 'out'/'inout'/'' -- non-empty + recipe.write_expr
        set triggers a copy-out loop.
    recipe: the FlattenRecipe describing the unpack.
    """
    outer_expr: str
    outer_type: str
    writeback_intent: str
    recipe: FlattenRecipe

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict (recurses into the recipe)."""
        return {
            'outer_expr': self.outer_expr,
            'outer_type': self.outer_type,
            'writeback_intent': self.writeback_intent,
            'recipe': self.recipe.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'FlattenEntry':
        """Rebuild from a :meth:`to_dict` mapping."""
        return cls(
            outer_expr=d['outer_expr'],
            outer_type=d['outer_type'],
            writeback_intent=d.get('writeback_intent', ''),
            recipe=FlattenRecipe.from_dict(d['recipe']),
        )


@dataclass(frozen=True)
class FlattenPlan:
    """All unpacks ``hlfir-flatten-structs`` performed for one entry
    subroutine.  One entry per flattened outer dummy; untouched
    scalars/plain-arrays don't appear.

    entries: tuple of FlattenEntry in argument order.
    """
    entries: Tuple[FlattenEntry, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict (one entry per flattened dummy)."""
        return {'entries': [e.to_dict() for e in self.entries]}

    @classmethod
    def from_dict(cls, d: dict) -> 'FlattenPlan':
        """Rehydrate from a plain dict -- the bridge returns the MLIR-side
        ``hlfir.flatten_plan`` attribute in this same nested shape."""
        return cls(entries=tuple(FlattenEntry.from_dict(e) for e in d.get('entries', [])))

    def to_json(self, path: str):
        """Write the plan to ``path`` as indented JSON."""
        with open(path, 'w') as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'FlattenPlan':
        """Load a plan previously written by :meth:`to_json`."""
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INDEX_RE = re.compile(r'\$i(\d+)')


def substitute_indices(expr: str, names: Tuple[str, ...]) -> str:
    """Replace ``$i1``, ``$i2``, ... placeholders with concrete loop
    variable names (``$i1`` -> ``names[0]``, etc).

    :raises IndexError: placeholder references past the end of ``names``.
    """

    def repl(m: re.Match) -> str:
        idx = int(m.group(1)) - 1
        if idx >= len(names):
            raise IndexError(f"placeholder $i{idx + 1} referenced but only {len(names)} names supplied")
        return names[idx]

    return _INDEX_RE.sub(repl, expr)


def strip_index_args(expr: str) -> str:
    """Strip the ``($i1, ...)`` suffix so the expr names the base storage
    path alone -- ``c_loc`` needs the array, not an element.  Returns the
    input unchanged if it has no parenthesised placeholder suffix.
    """
    m = re.match(r'^(.+?)\(\s*\$i\d+(?:\s*,\s*\$i\d+)*\s*\)\s*$', expr)
    return m.group(1) if m else expr
