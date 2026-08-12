"""Make every variant SDFG expose the same top-level signature as the
pre-specialise baseline.

``propagate_if_cond`` + ``simplify`` prune per-variant arrays and
symbols that the specialisation decides aren't reachable, which diverges
the 4 variant kernel signatures. ``main_per.cu`` needs one signature
shape to call, so we pad each variant with the baseline's missing
non-transient arrays and symbols, plus the two config symbols
``lvn_only`` and ``istep`` (which specialisation folds to literals but
the caller passes every time).

Arrays and non-transient scalars are padded **without** anchoring --
DaCe's ``arglist()`` picks them up automatically (arrays are always in
``data_args``; non-transient scalars always in ``scalar_args``). The
per-variant ``__sig_anchor`` state only holds a ``side_effects=True``
tasklet that references every **symbol** that must appear in the
kernel signature: the union's ``sdfg.symbols`` entries plus every free
symbol appearing in the shape expressions of the union's non-transient
arrays. Symbols need an explicit use site (free-symbol path) to land in
``arglist()``, so without the tasklet a variant would silently drop
symbols that were only used via a sibling's array shape.
"""

import copy
import re
from typing import Dict, Iterable, List, Set

import dace
from dace import SDFG


_DEAD_SYMBOLS = ("lvn_only", "istep")

# Names matching these patterns are specialisation-local loop iterators /
# loop bounds that disappear entirely from some variants (loop removed by
# specialise + simplify). Drift on these is expected and tolerated: they are
# excluded from the anchor union and from the variant-vs-variant equality
# check.
_IGNORED_DRIFT_RE = re.compile(r"^(?:_for_it_|i_(?:start|end)blk)")


def _is_ignored_drift_name(name: str) -> bool:
    return bool(_IGNORED_DRIFT_RE.match(name))


def _shape_free_symbols(desc) -> Set[str]:
    """Every free-symbol name appearing in ``desc.shape``. Integer dims
    have no ``free_symbols``; symbolic dims surface as ``sympy.Symbol``
    objects we stringify."""
    names: Set[str] = set()
    for dim in getattr(desc, "shape", ()) or ():
        for s in getattr(dim, "free_symbols", ()) or ():
            names.add(str(s))
    return names


def unify_variant_signatures(baseline: SDFG, variants: List[SDFG]):
    """Make every variant's ``SDFG.arglist()`` the same.

    Specialisation + ``simplify`` prune per-variant arrays and symbols
    (some variants lose kernel inputs) and lift per-variant loop
    iterators / conditional flags into ``sdfg.symbols`` (making them
    free). Neither is uniform across the 4 variants, so the 4 kernel
    signatures diverge.

    This pass:

    1. Pools the union of every SDFG's non-transient arrays (pool A),
       every SDFG's ``sdfg.symbols`` (pool S), and every free symbol
       appearing in the shape of any pooled array (pool shape-S).
    2. Pads each variant with every missing array from A and every
       missing symbol from S (+ ``lvn_only`` / ``istep`` as always-
       present dead symbols).
    3. Installs a ``side_effects=True`` anchor tasklet per variant that
       references every name in S ∪ shape-S as a bare identifier so
       ``simplify`` / ``used_symbols`` can't drop them. Arrays and
       non-transient scalars are left unreferenced: DaCe picks them
       up from ``sdfg.arrays`` for the kernel signature directly.
    4. Verifies via ``SDFG.arglist()`` that every variant's signature
       is a subset of the baseline's + DEAD_SYMBOLS, and that the 4
       variants agree with each other; raises ``ValueError`` with the
       drift detail otherwise.
    """
    union_arrays: Dict[str, object] = {}
    union_symbols: Dict[str, object] = {}

    def _add_arglist_of(sdfg: SDFG):
        for name in sdfg.arglist().keys():
            if _is_ignored_drift_name(name):
                continue
            if name in sdfg.arrays:
                desc = sdfg.arrays[name]
                if not getattr(desc, "transient", False):
                    union_arrays.setdefault(name, desc)
            elif name in sdfg.symbols:
                union_symbols.setdefault(name, sdfg.symbols[name])

    _add_arglist_of(baseline)
    for v in variants:
        _add_arglist_of(v)
    # A frontend may model these config knobs as length-1 data descriptors rather than
    # symbols; padding them as symbols would then collide with the descriptor.
    for s in _DEAD_SYMBOLS:
        if s in baseline.arrays or any(s in v.arrays for v in variants):
            continue
        union_symbols.setdefault(s, dace.int32)

    # Free symbols that appear in the shape of any pooled non-transient
    # array. Every variant must keep these alive -- they're implicit
    # kernel parameters whenever the array they size is in the arglist.
    shape_symbols: Set[str] = set()
    for desc in union_arrays.values():
        shape_symbols |= _shape_free_symbols(desc)
    shape_symbols = {
        n for n in shape_symbols if not _is_ignored_drift_name(n)
    }

    for v in variants:
        _unify_one(v, union_arrays, union_symbols, shape_symbols)

    _verify_arglists_equal(baseline, variants)


def _verify_arglists_equal(baseline: SDFG, variants: List[SDFG]):
    """Raise ``ValueError`` if any variant's arglist isn't a subset of the
    baseline signature, or if the 4 variant arglists disagree.

    The baseline reference is ``non-transient arrays + sdfg.symbols``
    (not just ``arglist()``): specialisation + ``simplify`` frees symbols
    the baseline had defined via init-edge assignments, so variants
    legitimately surface names that are in ``baseline.symbols`` without
    being in ``baseline.arglist()``. Those are not drift.

    Names matching ``_IGNORED_DRIFT_RE`` (per-variant loop iterators /
    bounds that vanish when specialisation removes the enclosing loop)
    are filtered out before comparing.
    """
    ref = {
        n for n, d in baseline.arrays.items()
        if not getattr(d, "transient", False) and not _is_ignored_drift_name(n)
    }
    ref |= {n for n in baseline.symbols if not _is_ignored_drift_name(n)}
    ref |= set(_DEAD_SYMBOLS)

    arglists = [
        {k for k in v.arglist().keys() if not _is_ignored_drift_name(k)}
        for v in variants
    ]

    mismatches = []
    for v, al in zip(variants, arglists):
        beyond_baseline = al - ref
        if beyond_baseline:
            mismatches.append((v.name, [], sorted(beyond_baseline)))
    if arglists and any(al != arglists[0] for al in arglists[1:]):
        for v, al in zip(variants[1:], arglists[1:]):
            if al != arglists[0]:
                mismatches.append(
                    (v.name,
                     sorted(arglists[0] - al),
                     sorted(al - arglists[0])),
                )
    if not mismatches:
        return
    lines = [f"arglist drift from baseline (ref: {baseline.name}):"]
    for name, missing, extra in mismatches:
        lines.append(f"  {name}:")
        if missing:
            lines.append(
                f"    missing {len(missing)}: {missing[:6]}{'...' if len(missing) > 6 else ''}"
            )
        if extra:
            lines.append(
                f"    extra   {len(extra)}: {extra[:6]}{'...' if len(extra) > 6 else ''}"
            )
    raise ValueError("\n".join(lines))


def _unify_one(sdfg: SDFG,
               ref_arrays: Dict[str, object],
               ref_symbols: Dict[str, object],
               shape_symbols: Set[str]):
    # Pad missing arrays (no anchor needed -- DaCe always includes
    # non-transient non-Scalar arrays in ``data_args``; non-transient
    # Scalars are likewise always in ``scalar_args``).
    for n, desc in ref_arrays.items():
        if n not in sdfg.arrays:
            sdfg.add_datadesc(n, copy.deepcopy(desc))
    # Pad missing symbols.
    for n, stype in ref_symbols.items():
        if n not in sdfg.symbols and n not in sdfg.arrays:
            sdfg.add_symbol(n, stype)
    # Also pad any shape-symbol not yet present -- it comes from a
    # sibling's array shape and may not have been surfaced in *this*
    # variant's ``sdfg.symbols`` yet.
    for n in shape_symbols:
        if n not in sdfg.symbols and n not in sdfg.arrays:
            sdfg.add_symbol(n, dace.int32)

    # Anchor only symbols: the union's ``sdfg.symbols`` set plus the
    # free-symbol set of all union-array shapes. Arrays and scalars
    # don't need anchoring.
    anchor_symbols = set(ref_symbols.keys()) | shape_symbols
    # Drop anything that ended up owned by this variant as an array or
    # scalar descriptor (those already appear in arglist on their own).
    anchor_symbols = {
        n for n in anchor_symbols
        if n in sdfg.symbols and n not in sdfg.arrays
    }
    _install_anchor_state(sdfg, sorted(anchor_symbols))


_ANCHOR_BATCH = 32  # keep per-tasklet `a + b + ...` BinOp chains shallow
                    # enough for DaCe's recursive AST visitors.


def _install_anchor_state(sdfg: SDFG, symbols: Iterable[str]):
    """One ``__sig_anchor`` state with one ``side_effects=True`` tasklet
    per batch that references the given symbols as bare identifiers.
    Each batch writes to its own transient ``_anchor_<i>`` scalar; the
    compiler DCEs them at -O3."""
    symbols = list(symbols)
    if not symbols:
        return

    old_start = sdfg.start_block
    anchor = sdfg.add_state("__sig_anchor", is_start_block=True)
    sdfg.add_edge(anchor, old_start, dace.InterstateEdge())

    for i in range(0, len(symbols), _ANCHOR_BATCH):
        batch = symbols[i:i + _ANCHOR_BATCH]
        _emit_anchor_batch(sdfg, anchor, batch, i // _ANCHOR_BATCH)


def _emit_anchor_batch(sdfg: SDFG, anchor, batch: List[str], idx: int):
    """One tasklet whose body sums the batch's symbols into a private
    transient ``_anchor_<i>`` scalar."""
    anchor_name, _ = sdfg.add_scalar(
        f"_anchor_{idx}",
        dtype=dace.int32,
        transient=True,
        find_new_name=True,
        storage=dace.StorageType.Register,
    )

    rhs = " + ".join(batch) if batch else "0"
    tasklet = anchor.add_tasklet(
        f"__sig_anchor_use_{idx}",
        set(),
        {"_out"},
        f"_out = {rhs}",
        side_effects=True,
    )
    sink = anchor.add_write(anchor_name)
    anchor.add_edge(tasklet, "_out", sink, None,
                    dace.Memlet(data=anchor_name, subset="0"))
