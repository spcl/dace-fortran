"""Fold ``SA`` / ``A`` size symbols to well-known scalar names.

After ``StructToContainerGroups`` the SDFG is full of dimension-size
symbols like ``__f2dace_SA_<field>_d_<dim>_s_<idx>`` and their plain-array
counterparts ``__f2dace_A_<field>_d_<dim>_s_<idx>`` -- one per (array,
dim). In ICON velocity every one of them is, in practice, equal to one
of a small set of global scalars: ``nproma``, ``nlev``, ``nlev + 1``,
``nblks_e``, ``nblks_c``, ``nblks_v``. This pass replaces each such
symbol by the appropriate scalar name, registering that scalar as an
SDFG symbol and installing a runtime assignment on the init interstate
edge so the generated code still reads the value from the struct
pointers at entry (``nproma = global_data->nproma`` etc.).

Special case: when both ``nblks_c`` and ``nblks_v`` are observed to
equal 1 (ICON's ICON_dsl_velocity_tendencies R02B05 config) a single
``nblks_cv`` symbol -- also equal to 1 -- is introduced and every
size-1 dimension resolves to it. The original ``nblks_c`` / ``nblks_v``
pair stays available for sites that refer to them explicitly.

Inputs are read from the CSV produced by ``tools/analyze_lbounds``.

Errors:
- If any of ``nproma``, ``nlev``, ``nblks_c``, ``nblks_e``, ``nblks_v``
  has more than one observed scalar value across the data folder, the
  pass raises (values must be constant across the dataset).
- If two distinct scalar names resolve to the same concrete value
  *other than* the ``nblks_c == nblks_v == 1`` case, the pass raises.

Anything that doesn't match a known value is left alone -- the SA/A
symbol stays in the SDFG unchanged.
"""

import csv
import re
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import dace
from dace import SDFG, properties
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf


_SA_RE = re.compile(r"^__f2dace_(SA|A)_(.+?)_d_(\d+)_s(?:_\d+)?(?:_.+)?$")

# Default values for the scalars when a data file doesn't pin them down.
_DEFAULTS = {"nlev": 90, "nblks_c": 1, "nblks_e": 2, "nblks_v": 1}

# ``StructToContainerGroups``-mangled forms for each well-known scalar.
# These are the *exact* names SCG emits when the struct member itself is
# already an SoA leaf, so using them here keeps ``nproma`` / ``nblks_*``
# as a single kernel parameter (rather than ``nproma`` + ``__CG_..._m_nproma``
# living side by side in the signature).
_MANGLED = {
    "nproma":  "__CG_global_data__m_nproma",
    "nlev":    "__CG_global_data__m_nlev",
    "nblks_c": "__CG_p_patch__m_nblks_c",
    "nblks_e": "__CG_p_patch__m_nblks_e",
    "nblks_v": "__CG_p_patch__m_nblks_v",
    # "nlev + 1" is an expression -- we don't mangle it; callers fold to
    # the mangled ``nlev`` then add 1.
    "nlev + 1": "__CG_global_data__m_nlev + 1",
}


def load_csv(path: Path):
    """Return ``(sizes, lbounds, scalars)`` where each value is a
    ``{key: {observed values}}`` dict."""
    sizes: Dict[Tuple[str, int], Set[int]] = {}
    lbounds: Dict[Tuple[str, int], Set[int]] = {}
    scalars: Dict[str, Set[int]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            vals = {int(v) for v in row["values"].split(";") if v}
            if row["kind"] == "size":
                sizes[(row["field"], int(row["dim"]))] = vals
            elif row["kind"] == "lbound":
                lbounds[(row["field"], int(row["dim"]))] = vals
            elif row["kind"] == "scalar":
                scalars[row["field"]] = vals
    return sizes, lbounds, scalars


def _concrete_scalars(scalars: Dict[str, Set[int]]) -> Dict[str, int]:
    """Resolve each well-known scalar to a single concrete integer.

    Raises ``ValueError`` if a scalar is observed with more than one
    distinct value. Missing scalars fall back to the Fortran default
    where one exists; scalars with no default and no observation are
    omitted from the output dict.
    """
    out: Dict[str, int] = {}
    for name in ("nproma", "nlev", "nblks_c", "nblks_e", "nblks_v"):
        observed = scalars.get(name)
        if observed is None:
            if name in _DEFAULTS:
                out[name] = _DEFAULTS[name]
            continue
        if len(observed) != 1:
            raise ValueError(
                f"scalar {name!r} has conflicting observations {sorted(observed)}"
            )
        (val,) = observed
        out[name] = val
    return out


def _build_value_to_name(concrete: Dict[str, int]) -> Dict[int, str]:
    """Invert ``{scalar_name: value}`` to ``{value: <CG-mangled-name>}``
    with the nblks_cv collapse rule baked in.

    Rewrite targets use the ``StructToContainerGroups`` mangled form
    (``__CG_<container>__m_<member>``) so they coincide with the symbol
    ``DealiasSymbols`` already produced from the init-code struct walks.
    That keeps ``nproma`` / ``nblks_*`` as exactly one kernel parameter
    each.

    Raises ``ValueError`` if two different scalars share a value except
    for the explicitly-handled ``nblks_c == nblks_v`` case.
    """
    value_to_name: Dict[int, str] = {}

    # nblks_c / nblks_v collapse.
    nbc = concrete.get("nblks_c")
    nbv = concrete.get("nblks_v")
    if nbc is not None and nbv is not None and nbc == nbv:
        # Pick a single mangled target for the collapsed pair.
        value_to_name[nbc] = _MANGLED["nblks_c"]

    def _set(val: Optional[int], name: str):
        if val is None:
            return
        mangled = _MANGLED.get(name, name)
        if val in value_to_name and value_to_name[val] != mangled:
            existing = value_to_name[val]
            # The nblks_c / nblks_v collapse is the only permitted collision.
            collapsed = {_MANGLED["nblks_c"], _MANGLED["nblks_v"]}
            if {existing, mangled} <= collapsed:
                return
            raise ValueError(
                f"scalar-name collision: {mangled!r} and {existing!r} both equal {val}"
            )
        value_to_name.setdefault(val, mangled)

    _set(concrete.get("nproma"), "nproma")
    _set(concrete.get("nlev"), "nlev")
    if "nlev" in concrete:
        _set(concrete["nlev"] + 1, "nlev + 1")
    _set(concrete.get("nblks_e"), "nblks_e")
    # If nblks_c != nblks_v, register them separately.
    if nbc is not None and nbv is not None and nbc != nbv:
        _set(nbc, "nblks_c")
        _set(nbv, "nblks_v")

    return value_to_name


@properties.make_properties
@xf.explicit_cf_compatible
class ResolveExtentSizes(ppl.Pass):
    """Replace SA/A size symbols with well-known scalar names."""

    CATEGORY: str = "Simplification"

    csv_path = properties.Property(
        dtype=str,
        default="baseline/lbounds.csv",
        desc="CSV produced by tools/analyze_lbounds",
    )

    def __init__(self, csv_path: str = "baseline/lbounds.csv"):
        super().__init__()
        self.csv_path = csv_path

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _):
        path = Path(self.csv_path)
        if not path.exists():
            return None
        sizes, _lbounds, scalars = load_csv(path)

        concrete = _concrete_scalars(scalars)
        value_to_name = _build_value_to_name(concrete)

        rewrites: Dict[str, str] = {}
        for sym in list(sdfg.symbols):
            m = _SA_RE.match(sym)
            if not m:
                continue
            field, dim_str = m.group(2), m.group(3)
            observed = sizes.get((field, int(dim_str)))
            if observed is None or len(observed) != 1:
                continue
            (size,) = observed
            target = value_to_name.get(size)
            if target is not None:
                rewrites[sym] = target

        if not rewrites:
            return None

        # Ensure every rewrite target (an already-mangled symbol) exists as
        # a global ``int32`` symbol. No init-edge install: these names are
        # caller-supplied kernel parameters.
        base_targets: Set[str] = set()
        for target in rewrites.values():
            # ``nlev + 1`` is an expression, the base symbol is what we
            # want registered.
            base_targets.add(target.split(" + ")[0])
        for name in base_targets:
            if name not in sdfg.symbols and name not in sdfg.arrays:
                sdfg.add_symbol(name, dace.int32)

        # ``replace_keys=False`` keeps every folded LHS in ``sdfg.symbols``
        # as an (unused) declaration. Complete rewrites drop naturally
        # from ``arglist()``; partial rewrites still find the LHS declared
        # and codegen stays consistent.
        sdfg.replace_dict(rewrites, replace_keys=False)
        from utils.prune_names import replace_on_interstate_edges
        replace_on_interstate_edges(sdfg, rewrites)

        return rewrites
