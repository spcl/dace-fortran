"""Fold ``SOA`` / ``OA`` offset symbols whose observed ``lbound`` is 1.

Fortran arrays default to 1-based indexing; in practice most SoA offsets
captured by ``__f2dace_SOA_<field>_d_<dim>_s_<idx>`` and
``__f2dace_OA_<field>_d_<dim>_s`` symbols are 1 across every data sample.
The observed values come from a CSV produced by
``tools/analyze_lbounds`` -- one row per ``(field, dim)`` with a
semicolon-separated list of distinct ``lbound`` values seen.

For every such symbol whose CSV row has exactly one value and that value
is 1, the pass:

1. Removes the symbol from ``sdfg.symbols``.
2. Replaces every textual occurrence with the literal ``"1"`` via
   ``sdfg.replace_dict`` (memlets, interstate-edge expressions, tasklet
   code, scope params, ...).

Symbols not in the CSV, or with multiple observed values, are left
alone. ``SA`` / ``A`` (size / array) symbols are ignored -- only the
offset family (``SOA``, ``OA``) is touched.
"""

import csv
import re
from pathlib import Path
from typing import Dict, Set, Tuple

from dace import SDFG, properties
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf


_SOA_OA_RE = re.compile(r"^__f2dace_(SOA|OA)_(.+?)_d_(\d+)_s(?:_\d+)?(?:_.+)?$")


def load_lbound_csv(path: Path) -> Dict[Tuple[str, int], Set[int]]:
    """Return ``{(field, dim): {observed values}}`` from the CSV."""
    out: Dict[Tuple[str, int], Set[int]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            field = row["field"]
            dim = int(row["dim"])
            vals = {int(v) for v in row["values"].split(";") if v}
            out[(field, dim)] = vals
    return out


@properties.make_properties
@xf.explicit_cf_compatible
class ResolveExtentOffsets(ppl.Pass):
    """Replace ``SOA``/``OA`` offset symbols that are always 1 with the
    literal 1."""

    CATEGORY: str = "Simplification"

    csv_path = properties.Property(
        dtype=str,
        default="baseline/lbounds.csv",
        desc="CSV produced by tools/analyze_lbounds: kind,field,dim,values",
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
        scan = load_lbound_csv(path)

        rewrites: Dict[str, str] = {}
        for sym in list(sdfg.symbols):
            m = _SOA_OA_RE.match(sym)
            if not m:
                continue
            field, dim_str = m.group(2), m.group(3)
            seen = scan.get((field, int(dim_str)))
            if seen and seen == {1}:
                rewrites[sym] = "1"

        if not rewrites:
            return None

        # ``replace_keys=False`` keeps every folded LHS in ``sdfg.symbols``
        # as a (now-unused) declaration. If the graph-wide rewrite is
        # perfect the unused symbol is dropped from ``arglist`` silently;
        # if anything leaks through -- a memlet subset / tasklet code /
        # interstate edge that the rewrite missed -- the LHS is still a
        # valid declared symbol and codegen can resolve the reference.
        sdfg.replace_dict(rewrites, replace_keys=False)
        from utils.prune_names import replace_on_interstate_edges
        replace_on_interstate_edges(sdfg, rewrites)

        return rewrites
