"""Fortran SELECT CASE -> chain of nested kind="conditional" AST nodes.

Exercises every case-label shape the bridge recognises:

    case (v)          -> x == v
    case (v1, v2, v3) -> x == v1 or x == v2 or x == v3   (merged group)
    case (lo:hi)      -> lo <= x <= hi
    case (lo:)        -> x >= lo
    case (:hi)        -> x <= hi
    case default      -> else branch at the innermost nesting
"""

from pathlib import Path

import numpy as np
import pytest

from tests._util import build_sdfg, f2py_compile

_HERE = Path(__file__).resolve().parent
_SRC_PATH = _HERE / "select_case.f90"


@pytest.mark.parametrize(
    "x,expected",
    [
        (1, 100),
        (2, 200),
        (3, 200),
        (5, 200),
        (4, 0),  # falls to default
        (10, 300),
        (15, 300),
        (20, 300),
        (100, 400),
        (250, 400),
        (-1, 500),
        (-50, 500),
        (7, 0),
    ],
)
def test_select_case_all_shapes(tmp_path, x, expected):
    mod = f2py_compile(_SRC_PATH, tmp_path / "ref", "sel_all_ref")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    # lift-cf-to-scf refuses to walk past fir.select_case ("terminator with side effects"), so
    # use the minimal pipeline -- buildSelectCaseChain consumes the op directly, no CFG lifting needed.
    sdfg = build_sdfg(_SRC_PATH.read_text(), sdfg_dir, name="sel_all", pipeline="hlfir-propagate-shapes").build()
    sdfg.validate()

    out_ref = np.zeros(1, order="F", dtype=np.int32)
    mod.sel_all(np.int32(x), out_ref)
    assert out_ref[0] == expected, f"reference mismatch: x={x} -> {out_ref[0]}"

    out_sdfg = np.zeros(1, dtype=np.int32)
    sdfg(x=int(x), out=out_sdfg)
    assert out_sdfg[0] == expected, f"SDFG mismatch: x={x} -> {out_sdfg[0]}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
