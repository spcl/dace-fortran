"""IF/ELSE branching inside a DO loop: exercises ``_emit_cond`` on a THEN/ELSE pair
(writing ``b``) and a THEN-only IF (writing ``c``); SDFG must match the gfortran reference.
"""

from pathlib import Path

import numpy as np

from tests._util import build_sdfg, f2py_compile

_HERE = Path(__file__).resolve().parent
_SRC_PATH = _HERE / "if_else.f90"


def test_if_else_branch_numerical(tmp_path):
    mod = f2py_compile(_SRC_PATH, tmp_path / "ref", "if_else_ref")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC_PATH.read_text(), sdfg_dir, name="if_else_branch", pipeline="hlfir-propagate-shapes").build()
    sdfg.validate()

    rng = np.random.default_rng(17)
    n = 16
    a = rng.standard_normal(n)

    # Sentinel values so dropped branches are visible.
    b_ref = np.full(n, 9.0, order="F", dtype=np.float64)
    c_ref = np.full(n, 9.0, order="F", dtype=np.float64)
    mod.if_else_branch(np.asfortranarray(a), b_ref, c_ref)

    b_sdfg = np.full(n, 9.0, dtype=np.float64)
    c_sdfg = np.full(n, 9.0, dtype=np.float64)
    sdfg(a=np.ascontiguousarray(a), b=b_sdfg, c=c_sdfg, n=n)

    np.testing.assert_allclose(b_sdfg, b_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(c_sdfg, c_ref, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
