"""Fortran EXIT inside a counted DO.

Flang lowers a loop body with EXIT to cf.br/cf.cond_br (no structured
fir.do_loop); lift-cf-to-scf folds it into an scf.while whose scf.condition
encodes the combined keep-going predicate. Pinned e2e against gfortran.
"""

from pathlib import Path

import numpy as np

from tests._util import build_sdfg, f2py_compile

_HERE = Path(__file__).resolve().parent
_SRC_PATH = _HERE / "do_loop_exit.f90"


def test_do_loop_exit_numerical(tmp_path):
    mod = f2py_compile(_SRC_PATH, tmp_path / "ref", "do_exit_ref")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC_PATH.read_text(), sdfg_dir, name="do_exit").build()
    sdfg.validate()

    rng = np.random.default_rng(23)
    n = 12
    # Mix values below/above the 100.0 exit threshold so an early exit actually fires.
    a = rng.uniform(-50.0, 150.0, size=n)

    b_ref = np.full(n, 9.0, order="F", dtype=np.float64)
    mod.do_exit(np.asfortranarray(a), b_ref)

    b_sdfg = np.full(n, 9.0, dtype=np.float64)
    # i is a local counter the SDFG initialises via an interstate edge, but
    # DaCe's free-symbol analysis still lists it, so callers pass a placeholder.
    sdfg(a=np.ascontiguousarray(a), b=b_sdfg, n=n, i=0)

    np.testing.assert_allclose(b_sdfg, b_ref, rtol=1e-12, atol=1e-12)


def test_exit_condition_reads_array_elements(tmp_path: Path):
    """QE gk_sort's ``IF (SUM(g(:, ng) ** 2) > q2x) EXIT``: the condition reads array elements, so the scalar holding
    it must read them through connectors, not name the array in its tasklet."""
    src = """
subroutine exit_on_norm(n, g, x, r)
  integer, intent(in) :: n
  double precision, intent(in) :: g(3, n), x
  integer, intent(out) :: r
  integer :: ng
  r = 0
  do ng = 1, n
    if (sum(g(:, ng) ** 2) > (x + 1d-8)) exit
    r = r + 1
  end do
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="exit_on_norm").build()
    g = np.array([[0.1] * 3, [0.2] * 3, [3.0] * 3, [0.1] * 3]).T.copy(order="F")
    r = np.zeros(1, np.int32)
    sdfg(n=np.int32(4), g=g, x=1.0, r=r)
    assert r[0] == 2


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
