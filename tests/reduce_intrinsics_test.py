"""Whole-array scalar reductions -> DaCe ``standard.Reduce`` library node: sum/product/
minval/maxval each lower through Flang's HLFIR reduce op, matching the gfortran/f2py reference.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

from tests._util import build_sdfg


_HERE = Path(__file__).resolve().parent
_SRC_PATH = _HERE / "reduce_intrinsics.f90"


def _f2py(src: Path, out_dir: Path, mod_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "numpy.f2py", "-c", str(src), "-m", mod_name, "--quiet"], cwd=out_dir)
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    __import__(mod_name)
    return sys.modules[mod_name]


def test_scalar_reductions_numerical(tmp_path):
    mod = _f2py(_SRC_PATH, tmp_path / "ref", "reduce_scalar_ref")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC_PATH.read_text(), sdfg_dir, name="reduce_scalar", pipeline="hlfir-propagate-shapes").build()
    sdfg.validate()

    rng = np.random.default_rng(7)
    n = 24
    # Positive values so product / minval / maxval are well-conditioned.
    a = rng.uniform(0.1, 2.0, size=n)

    # Reference.
    t_ref, p_ref, lo_ref, hi_ref = (
        np.zeros(1, order="F"),
        np.zeros(1, order="F"),
        np.zeros(1, order="F"),
        np.zeros(1, order="F"),
    )
    mod.reduce_scalar(np.asfortranarray(a), t_ref, p_ref, lo_ref, hi_ref)

    # SDFG: intent(inout) scalars land as size-1 Array descriptors (DaCe can't put
    # Scalars on the external signature), so the caller binds size-1 numpy arrays.
    t_sdfg = np.zeros(1, dtype=np.float64)
    p_sdfg = np.zeros(1, dtype=np.float64)
    lo_sdfg = np.zeros(1, dtype=np.float64)
    hi_sdfg = np.zeros(1, dtype=np.float64)
    sdfg(a=np.ascontiguousarray(a), total=t_sdfg, prod=p_sdfg, lo=lo_sdfg, hi=hi_sdfg, n=n)

    np.testing.assert_allclose(t_sdfg[0], t_ref[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(p_sdfg[0], p_ref[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(lo_sdfg[0], lo_ref[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(hi_sdfg[0], hi_ref[0], rtol=1e-12, atol=1e-12)


def test_scalar_reductions_structure(tmp_path):
    """SDFG contains four Reduce library nodes (one per sum/product/minval/maxval
    call) plus the scalar outputs as non-transient SDFG data."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC_PATH.read_text(), sdfg_dir, name="reduce_scalar", pipeline="hlfir-propagate-shapes").build()

    from dace.sdfg.state import LoopRegion, SDFGState
    from dace.libraries.standard.nodes.reduce import Reduce

    def iter_states(region):
        for n in region.nodes():
            if isinstance(n, LoopRegion):
                yield from iter_states(n)
            elif isinstance(n, SDFGState):
                yield n

    reduces = [n for s in iter_states(sdfg) for n in s.nodes() if isinstance(n, Reduce)]
    assert len(reduces) == 4, f"expected 4 Reduce library nodes; got {len(reduces)}"

    wcrs = sorted(r.wcr for r in reduces)
    assert "lambda a, b: a + b" in wcrs
    assert "lambda a, b: a * b" in wcrs
    assert "lambda a, b: min(a, b)" in wcrs
    assert "lambda a, b: max(a, b)" in wcrs


_SYMBOL_TARGET_SRC = """
subroutine number_sticks(n, st, index_map, total)
  implicit none
  integer, intent(in) :: n
  integer, intent(in) :: st(n)
  integer, intent(inout) :: index_map(n)
  integer, intent(out) :: total
  integer :: i, nct, nst
  nct = maxval(index_map)
  do i = 1, n
    if (st(i) > 0) then
      if (index_map(i) == 0) then
        nct = nct + 1
        index_map(i) = nct
      end if
    end if
  end do
  total = 0
  do i = 1, nct
    total = total + i
  end do
  nst = count(st > 0)
  do i = 1, nst
    total = total + 100
  end do
end subroutine
"""


def test_reductions_into_loop_bound_symbols(tmp_path):
    """``nct = MAXVAL(index_map)`` / ``nst = COUNT(st > 0)`` where the result later bounds a loop, so it is a symbol
    (QE ``sticks_map_index`` / ``sticks_map_set``): the library node writes a transient and the symbol is assigned
    from it (``KeyError: 'nct'`` / ``'nst'`` before)."""
    sdfg = build_sdfg(_SYMBOL_TARGET_SRC, tmp_path, name="number_sticks", entry="number_sticks").build()
    st = np.array([1, 0, 1, 1], dtype=np.int32)
    index_map = np.array([0, 0, 5, 0], dtype=np.int32)
    total = np.zeros(1, dtype=np.int32)
    sdfg(n=4, st=st, index_map=index_map, total=total)
    np.testing.assert_array_equal(index_map, [6, 0, 5, 7])
    assert total[0] == 28 + 3 * 100  # 1 + ... + 7, then one 100 per positive stick


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
