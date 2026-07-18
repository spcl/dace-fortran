"""Section reductions in an ``IF`` condition.

Design (2026-06-16, "reductions are lib-nodes"): a SUM/MINVAL/MAXVAL/PRODUCT in
a condition materialises into a ``Reduce`` LIBRARY NODE writing a scalar transient
before the branch (condition reads ``__reduce_cond_N``), not inline-unrolled. A
SECTION operand (``kmin(iv,:)``, ``m(:,j)``) becomes a DaCe VIEW with the correct
stride/shape.  Replaces the earlier const-extent inline-unroll, which dropped
per-element subscripts and couldn't handle runtime extents.

Tests anchor both the structure (Reduce + View, no inline reduction) and e2e correctness.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from _util import build_sdfg, have_flang  # noqa: E402

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not available")


def _build(src_text: str, tmp_path, name: str):
    sdfg = build_sdfg(src_text, tmp_path / "sdfg", name=name).build()
    sdfg.validate()
    return sdfg


def _reduce_nodes(sdfg):
    from dace.libraries.standard.nodes import Reduce
    return [n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, Reduce)]


def _has_view(sdfg):
    from dace.data import View
    return any(isinstance(d, View) for d in sdfg.arrays.values())


def _all_conditions(sdfg):
    out = []
    for edge in sdfg.all_interstate_edges():
        for _, value in (edge.data.assignments or {}).items():
            out.append(value)
    return out


def _assert_materialised(sdfg, inline_fragment):
    """Reduction lowered to a Reduce lib-node + section View; no condition
    inline-unrolls it (no nested min/max, no bare array occurrence)."""
    assert _reduce_nodes(sdfg), "section reduction did not become a Reduce lib-node"
    assert _has_view(sdfg), "section operand did not become a View"
    conds = _all_conditions(sdfg)
    assert any("__reduce_cond" in c for c in conds), \
        f"condition does not read the materialised reduction scalar; got {conds}"
    for c in conds:
        assert inline_fragment not in c, f"reduction inline-unrolled into condition: {c!r}"


_MINVAL_SOURCE = """
MODULE m
  USE iso_fortran_env, ONLY: real64
CONTAINS
  SUBROUTINE minval_run(n, ke, kmin, out)
    INTEGER, INTENT(IN) :: n, ke
    REAL(real64), INTENT(INOUT) :: out(n)
    INTEGER :: kmin(n, 4)
    INTEGER :: iv, k
    kmin(:, :) = ke + 1
    DO iv = 1, n
      DO k = 1, ke
        IF (k >= MINVAL(kmin(iv, :))) THEN
          out(iv) = out(iv) + 1.0_real64
        END IF
      END DO
    END DO
  END SUBROUTINE minval_run
END MODULE m
"""


def test_minval_const_extent_section_in_if_condition(tmp_path):
    """``IF (k >= MINVAL(kmin(iv, :)))`` -- ICON ``mo_aes_graupel:341``; row becomes
    a View, MINVAL a Reduce lib-node."""
    sdfg = _build(_MINVAL_SOURCE, tmp_path, "minval_run")
    _assert_materialised(sdfg, "min(")
    # kmin is all ke+1 -> MINVAL = ke+1; k in 1..ke never reaches ke+1 -> no hits.
    n, ke = 3, 4
    kmin = np.zeros((n, 4), dtype=np.int32, order='F')  # set to ke+1 inside the kernel
    out = np.zeros(n, dtype=np.float64)
    sdfg(n=np.int32(n), ke=np.int32(ke), kmin=kmin, out=out)
    assert out.tolist() == [0.0, 0.0, 0.0]


_MAXVAL_SOURCE = """
MODULE m
  USE iso_fortran_env, ONLY: real64
CONTAINS
  SUBROUTINE maxval_run(n, ke, kmax, out)
    INTEGER, INTENT(IN) :: n, ke
    REAL(real64), INTENT(INOUT) :: out(n)
    INTEGER :: kmax(n, 4)
    INTEGER :: iv, k
    kmax(:, :) = ke
    DO iv = 1, n
      DO k = 1, ke
        IF (k <= MAXVAL(kmax(iv, :))) THEN
          out(iv) = out(iv) + 1.0_real64
        END IF
      END DO
    END DO
  END SUBROUTINE maxval_run
END MODULE m
"""


def test_maxval_const_extent_section_in_if_condition(tmp_path):
    sdfg = _build(_MAXVAL_SOURCE, tmp_path, "maxval_run")
    _assert_materialised(sdfg, "max(")
    # kmax all = ke -> MAXVAL = ke; k in 1..ke always <= ke -> ke hits per row.
    n, ke = 3, 4
    kmax = np.zeros((n, 4), dtype=np.int32, order='F')  # set to ke inside the kernel
    out = np.zeros(n, dtype=np.float64)
    sdfg(n=np.int32(n), ke=np.int32(ke), kmax=kmax, out=out)
    assert out.tolist() == [float(ke)] * n


_SUM_SOURCE = """
MODULE m
  USE iso_fortran_env, ONLY: real64
CONTAINS
  SUBROUTINE sum_run(n, weights, out)
    INTEGER, INTENT(IN) :: n
    INTEGER, INTENT(IN) :: weights(n, 3)
    REAL(real64), INTENT(INOUT) :: out(n)
    INTEGER :: iv
    DO iv = 1, n
      IF (SUM(weights(iv, :)) > 5) THEN
        out(iv) = out(iv) + 1.0_real64
      END IF
    END DO
  END SUBROUTINE sum_run
END MODULE m
"""


def test_sum_const_extent_section_in_if_condition(tmp_path):
    sdfg = _build(_SUM_SOURCE, tmp_path, "sum_run")
    _assert_materialised(sdfg, "weights[")
    n = 3
    weights = np.asfortranarray(
        np.array(
            [
                [1, 2, 3],  # sum 6 > 5 -> hit
                [1, 1, 1],  # sum 3 -> no
                [2, 2, 2]
            ],
            dtype=np.int32))  # sum 6 -> hit
    out = np.zeros(n, dtype=np.float64)
    sdfg(n=np.int32(n), weights=weights, out=out)
    assert out.tolist() == [1.0, 0.0, 1.0]


_PRODUCT_SOURCE = """
MODULE m
  USE iso_fortran_env, ONLY: real64
CONTAINS
  SUBROUTINE product_run(n, factors, out)
    INTEGER, INTENT(IN) :: n
    INTEGER, INTENT(IN) :: factors(n, 3)
    REAL(real64), INTENT(INOUT) :: out(n)
    INTEGER :: iv
    DO iv = 1, n
      IF (PRODUCT(factors(iv, :)) > 8) THEN
        out(iv) = out(iv) + 1.0_real64
      END IF
    END DO
  END SUBROUTINE product_run
END MODULE m
"""


def test_product_const_extent_section_in_if_condition(tmp_path):
    sdfg = _build(_PRODUCT_SOURCE, tmp_path, "product_run")
    _assert_materialised(sdfg, "factors[")
    n = 3
    factors = np.asfortranarray(
        np.array(
            [
                [2, 2, 3],  # prod 12 > 8 -> hit
                [1, 2, 2],  # prod 4 -> no
                [3, 3, 1]
            ],
            dtype=np.int32))  # prod 9 -> hit
    out = np.zeros(n, dtype=np.float64)
    sdfg(n=np.int32(n), factors=factors, out=out)
    assert out.tolist() == [1.0, 0.0, 1.0]


_RUNTIME_EXTENT_SOURCE = """
MODULE m
  USE iso_fortran_env, ONLY: real64
CONTAINS
  SUBROUTINE runtime_run(n, mm, arr, out)
    INTEGER, INTENT(IN) :: n, mm
    INTEGER, INTENT(IN) :: arr(n, mm)
    REAL(real64), INTENT(INOUT) :: out(n)
    INTEGER :: iv
    DO iv = 1, n
      IF (1 >= MINVAL(arr(iv, :))) THEN
        out(iv) = out(iv) + 1.0_real64
      END IF
    END DO
  END SUBROUTINE runtime_run
END MODULE m
"""


def test_runtime_extent_section_materialises(tmp_path):
    """A RUNTIME-extent section ``arr(iv, 1:mm)`` -- unhandled by the old
    inline-unroll (bailed to ``?``) -- now lowers to a View + Reduce lib-node."""
    sdfg = _build(_RUNTIME_EXTENT_SOURCE, tmp_path, "runtime_run")
    _assert_materialised(sdfg, "min(")
    for value in _all_conditions(sdfg):
        assert "?" not in value, f"runtime-extent reduction left a ?-placeholder: {value!r}"
    n, mm = 3, 4
    arr = np.asfortranarray(
        np.array(
            [
                [1, 5, 5, 5],  # min 1 -> 1>=1 hit
                [2, 3, 4, 5],  # min 2 -> 1>=2 no
                [0, 9, 9, 9]
            ],
            dtype=np.int32))  # min 0 -> 1>=0 hit
    out = np.zeros(n, dtype=np.float64)
    sdfg(n=np.int32(n), mm=np.int32(mm), arr=arr, out=out)
    assert out.tolist() == [1.0, 0.0, 1.0]
