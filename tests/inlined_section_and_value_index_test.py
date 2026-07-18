"""Two isolated facets surfaced by the QE ``vexx_bp_k_gpu`` /
``test_pointer_member_indexed_inlined_function`` work:

(a) **section-alias of an inlined FUNCTION dummy**: an inlined function whose
    array dummy binds to a caller column section (``out(j) = colsum(a(:, j))``).
    Flang boxes the section (``fir.embox``) on the FUNCTION-result inline path,
    so the bridge must peel the box to reach ``hlfir.designate`` and register
    the dummy as a ``section_alias`` (else it leaks as a free program arg).
    Fixed via the fir.embox/fir.load/fir.rebox peels in
    ``bridge/extract_vars.cpp``'s view-alias loop.

(b) **indexed read whose index is a local-array element (value-symbol) on a
    flattened POINTER struct member**: ``i = NINT(sel(:, j))`` then
    ``out(j) = t%data(i(1), i(2), i(3))``.  Each ``i(k)`` becomes a value-symbol
    rendered as ``t_data[i_atK - offset_t_data_dK, ...]``; unlike a plain array
    dummy (literal ``-1``), the flattened POINTER member routes the shift
    through ``offset_t_data_d<k>``, which had no passed array to infer from and
    auto-filled to 0 -- an off-by-one.  Fixed: ``auto_dim_symbols`` now defaults
    a free offset symbol to the Fortran 1-based lower bound (the bridge keeps
    ``offset_<arr>_d<i>`` free on purpose for dummy ALLOC/POINTER bounds the
    bindings emitter fills via ``lbound``, e.g. ICON's ``end_block(min_rl:)``).
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# (a) -------------------------------------------------------------------
_SRC_SECTION = """
MODULE m_seca
  IMPLICIT NONE
CONTAINS
  FUNCTION colsum(v) RESULT(r)
    REAL(8), INTENT(IN) :: v(2)
    REAL(8) :: r
    r = v(1) + 10.0D0 * v(2)
  END FUNCTION colsum

  SUBROUTINE run(a, out, n)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: n
    REAL(8), INTENT(IN) :: a(2, n)
    REAL(8), INTENT(OUT) :: out(n)
    INTEGER :: j
    DO j = 1, n
      out(j) = colsum(a(:, j))
    END DO
  END SUBROUTINE run
END MODULE m_seca
"""


def test_inlined_function_section_alias_reads_correct_column(tmp_path):
    """(a) The inlined FUNCTION dummy bound to ``a(:, j)`` reads the correct column
    -- regression guard for the box-peel section-alias fix."""
    sdfg = build_sdfg(_SRC_SECTION, tmp_path / "sdfg", name="run", entry="m_seca::run").build()
    n = 4
    # asfortranarray of a C-order reshape -> owned F-contiguous copy (reshape(order="F")
    # would be a non-owning view, which DaCe rejects as a program arg).
    a = np.asfortranarray(np.arange(1.0, 2 * n + 1).reshape((2, n)))
    out = np.zeros(n, dtype=np.float64, order="F")
    sdfg(a=a, out=out, n=np.int32(n))
    expected = a[0, :] + 10.0 * a[1, :]
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


# (b) -------------------------------------------------------------------
_SRC_VALUE_INDEX = """
MODULE m_idxb
  IMPLICIT NONE
  TYPE :: tbl
    REAL(8), POINTER :: data(:, :, :)
  END TYPE
CONTAINS
  SUBROUTINE run(t, sel, out, n)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: n
    TYPE(tbl), INTENT(IN) :: t
    REAL(8), INTENT(IN) :: sel(3, n)
    REAL(8), INTENT(OUT) :: out(n)
    INTEGER :: j, i(3)
    DO j = 1, n
      i = NINT(sel(:, j))
      out(j) = t % data(i(1), i(2), i(3))
    END DO
  END SUBROUTINE run
END MODULE m_idxb
"""


def test_value_symbol_index_on_pointer_member_is_one_based(tmp_path):
    """(b) ``i = NINT(sel(:,j))`` then ``t%data(i(1),i(2),i(3))`` -- the pointer-member
    read must map ``data(k)`` -> ``data[k-1]``."""
    sdfg = build_sdfg(_SRC_VALUE_INDEX, tmp_path / "sdfg", name="run", entry="m_idxb::run").build()
    # sz > largest selector so an off-by-one read stays in bounds -> clean value
    # mismatch, not a segfault.  Mirrors graupel/getv's data path.
    n, sz = 2, 4
    data = np.asfortranarray(np.arange(float(sz**3)).reshape((sz, sz, sz)))
    # selector j -> index (j, j, j), 1-based; j in 1..n, all < sz.
    sel = np.asfortranarray(np.array([[1.0, 2.0]] * 3, dtype=np.float64))  # (3, n)
    out = np.zeros(n, dtype=np.float64, order="F")
    sdfg(t_data=data, sel=sel, out=out, n=np.int32(n))
    # out(j) = data(i,i,i) with i = j -> numpy data[j-1, j-1, j-1].
    expected = np.array([data[j - 1, j - 1, j - 1] for j in range(1, n + 1)])
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)
