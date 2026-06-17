"""Two isolated facets surfaced by the QE ``vexx_bp_k_gpu`` /
``test_pointer_member_indexed_inlined_function`` work, split so each
mechanism has its own minimal reproducer:

  (a) **section-alias of an inlined FUNCTION dummy** -- an inlined
      function whose array dummy binds to a *column section* of the
      caller's array (``out(j) = colsum(a(:, j))``).  Flang boxes the
      section (``fir.embox``) on the FUNCTION-result inline path, so the
      bridge must peel the box to reach the ``hlfir.designate`` and
      register the dummy as a ``section_alias`` (otherwise it leaks as a
      free program argument).  This is the path fixed by the
      ``fir.embox``/``fir.load``/``fir.rebox`` peels added to the
      view-alias loop in ``bridge/extract_vars.cpp``.  EXPECTED: PASS --
      the section is read at the correct (1-based) column.

  (b) **indexed read whose index is a local-array element (value-symbol)
      on a flattened POINTER struct member** -- ``i = NINT(sel(:, j))``
      then ``out(j) = t%data(i(1), i(2), i(3))`` (graupel/getv's data
      path).  Each ``i(k)`` is promoted to a value-symbol (``i_at0`` ...)
      and the read renders as ``t_data[i_at0 - offset_t_data_d0, ...]``.
      For a *plain* array dummy the same value-symbol index renders with a
      literal ``-1`` (correct); only the flattened POINTER member routes
      the 1-based shift through ``offset_t_data_d<k>``.  That offset symbol
      has no passed array to infer it from and auto-fills to ``0`` -- so
      the read is off by one (``data[i]`` instead of ``data[i-1]``).  Same
      class as the graupel extent bug, but for an *offset* symbol.  FIXED:
      ``auto_dim_symbols`` now defaults a free offset symbol to the Fortran
      1-based lower bound (1) for direct ``sdfg()`` calls (the bridge keeps
      ``offset_<arr>_d<i>`` free on purpose for dummy ALLOC/POINTER bounds
      the bindings emitter fills via ``lbound`` -- e.g. ICON's
      ``end_block(min_rl:)`` -- so we default rather than bake).  EXPECTED:
      PASS.
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
    """(a) The inlined FUNCTION dummy bound to ``a(:, j)`` reads the
    correct column.  Regression guard for the box-peel section-alias
    fix."""
    sdfg = build_sdfg(_SRC_SECTION, tmp_path / "sdfg", name="run", entry="m_seca::run").build()
    n = 4
    # ``asfortranarray`` of a C-order reshape -> owned F-contiguous copy.
    # (``reshape(order="F")`` would return a non-owning view, which DaCe
    # rejects as a program argument.)
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
    """(b) ``i = NINT(sel(:,j))`` then ``out(j) = t%data(i(1),i(2),i(3))``.
    The 1-based pointer-member read must map ``data(k)`` -> ``data[k-1]``."""
    sdfg = build_sdfg(_SRC_VALUE_INDEX, tmp_path / "sdfg", name="run", entry="m_idxb::run").build()
    # ``sz`` strictly larger than the largest selector so a (suspected)
    # off-by-one read ``data[i]`` stays in bounds -> clean value mismatch,
    # not an out-of-bounds segfault.  Mirrors graupel/getv's data path
    # (``i = NINT(section)`` then a 3-D pointer-member read).
    n, sz = 2, 4
    data = np.asfortranarray(np.arange(float(sz ** 3)).reshape((sz, sz, sz)))
    # selector j -> index (j, j, j), 1-based; j in 1..n, all < sz.
    sel = np.asfortranarray(np.array([[1.0, 2.0]] * 3, dtype=np.float64))  # (3, n)
    out = np.zeros(n, dtype=np.float64, order="F")
    sdfg(t_data=data, sel=sel, out=out, n=np.int32(n))
    # out(j) = data(i,i,i) with i = j -> numpy data[j-1, j-1, j-1].
    expected = np.array([data[j - 1, j - 1, j - 1] for j in range(1, n + 1)])
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)
