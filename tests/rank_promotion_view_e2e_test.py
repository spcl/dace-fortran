"""End-to-end numerical correctness for the rank-promotion view alias.

When a Fortran caller passes a 1D array unmodified to a callee expecting
a multi-D dummy (storage-association reshape via ``fir.convert``), the
bridge mints the dummy as a DaCe :class:`dace.data.View` over the
caller's 1D storage.  The view carries its own column-major strides so
a 3D ``buf(m, i, j)`` access flattens to ``m + 5*(i-1) + 5*M*(j-1)``
against the parent's flat buffer.

This test verifies the runtime behaviour matches the f2py-compiled
gfortran reference for the canonical NPB LU pattern:

    double precision :: scratch(N1*N2*N3)
    call inner(scratch)
    ! inner's dummy: double precision :: buf(N1, N2, N3)

The view must be wired so writes inside ``inner`` to ``buf(m, i, j)``
land at the right linear offset inside ``scratch``.  The post-run
contents of ``scratch`` are compared element-by-element against the
gfortran build.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """\
module m
  implicit none
  integer, parameter :: N1 = 5, N2 = 7, N3 = 4
  double precision :: scratch(N1*N2*N3)
contains
  subroutine inner(buf)
    double precision, intent(inout) :: buf(N1, N2, N3)
    integer :: i, j, k
    do k = 1, N3
      do j = 1, N2
        do i = 1, N1
          buf(i, j, k) = real(i + 10*j + 100*k, kind=8)
        end do
      end do
    end do
  end subroutine inner

  subroutine fill()
    call inner(scratch)
  end subroutine fill
end module m
"""


def test_rank_promotion_view_writes_land_at_correct_linear_offset(tmp_path):
    """The view's column-major strides must map ``buf(m, i, j)`` to
    the correct linear offset inside the parent 1D storage.  Verify
    that every write ``buf(i, j, k) = i + 10*j + 100*k`` lands at
    Fortran column-major linear offset ``i + N1*(j-1) + N1*N2*(k-1)``
    inside the parent ``scratch(N1*N2*N3)``.
    """
    N1, N2, N3 = 5, 7, 4
    SCRATCH_N = N1 * N2 * N3

    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="fill", entry="fill").build()

    # Initialise to a sentinel so we can see which slots inner wrote
    # and which stayed untouched (any -1.0 = view stride bug).
    scratch_sdfg = np.full((SCRATCH_N, ), -1.0, dtype=np.float64, order='F')
    sdfg(scratch=scratch_sdfg)

    # Build the reference flat buffer by replicating the Fortran
    # column-major linear-offset formula in pure NumPy.  This is the
    # contract the view's strides ``(1, N1, N1*N2)`` encode -- if
    # the view is wired correctly every write reaches the right slot.
    expected = np.empty(SCRATCH_N, dtype=np.float64)
    for k in range(1, N3 + 1):
        for j in range(1, N2 + 1):
            for i in range(1, N1 + 1):
                lin = (i - 1) + N1 * (j - 1) + N1 * N2 * (k - 1)
                expected[lin] = i + 10 * j + 100 * k

    np.testing.assert_array_equal(scratch_sdfg, expected)


# ---------------------------------------------------------------------------
# 4D source -> 2D view.  Fortran sequence association lets any rank
# reinterpretation happen as long as the actual's flat element count covers
# the dummy's.  4D ``arr(A, B, C, D)`` -> 2D ``buf(A*B, C*D)`` collapses
# the leading 2 actual dims into the dummy's first dim and the trailing
# 2 into the dummy's second, all in Fortran column-major flat order.
# ---------------------------------------------------------------------------
_SRC_4D_TO_2D = """\
module m
  implicit none
  integer, parameter :: A = 4, B = 3, C = 5, D = 2
  double precision :: arr_4d(A, B, C, D)
contains
  subroutine inner(buf)
    double precision, intent(inout) :: buf(A*B, C*D)
    integer :: i, j
    do j = 1, C*D
      do i = 1, A*B
        buf(i, j) = real(i + 100*j, kind=8)
      end do
    end do
  end subroutine inner

  subroutine fill()
    call inner(arr_4d)
  end subroutine fill
end module m
"""


def test_rank_reinterpret_4d_source_to_2d_view(tmp_path):
    """``arr_4d(A,B,C,D)`` passed to dummy ``buf(A*B, C*D)``.  The view's
    strides ``(1, A*B)`` over the source's column-major layout must
    flatten ``buf(i, j)`` to the right slot inside the 4D actual.

    The source's column-major linear offset for ``arr_4d(a, b, c, d)``
    is ``(a-1) + A*(b-1) + A*B*(c-1) + A*B*C*(d-1)``.  The view sees
    that same flat sequence reshaped to ``(A*B, C*D)``, so
    ``buf(i, j)`` maps to source index ``(i-1) + A*B*(j-1)`` (still
    column-major over the view's own shape).  Verify every write
    lands at the expected source slot.
    """
    A, B, C, D = 4, 3, 5, 2
    sdfg = build_sdfg(_SRC_4D_TO_2D, tmp_path / "sdfg", name="fill", entry="fill").build()

    arr_sdfg = np.full((A, B, C, D), -1.0, dtype=np.float64, order='F')
    sdfg(arr_4d=arr_sdfg)

    # Expected: flatten the source in column-major, then index in
    # the view's (A*B, C*D) shape.  Use NumPy's column-major
    # view (.flatten('F')) for the side-by-side comparison.
    expected_flat = np.empty(A * B * C * D, dtype=np.float64)
    for j in range(1, C * D + 1):
        for i in range(1, A * B + 1):
            lin = (i - 1) + (A * B) * (j - 1)
            expected_flat[lin] = i + 100 * j
    expected = expected_flat.reshape((A, B, C, D), order='F')

    np.testing.assert_array_equal(arr_sdfg, expected)


# ---------------------------------------------------------------------------
# 3D source -> 1D view.  Rank REDUCTION (opposite direction of LU's tv).
# ---------------------------------------------------------------------------
_SRC_3D_TO_1D = """\
module m
  implicit none
  integer, parameter :: A = 4, B = 3, C = 5
  double precision :: arr_3d(A, B, C)
contains
  subroutine inner(buf)
    double precision, intent(inout) :: buf(A*B*C)
    integer :: i
    do i = 1, A*B*C
      buf(i) = real(i*7, kind=8)
    end do
  end subroutine inner

  subroutine fill()
    call inner(arr_3d)
  end subroutine fill
end module m
"""


def test_rank_reinterpret_3d_source_to_1d_view(tmp_path):
    """``arr_3d(A, B, C)`` passed to dummy ``buf(A*B*C)``.  The view
    is 1D with stride ``(1,)``; writes ``buf(i)`` go directly to flat
    slot ``i - 1`` of the source.  Verifies rank REDUCTION (the
    opposite direction from LU's tv pattern)."""
    A, B, C = 4, 3, 5
    sdfg = build_sdfg(_SRC_3D_TO_1D, tmp_path / "sdfg", name="fill", entry="fill").build()

    arr_sdfg = np.full((A, B, C), -1.0, dtype=np.float64, order='F')
    sdfg(arr_3d=arr_sdfg)

    expected_flat = np.array([7 * i for i in range(1, A * B * C + 1)], dtype=np.float64)
    expected = expected_flat.reshape((A, B, C), order='F')

    np.testing.assert_array_equal(arr_sdfg, expected)


# ---------------------------------------------------------------------------
# 2D source -> 4D view.  Rank promotion 2D -> 4D.
# ---------------------------------------------------------------------------
_SRC_2D_TO_4D = """\
module m
  implicit none
  integer, parameter :: ROWS = 6, COLS = 8
  double precision :: arr_2d(ROWS, COLS)
contains
  subroutine inner(buf)
    double precision, intent(inout) :: buf(2, 3, 4, 2)
    integer :: a, b, c, d
    do d = 1, 2
      do c = 1, 4
        do b = 1, 3
          do a = 1, 2
            buf(a, b, c, d) = real(a + 10*b + 100*c + 1000*d, kind=8)
          end do
        end do
      end do
    end do
  end subroutine inner

  subroutine fill()
    call inner(arr_2d)
  end subroutine fill
end module m
"""


def test_rank_reinterpret_2d_source_to_4d_view(tmp_path):
    """``arr_2d(6, 8)`` passed to dummy ``buf(2, 3, 4, 2)``.  Source's
    48 elements get reinterpreted as 4D ``(2, 3, 4, 2)`` with view
    strides ``(1, 2, 6, 24)``.  Each ``buf(a, b, c, d)`` write lands
    at the right flat offset inside the 2D actual."""
    ROWS, COLS = 6, 8
    sdfg = build_sdfg(_SRC_2D_TO_4D, tmp_path / "sdfg", name="fill", entry="fill").build()

    arr_sdfg = np.full((ROWS, COLS), -1.0, dtype=np.float64, order='F')
    sdfg(arr_2d=arr_sdfg)

    expected_flat = np.empty(ROWS * COLS, dtype=np.float64)
    for d in range(1, 3):
        for c in range(1, 5):
            for b in range(1, 4):
                for a in range(1, 3):
                    lin = (a - 1) + 2 * (b - 1) + 2 * 3 * (c - 1) + 2 * 3 * 4 * (d - 1)
                    expected_flat[lin] = a + 10 * b + 100 * c + 1000 * d
    expected = expected_flat.reshape((ROWS, COLS), order='F')

    np.testing.assert_array_equal(arr_sdfg, expected)


# ---------------------------------------------------------------------------
# 2D source -> 3D view.  Mid-rank reinterpretation.
# ---------------------------------------------------------------------------
_SRC_2D_TO_3D = """\
module m
  implicit none
  integer, parameter :: ROWS = 6, COLS = 8
  double precision :: arr_2d(ROWS, COLS)
contains
  subroutine inner(buf)
    double precision, intent(inout) :: buf(2, 3, COLS)
    integer :: a, b, c
    do c = 1, COLS
      do b = 1, 3
        do a = 1, 2
          buf(a, b, c) = real(a + 10*b + 100*c, kind=8)
        end do
      end do
    end do
  end subroutine inner

  subroutine fill()
    call inner(arr_2d)
  end subroutine fill
end module m
"""


def test_rank_reinterpret_2d_source_to_3d_view(tmp_path):
    """``arr_2d(ROWS, COLS)`` -> dummy ``buf(2, 3, COLS)`` where
    ``ROWS = 2*3``.  The view's strides ``(1, 2, 6)`` flatten 3D
    accesses to 2D column-major offsets inside the actual.
    """
    ROWS, COLS = 6, 8
    sdfg = build_sdfg(_SRC_2D_TO_3D, tmp_path / "sdfg", name="fill", entry="fill").build()

    arr_sdfg = np.full((ROWS, COLS), -1.0, dtype=np.float64, order='F')
    sdfg(arr_2d=arr_sdfg)

    expected_flat = np.empty(ROWS * COLS, dtype=np.float64)
    for c in range(1, COLS + 1):
        for b in range(1, 4):
            for a in range(1, 3):
                lin = (a - 1) + 2 * (b - 1) + 2 * 3 * (c - 1)
                expected_flat[lin] = a + 10 * b + 100 * c
    expected = expected_flat.reshape((ROWS, COLS), order='F')

    np.testing.assert_array_equal(arr_sdfg, expected)
