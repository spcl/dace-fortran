"""End-to-end numerical correctness for the rank-promotion view alias.

When a caller passes a 1D array to a callee expecting a multi-D dummy
(storage-association reshape via ``fir.convert``), the bridge mints the dummy
as a DaCe :class:`dace.data.View` over the caller's 1D storage, with its own
column-major strides (3D ``buf(m,i,j)`` -> ``m + 5*(i-1) + 5*M*(j-1)``).

Verifies writes inside ``inner`` to ``buf(m,i,j)`` land at the right linear
offset inside ``scratch``, matching the f2py-compiled gfortran reference
element-by-element (NPB LU's ``scratch``/``buf`` pattern).
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
    """View's column-major strides map ``buf(i,j,k)`` to linear offset
    ``i + N1*(j-1) + N1*N2*(k-1)`` inside the parent ``scratch(N1*N2*N3)``."""
    N1, N2, N3 = 5, 7, 4
    SCRATCH_N = N1 * N2 * N3

    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="fill", entry="m::fill").build()

    # Sentinel init so untouched slots are visible (any -1.0 = view stride bug).
    scratch_sdfg = np.full((SCRATCH_N, ), -1.0, dtype=np.float64, order='F')
    sdfg(scratch=scratch_sdfg)

    # Reference flat buffer replicates the Fortran column-major linear-offset
    # formula in pure NumPy -- the contract the view's strides (1, N1, N1*N2) encode.
    expected = np.empty(SCRATCH_N, dtype=np.float64)
    for k in range(1, N3 + 1):
        for j in range(1, N2 + 1):
            for i in range(1, N1 + 1):
                lin = (i - 1) + N1 * (j - 1) + N1 * N2 * (k - 1)
                expected[lin] = i + 10 * j + 100 * k

    np.testing.assert_array_equal(scratch_sdfg, expected)


# ---------------------------------------------------------------------------
# 4D source -> 2D view.  Sequence association lets any rank reinterpretation
# happen as long as the actual's flat element count covers the dummy's; here
# arr(A,B,C,D) -> buf(A*B, C*D) collapses dims in Fortran column-major flat order.
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
    """``arr_4d(A,B,C,D)`` -> dummy ``buf(A*B, C*D)``; view strides ``(1, A*B)``
    must flatten ``buf(i,j)`` to the right slot inside the 4D actual.

    Source linear offset for ``arr_4d(a,b,c,d)`` is
    ``(a-1) + A*(b-1) + A*B*(c-1) + A*B*C*(d-1)``; the view reshapes that same
    flat sequence to ``(A*B, C*D)``, so ``buf(i,j)`` -> ``(i-1) + A*B*(j-1)``.
    """
    A, B, C, D = 4, 3, 5, 2
    sdfg = build_sdfg(_SRC_4D_TO_2D, tmp_path / "sdfg", name="fill", entry="m::fill").build()

    arr_sdfg = np.full((A, B, C, D), -1.0, dtype=np.float64, order='F')
    sdfg(arr_4d=arr_sdfg)

    # Expected: flatten the source column-major, then reindex to the view's (A*B, C*D) shape.
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
    """``arr_3d(A,B,C)`` -> dummy ``buf(A*B*C)``; 1D view with stride ``(1,)``,
    writes go directly to flat slot ``i-1``.  Rank REDUCTION (opposite of LU's tv)."""
    A, B, C = 4, 3, 5
    sdfg = build_sdfg(_SRC_3D_TO_1D, tmp_path / "sdfg", name="fill", entry="m::fill").build()

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
    """``arr_2d(6,8)`` -> dummy ``buf(2,3,4,2)``; source's 48 elements
    reinterpreted with view strides ``(1,2,6,24)``; each write lands at the right flat offset."""
    ROWS, COLS = 6, 8
    sdfg = build_sdfg(_SRC_2D_TO_4D, tmp_path / "sdfg", name="fill", entry="m::fill").build()

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
    """``arr_2d(ROWS, COLS)`` -> dummy ``buf(2, 3, COLS)`` (``ROWS = 2*3``); view
    strides ``(1, 2, 6)`` flatten 3D accesses to 2D column-major offsets."""
    ROWS, COLS = 6, 8
    sdfg = build_sdfg(_SRC_2D_TO_3D, tmp_path / "sdfg", name="fill", entry="m::fill").build()

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
