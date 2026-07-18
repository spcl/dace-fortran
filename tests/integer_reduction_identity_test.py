"""Bridge must emit TYPE-CORRECT reduction identities for MINVAL/MAXVAL on integer arrays.

Root cause (surfaced via NPB LU's -O0 compile-flag pin): the bridge emitted ``inf``/``-inf``
as the identity even for ints; casting a non-finite double to int is UB in C++, giving
INT_MIN at -O0 regardless of intent. Fix: ``dispatch.cpp::identityForType`` now picks
``2^(w-1)-1`` / ``-2^(w-1)`` by integer bit-width; float types still get inf/-inf.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_minval_int32_whole_array(tmp_path):
    """``MINVAL(arr)`` on i32: identity must be INT_MAX, not ``INFINITY`` (whose int-cast is garbage)."""
    src = """
SUBROUTINE f(arr, res)
integer, dimension(5) :: arr
integer :: res
res = MINVAL(arr)
END SUBROUTINE f
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name='f').build()
    arr = np.array([5, 3, -1, 7, 2], dtype=np.int32, order='F')
    res = np.zeros((1, ), dtype=np.int32, order='F')
    sdfg(arr=arr, res=res)
    assert res[0] == -1, f"MINVAL should be -1, got {res[0]}"


def test_maxval_int32_whole_array(tmp_path):
    """``MAXVAL(arr)`` on i32.  Identity must be INT_MIN."""
    src = """
SUBROUTINE f(arr, res)
integer, dimension(5) :: arr
integer :: res
res = MAXVAL(arr)
END SUBROUTINE f
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name='f').build()
    arr = np.array([5, 3, -1, 7, 2], dtype=np.int32, order='F')
    res = np.zeros((1, ), dtype=np.int32, order='F')
    sdfg(arr=arr, res=res)
    assert res[0] == 7, f"MAXVAL should be 7, got {res[0]}"


def test_minval_int32_slice(tmp_path):
    """``MINVAL(arr(2:4))``: slice path goes through ``buildSectionReduceAssign`` (loop-accumulator)
    rather than the whole-array Reduce libnode; same identity-type contract."""
    src = """
SUBROUTINE f(arr, res)
integer, dimension(5) :: arr
integer :: res
res = MINVAL(arr(2:4))
END SUBROUTINE f
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name='f').build()
    arr = np.array([10, 3, -1, 7, 100], dtype=np.int32, order='F')
    res = np.zeros((1, ), dtype=np.int32, order='F')
    sdfg(arr=arr, res=res)
    assert res[0] == -1, f"MINVAL of [3, -1, 7] should be -1, got {res[0]}"


def test_maxval_int32_slice(tmp_path):
    """``MAXVAL(arr(2:4))`` -- slice path with MAXVAL."""
    src = """
SUBROUTINE f(arr, res)
integer, dimension(5) :: arr
integer :: res
res = MAXVAL(arr(2:4))
END SUBROUTINE f
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name='f').build()
    arr = np.array([10, 3, -1, 7, 100], dtype=np.int32, order='F')
    res = np.zeros((1, ), dtype=np.int32, order='F')
    sdfg(arr=arr, res=res)
    assert res[0] == 7, f"MAXVAL of [3, -1, 7] should be 7, got {res[0]}"


def test_minval_empty_int32_returns_max(tmp_path):
    """``MINVAL`` of a zero-length array returns the identity itself (Fortran: HUGE(int)) -- must be INT_MAX, not INT_MIN."""
    src = """
SUBROUTINE f(empty, res)
integer, dimension(0) :: empty
integer :: res
res = MINVAL(empty)
END SUBROUTINE f
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name='f').build()
    empty = np.zeros((0, ), dtype=np.int32, order='F')
    res = np.zeros((1, ), dtype=np.int32, order='F')
    sdfg(empty=empty, res=res)
    assert res[0] == np.iinfo(np.int32).max, \
        f"MINVAL of empty int32 should be INT_MAX, got {res[0]}"


def test_maxval_empty_int32_returns_min(tmp_path):
    """``MAXVAL`` of a zero-length array returns the identity itself (Fortran: -HUGE(int)-1) -- must be INT_MIN, not INT_MAX."""
    src = """
SUBROUTINE f(empty, res)
integer, dimension(0) :: empty
integer :: res
res = MAXVAL(empty)
END SUBROUTINE f
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name='f').build()
    empty = np.zeros((0, ), dtype=np.int32, order='F')
    res = np.zeros((1, ), dtype=np.int32, order='F')
    sdfg(empty=empty, res=res)
    assert res[0] == np.iinfo(np.int32).min, \
        f"MAXVAL of empty int32 should be INT_MIN, got {res[0]}"


def test_minval_float64_still_uses_inf(tmp_path):
    """Regression: float MINVAL identity stays ``inf`` -- type-aware path must not break the pre-existing float case."""
    src = """
SUBROUTINE f(arr, res)
double precision, dimension(5) :: arr
double precision :: res
res = MINVAL(arr)
END SUBROUTINE f
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name='f').build()
    arr = np.array([5.5, 3.3, -1.1, 7.7, 2.2], dtype=np.float64, order='F')
    res = np.zeros((1, ), dtype=np.float64, order='F')
    sdfg(arr=arr, res=res)
    assert res[0] == -1.1, f"MINVAL float should be -1.1, got {res[0]}"


def test_sum_int32_still_uses_zero(tmp_path):
    """Regression: SUM identity stays ``0`` -- type-aware logic must not touch it."""
    src = """
SUBROUTINE f(arr, res)
integer, dimension(5) :: arr
integer :: res
res = SUM(arr)
END SUBROUTINE f
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name='f').build()
    arr = np.array([1, 2, 3, 4, 5], dtype=np.int32, order='F')
    res = np.zeros((1, ), dtype=np.int32, order='F')
    sdfg(arr=arr, res=res)
    assert res[0] == 15, f"SUM should be 15, got {res[0]}"
