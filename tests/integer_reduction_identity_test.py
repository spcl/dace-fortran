"""Verify the bridge emits TYPE-CORRECT reduction identities for MINVAL /
MAXVAL on integer arrays.

The bug surfaced via NPB LU's compile-flag pin (commit ``1fc8262`` set
the DaCe CPU args to ``-O0 -fno-fast-math -ffp-contract=off`` for fair
reference parity).  At -O0 the test
``tests/intrinsic_minmaxval_test.py::test_fortran_frontend_minval_int``
failed with ``res[1] == -2147483648`` instead of the expected min.

Root cause: the bridge emitted ``inf`` / ``-inf`` as the reduction
identity for MINVAL / MAXVAL on integer arrays.  ``inf`` flows through
DaCe's cppunparse to ``INFINITY`` (a ``double``), then the codegen
emits ``int_accumulator = INFINITY;`` -- C++ conversion of a non-finite
double to an integer is undefined behaviour.  At -O3 the optimizer
folded the conversion to INT_MAX/INT_MIN; at -O0 it produced INT_MIN
regardless of intent, silently breaking the reduction.

Fix: ``bridge/ast/dispatch.cpp::identityForType`` picks the literal
``2^(w-1) - 1`` (MAXVAL identity for MINVAL on int) or ``-2^(w-1)``
(MINVAL identity for MAXVAL on int) per the integer's bit-width.  For
float types it passes ``inf`` / ``-inf`` through unchanged.

These probes pin the contract per Fortran INTEGER kind and reduction
shape (whole-array, slice, empty array).
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_minval_int32_whole_array(tmp_path):
    """``MINVAL(arr)`` on a default-INTEGER (i32) array.  Identity
    must be INT_MAX (= 2^31 - 1), not ``INFINITY`` -- else the first
    ``min(INFINITY_as_int, x)`` returns the cast garbage."""
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
    """``MINVAL(arr(2:4))`` -- the slice path goes through
    ``buildSectionReduceAssign`` (loop-accumulator) rather than the
    whole-array Reduce libnode.  Same identity-type contract."""
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
    """``MINVAL`` of a zero-length array returns the identity itself
    (Fortran semantic: HUGE(int)).  Must be INT_MAX, not INT_MIN."""
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
    """``MAXVAL`` of a zero-length array returns the identity itself
    (Fortran semantic: -HUGE(int) - 1).  Must be INT_MIN, not INT_MAX."""
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
    """Regression: float MINVAL identity stays ``inf`` -- the
    type-aware path must not break the pre-existing float case."""
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
    """Regression: SUM identity stays ``0`` -- type-aware logic must
    not touch the integer-compatible literal ``0``."""
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
