"""Simple FaCe-native tests for Fortran bitwise intrinsics.

Flang lowers each of these to a small ``arith.*`` op tree on the integer
operand types:

- ``IBSET(x, p)``  -> ``x | (1 << p)``         (set bit p).
- ``IBCLR(x, p)``  -> ``x & ~(1 << p)``         (clear bit p; the ``~``
  comes out as ``arith.xori a, -1``).
- ``IEOR(a, b)``   -> ``a ^ b``                 (bitwise XOR).
- ``ISHFT(x, n)``  -> ``logical_left_shift(x, n)`` for ``n>0``,
                     ``logical_right_shift(x, -n)`` for ``n<0``  (Flang
                     inlines via ``arith.shli`` / ``arith.shrui`` + a
                     sign select; the right shift is *logical* / zero-fill,
                     not the sign-extending ``arith.shrsi``).
- ``IAND(a, b)``   -> ``a & b``.
- ``IOR(a, b)``    -> ``a | b``.
- ``BTEST(x, p)``  -> ``(x >> p) & 1`` (returned as i1 / .true.).
- ``IBITS(x, p, n)`` -> ``(x >> p) & ((1 << n) - 1)``.

The bridge's ``buildExpr`` recognises the underlying ``arith.shli`` /
``arith.shrui`` (-> ``logical_left_shift`` / ``logical_right_shift``
runtime helpers, which shift via the unsigned type so a negative
operand zero-fills) / ``arith.shrsi`` (-> arithmetic ``>>``) /
``arith.andi`` / ``arith.ori`` / ``arith.xori`` ops on non-i1
operands; the tests below verify the full chain end-to-end.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_bitwise_set_clear_xor_shift_and(tmp_path: Path):
    src = """
subroutine probe(x, y, out)
  integer, intent(in)  :: x, y
  integer, intent(out) :: out(6)
  out(1) = ibset(x, 2)
  out(2) = ibclr(x, 2)
  out(3) = ieor(x, y)
  out(4) = ishft(x, 3)
  out(5) = iand(x, y)
  out(6) = ior(x, y)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name='probe').build()
    x_in, y_in = 0b1010, 0b1100
    out = np.zeros(6, dtype=np.int32)
    sdfg(x=x_in, y=y_in, out=out)
    assert int(out[0]) == x_in | (1 << 2)  # ibset
    assert int(out[1]) == x_in & ~(1 << 2)  # ibclr
    assert int(out[2]) == x_in ^ y_in  # ieor
    assert int(out[3]) == x_in << 3  # ishft (positive)
    assert int(out[4]) == x_in & y_in  # iand
    assert int(out[5]) == x_in | y_in  # ior


def test_ishft_negative_shift_is_logical(tmp_path: Path):
    """``ISHFT`` with a negative shift is a *logical* (zero-fill) right
    shift, and applies to negative operands too.  A signed C++ ``>>``
    would sign-extend (``ishft(-182, -2) == -46`` instead of the
    correct ``1073741778``); the ``logical_right_shift`` helper makes
    the SDFG match Fortran.  Expected values are computed via numpy's
    unsigned 32-bit shift, which is the same zero-fill semantics."""
    src = """
subroutine probe(x, s, o_var, o_negconst, o_negx)
  integer, intent(in)  :: x, s
  integer, intent(out) :: o_var, o_negconst, o_negx
  o_var      = ishft(x, s)      ! runtime sign of shift
  o_negconst = ishft(x, -2)     ! constant negative -> logical right
  o_negx     = ishft(-100, -2)  ! negative operand, logical right
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name='probe').build()

    def fortran_ishft(x, s):
        u = np.uint32(np.int32(x))
        r = (u << np.uint32(s)) if s >= 0 else (u >> np.uint32(-s))
        return np.int32(r)

    for x_in, s_in in [(-182, -2), (-1, -1), (12345, -5), (-99999, -7),
                       (255, 4), (-255, 3), (1, 31)]:
        o_var = np.zeros(1, np.int32)
        o_negc = np.zeros(1, np.int32)
        o_negx = np.zeros(1, np.int32)
        sdfg(x=np.int32(x_in), s=np.int32(s_in),
             o_var=o_var, o_negconst=o_negc, o_negx=o_negx)
        assert int(o_var[0]) == int(fortran_ishft(x_in, s_in)), (x_in, s_in)
        assert int(o_negc[0]) == int(fortran_ishft(x_in, -2)), x_in
        assert int(o_negx[0]) == int(fortran_ishft(-100, -2))


def test_bit_query_and_extract(tmp_path: Path):
    """``btest(x, p)`` and ``ibits(x, p, n)``  --  bit query + slice extract."""
    src = """
subroutine probe(x, out_ibits, out_btest)
  integer, intent(in)  :: x
  integer, intent(out) :: out_ibits, out_btest
  out_ibits = ibits(x, 2, 3)
  if (btest(x, 1)) then
    out_btest = 1
  else
    out_btest = 0
  end if
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name='probe').build()
    x_in = 0b101110
    out_ibits = np.zeros(1, dtype=np.int32)
    out_btest = np.zeros(1, dtype=np.int32)
    sdfg(x=x_in, out_ibits=out_ibits, out_btest=out_btest)
    assert int(out_ibits[0]) == (x_in >> 2) & ((1 << 3) - 1)
    assert int(out_btest[0]) == 1 if (x_in & (1 << 1)) else 0
