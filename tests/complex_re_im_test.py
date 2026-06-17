"""Pin the COMPLEX(KIND=*) ``%re`` / ``%im`` field accessor lowering.

Fortran 2008's ``z%re`` and ``z%im`` are INTRINSIC accessors on the
``COMPLEX`` type -- equivalent to ``REAL(z, kind)`` and ``AIMAG(z)``
respectively, but with full LHS support (``z%re = 1.0`` is legal).
The bridge must:

  * Route the LOWER side (read) through ``fir.extract_value`` ->
    ``<z>.real()`` / ``<z>.imag()`` (``expressions.cpp:934``).
  * NOT confuse ``%re`` / ``%im`` for user-defined struct field
    accesses -- they're built-in on COMPLEX, with no underlying
    ``hlfir.declare`` to scope-qualify.
  * Preserve the COMPLEX dtype on the SDFG signature
    (``dace.complex128`` for ``KIND=8``, ``dace.complex64`` for
    ``KIND=4``) without splitting into a separate ``_re`` / ``_im``
    pair.

User concern (verbatim): "Since re and im are special accesses of a
special type 'complex' we should not detect them in the struct chain".
These tests pin that contract.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(),
                                reason="flang-new-21 not on PATH")


# ===========================================================================
# Basic %re / %im read paths
# ===========================================================================
# A SCALAR ``COMPLEX(8) :: z`` dummy is registered by the bridge as a
# length-1 ``Array`` (pass-by-reference) rather than a by-value
# ``Scalar``, because DaCe's ctypes interop mis-handles by-value
# complex scalars (``complex128.as_ctypes()`` returns ``c_longdouble``,
# dropping the imaginary part).  Fortran passes scalar dummies by
# reference anyway, so callers bind a 1-element numpy complex array.
# These tests cover BOTH the scalar declaration (``z``) AND the array
# declaration (``z(n)``) forms of the ``%re`` / ``%im`` accessor.
def test_complex_re_read_scalar(tmp_path):
    """``out = z%re`` on a SCALAR COMPLEX dummy returns the real part."""
    src = """
MODULE cplx_re_scalar_mod
CONTAINS
SUBROUTINE cplx_re_scalar(z, out)
  IMPLICIT NONE
  COMPLEX(8), INTENT(IN) :: z
  REAL(8), INTENT(OUT) :: out
  out = z%re
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="cplx_re_scalar",
                      entry="cplx_re_scalar_mod::cplx_re_scalar").build()
    z = np.array([3 + 4j], dtype=np.complex128)  # scalar dummy, by-ref bind
    out_arr = np.zeros(1, dtype=np.float64)
    sdfg(z=z, out=out_arr)
    assert out_arr[0] == 3.0


def test_complex_im_read_scalar(tmp_path):
    """``out = z%im`` on a SCALAR COMPLEX dummy returns the imag part."""
    src = """
MODULE cplx_im_scalar_mod
CONTAINS
SUBROUTINE cplx_im_scalar(z, out)
  IMPLICIT NONE
  COMPLEX(8), INTENT(IN) :: z
  REAL(8), INTENT(OUT) :: out
  out = z%im
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="cplx_im_scalar",
                      entry="cplx_im_scalar_mod::cplx_im_scalar").build()
    z = np.array([3 + 4j], dtype=np.complex128)
    out_arr = np.zeros(1, dtype=np.float64)
    sdfg(z=z, out=out_arr)
    assert out_arr[0] == 4.0


def test_complex_re_read_element(tmp_path):
    """``out = z(1)%re`` returns the real part of a COMPLEX element."""
    src = """
MODULE cplx_re_elem_mod
CONTAINS
SUBROUTINE cplx_re_elem(z, out)
  IMPLICIT NONE
  COMPLEX(8), INTENT(IN) :: z(1)
  REAL(8), INTENT(OUT) :: out
  out = z(1)%re
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="cplx_re_elem",
                      entry="cplx_re_elem_mod::cplx_re_elem").build()
    z = np.array([3 + 4j], dtype=np.complex128, order="F")
    out_arr = np.zeros(1, dtype=np.float64)
    sdfg(z=z, out=out_arr)
    assert out_arr[0] == 3.0


def test_complex_im_read_element(tmp_path):
    """``out = z(1)%im`` returns the imaginary part of a COMPLEX element."""
    src = """
MODULE cplx_im_elem_mod
CONTAINS
SUBROUTINE cplx_im_elem(z, out)
  IMPLICIT NONE
  COMPLEX(8), INTENT(IN) :: z(1)
  REAL(8), INTENT(OUT) :: out
  out = z(1)%im
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="cplx_im_elem",
                      entry="cplx_im_elem_mod::cplx_im_elem").build()
    z = np.array([3 + 4j], dtype=np.complex128, order="F")
    out_arr = np.zeros(1, dtype=np.float64)
    sdfg(z=z, out=out_arr)
    assert out_arr[0] == 4.0


def test_complex_re_im_read_array(tmp_path):
    """``out_re(i) = z(i)%re`` element-wise over a 1-D COMPLEX array."""
    src = """
MODULE cplx_arr_split_mod
CONTAINS
SUBROUTINE cplx_arr_split(z, out_re, out_im, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  COMPLEX(8), INTENT(IN) :: z(n)
  REAL(8), INTENT(OUT) :: out_re(n), out_im(n)
  INTEGER :: i
  DO i = 1, n
    out_re(i) = z(i)%re
    out_im(i) = z(i)%im
  END DO
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="cplx_arr_split",
                      entry="cplx_arr_split_mod::cplx_arr_split").build()
    z = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128, order="F")
    out_re = np.zeros(3, dtype=np.float64, order="F")
    out_im = np.zeros(3, dtype=np.float64, order="F")
    sdfg(z=z, out_re=out_re, out_im=out_im, n=np.int32(3))
    np.testing.assert_array_equal(out_re, [1.0, 3.0, 5.0])
    np.testing.assert_array_equal(out_im, [2.0, 4.0, 6.0])


# ===========================================================================
# Equivalent intrinsics REAL() / AIMAG() must produce the SAME result
# ===========================================================================
def test_complex_re_equivalent_to_real_intrinsic(tmp_path):
    """``z(1)%re`` and ``REAL(z(1), KIND=8)`` are semantically equal."""
    src = """
MODULE cplx_re_vs_real_mod
CONTAINS
SUBROUTINE cplx_re_vs_real(z, out_field, out_intr)
  IMPLICIT NONE
  COMPLEX(8), INTENT(IN) :: z(1)
  REAL(8), INTENT(OUT) :: out_field, out_intr
  out_field = z(1)%re
  out_intr  = REAL(z(1), KIND=8)
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="cplx_re_vs_real",
                      entry="cplx_re_vs_real_mod::cplx_re_vs_real").build()
    z = np.array([2.5 - 1.5j], dtype=np.complex128, order="F")
    field = np.zeros(1, dtype=np.float64)
    intr = np.zeros(1, dtype=np.float64)
    sdfg(z=z, out_field=field, out_intr=intr)
    assert field[0] == intr[0] == 2.5


def test_complex_im_equivalent_to_aimag(tmp_path):
    """``z(1)%im`` and ``AIMAG(z(1))`` produce the same value."""
    src = """
MODULE cplx_im_vs_aimag_mod
CONTAINS
SUBROUTINE cplx_im_vs_aimag(z, out_field, out_intr)
  IMPLICIT NONE
  COMPLEX(8), INTENT(IN) :: z(1)
  REAL(8), INTENT(OUT) :: out_field, out_intr
  out_field = z(1)%im
  out_intr  = AIMAG(z(1))
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="cplx_im_vs_aimag",
                      entry="cplx_im_vs_aimag_mod::cplx_im_vs_aimag").build()
    z = np.array([2.5 - 1.5j], dtype=np.complex128, order="F")
    field = np.zeros(1, dtype=np.float64)
    intr = np.zeros(1, dtype=np.float64)
    sdfg(z=z, out_field=field, out_intr=intr)
    assert field[0] == intr[0] == -1.5


# ===========================================================================
# SDFG signature shape -- COMPLEX must NOT split into separate re/im arrays
# ===========================================================================
def test_complex_array_stays_single_complex_descriptor(tmp_path):
    """A COMPLEX array dummy lands on the signature as ONE complex
    descriptor (``dace.complex128``), not split into ``z_re`` / ``z_im``
    real descriptors.  Pinning this so a future struct-flattening pass
    can't accidentally split complex into a struct."""
    src = """
MODULE cplx_signature_mod
CONTAINS
SUBROUTINE cplx_signature(z, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  COMPLEX(8), INTENT(INOUT) :: z(n)
  z(1) = (1.0_8, 0.0_8)
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="cplx_signature",
                      entry="cplx_signature_mod::cplx_signature").build()
    arglist = sdfg.arglist()
    assert 'z' in arglist
    # No spurious split arrays
    assert 'z_re' not in arglist
    assert 'z_im' not in arglist
    assert 're' not in arglist
    assert 'im' not in arglist
    # dtype is complex128 (16-byte)
    import dace
    z_arr = arglist['z']
    assert z_arr.dtype == dace.complex128, (
        f"expected dace.complex128, got {z_arr.dtype}")


# ===========================================================================
# kind=4 complex too (32-bit real per component)
# ===========================================================================
def test_complex_kind4_re_im(tmp_path):
    src = """
MODULE cplx_k4_mod
CONTAINS
SUBROUTINE cplx_k4(z, out_re, out_im)
  IMPLICIT NONE
  COMPLEX(4), INTENT(IN) :: z(1)
  REAL(4), INTENT(OUT) :: out_re, out_im
  out_re = z(1)%re
  out_im = z(1)%im
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="cplx_k4",
                      entry="cplx_k4_mod::cplx_k4").build()
    z = np.array([1.5 + 2.5j], dtype=np.complex64, order="F")
    out_re = np.zeros(1, dtype=np.float32)
    out_im = np.zeros(1, dtype=np.float32)
    sdfg(z=z, out_re=out_re, out_im=out_im)
    assert out_re[0] == 1.5
    assert out_im[0] == 2.5
    assert out_im[0] == 2.5


# ===========================================================================
# User variable named ``im`` (INTEGER) -- NOT a complex access; the
# Python ``_RESERVED_DACE_NAMES`` shield handles the SymPy collision.
# Verifies the two paths (complex-accessor vs user-var) stay independent.
# ===========================================================================
def test_user_integer_im_does_not_conflict_with_complex_im_accessor(tmp_path):
    """A kernel that uses BOTH a user-named ``im`` integer counter AND
    a COMPLEX ``%im`` accessor -- the two must not interfere.  The
    user ``im`` is renamed by the Python shield; the complex ``%im``
    routes through ``fir.extract_value`` and isn't seen as a user
    field at all."""
    src = """
MODULE im_dual_use_mod
CONTAINS
SUBROUTINE im_dual_use(z, sums, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  COMPLEX(8), INTENT(IN) :: z(n)
  REAL(8), INTENT(OUT) :: sums
  INTEGER :: im
  sums = 0.0_8
  DO im = 1, n
    sums = sums + z(im)%im
  END DO
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="im_dual_use",
                      entry="im_dual_use_mod::im_dual_use").build()
    z = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128, order="F")
    sums = np.zeros(1, dtype=np.float64)
    sdfg(z=z, sums=sums, n=np.int32(3))
    assert sums[0] == 12.0  # 2 + 4 + 6
