"""Two inlined call sites of the same callee against different arrays -- reduces the
cloudsc shape that broke view_test_2 (multi-callsite section-slice dummies) to a
minimal form.  ``..._whole_array``: no view-alias machinery involved.
``..._section_slice``: activates the section_alias path (Pass 0b multi-callsite
rename) so each call site gets its own VarInfo."""

import numpy as np

from tests._util import build_sdfg
from tests._helpers import f2py


_BAR_DEF = """
SUBROUTINE bar(x)
double precision, intent(inout) :: x(10)
integer i
DO i = 2, 9
    x(i) = x(i) * 2
ENDDO
DO i = 2, 9
    x(i) = sin(x(i))
ENDDO
END SUBROUTINE bar
"""


def test_fortran_frontend_multi_callsite_whole_array(tmp_path):
    """Two ``bar`` calls, one against ``a``, one against ``b``  --  whole arrays."""
    src = (
        """
MODULE kernel_mod
CONTAINS
"""
        + _BAR_DEF
        + """
SUBROUTINE driver(a, b)
double precision, intent(inout) :: a(10), b(10)
integer i

DO i = 1, 10
    a(i) = i
    b(i) = i * 100
ENDDO

CALL bar(a)
CALL bar(b)
END SUBROUTINE driver
END MODULE kernel_mod
"""
    )
    ref = f2py(src, tmp_path / "ref", "multi_callsite_whole_ref")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="multi_callsite_whole", entry="driver").build()

    a_ref = np.zeros(10, dtype=np.float64)
    b_ref = np.zeros(10, dtype=np.float64)
    ref.kernel_mod.driver(a_ref, b_ref)

    a = np.zeros(10, dtype=np.float64)
    b = np.zeros(10, dtype=np.float64)
    sdfg(a=a, b=b)
    np.testing.assert_allclose(a, a_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(b, b_ref, rtol=1e-12, atol=1e-12)


def test_fortran_frontend_multi_callsite_section_slice(tmp_path):
    """Two ``bar`` calls, each on a section slice of a different 2-D array."""
    src = (
        """
MODULE kernel_mod
CONTAINS
"""
        + _BAR_DEF
        + """
SUBROUTINE driver(a, b)
double precision, intent(inout) :: a(10, 5), b(10, 5)
integer i, j

DO j = 1, 5
    DO i = 1, 10
        a(i, j) = i + 10 * j
        b(i, j) = (i + 10 * j) * 100
    ENDDO
ENDDO

CALL bar(a(:, 2))
CALL bar(b(:, 2))
END SUBROUTINE driver
END MODULE kernel_mod
"""
    )
    ref = f2py(src, tmp_path / "ref", "multi_callsite_slice_ref")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="multi_callsite_slice", entry="driver").build()

    a_ref = np.asfortranarray(np.zeros((10, 5), dtype=np.float64))
    b_ref = np.asfortranarray(np.zeros((10, 5), dtype=np.float64))
    ref.kernel_mod.driver(a_ref, b_ref)

    a = np.asfortranarray(np.zeros((10, 5), dtype=np.float64))
    b = np.asfortranarray(np.zeros((10, 5), dtype=np.float64))
    sdfg(a=a, b=b)
    np.testing.assert_allclose(a, a_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(b, b_ref, rtol=1e-12, atol=1e-12)


def test_section_dummy_passed_on_as_a_section_and_sized(tmp_path):
    """QE ``wave_g2r``'s shape, called twice: the section dummy is sized with ``SIZE`` and passed on as a
    section of itself, and the callee zeroes the array the caller read just before the second call."""
    src = """
MODULE kernel_mod
CONTAINS
SUBROUTINE scatter(psi, c, igk, ngk)
double precision, intent(out) :: psi(:)
double precision, intent(in) :: c(:, :)
integer, intent(in) :: igk(:), ngk
integer ig
psi = 0
DO ig = 1, ngk
    psi(igk(ig)) = c(ig, 1)
ENDDO
END SUBROUTINE scatter
SUBROUTINE g2r(f_in, f_out, igk)
double precision, intent(in) :: f_in(:, :)
double precision :: f_out(:)
integer, intent(in) :: igk(:)
CALL scatter(f_out, f_in(:, 1:1), igk, SIZE(f_in, 1))
END SUBROUTINE g2r
SUBROUTINE driver(a, igk, psic, out)
double precision, intent(in) :: a(5, 3)
integer, intent(in) :: igk(5)
double precision, intent(inout) :: psic(5), out(5)
CALL g2r(a(1:5, 1:2), psic, igk)
out = psic
CALL g2r(a(:, 2:3), psic, igk)
out = out + 2 * psic
END SUBROUTINE driver
END MODULE kernel_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="section_of_section", entry="kernel_mod::driver").build()
    a = np.array(np.arange(1, 16, dtype=np.float64).reshape(5, 3, order="F"), order="F")
    igk = np.array([3, 1, 2, 5, 4], dtype=np.int32)
    psic = np.zeros(5)
    out = np.zeros(5)
    sdfg(a=a, igk=igk, psic=psic, out=out)
    ref = np.zeros(5)
    ref[igk - 1] = a[:, 0] + 2 * a[:, 1]
    np.testing.assert_array_equal(out, ref)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
