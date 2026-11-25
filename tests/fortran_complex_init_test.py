"""Complex counterpart of ``fortran_int_init_test.py``: single-precision COMPLEX and
DOUBLE COMPLEX arrays (kinds via ``KIND(1.0)``/``KIND(1.0D0)``) ingest as
``dace.complex64``/``dace.complex128``.

Frontend-only (type ingestion); SDFG->C++ codegen is DaCe's.  A single-precision complex
literal compiles since DaCe extended 1.0.1 -- see ``test_single_complex_literal_stores_its_value``.
"""

import dace
import numpy as np
import pytest

from tests._util import build_sdfg


def test_fortran_frontend_complex_init(tmp_path):
    """COMPLEX(kind=KIND(1.0)) -> complex64, COMPLEX(kind=KIND(1.0d0)) -> complex128."""
    src = """
subroutine main(c, z)
  integer, parameter :: sp = kind(1.0)
  integer, parameter :: dp = kind(1.0d0)
  complex(kind=sp) c(2)
  complex(kind=dp) z(2)
  c(1) = (1.0_sp, 2.0_sp)
  c(2) = (3.0_sp, -4.0_sp)
  z(1) = (5.0_dp, 6.0_dp)
  z(2) = (-7.0_dp, 8.0_dp)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name="complex_init").build()
    assert sdfg.arrays["c"].dtype == dace.complex64
    assert sdfg.arrays["z"].dtype == dace.complex128


def test_fortran_frontend_complex_arith(tmp_path):
    """Ingested complex64/complex128 arrays compute correctly end to end.  Values arrive
    via input arrays (no complex-literal construction), avoiding the codegen limitation
    isolated in ``test_single_complex_literal_codegen_raises``."""
    src = """
subroutine main(cin, cout, zin, zout)
  integer, parameter :: sp = kind(1.0)
  integer, parameter :: dp = kind(1.0d0)
  complex(kind=sp) cin(2), cout(2)
  complex(kind=dp) zin(2), zout(2)
  cout(1) = cin(1) + cin(2)
  cout(2) = cin(1) * cin(2)
  zout(1) = zin(1) + zin(2)
  zout(2) = zin(1) * zin(2)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name="complex_arith").build()
    assert sdfg.arrays["cin"].dtype == dace.complex64
    assert sdfg.arrays["zin"].dtype == dace.complex128

    cin = np.array([1 + 2j, 3 - 4j], order="F", dtype=np.complex64)
    zin = np.array([5 + 6j, -7 + 8j], order="F", dtype=np.complex128)
    cout = np.zeros(2, order="F", dtype=np.complex64)
    zout = np.zeros(2, order="F", dtype=np.complex128)
    sdfg(cin=cin, cout=cout, zin=zin, zout=zout)

    assert cout[0] == cin[0] + cin[1]
    assert cout[1] == cin[0] * cin[1]
    assert zout[0] == zin[0] + zin[1]
    assert zout[1] == zin[0] * zin[1]


def test_double_complex_literal_run(tmp_path):
    """A DOUBLE COMPLEX literal initializes and runs end to end -- works because DaCe's
    ``1j`` imaginary unit is itself complex128, so ``re + 1j*im`` is well-typed."""
    src = """
subroutine main(z)
  integer, parameter :: dp = kind(1.0d0)
  complex(kind=dp) z(2)
  z(1) = (5.0_dp, 6.0_dp)
  z(2) = (-7.0_dp, 8.0_dp)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name="dcomplex_lit").build()
    assert sdfg.arrays["z"].dtype == dace.complex128
    z = np.zeros(2, order="F", dtype=np.complex128)
    sdfg(z=z)
    assert z[0] == np.complex128(5.0 + 6.0j)
    assert z[1] == np.complex128(-7.0 + 8.0j)


# Reproducer: COMPLEX literal cast through decltype of a reference, readable codegen (dace 2514e17f17)
def test_single_complex_literal_stores_its_value(tmp_path):
    """A single-precision COMPLEX literal compiles and stores its value (DaCe once emitted an uncompilable
    ``float32 + complex128 * float32`` for it)."""
    src = """
subroutine main(c)
  integer, parameter :: sp = kind(1.0)
  complex(kind=sp) c(2)
  c(1) = (1.0_sp, 2.0_sp)
  c(2) = (3.0_sp, -4.0_sp)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name="scomplex_lit").build()
    assert sdfg.arrays["c"].dtype == dace.complex64
    c = np.zeros(2, order="F", dtype=np.complex64)
    sdfg(c=c)
    np.testing.assert_array_equal(c, np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex64))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
