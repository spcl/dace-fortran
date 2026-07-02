"""Complex counterpart of ``fortran_int_init_test.py``.

Initializes a single-precision COMPLEX array and a DOUBLE COMPLEX array in a
Fortran source string (kinds spelled via ``KIND(1.0)`` / ``KIND(1.0D0)``) and
checks that ``build_sdfg`` ingests them with the right element type
(``dace.complex64`` / ``dace.complex128``).

These tests only exercise the Fortran-frontend -> SDFG path (type ingestion);
the SDFG -> C++ codegen is DaCe's and is left untouched.  One consequence is
isolated below as an ``xfail``: DaCe's tasklet codegen renders a complex
literal via its ``1j`` (always complex128) imaginary unit, so a *single*
precision complex literal produces an uncompilable ``complex128 * float32``
expression.  See ``test_single_complex_literal_codegen_xfail``.
"""

import numpy as np
import pytest

import dace
from dace.codegen.exceptions import CompilationError

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_complex_init(tmp_path):
    """COMPLEX(kind=KIND(1.0)) -> complex64, COMPLEX(kind=KIND(1.0d0)) -> complex128.

    ``.build()`` returns the (uncompiled) SDFG; inspecting the descriptors is
    what proves the frontend ingested the types correctly.
    """
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
    sdfg = build_sdfg(src, tmp_path, name='complex_init').build()
    # COMPLEX(sp) -> 2xFP32 -> complex64; COMPLEX(dp) -> 2xFP64 -> complex128.
    assert sdfg.arrays['c'].dtype == dace.complex64
    assert sdfg.arrays['z'].dtype == dace.complex128


def test_fortran_frontend_complex_arith(tmp_path):
    """The ingested complex64/complex128 arrays compute correctly end to end.

    Values arrive via input arrays (no complex-literal construction), so this
    exercises both precisions at runtime without tripping the codegen
    limitation isolated in ``test_single_complex_literal_codegen_xfail``.
    """
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
    sdfg = build_sdfg(src, tmp_path, name='complex_arith').build()
    assert sdfg.arrays['cin'].dtype == dace.complex64
    assert sdfg.arrays['zin'].dtype == dace.complex128

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
    """A DOUBLE COMPLEX literal initializes and runs end to end.

    The double-precision path works because DaCe's ``1j`` imaginary unit is
    itself complex128, so ``re + 1j*im`` is a well-typed complex128 expression.
    """
    src = """
subroutine main(z)
  integer, parameter :: dp = kind(1.0d0)
  complex(kind=dp) z(2)
  z(1) = (5.0_dp, 6.0_dp)
  z(2) = (-7.0_dp, 8.0_dp)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='dcomplex_lit').build()
    assert sdfg.arrays['z'].dtype == dace.complex128
    z = np.zeros(2, order="F", dtype=np.complex128)
    sdfg(z=z)
    assert z[0] == np.complex128(5.0 + 6.0j)
    assert z[1] == np.complex128(-7.0 + 8.0j)


@pytest.mark.xfail(
    strict=True,
    raises=CompilationError,
    reason="DaCe's tasklet codegen renders a complex literal with its complex128 "
           "`1j` imaginary unit, so a single-precision COMPLEX literal emits an "
           "uncompilable `float32 + complex128 * float32`.  Frontend ingestion is "
           "correct (see test_fortran_frontend_complex_init); this is a DaCe "
           "SDFG->C++ codegen limitation, not a frontend one.",
)
def test_single_complex_literal_codegen_xfail(tmp_path):
    """Building the SDFG succeeds (types ingest); *running* it hits the codegen bug."""
    src = """
subroutine main(c)
  integer, parameter :: sp = kind(1.0)
  complex(kind=sp) c(2)
  c(1) = (1.0_sp, 2.0_sp)
  c(2) = (3.0_sp, -4.0_sp)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='scomplex_lit').build()
    # Frontend ingestion is still correct even though codegen will fail.
    assert sdfg.arrays['c'].dtype == dace.complex64
    c = np.zeros(2, order="F", dtype=np.complex64)
    sdfg(c=c)  # <- triggers DaCe codegen + C++ compile -> CompilationError
