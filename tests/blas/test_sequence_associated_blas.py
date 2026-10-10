# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""BLAS operands passed by sequence association (QE ``calbec_gamma``): COMPLEX arrays reach ``REAL(8) a(lda, *)`` /
``x(*)`` / ``y(*)`` dummies of the ``mydgemv`` / ``mydgemm`` / ``mydger`` wrappers, which forward them to the real
routine. Each operand is the actual's storage read as REAL(8) in the routine's own column-major shape."""

import numpy as np

from tests._util import build_sdfg

_SRC = """
SUBROUTINE mydgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
  DOUBLE PRECISION, INTENT(IN) :: alpha, beta
  INTEGER, INTENT(IN) :: incx, incy, lda, m, n
  CHARACTER*1, INTENT(IN) :: trans
  DOUBLE PRECISION :: a(lda, *), x(*), y(*)
  CALL dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
END SUBROUTINE mydgemv

SUBROUTINE mydgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
  DOUBLE PRECISION, INTENT(IN) :: alpha, beta
  INTEGER, INTENT(IN) :: k, lda, ldb, ldc, m, n
  CHARACTER*1, INTENT(IN) :: transa, transb
  DOUBLE PRECISION :: a(lda, *), b(ldb, *), c(ldc, *)
  CALL dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
END SUBROUTINE mydgemm

SUBROUTINE mydger(m, n, alpha, x, incx, y, incy, a, lda)
  DOUBLE PRECISION, INTENT(IN) :: alpha
  INTEGER, INTENT(IN) :: incx, incy, lda, m, n
  DOUBLE PRECISION :: x(*), y(*), a(lda, *)
  CALL dger(m, n, alpha, x, incx, y, incy, a, lda)
END SUBROUTINE mydger

MODULE calbec_mod
CONTAINS
  SUBROUTINE calbec_gemv(npw, npwx, nkb, beta, psi, betapsi)
    INTEGER, INTENT(IN) :: npw, npwx, nkb
    COMPLEX(8), INTENT(IN) :: beta(npwx, nkb), psi(npwx, 1)
    REAL(8), INTENT(OUT) :: betapsi(nkb, 1)
    CALL mydgemv('C', 2 * npw, nkb, 2.0D0, beta, 2 * npwx, psi, 1, 0.0D0, betapsi, 1)
  END SUBROUTINE calbec_gemv

  SUBROUTINE calbec_gemm(npw, npwx, nkb, m, beta, psi, betapsi)
    INTEGER, INTENT(IN) :: npw, npwx, nkb, m
    COMPLEX(8), INTENT(IN) :: beta(npwx, nkb), psi(npwx, m)
    REAL(8), INTENT(OUT) :: betapsi(nkb, m)
    CALL mydgemm('C', 'N', nkb, m, 2 * npw, 2.0D0, beta, 2 * npwx, psi, 2 * npwx, 0.0D0, betapsi, nkb)
  END SUBROUTINE calbec_gemm

  SUBROUTINE hpsi_gemm(npw, npwx, nkb, m, vkb, ps, hpsi)
    INTEGER, INTENT(IN) :: npw, npwx, nkb, m
    COMPLEX(8), INTENT(IN) :: vkb(npwx, nkb)
    REAL(8), INTENT(IN) :: ps(nkb, m)
    COMPLEX(8), INTENT(INOUT) :: hpsi(npwx, m)
    CALL mydgemm('N', 'N', 2 * npw, m, nkb, 1.0D0, vkb, 2 * npwx, ps, nkb, 1.0D0, hpsi, 2 * npwx)
  END SUBROUTINE hpsi_gemm

  SUBROUTINE calbec_ger(npwx, nkb, m, beta, psi, betapsi)
    INTEGER, INTENT(IN) :: npwx, nkb, m
    COMPLEX(8), INTENT(IN) :: beta(npwx, nkb), psi(npwx, m)
    REAL(8), INTENT(INOUT) :: betapsi(nkb, m)
    CALL mydger(nkb, m, -1.0D0, beta, 2 * npwx, psi, 2 * npwx, betapsi, nkb)
  END SUBROUTINE calbec_ger

  SUBROUTINE add_vuspsi(nh, nhm, nkb, m, ofs, na, deeq, becp, ps)
    INTEGER, INTENT(IN) :: nh, nhm, nkb, m, ofs, na
    REAL(8), INTENT(IN) :: deeq(nhm, nhm, 2), becp(nkb, m)
    REAL(8), INTENT(INOUT) :: ps(nkb, m)
    CALL mydgemm('N', 'N', nh, m, nh, 1.0D0, deeq(1, 1, na), nhm, becp(ofs + 1, 1), nkb, 0.0D0, ps(ofs + 1, 1), nkb)
  END SUBROUTINE add_vuspsi
END MODULE calbec_mod
"""

NPW, NPWX, NKB, M = 3, 5, 4, 2


def _complex(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return np.asfortranarray(rng.random(shape) + 1j * rng.random(shape))


def _real_rows(z: np.ndarray, rows: int) -> np.ndarray:
    """The first ``rows`` REAL(8) entries of each column of ``z`` read as REAL(8) storage."""
    return np.asfortranarray(z).reshape(-1, order="F").view(np.float64).reshape(2 * z.shape[0], -1, order="F")[:rows]


def test_complex_actuals_to_dgemv(tmp_path):
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="calbec_gemv", entry="calbec_mod::calbec_gemv").build()
    rng = np.random.default_rng(0)
    beta, psi = _complex(rng, NPWX, NKB), _complex(rng, NPWX, 1)
    out = np.zeros((NKB, 1), order="F")
    sdfg(npw=np.int32(NPW), npwx=np.int32(NPWX), nkb=np.int32(NKB), beta=beta, psi=psi, betapsi=out)
    ref = 2.0 * _real_rows(beta, 2 * NPW).T @ _real_rows(psi, 2 * NPW)
    np.testing.assert_allclose(out, ref, rtol=1e-12)


def test_complex_actuals_to_dgemm(tmp_path):
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="calbec_gemm", entry="calbec_mod::calbec_gemm").build()
    rng = np.random.default_rng(1)
    beta, psi = _complex(rng, NPWX, NKB), _complex(rng, NPWX, M)
    out = np.zeros((NKB, M), order="F")
    sdfg(npw=np.int32(NPW), npwx=np.int32(NPWX), nkb=np.int32(NKB), m=np.int32(M), beta=beta, psi=psi, betapsi=out)
    ref = 2.0 * _real_rows(beta, 2 * NPW).T @ _real_rows(psi, 2 * NPW)
    np.testing.assert_allclose(out, ref, rtol=1e-12)


def _real_view(z: np.ndarray) -> np.ndarray:
    """The REAL(8) storage of a column-major COMPLEX(8) ``z``, one column per column of ``z``."""
    return z.reshape(-1, order="F").view(np.float64).reshape(2 * z.shape[0], -1, order="F")


def test_complex_actual_accumulates_into_dgemm_c(tmp_path):
    """QE ``add_vuspsi_k``: ``beta = 1`` reads ``C``, a window of a COMPLEX actual, and writes it back."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="hpsi_gemm", entry="calbec_mod::hpsi_gemm").build()
    rng = np.random.default_rng(5)
    vkb, hpsi = _complex(rng, NPWX, NKB), _complex(rng, NPWX, M)
    ps = np.asfortranarray(rng.random((NKB, M)))
    ref = hpsi.copy(order="F")
    _real_view(ref)[: 2 * NPW] += _real_rows(vkb, 2 * NPW) @ ps
    sdfg(npw=np.int32(NPW), npwx=np.int32(NPWX), nkb=np.int32(NKB), m=np.int32(M), vkb=vkb, ps=ps, hpsi=hpsi)
    np.testing.assert_allclose(hpsi, ref, rtol=1e-12)


def test_complex_actuals_to_dger(tmp_path):
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="calbec_ger", entry="calbec_mod::calbec_ger").build()
    rng = np.random.default_rng(2)
    beta, psi = _complex(rng, NPWX, NKB), _complex(rng, NPWX, M)
    out = np.asfortranarray(rng.random((NKB, M)))
    ref = out - np.outer(beta[0].real, psi[0].real)
    sdfg(npwx=np.int32(NPWX), nkb=np.int32(NKB), m=np.int32(M), beta=beta, psi=psi, betapsi=out)
    np.testing.assert_allclose(out, ref, rtol=1e-12)


def test_element_actuals_to_dgemm(tmp_path):
    """QE ``add_vuspsi_gamma``: the operands start at elements (``deeq(1, 1, na)``, ``becp(ofs + 1, 1)``)."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="add_vuspsi", entry="calbec_mod::add_vuspsi").build()
    rng = np.random.default_rng(3)
    nh, nhm, nkb, m, ofs, na = 2, 3, 5, 2, 1, 2
    deeq = np.asfortranarray(rng.random((nhm, nhm, 2)))
    becp = np.asfortranarray(rng.random((nkb, m)))
    ps = np.asfortranarray(rng.random((nkb, m)))
    ref = ps.copy(order="F")
    ref[ofs : ofs + nh] = deeq[:nh, :nh, na - 1] @ becp[ofs : ofs + nh]
    args = dict(nh=nh, nhm=nhm, nkb=nkb, m=m, ofs=ofs, na=na)
    sdfg(**{k: np.int32(v) for k, v in args.items()}, deeq=deeq, becp=becp, ps=ps)
    np.testing.assert_allclose(ps, ref, rtol=1e-12)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
