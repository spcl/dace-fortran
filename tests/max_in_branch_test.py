"""Regression coverage for MAX(complex_expr, 0) feeding a local scalar later read inside an
IF condition (extracted from ice_supersaturation_adjustment):

    REAL(8) :: zsupsat
    DO jl = 1, n
      IF (...) THEN
        zsupsat = MAX((a(jl) - b * c(jl)) / d(jl), 0.0D0)
      ELSE
        zsupsat = MAX((1.0D0 - e(jl)) * (...) / d(jl), 0.0D0)
      END IF
      IF (zsupsat > eps) THEN
        ! consume zsupsat ...
      END IF
    END DO

Bug: a float scalar read in an IF condition was misclassified as an SDFG symbol
(extract_vars.cpp Pass 2d), routing its assignment through array_read_to_dace_expr -- which
only recognises a single-array-read RHS, so the MAX expression collapsed to its first operand.

Fix: keep float scalars as plain scalars even in branch conditions; only integer scalars
(loop counters, do-while guards, indices) need symbol promotion for interstate edges.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_max_complex_expr_in_branch_to_scalar(tmp_path: Path):
    """Float scalar gets MAX((a-b*c)/d, 0) in a per-iter IF, then is read in a downstream IF --
    verifies the full MAX expression survives instead of collapsing to its first operand."""
    src = """
SUBROUTINE max_in_branch(a, b, c, d, e, out, n, eps)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(IN) :: a(n), c(n), d(n), e(n)
  REAL(KIND=8), VALUE :: b, eps
  REAL(KIND=8), INTENT(OUT) :: out(n)
  INTEGER :: jl
  REAL(KIND=8) :: zsupsat
  DO jl = 1, n
    IF (e(jl) > 0.5D0) THEN
      zsupsat = MAX((a(jl) - b * c(jl)) / d(jl), 0.0D0)
    ELSE
      zsupsat = MAX((1.0D0 - e(jl)) * (a(jl) - b * c(jl)) / d(jl), 0.0D0)
    END IF
    IF (zsupsat > eps) THEN
      out(jl) = zsupsat
    ELSE
      out(jl) = -1.0D0
    END IF
  END DO
END SUBROUTINE
"""
    sdfg = build_sdfg(src, tmp_path, name="max_in_branch").build()

    n = 8
    rng = np.random.default_rng(0)
    a = np.asfortranarray(rng.uniform(0.0, 5.0, n))
    c = np.asfortranarray(rng.uniform(0.0, 1.0, n))
    d = np.asfortranarray(rng.uniform(0.5, 1.5, n))
    e = np.asfortranarray(rng.uniform(0.0, 1.0, n))
    out = np.zeros(n, dtype=np.float64, order="F")
    sdfg(a=a, b=2.0, c=c, d=d, e=e, out=out, n=n, eps=1e-6)

    # NumPy reference -- mirrors the Fortran branches exactly.
    zsupsat = np.where(
        e > 0.5,
        np.maximum((a - 2.0 * c) / d, 0.0),
        np.maximum((1.0 - e) * (a - 2.0 * c) / d, 0.0),
    )
    expected = np.where(zsupsat > 1e-6, zsupsat, -1.0)

    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-15)


def test_max_in_outer_branch_only(tmp_path: Path):
    """Trimmed reproducer without the downstream condition -- catches the regression even when
    the float scalar is never re-read inside another condition."""
    src = """
SUBROUTINE max_in_outer_branch(a, b, c, d, mask, out, n)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(IN) :: a(n), c(n), d(n), mask(n)
  REAL(KIND=8), VALUE :: b
  REAL(KIND=8), INTENT(OUT) :: out(n)
  INTEGER :: jl
  REAL(KIND=8) :: zsupsat
  DO jl = 1, n
    IF (mask(jl) > 0.0D0) THEN
      zsupsat = MAX((a(jl) - b * c(jl)) / d(jl), 0.0D0)
      IF (zsupsat > 0.0D0) out(jl) = zsupsat
    END IF
  END DO
END SUBROUTINE
"""
    sdfg = build_sdfg(src, tmp_path, name="max_in_outer_branch").build()

    n = 6
    rng = np.random.default_rng(1)
    a = np.asfortranarray(rng.uniform(0.0, 5.0, n))
    c = np.asfortranarray(rng.uniform(0.0, 1.0, n))
    d = np.asfortranarray(rng.uniform(0.5, 1.5, n))
    mask = np.asfortranarray(rng.uniform(-1.0, 1.0, n))
    out = np.zeros(n, dtype=np.float64, order="F")
    sdfg(a=a, b=2.0, c=c, d=d, mask=mask, out=out, n=n)

    zs = np.maximum((a - 2.0 * c) / d, 0.0)
    expected = np.where((mask > 0.0) & (zs > 0.0), zs, 0.0)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-15)
