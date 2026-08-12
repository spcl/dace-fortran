# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""BLAS call nested inside a conditional branch, with a scalar arriving as a
length-1 array (QE ``h_psi`` ``dgemv`` pattern).

``emit_blas`` stages such scalars (``alpha``/``beta``) as symbol promotions on
the BLAS state's inbound interstate edge.  It queried ``ctx.sdfg.in_edges``,
but a BLAS state created inside an ``if`` branch lives in the branch's
ControlFlowRegion, not the top-level SDFG graph --
``KeyError: SDFGState (s_NNNN)`` (the QE h_psi build's third fatal error).
Pinned: build + validate + numerics.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH")

_SRC = """
subroutine probe_gemv_cond(A, x, y, alpha, n, flag)
  implicit none
  integer, intent(in) :: n, flag
  real(8), intent(in) :: A(n,n), x(n), alpha(1)
  real(8), intent(inout) :: y(n)
  real(8) :: beta(1)
  if (flag > 0) then
    beta(1) = 0.0d0
    call dgemv('N', n, n, alpha, A, n, x, 1, beta, y, 1)
  end if
end subroutine probe_gemv_cond
"""


def test_gemv_in_branch_builds(tmp_path):
    """The SDFG builds and validates (was: KeyError on the nested BLAS state)."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="probe_gemv_cond").build()
    sdfg.validate()


def test_gemv_in_branch_numerical(tmp_path):
    """Promoted alpha/beta reach the lib node; y == alpha*A@x in the taken branch."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="probe_gemv_cond").build()
    rng = np.random.default_rng(3)
    n = 5
    A = np.asfortranarray(rng.standard_normal((n, n)))
    x = np.asfortranarray(rng.standard_normal(n))
    alpha = np.array([2.5], dtype=np.float64)
    y = np.asfortranarray(rng.standard_normal(n))
    y_ref = alpha[0] * (A @ x)
    sdfg(a=A, x=x, y=y, alpha=alpha, n=n, flag=1)  # Fortran descriptors register lowercase
    np.testing.assert_allclose(y, y_ref, rtol=1e-12, atol=1e-12)
