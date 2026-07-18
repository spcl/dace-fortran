"""Reused-scalar WAR/WAW hazard: a Fortran scalar reassigned with live reads of the prior
value must not collapse onto one unordered DaCe scalar.

Fortran reuses a scalar temp freely::

    tmp = a(i)
    x   = tmp * b(i)      ! reads the FIRST value
    tmp = a(i-1)          ! re-write
    y   = tmp * b(i-1)    ! reads the SECOND value

DaCe scalars are values not ordered memory, so two writes to one scalar in a state carry no
ordering -- the scheduler may hoist the second write ahead of the first write's reads. Exact
NPB-LU viscous-flux miscompile (``u21i`` read ``rho_i(i-1)`` instead of ``rho_i(i)`` -> ~34%
residual error at Class S).

Fix (``emit_cfg._scalar_reassign_in_state``): start a new state at the re-write so each scalar
writes at most once per state; benign scalar RAW does not split.

Checked e2e against an f2py-compiled gfortran reference: bugged = diverges, fixed = bit-identical.
"""

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from _helpers import f2py

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_HEAD = "MODULE kernel_mod\nCONTAINS\n"
_TAIL = "END MODULE kernel_mod\n"


def _run(tmp_path, body, name, n=8, seed=11, nvars=1):
    src = _HEAD + body + _TAIL
    ref = f2py(src, tmp_path / "ref", f"{name}_ref")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name=name, entry="driver").build()

    rng = np.random.default_rng(seed)
    shape = (n, ) if nvars == 1 else (nvars, n)
    a = np.asfortranarray(rng.random(shape) + 0.5)
    b = np.asfortranarray(rng.random(shape) + 0.5)
    c0 = np.zeros(shape, dtype=np.float64, order="F")

    a_ref, b_ref, c_ref = (x.copy(order="F") for x in (a, b, c0))
    ref.kernel_mod.driver(a_ref, b_ref, c_ref, n)

    a_s, b_s, c_s = (x.copy(order="F") for x in (a, b, c0))
    sdfg(a=a_s, b=b_s, c=c_s, n=n)
    np.testing.assert_allclose(c_s,
                               c_ref,
                               rtol=1e-12,
                               atol=1e-12,
                               err_msg=f"{name}: SDFG diverged from gfortran reference")


def test_scalar_reuse_minimal(tmp_path):
    """Minimal LU pattern: tmp=a(i); x=tmp*b(i); tmp=a(i-1); y=tmp*b(i-1).

    Correct: c(i) = a(i)*b(i) - a(i-1)*b(i-1). Bugged: c(i) = a(i-1)*b(i) - a(i-1)*b(i-1) (x read the second tmp)."""
    _run(
        tmp_path, """
SUBROUTINE driver(a, b, c, n)
integer, intent(in) :: n
double precision, intent(in) :: a(n), b(n)
double precision, intent(inout) :: c(n)
double precision tmp, x, y
integer i
DO i = 2, n
    tmp = a(i)
    x   = tmp * b(i)
    tmp = a(i-1)
    y   = tmp * b(i-1)
    c(i) = x - y
ENDDO
END SUBROUTINE driver
""", "scalar_reuse_minimal")


def test_scalar_reuse_multicomponent(tmp_path):
    """NPB-LU shape: one scalar ``tmp`` feeds five ``*i`` reads, is re-written, then feeds five ``*im1`` reads -- five separate clobbered values."""
    _run(tmp_path,
         """
SUBROUTINE driver(a, b, c, n)
integer, intent(in) :: n
double precision, intent(in) :: a(5,n), b(5,n)
double precision, intent(inout) :: c(5,n)
double precision tmp, ui1, ui2, ui3, ui4, ui5
double precision um1, um2, um3, um4, um5
integer i
DO i = 2, n
    tmp = b(1,i)
    ui1 = tmp * a(1,i)
    ui2 = tmp * a(2,i)
    ui3 = tmp * a(3,i)
    ui4 = tmp * a(4,i)
    ui5 = tmp * a(5,i)
    tmp = b(1,i-1)
    um1 = tmp * a(1,i-1)
    um2 = tmp * a(2,i-1)
    um3 = tmp * a(3,i-1)
    um4 = tmp * a(4,i-1)
    um5 = tmp * a(5,i-1)
    c(1,i) = ui1 - um1
    c(2,i) = ui2 - um2
    c(3,i) = ui3 - um3
    c(4,i) = ui4 - um4
    c(5,i) = ui5 - um5
ENDDO
END SUBROUTINE driver
""",
         "scalar_reuse_multicomp",
         nvars=5)


def test_scalar_reuse_three_writes(tmp_path):
    """Three re-writes of the same scalar in one iteration: two splits."""
    _run(
        tmp_path, """
SUBROUTINE driver(a, b, c, n)
integer, intent(in) :: n
double precision, intent(in) :: a(n), b(n)
double precision, intent(inout) :: c(n)
double precision tmp, x, y, z
integer i
DO i = 1, n
    tmp = a(i)
    x   = tmp * 2.0d0
    tmp = b(i)
    y   = tmp * 3.0d0
    tmp = a(i) + b(i)
    z   = tmp * 4.0d0
    c(i) = x + y + z
ENDDO
END SUBROUTINE driver
""", "scalar_reuse_three")


def test_scalar_reuse_in_if(tmp_path):
    """Re-write inside an IF body -- exercises ``emit_assign``'s realised-graph guard (the has_structured path), not the loop batch path."""
    _run(
        tmp_path, """
SUBROUTINE driver(a, b, c, n)
integer, intent(in) :: n
double precision, intent(in) :: a(n), b(n)
double precision, intent(inout) :: c(n)
double precision tmp, x, y
integer i
DO i = 2, n
    IF (a(i) > 0.7d0) THEN
        tmp = a(i)
        x   = tmp * b(i)
        tmp = a(i-1)
        y   = tmp * b(i-1)
        c(i) = x - y
    ENDIF
ENDDO
END SUBROUTINE driver
""", "scalar_reuse_in_if")


def test_loop_carried_scalar_preserved(tmp_path):
    """A loop-CARRIED scalar accumulator must NOT be broken by the split: ``s`` reads its
    previous-iteration value, so it stays one scalar. Guards against over-eager splitting."""
    _run(
        tmp_path, """
SUBROUTINE driver(a, b, c, n)
integer, intent(in) :: n
double precision, intent(in) :: a(n), b(n)
double precision, intent(inout) :: c(n)
double precision s
integer i
s = 0.0d0
DO i = 1, n
    s = s + a(i)
    c(i) = s
ENDDO
END SUBROUTINE driver
""", "loop_carried_scalar")
