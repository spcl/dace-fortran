"""Regression cover for the intrinsic-name-shadow family (backlog M4 / Ext 5).

A user variable named after a Fortran intrinsic (``max``, ``sum``, ``sqrt``, ...)
is handled at extraction time by ``rejectOrRenameReservedShortNames``
(``bridge/extract_vars.cpp``): a LOCAL variable is RENAMED to ``var_<name>``
(mangling override, landed in de9348e) since the genuine intrinsic renders
through a separate ``hlfir`` op; a DUMMY argument is HARD-REJECTED instead,
since its short name is the user-facing SDFG signature arg and silently
renaming it would break the caller ABI.

M4's hypothesis -- that QE's ``DOUBLE PRECISION :: max`` leaks because it has
"no hlfir.declare so the hard-reject misses" -- is stale: every LOCAL
intrinsic-shadow form (tasklet body, loop bound/array size/interstate
condition via symbol/sympy, EQUIVALENCE/BLOCK/COMMON, inlined callee beside a
sibling using the GENUINE intrinsic) surfaces as an ``hlfir.declare`` and is
renamed correctly.  These tests pin the parts of that family not already
covered in ``design_failures_test.py`` (Latent #8): reduction names, the
symbol-context path, rename scope-locality, dead-shadow/genuine coexistence,
and the dummy diagnostic.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_reduction_name_variables_build_and_compute(tmp_path):
    """``sum``/``product``/``minval``/``maxval`` as LOCAL variables (the reduction
    half of ``kRejectedIntrinsicNames``) all rename to ``var_<name>`` and compute
    correctly."""
    src = """
MODULE red_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE red(out, a, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a(n)
  REAL(8), INTENT(OUT) :: out(4)
  REAL(8) :: sum, product, minval, maxval
  INTEGER :: i
  sum = 0.0D0
  product = 1.0D0
  minval = a(1)
  maxval = a(1)
  DO i = 1, n
    sum = sum + a(i)
    product = product * a(i)
    IF (a(i) < minval) minval = a(i)
    IF (a(i) > maxval) maxval = a(i)
  END DO
  out(1) = sum
  out(2) = product
  out(3) = minval
  out(4) = maxval
END SUBROUTINE
END MODULE red_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="red", entry="red_mod::red").build()
    n = 4
    a = np.array([2.0, 3.0, 1.0, 4.0], dtype=np.float64)
    out = np.zeros(4, dtype=np.float64, order="F")
    sdfg(out=out, a=np.asfortranarray(a), n=np.int32(n))
    np.testing.assert_allclose(out, [a.sum(), a.prod(), a.min(), a.max()])


def test_multiple_intrinsic_shadows_in_one_scope(tmp_path):
    """``max``, ``sum``, ``abs`` declared in the SAME scope each get an independent
    ``var_<name>`` rename (keyed per-declare, so they don't alias)."""
    src = """
MODULE multi_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE multi(res, a, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a(n)
  REAL(8), INTENT(OUT) :: res(n)
  REAL(8) :: max, sum, abs
  INTEGER :: i
  max = 10.0D0
  abs = 3.0D0
  sum = 0.0D0
  DO i = 1, n
    sum = sum + a(i)
  END DO
  DO i = 1, n
    res(i) = (a(i) + sum) * max - abs
  END DO
END SUBROUTINE
END MODULE multi_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="multi", entry="multi_mod::multi").build()
    n = 4
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    res = np.zeros(n, dtype=np.float64, order="F")
    sdfg(res=res, a=np.asfortranarray(a), n=np.int32(n))
    np.testing.assert_allclose(res, (a + a.sum()) * 10.0 - 3.0)


def test_intrinsic_shadow_in_symbol_contexts(tmp_path):
    """``INTEGER :: max`` as a loop bound, an interstate IF condition, and a
    local-array size -- the symbol/sympy path, distinct from the tasklet-body
    rewriter; the mangling override feeds ``extractName`` so all three render
    ``var_max``."""
    src = """
MODULE sym_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE sym(res, a, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a(n)
  REAL(8), INTENT(OUT) :: res(n)
  INTEGER :: max, i
  REAL(8), ALLOCATABLE :: work(:)
  max = n
  IF (max > 0) THEN
    ALLOCATE(work(max))
    DO i = 1, max
      work(i) = a(i) * REAL(max, 8)
    END DO
    res = work
    DEALLOCATE(work)
  END IF
END SUBROUTINE
END MODULE sym_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="sym", entry="sym_mod::sym").build()
    n = 4
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    res = np.zeros(n, dtype=np.float64, order="F")
    sdfg(res=res, a=np.asfortranarray(a), n=np.int32(n))
    np.testing.assert_allclose(res, a * float(n))


def test_sibling_scope_keeps_genuine_intrinsic(tmp_path):
    """The rename is SCOPE-LOCAL: an inlined callee shadows ``sum`` as a variable
    while the entry scope calls the GENUINE ``SUM(a)``.  Only the callee's declare
    is renamed, so ``SUM(a)`` still reduces -- a too-broad rename would silently
    turn it into a scalar read."""
    src = """
MODULE sib_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE scale(x, m, y)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: m
  REAL(8), INTENT(IN) :: x(m)
  REAL(8), INTENT(OUT) :: y
  REAL(8) :: sum
  INTEGER :: i
  sum = 3.0D0
  y = 0.0D0
  DO i = 1, m
    y = y + sum * x(i)
  END DO
END SUBROUTINE
SUBROUTINE sib(out, a, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a(n)
  REAL(8), INTENT(OUT) :: out(2)
  REAL(8) :: t
  CALL scale(a, n, t)
  out(1) = t
  out(2) = SUM(a)
END SUBROUTINE
END MODULE sib_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="sib", entry="sib_mod::sib").build()
    n = 4
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    out = np.zeros(2, dtype=np.float64, order="F")
    sdfg(out=out, a=np.asfortranarray(a), n=np.int32(n))
    np.testing.assert_allclose(out, [3.0 * a.sum(), a.sum()])


def test_dead_shadow_var_does_not_corrupt_genuine_intrinsic(tmp_path):
    """A DEAD ``REAL(8) :: max`` (declared, never read) beside a genuine ``MAX(a, b)``
    call: flang keeps the unused alloca so the rename still fires, but the
    intrinsic is a separate ``hlfir`` op and renders untouched."""
    src = """
MODULE dead_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE dead(res, a, b, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a(n), b(n)
  REAL(8), INTENT(OUT) :: res(n)
  REAL(8) :: max
  INTEGER :: i
  DO i = 1, n
    res(i) = MAX(a(i), b(i))
  END DO
END SUBROUTINE
END MODULE dead_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="dead", entry="dead_mod::dead").build()
    n = 4
    a = np.array([3.0, 1.0, 4.0, 1.5], dtype=np.float64)
    b = np.array([1.0, 2.0, 1.0, 5.0], dtype=np.float64)
    res = np.zeros(n, dtype=np.float64, order="F")
    sdfg(res=res, a=np.asfortranarray(a), b=np.asfortranarray(b), n=np.int32(n))
    np.testing.assert_allclose(res, np.maximum(a, b))


def test_dummy_named_after_intrinsic_is_clear_error(tmp_path):
    """A DUMMY argument named after an intrinsic is hard-rejected with an actionable
    diagnostic rather than silently renamed (the short name is the user-facing SDFG
    signature arg).  Pins the diagnostic so a future ABI-miscompiling relaxation is caught."""
    src = """
MODULE dummy_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE find_max(max, a, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a(n)
  REAL(8), INTENT(OUT) :: max
  INTEGER :: i
  max = a(1)
  DO i = 2, n
    IF (a(i) > max) max = a(i)
  END DO
END SUBROUTINE
END MODULE dummy_mod
"""
    with pytest.raises(RuntimeError, match=r"collide with bridge-rendered intrinsics"):
        build_sdfg(src, tmp_path / "sdfg", name="find_max", entry="dummy_mod::find_max").build()
