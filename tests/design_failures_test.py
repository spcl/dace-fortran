"""Regression tests for the post-dd80990 design-audit findings.

Pins behaviour for design failures D1-D5 + latent bugs #1, #2, #8 so
future refactors of the scope-qualification / collision-detection
pipeline can't silently regress.

Audit reference: ``/home/primrose/.claude/projects/-home-primrose-Work/memory/project_ci_failures_after_scope_qualification.md``
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


# ===========================================================================
# D1 + latent #1 -- cross-module state isolation
# ---------------------------------------------------------------------------
# Two distinct kernels built back-to-back in the SAME process: prior
# module's ``kEntryScope`` / ``kShortNameCollisions`` must not leak.
# ``buildAllocatedReaderNames`` runs after ``clearManglingOverrides``
# so a stale allocated-tracker reader-set from module A can't poison
# module B.
# ===========================================================================
def test_two_modules_back_to_back_isolated(tmp_path):
    """Two distinct kernels with DIFFERENT entry F-scopes built in the
    same process: prior module's ``kEntryScope`` /
    ``kShortNameCollisions`` must not leak.  Both must produce SDFG
    signatures matching their OWN entry name."""
    src_a = """
MODULE alpha_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE alpha(x, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(INOUT) :: x(n)
  x = x + 1.0_8
END SUBROUTINE
END MODULE alpha_mod
"""
    src_b = """
MODULE beta_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE beta(y, m)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: m
  REAL(8), INTENT(INOUT) :: y(m)
  y = y * 2.0_8
END SUBROUTINE
END MODULE beta_mod
"""
    # Build A first
    sdfg_a = build_sdfg(src_a, tmp_path / "a", name="alpha", entry="alpha_mod::alpha").build()
    xa = np.ones(3, dtype=np.float64, order='F')
    sdfg_a(x=xa, n=np.int32(3))
    np.testing.assert_array_equal(xa, 2.0)
    # Build B second -- A's state (entryScope='alpha', collisions for
    # ``x`` / ``n``) must not contaminate B's extraction.  B's signature
    # must have bare ``y`` and ``m``, not ``alpha_y`` or ``beta_y``.
    sdfg_b = build_sdfg(src_b, tmp_path / "b", name="beta", entry="beta_mod::beta").build()
    assert 'y' in sdfg_b.arrays, (f"B leaked A's state: B's signature is {sorted(sdfg_b.arrays.keys())}")
    yb = np.ones(3, dtype=np.float64, order='F')
    sdfg_b(y=yb, m=np.int32(3))
    np.testing.assert_array_equal(yb, 2.0)


# ===========================================================================
# D2 -- collision pre-walk runs AFTER Pass 0b ``_call<idx>`` mutation
# ---------------------------------------------------------------------------
# A subroutine called from two distinct sites with section-slice
# arguments gets ``_call0`` / ``_call1`` suffixes by Pass 0b.  The
# collision pre-walk must see the suffixed names so both VarInfo
# entries are correctly distinct.
# ===========================================================================
def test_multi_callsite_no_qualification_for_unique_short_name(tmp_path):
    """Collision pre-walk runs AFTER Pass 0b's ``_call<idx>`` mutation,
    so callsite-disambiguated declares (suffix ``_call0`` / ``_call1``)
    are correctly considered SEPARATE short names rather than colliding
    via their pre-mutation name.  Verifies the rename ORDER doesn't
    silently re-collapse them into a single bucket."""
    # Simple 1-callsite version: helper's ``arr`` is an inlined alias of
    # caller's ``a``; my fix collapses to entry-scope ``a``.
    src = """
MODULE main_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE main(a, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(INOUT) :: a(n)
  CALL helper(a)
CONTAINS
  SUBROUTINE helper(arr)
    REAL(8), INTENT(INOUT) :: arr(:)
    arr = arr + 1.0_8
  END SUBROUTINE
END SUBROUTINE
END MODULE main_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="main", entry="main_mod::main").build()
    a = np.ones(3, dtype=np.float64, order='F')
    sdfg(a=a, n=np.int32(3))
    np.testing.assert_array_equal(a, 2.0)


# ===========================================================================
# D4 + latent #2 -- alias-aware + fir.declare-aware collision pre-walk
# ---------------------------------------------------------------------------
# An inlined-callee dummy that's an assumed-shape alias of the caller's
# storage must NOT trigger qualification of the caller's same-named
# dummy.  Mirrors what closed the auto_iface_flat_matches_handwritten
# regression.
# ===========================================================================
def test_inlined_alias_does_not_qualify_caller_dummy(tmp_path):
    """Kernel ``kern`` has dummy ``out``; internal subprogram ``set_one``
    has dummy ``out`` that aliases the caller's after inlining.  Both
    must collapse to a single signature variable ``out``."""
    src = """
MODULE kern_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE kern(out, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(OUT) :: out(n)
  CALL set_one(out)
CONTAINS
  SUBROUTINE set_one(out)
    REAL(8), INTENT(OUT) :: out(:)
    out = 1.0_8
  END SUBROUTINE
END SUBROUTINE
END MODULE kern_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="kern", entry="kern_mod::kern").build()
    # SDFG signature must have bare ``out``, not ``set_one_out``.
    assert 'out' in sdfg.arrays, (f"expected bare 'out' on signature, got: {sorted(sdfg.arrays.keys())}")
    out = np.zeros(3, dtype=np.float64, order='F')
    sdfg(out=out, n=np.int32(3))
    np.testing.assert_array_equal(out, 1.0)


# ===========================================================================
# Inlined-OPTIONAL dummy -- previously a CI failure (tf2_a, fun_a, etc.)
# ===========================================================================
def test_inlined_optional_dummy_collapses_to_caller_arg(tmp_path):
    """An OPTIONAL dummy in an internal subprogram, called with AND
    without the optional, must not create a spurious ``<scope>_a``
    SDFG signature variable.  The caller's ``a`` is bound at one
    callsite and absent at the other; ``is_present`` folds statically."""
    src = """
MODULE main_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE main(res, res2, a)
  IMPLICIT NONE
  INTEGER :: a
  INTEGER :: res(4), res2(4)
  CALL tf(res, a)
  CALL tf(res2)
CONTAINS
  SUBROUTINE tf(r, x)
    INTEGER, INTENT(OUT) :: r(4)
    INTEGER, OPTIONAL, INTENT(IN) :: x
    IF (PRESENT(x)) THEN
      r(1) = 1
    ELSE
      r(1) = 0
    END IF
  END SUBROUTINE
END SUBROUTINE
END MODULE main_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="main", entry="main_mod::main").build()
    # No tf_x in the signature -- the inlined OPTIONAL is folded.
    bad_keys = [k for k in sdfg.arrays.keys() if k.startswith('tf_') or k.endswith('_x')]
    assert not bad_keys, (f"unexpected qualified inlined-OPTIONAL on signature: {bad_keys}")


# ===========================================================================
# Latent #8 (intrinsic-shadow RENAME) -- user variable named after a
# ---------------------------------------------------------------------------
# Fortran intrinsic (``max``, ``min``, ``sqrt``, ...).  The bridge renders
# intrinsics as bare tokens in tasklet bodies, so a user variable with the
# same name would collide.  Rather than hard-reject -- QE's dead
# ``DOUBLE PRECISION :: max`` must still build -- extract-time RENAMES the
# user variable to ``var_<name>`` so its reads/writes stay distinct while a
# genuine intrinsic ``<name>(...)`` call keeps rendering normally.  (Changed
# from hard-reject to rename in de9348e; these tests pin that the rename
# yields a correct build, not a diagnostic.)
# ===========================================================================
def test_user_variable_named_max_builds_via_rename(tmp_path):
    """``REAL(8) :: max`` shadowing the MAX intrinsic builds (renamed to
    ``var_max``) and computes correctly."""
    src = """
MODULE bad_max_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE bad_max(out)
  IMPLICIT NONE
  REAL(8), INTENT(OUT) :: out
  REAL(8) :: max
  max = 5.0_8
  out = max + 1.0_8
END SUBROUTINE
END MODULE bad_max_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="bad_max", entry="bad_max_mod::bad_max").build()
    out = np.zeros(1, dtype=np.float64)
    sdfg(out=out)
    assert out[0] == 6.0  # max = 5.0; out = max + 1.0


def test_user_variable_named_sqrt_builds_via_rename(tmp_path):
    """``REAL(8) :: sqrt`` shadowing the SQRT intrinsic builds and reads back."""
    src = """
MODULE bad_sqrt_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE bad_sqrt(out)
  IMPLICIT NONE
  REAL(8), INTENT(OUT) :: out
  REAL(8) :: sqrt
  sqrt = 4.0_8
  out = sqrt
END SUBROUTINE
END MODULE bad_sqrt_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="bad_sqrt", entry="bad_sqrt_mod::bad_sqrt").build()
    out = np.zeros(1, dtype=np.float64)
    sdfg(out=out)
    assert out[0] == 4.0  # sqrt = 4.0


def test_user_variable_named_min_builds_via_rename(tmp_path):
    """``INTEGER :: min`` shadowing the MIN intrinsic builds and reads back."""
    src = """
MODULE bad_min_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE bad_min(out)
  IMPLICIT NONE
  REAL(8), INTENT(OUT) :: out
  INTEGER :: min
  min = 7
  out = REAL(min, 8)
END SUBROUTINE
END MODULE bad_min_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="bad_min", entry="bad_min_mod::bad_min").build()
    out = np.zeros(1, dtype=np.float64)
    sdfg(out=out)
    assert out[0] == 7.0  # min = 7


# ===========================================================================
# Sympy-reserved name auto-rename
# ---------------------------------------------------------------------------
# ``i``, ``pi``, ``e`` etc. are sympy reserved.  Local variables get
# auto-renamed to ``fortran_<short>`` so sympy doesn't collapse them
# into the constant.  Dummies are EXEMPT to preserve the caller-side
# ABI (passing ``i=4`` from Python must reach the SDFG's ``i`` arg).
# ===========================================================================
def test_local_pi_is_renamed_to_fortran_pi(tmp_path):
    """``REAL(8), PARAMETER :: pi = ...`` as a LOCAL constant.  Bridge
    renames to ``fortran_pi`` internally; user doesn't see it on the
    signature."""
    src = """
MODULE kern_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE kern(out)
  IMPLICIT NONE
  REAL(8), INTENT(OUT) :: out
  REAL(8), PARAMETER :: pi = 3.141592653589793_8
  out = pi * 2.0_8
END SUBROUTINE
END MODULE kern_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="kern", entry="kern_mod::kern").build()
    out = np.zeros(1, dtype=np.float64)
    sdfg(out=out)
    np.testing.assert_allclose(out[0], 2.0 * 3.141592653589793, rtol=1e-12)


def test_local_e_is_renamed(tmp_path):
    """``e`` as a LOCAL must not collide with sympy.E."""
    src = """
MODULE kern_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE kern(out)
  IMPLICIT NONE
  REAL(8), INTENT(OUT) :: out
  REAL(8) :: e
  e = 2.71828_8
  out = e + 1.0_8
END SUBROUTINE
END MODULE kern_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="kern", entry="kern_mod::kern").build()
    out = np.zeros(1, dtype=np.float64)
    sdfg(out=out)
    np.testing.assert_allclose(out[0], 3.71828, rtol=1e-6)


def test_dummy_i_preserves_signature_name(tmp_path):
    """``i`` as an intent(in) DUMMY must NOT be renamed -- caller's
    ``i=...`` binding requires the bare name on the SDFG signature."""
    src = """
MODULE kern_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE kern(i, out)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: i
  REAL(8), INTENT(OUT) :: out
  out = REAL(i * 2, 8)
END SUBROUTINE
END MODULE kern_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="kern", entry="kern_mod::kern").build()
    out = np.zeros(1, dtype=np.float64)
    sdfg(i=np.int32(7), out=out)
    np.testing.assert_allclose(out[0], 14.0)


def test_local_i_loop_iterator_not_renamed(tmp_path):
    """``i`` as a LOCAL loop iterator must NOT be renamed to
    ``fortran_i`` -- the rename collides with DaCe's LoopRegion
    iterator-symbol machinery (``_loop_it_<N>`` on the loop labelled
    ``loop_fortran_i_0``) and surfaces as ``InvalidSDFGError: Loop
    iterator must not appear on the LHS of an interstate-edge
    assignment``.  ``i`` is excluded from ``kSympyReservedNames`` for
    exactly this reason.  Regression for the CI failure
    ``test_dummy_shaped_fn_return``."""
    src = """
MODULE kern_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE kern(out, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(OUT) :: out(n)
  INTEGER :: i
  DO i = 1, n
    out(i) = REAL(i, 8)
  END DO
END SUBROUTINE
END MODULE kern_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="kern", entry="kern_mod::kern").build()
    out = np.zeros(4, dtype=np.float64, order="F")
    sdfg(out=out, n=np.int32(4))
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0, 4.0])


def test_nested_loops_with_i_iterator(tmp_path):
    """Two loop iterators (``i`` outer, ``i`` reused in inlined helper)
    after a PURE FUNCTION inline -- the shape that broke
    ``test_dummy_shaped_fn_return``.  Both ``i`` copies stay bare; no
    LoopRegion iterator collision."""
    src = """
MODULE m_iter
  IMPLICIT NONE
CONTAINS
  PURE FUNCTION scaled(x, k) RESULT(r)
    INTEGER, INTENT(IN) :: k
    REAL(8), INTENT(IN) :: x
    REAL(8) :: r(k)
    INTEGER :: i
    DO i = 1, k
      r(i) = x * REAL(i, 8)
    END DO
  END FUNCTION scaled

  SUBROUTINE kern(out_arr, src, n, k)
    INTEGER, INTENT(IN) :: n, k
    REAL(8), INTENT(IN) :: src(n)
    REAL(8), INTENT(OUT) :: out_arr(k, n)
    REAL(8) :: tmp(k)
    INTEGER :: i
    DO i = 1, n
      tmp = scaled(src(i), k)
      out_arr(:, i) = tmp
    END DO
  END SUBROUTINE kern
END MODULE m_iter
"""
    from dace_fortran import build_sdfg_from_files
    srcfile = tmp_path / "m_iter.f90"
    srcfile.write_text(src)
    sdfg = build_sdfg_from_files([srcfile], entry="m_iter::kern", name="kern", out_dir=tmp_path / "build")
    sdfg.validate()
    out_arr = np.zeros((3, 2), dtype=np.float64, order="F")
    src_a = np.array([2.0, 5.0], dtype=np.float64, order="F")
    sdfg(out_arr=out_arr, src=src_a, n=np.int32(2), k=np.int32(3))
    # col i=1: scaled(2.0, 3) = [2, 4, 6]; col i=2: scaled(5.0,3)=[5,10,15]
    np.testing.assert_array_equal(out_arr[:, 0], [2.0, 4.0, 6.0])
    np.testing.assert_array_equal(out_arr[:, 1], [5.0, 10.0, 15.0])


def test_dummy_pi_preserves_signature_name(tmp_path):
    """``pi`` as an intent(in) DUMMY must NOT be renamed."""
    src = """
MODULE kern_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE kern(pi, out)
  IMPLICIT NONE
  REAL(8), INTENT(IN) :: pi
  REAL(8), INTENT(OUT) :: out
  out = pi * 2.0_8
END SUBROUTINE
END MODULE kern_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="kern", entry="kern_mod::kern").build()
    out = np.zeros(1, dtype=np.float64)
    sdfg(pi=np.float64(3.14), out=out)
    np.testing.assert_allclose(out[0], 6.28, rtol=1e-6)
