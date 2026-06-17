"""Reproducers for the remaining QE ``vexx_bp_k_gpu`` bring-up gates:
pointer/allocatable struct-member ARRAY indexed reads and inlined
FUNCTION dummies bound to array sections.

Each kernel is reduced from QE ``vcut_get`` (coulomb_vcut_module):

    FUNCTION vcut_get(vcut, q) RESULT(res)
      REAL(8), POINTER :: vcut % corrected(:, :, :)
      INTEGER :: i(3)
      i = NINT(...)
      IF (...) THEN ... ELSE
        IF (i(1) > UBOUND(vcut % corrected, 1) ...) ...
        res = vcut % corrected(i(1), i(2), i(3))
      END IF
    END FUNCTION
    ! called as: fac(ig) = vcut_get(vcut, q(:, ig))

These currently FAIL (free-symbol / unresolved-read) and pin the fixes:
walking ``expandDesignateChain`` / ``buildExpr`` through the pointer-box
``fir.load`` between ``designate{component}`` and ``designate(indices)``,
and section-aliasing an inlined FUNCTION-RESULT section dummy.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_pointer_member_array_indexed_by_local(tmp_path):
    """MODULE-scope derived-type var with a POINTER 3-D array member,
    read indexed by a LOCAL index array (faithful to QE ``vcut`` which
    is a module variable).  Isolates the
    ``designate{component} -> load(box) -> designate(i(1),i(2),i(3))``
    chain: the read currently fails to wire (``_in_vcut_corrected_0``
    free symbol) because ``expandDesignateChain`` / ``buildExpr`` drop
    the element indices when the component sits on the PARENT designate
    and the pointer-box ``fir.load`` separates it from the element
    designate."""
    src = """
MODULE m
  TYPE :: tbl
    REAL(8), POINTER :: corrected(:, :, :)
  END TYPE
  TYPE(tbl) :: vcut
CONTAINS
  SUBROUTINE run(j1, j2, j3, res)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: j1, j2, j3
    REAL(8), INTENT(OUT) :: res
    INTEGER :: i(3)
    i(1) = j1
    i(2) = j2
    i(3) = j3
    res = vcut % corrected(i(1), i(2), i(3))
  END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="run", entry="m::run").build()
    nx, ny, nz = 3, 4, 5
    corrected = np.asfortranarray(
        np.arange(nx * ny * nz, dtype=np.float64).reshape((nx, ny, nz)))
    res = np.zeros(1, dtype=np.float64)
    # Fortran 1-based (2,3,4) -> 0-based (1,2,3)
    sdfg(vcut_corrected=corrected, j1=np.int32(2), j2=np.int32(3),
         j3=np.int32(4), res=res)
    np.testing.assert_allclose(res[0], corrected[1, 2, 3])


def test_pointer_member_indexed_inlined_function(tmp_path):
    """Faithful ``vcut_get`` shape: a RESULT function with a pointer
    struct-member 3-D read, a bounds-check IF, and a section-arg call
    site ``fac(ig) = getv(t, q(:, ig))``."""
    src = """
MODULE m
  TYPE :: tbl
    REAL(8), POINTER :: corrected(:, :, :)
    REAL(8) :: cutoff
  END TYPE
CONTAINS
  FUNCTION getv(t, q) RESULT(res)
    TYPE(tbl), INTENT(IN) :: t
    REAL(8), INTENT(IN) :: q(3)
    REAL(8) :: res
    INTEGER :: i(3)
    i = NINT(q)
    IF (SUM(q ** 2) > t % cutoff ** 2) THEN
      res = 1.0D0 / SUM(q ** 2)
    ELSE
      res = t % corrected(i(1), i(2), i(3))
    END IF
  END FUNCTION

  SUBROUTINE run(t, q, fac, ng)
    IMPLICIT NONE
    TYPE(tbl), INTENT(IN) :: t
    INTEGER, INTENT(IN) :: ng
    REAL(8), INTENT(IN) :: q(3, ng)
    REAL(8), INTENT(OUT) :: fac(ng)
    INTEGER :: ig
    DO ig = 1, ng
      fac(ig) = getv(t, q(:, ig))
    END DO
  END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="run", entry="m::run").build()
    nx, ny, nz = 3, 3, 3
    # ``asfortranarray`` of the reshape, not ``reshape(order="F")`` -- the
    # latter returns a non-owning view, which DaCe rejects as a program
    # argument (analyzability).  Mirrors the sibling test above.
    corrected = np.asfortranarray(
        np.arange(nx * ny * nz, dtype=np.float64).reshape((nx, ny, nz)))
    ng = 2
    # q chosen so SUM(q**2) <= cutoff**2 (take the ELSE branch) and
    # NINT(q) lands inside [1, n].
    q = np.asarray([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=np.float64, order="F")
    fac = np.zeros(ng, dtype=np.float64, order="F")
    # ``t_cutoff`` (the flattened scalar member ``t%cutoff``) is exposed as
    # a (1,)-Array on the SDFG surface, so pass a 1-element array, not a
    # bare Python float.
    t_cutoff = np.array([100.0], dtype=np.float64)
    sdfg(t_corrected=corrected, t_cutoff=t_cutoff, q=q, fac=fac, ng=np.int32(ng))
    expected = np.array([corrected[0, 0, 0], corrected[1, 1, 1]])
    np.testing.assert_allclose(fac, expected)


def test_aor_module_member_array_rank(tmp_path):
    """MODULE-scope ALLOCATABLE array-of-records with an array member,
    indexed ``tabxx(ia) % box(ir)`` (faithful to QE ``tabxx`` / ``ke``).
    The flattened member ``tabxx_box`` must be rank 2 -- record dim
    PREPENDED to the member's own dim -- else the memlet carries more
    dims than the descriptor and ``offset_tabxx_box_d1`` leaks."""
    src = """
MODULE m
  TYPE :: rec
    REAL(8), ALLOCATABLE :: box(:)
  END TYPE
  TYPE(rec), ALLOCATABLE :: tabxx(:)
CONTAINS
  SUBROUTINE run(ia, ir, res)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: ia, ir
    REAL(8), INTENT(OUT) :: res
    res = tabxx(ia) % box(ir)
  END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="run", entry="m::run").build()
    assert "tabxx_box" in sdfg.arrays
    # record dim + member dim
    assert len(sdfg.arrays["tabxx_box"].shape) == 2
