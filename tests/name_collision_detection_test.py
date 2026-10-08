"""Verify the bridge's name-collision detection between builder.arrays/scalars/symbols.

A name appearing as multiple VarInfos with different roles (e.g. an ARRAY
block-arg whose name matches a SCALAR dummy of an inlined helper) used to leak
into both dicts, causing a spurious 1D memlet on top of the correct ND one --
graupel's ``InvalidSDFGEdgeError: Memlet subset does not match node dimension
(expected 2, got 1)``. Two-layer defense: (1) ARRAY wins over SCALAR over
SYMBOL on collision; (2) RuntimeError at builder-init if the three role-keyed
dicts aren't disjoint, caught at extract time instead of 200 states later.
"""

from pathlib import Path

import numpy as np

from tests._util import build_sdfg


def test_inlined_callee_scalar_shadowing_outer_array(tmp_path):
    """Reproduces the graupel pattern: outer 2D array ``arr`` + an inlined helper's
    scalar ``arr`` dummy collide post-inline; bridge must classify as ARRAY
    (rank>0 wins), not duplicate across role dicts."""
    src = """
module m
contains
  subroutine helper(arr)
    real(kind=8), intent(inout) :: arr   ! scalar dummy with name 'arr'
    arr = arr * 2.0d0
  end subroutine
  subroutine outer(arr, out)
    real(kind=8), intent(inout) :: arr(3, 3)   ! 2D array with name 'arr'
    real(kind=8), intent(out) :: out
    real(kind=8) :: tmp
    tmp = arr(1, 1)
    call helper(tmp)
    out = tmp + arr(2, 2)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="outer", entry="m::outer").build()
    # arr must be in arrays (2D) not scalars -- inlined callee's collision loses to the outer array
    assert "arr" in sdfg.arrays
    assert tuple(int(s) for s in sdfg.arrays["arr"].shape) == (3, 3)
    # spurious 1D arr[0] memlet must not appear, else validate() raises InvalidSDFGEdgeError
    sdfg.validate()


def test_collision_loud_fail_is_loud(tmp_path):
    """Role-keyed dicts must stay disjoint after de-collision (builder init raises
    RuntimeError if not). Can't easily force the failure without a synthetic VarInfo
    list, so this just smoke-tests no false-positive raise on a collision-free kernel."""
    src = """
module m
contains
  subroutine f(a, b, out)
    real(kind=8), intent(in) :: a(3), b(3)
    real(kind=8), intent(out) :: out
    out = sum(a * b)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f", entry="m::f").build()
    # No exception -> no false-positive collision.
    assert "a" in sdfg.arrays
    assert "b" in sdfg.arrays


def test_inlined_callee_local_array_shadowing_module_array(tmp_path):
    """QE h_psi: ``grid_set``'s local ``g(3)`` and ``gvect``'s module ``g(:, :)`` share a name once
    ``grid_set`` is inlined; each write must land on its own array (``g(1) = ...`` is not a 1-D write
    into the 2-D module array, which surfaces in the signature as module globals do)."""
    src = """
module gvect
  real(kind=8) :: g(3, 2)
end module
module m
contains
  subroutine grid_set(bg, gsq)
    real(kind=8), intent(in) :: bg(3)
    real(kind=8), intent(out) :: gsq
    real(kind=8) :: g(3)
    g(1) = 2.0d0 * bg(1)
    g(2) = 2.0d0 * bg(2)
    g(3) = 2.0d0 * bg(3)
    gsq = g(1) ** 2 + g(2) ** 2 + g(3) ** 2
  end subroutine
  subroutine outer(bg, gsq, s)
    use gvect, only: g
    real(kind=8), intent(in) :: bg(3)
    real(kind=8), intent(out) :: gsq, s
    g = 1.0d0
    call grid_set(bg, gsq)
    s = sum(g)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="outer", entry="m::outer").build()
    sdfg.validate()
    bg = np.array([1.0, 2.0, 3.0])
    gsq = np.zeros(1)
    s = np.zeros(1)
    g = np.zeros((3, 2), order="F")
    sdfg(bg=bg, gsq=gsq, s=s, g=g)
    assert gsq[0] == 4.0 * 14.0
    assert s[0] == 6.0
    assert np.all(g == 1.0)


if __name__ == "__main__":
    import tempfile

    for test in (
        test_inlined_callee_scalar_shadowing_outer_array,
        test_collision_loud_fail_is_loud,
        test_inlined_callee_local_array_shadowing_module_array,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            test(Path(tmp))
