"""ELEMENTAL procedures -> loop-over-array + scalar-body tasklet.

After ``hlfir-inline-all`` splices an elemental callee's body into the ``fir.do_loop``,
``hlfir-fold-element-aliases`` erases the element-scoped alias declares so the SDFG builder
sees plain indexed access into the outer array. Covers both the subroutine (inout) and
function (``hlfir.elemental``) forms. References are NumPy, not f2py -- module-contained
elemental parsing is shaky there.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_elemental_subroutine_with_inout(tmp_path: Path):
    """Elemental subroutine with inout scalars on three arrays: inline-all + fold-element-aliases
    collapse the ``fir.do_loop`` + ``fir.call`` body into indexed updates on the outer arrays."""
    src = """
module apply_delta_mod
contains
subroutine apply_delta(od, scat_od, g)
  implicit none
  real(8), intent(inout) :: od(14), scat_od(14), g(14)
  call delta(od, scat_od, g)
contains
  elemental subroutine delta(a, b, c)
    real(8), intent(inout) :: a, b, c
    real(8) :: f
    f = c * c
    a = a - b * f
    b = b * (1.0d0 - f)
    c = c / (1.0d0 + c)
  end subroutine delta
end subroutine apply_delta
end module apply_delta_mod
"""
    sdfg = build_sdfg(src, tmp_path, name="apply_delta", entry="apply_delta_mod::apply_delta").build()

    rng = np.random.default_rng(0)
    od = np.asfortranarray(rng.random(14, dtype=np.float64))
    scat_od = np.asfortranarray(rng.random(14, dtype=np.float64))
    g = np.asfortranarray(rng.random(14, dtype=np.float64))

    od_ref, sod_ref, g_ref = od.copy(), scat_od.copy(), g.copy()
    f_ref = g_ref * g_ref
    od_ref = od_ref - sod_ref * f_ref
    sod_ref = sod_ref * (1.0 - f_ref)
    g_ref = g_ref / (1.0 + g_ref)

    sdfg(od=od, scat_od=scat_od, g=g)

    np.testing.assert_allclose(od, od_ref, atol=1e-12, rtol=0)
    np.testing.assert_allclose(scat_od, sod_ref, atol=1e-12, rtol=0)
    np.testing.assert_allclose(g, g_ref, atol=1e-12, rtol=0)


def test_elemental_function_via_hlfir_elemental(tmp_path: Path):
    """Pointwise array expression: flang wraps the RHS in ``hlfir.elemental`` +
    ``hlfir.yield_element``, the shape ``buildElementalAssign`` consumes. Guards that
    ``FoldElementAliases`` (targets inlined-scalar-body aliases) leaves this path untouched."""
    src = """
module apply_square_shift_mod
contains
subroutine apply_square_shift(x, y, n)
  implicit none
  integer, intent(in)  :: n
  real(8), intent(in)  :: x(n)
  real(8), intent(out) :: y(n)
  y = x * x - 1.0d0
end subroutine apply_square_shift
end module apply_square_shift_mod
"""
    sdfg = build_sdfg(src, tmp_path, name="apply_square_shift",
                      entry="apply_square_shift_mod::apply_square_shift").build()

    rng = np.random.default_rng(1)
    n = 32
    x = np.asfortranarray(rng.standard_normal(n, dtype=np.float64))
    y = np.zeros(n, dtype=np.float64, order="F")

    sdfg(x=x, y=y, n=n)

    np.testing.assert_allclose(y, x * x - 1.0, atol=1e-12, rtol=0)


def test_fold_element_aliases_drops_inlined_declares(tmp_path: Path):
    """Inlined elemental body must not leak its Fortran-named scalar as a separate SDFG array
    (pre-FoldElementAliases, callee dummies showed up as stray scalars on the SDFG argslist)."""
    src = """
module driver_mod
contains
subroutine driver(x)
  implicit none
  real(8), intent(inout) :: x(8)
  call doubler(x)
contains
  elemental subroutine doubler(v)
    real(8), intent(inout) :: v
    v = v * 2.0d0
  end subroutine doubler
end subroutine driver
end module driver_mod
"""
    b = build_sdfg(src, tmp_path, name="driver", entry="driver_mod::driver")
    sdfg = b.build()

    # inlined callee's scalar v must NOT show up as its own array
    assert "v" not in b.arrays, \
        f"elemental inner dummy 'v' leaked into arrays: {list(b.arrays.keys())}"
    assert "x" in sdfg.arrays, list(sdfg.arrays.keys())


def test_elemental_body_with_intrinsic(tmp_path: Path):
    """Elemental scalar body invokes ``exp``: must survive inline-all + FoldElementAliases and land in the tasklet as a Python call."""
    src = """
module apply_soft_mod
contains
subroutine apply_soft(x, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(inout) :: x(n)
  call soft(x)
contains
  elemental subroutine soft(v)
    real(8), intent(inout) :: v
    v = exp(v) - 1.0d0
  end subroutine soft
end subroutine apply_soft
end module apply_soft_mod
"""
    sdfg = build_sdfg(src, tmp_path, name="apply_soft", entry="apply_soft_mod::apply_soft").build()

    rng = np.random.default_rng(2)
    n = 16
    x = np.asfortranarray(rng.random(n, dtype=np.float64))
    x_ref = np.exp(x) - 1.0

    sdfg(x=x, n=n)

    np.testing.assert_allclose(x, x_ref, atol=1e-12, rtol=0)


def test_elemental_subroutine_relu(tmp_path: Path):
    """Elemental ReLU via if/else: exercises conditional control flow inside the inlined scalar body."""
    src = """
module apply_relu_mod
contains
subroutine apply_relu(x, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(inout) :: x(n)
  call relu(x)
contains
  elemental subroutine relu(v)
    real(8), intent(inout) :: v
    if (v <= 0.0d0) v = 0.0d0
  end subroutine relu
end subroutine apply_relu
end module apply_relu_mod
"""
    sdfg = build_sdfg(src, tmp_path, name="apply_relu", entry="apply_relu_mod::apply_relu").build()

    rng = np.random.default_rng(3)
    n = 32
    x = np.asfortranarray(rng.standard_normal(n, dtype=np.float64))
    x_ref = np.maximum(x, 0.0)

    sdfg(x=x, n=n)

    np.testing.assert_allclose(x, x_ref, atol=1e-12, rtol=0)


def test_elemental_subroutine_softmax_step(tmp_path: Path):
    """Two-statement elemental body (``t = exp(x); x = t / s``) where the second reads the
    first's write. Drives RAW-hazard serialisation in ``emit_loop``'s child-assigns path --
    without a fresh state per statement, the tasklets would race.
    """
    src = """
module apply_softmax_step_mod
contains
subroutine apply_softmax_step(x, s, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: s
  real(8), intent(inout) :: x(n)
  call smstep(x, s)
contains
  elemental subroutine smstep(v, norm)
    real(8), intent(inout) :: v
    real(8), intent(in)    :: norm
    real(8) :: t
    t = exp(v)
    v = t / norm
  end subroutine smstep
end subroutine apply_softmax_step
end module apply_softmax_step_mod
"""
    sdfg = build_sdfg(src, tmp_path, name="apply_softmax_step",
                      entry="apply_softmax_step_mod::apply_softmax_step").build()

    rng = np.random.default_rng(4)
    n = 16
    x = np.asfortranarray(rng.standard_normal(n, dtype=np.float64))
    s_val = float(np.sum(np.exp(x)))
    x_ref = np.exp(x) / s_val

    sdfg(x=x, s=s_val, n=n)

    np.testing.assert_allclose(x, x_ref, atol=1e-12, rtol=0)
