"""Runtime-shape variants of the static struct-flatten e2e tests (``flatten_structs_test.py``,
``array_of_records_test.py``): same scenarios with RUNTIME-sized outer arrays, so the
flattener must thread live extents through the companion alloca/shape
(``solve_free_sfc_ab_mimetic`` blocker #3 fix -- ``outerExtentValues`` +
``makeCompanionAlloca`` in ``FlattenStructs.cpp``) instead of baking static literals.
Each checks a closed form so a wrong companion shape shows up as a numeric mismatch.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# ---------------------------------------------------------------------------
# Local runtime-sized AoS with SCALAR members -> per-member SoA companions (non-concat
# dynamic path: ``rewrapWith`` produces box result#0, alloca ``array<?x?xf64>``).
# Static counterpart: ``array_of_records_test.py::test_aor_l1_static_scalar_member``.
# ---------------------------------------------------------------------------
_SCALAR_MEMBER_SRC = """
module lib
  implicit none
  type pt
    real(8) :: a
    real(8) :: b
  end type pt
contains
  subroutine kern(n, nb, src, out)
    integer, intent(in) :: n, nb
    real(8), intent(in) :: src(n, nb)
    real(8), intent(out) :: out(n, nb)
    type(pt) :: p(n, nb)          ! LOCAL, runtime-sized
    integer :: i, jb
    do jb = 1, nb
      do i = 1, n
        p(i, jb) % a = src(i, jb)
        p(i, jb) % b = 10.0d0 * src(i, jb)
      end do
    end do
    do jb = 1, nb
      do i = 1, n
        out(i, jb) = p(i, jb) % a + p(i, jb) % b
      end do
    end do
  end subroutine kern
end module lib
"""


def test_runtime_local_aos_scalar_members(tmp_path: Path):
    sdfg = build_sdfg(_SCALAR_MEMBER_SRC, tmp_path / "sdfg", name="kern", entry="lib::kern").build()
    sdfg.validate()
    for comp in ("p_a", "p_b"):
        assert comp in sdfg.arrays, f"missing companion {comp}; arrays={sorted(sdfg.arrays)}"
        assert len(sdfg.arrays[comp].shape) == 2, sdfg.arrays[comp].shape

    n, nb = 4, 3
    rng = np.random.default_rng(1)
    src = np.asfortranarray(rng.standard_normal((n, nb)))
    out = np.asfortranarray(np.zeros((n, nb)))
    sdfg(src=src, out=out, n=np.int32(n), nb=np.int32(nb), src_d0=n, src_d1=nb, out_d0=n, out_d1=nb)
    np.testing.assert_allclose(out, 11.0 * src, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Local runtime-sized AoS with a scalar member AND a static-array member -- exercises the
# non-concat (scalar) and concat (array) branches of the same ``splitLocal`` pass.
# ---------------------------------------------------------------------------
_MIXED_MEMBER_SRC = """
module lib
  implicit none
  type mix
    real(8) :: s
    real(8) :: v(3)
  end type mix
contains
  subroutine kern(n, nb, src, out)
    integer, intent(in) :: n, nb
    real(8), intent(in) :: src(n, nb)
    real(8), intent(out) :: out(n, nb)
    type(mix) :: p(n, nb)
    integer :: i, jb
    do jb = 1, nb
      do i = 1, n
        p(i, jb) % s = src(i, jb)
        p(i, jb) % v(1) = src(i, jb)
        p(i, jb) % v(2) = 2.0d0 * src(i, jb)
        p(i, jb) % v(3) = 3.0d0 * src(i, jb)
      end do
    end do
    do jb = 1, nb
      do i = 1, n
        out(i, jb) = p(i, jb) % s + p(i, jb) % v(1) + p(i, jb) % v(2) + p(i, jb) % v(3)
      end do
    end do
  end subroutine kern
end module lib
"""


def test_runtime_local_aos_mixed_scalar_and_array_members(tmp_path: Path):
    sdfg = build_sdfg(_MIXED_MEMBER_SRC, tmp_path / "sdfg", name="kern", entry="lib::kern").build()
    sdfg.validate()
    assert "p_s" in sdfg.arrays and "p_v" in sdfg.arrays, sorted(sdfg.arrays)
    assert len(sdfg.arrays["p_s"].shape) == 2, sdfg.arrays["p_s"].shape
    assert len(sdfg.arrays["p_v"].shape) == 3 and int(sdfg.arrays["p_v"].shape[-1]) == 3

    n, nb = 5, 2
    rng = np.random.default_rng(2)
    src = np.asfortranarray(rng.standard_normal((n, nb)))
    out = np.asfortranarray(np.zeros((n, nb)))
    sdfg(src=src, out=out, n=np.int32(n), nb=np.int32(nb), src_d0=n, src_d1=nb, out_d0=n, out_d1=nb)
    # s + (v1+v2+v3) = src + 6*src = 7*src
    np.testing.assert_allclose(out, 7.0 * src, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Runtime-dim mirror of ``flatten_structs_test.py::test_inlined_multidim_aos_array_member_flattens_to_soa``
# -- canonical cartesian ``x(3)`` dummy flattened through an inlined callee, outer extents at runtime.
# ---------------------------------------------------------------------------
_DUMMY_CARTESIAN_RUNTIME_SRC = """
module lib
  implicit none
  type cc
    real(8) :: x(3)
  end type cc
contains
  subroutine inner(n1, n2, n3, field, out)
    integer, intent(in) :: n1, n2, n3
    type(cc), intent(in) :: field(n1, n2, n3)
    real(8), intent(out) :: out
    out = dot_product(field(2, 1, 3) % x, field(2, 1, 3) % x)
  end subroutine inner

  subroutine driver(n1, n2, n3, field, out)
    integer, intent(in) :: n1, n2, n3
    type(cc), intent(in) :: field(n1, n2, n3)
    real(8), intent(out) :: out
    call inner(n1, n2, n3, field, out)
  end subroutine driver
end module lib
"""


def test_runtime_dummy_cartesian_aos_flattens_to_soa(tmp_path: Path):
    sdfg = build_sdfg(_DUMMY_CARTESIAN_RUNTIME_SRC, tmp_path / "sdfg", name="driver", entry="lib::driver").build()
    sdfg.validate()
    assert "field_x" in sdfg.arrays, f"missing SoA companion; arrays={sorted(sdfg.arrays)}"
    assert int(sdfg.arrays["field_x"].shape[-1]) == 3, sdfg.arrays["field_x"].shape

    n1, n2, n3 = 4, 2, 5
    field_x = np.zeros((n1, n2, n3, 3), order="F", dtype=np.float64)
    field_x[1, 0, 2, :] = [3.0, 4.0, 12.0]
    out = np.zeros(1, dtype=np.float64)
    sdfg(field_x=field_x,
         out=out,
         n1=np.int32(n1),
         n2=np.int32(n2),
         n3=np.int32(n3),
         field_x_d0=n1,
         field_x_d1=n2,
         field_x_d2=n3)
    assert abs(out[0] - 169.0) < 1e-12, out[0]


# ---------------------------------------------------------------------------
# Local runtime-sized cartesian AoS initialised via a WHOLE-ARRAY SECTION of a component
# element passed to an inlined callee (ICON-O ``CALL init_zero_3d(z_adv_u_i(:,:,:)%x(1))``
# shape) -- the ocean ``solve_free_sfc`` pattern: section-of-component composes onto the
# rank-4 companion ``z_x(:,:,:,1)``, which must register as an SDFG transient.
# ---------------------------------------------------------------------------
_SECTION_ARG_SRC = """
module lib
  implicit none
  type cc
    real(8) :: x(3)
  end type cc
contains
  subroutine zero3d(a)
    real(8), intent(out) :: a(:, :, :)
    a = 0.0d0
  end subroutine zero3d

  subroutine kern(n, m, k, out)
    integer, intent(in) :: n, m, k
    real(8), intent(out) :: out(n, m, k)
    type(cc) :: z(n, m, k)
    integer :: i, j, l
    call zero3d(z(:, :, :) % x(1))      ! whole-array section of component x(1)
    do l = 1, k
      do j = 1, m
        do i = 1, n
          z(i, j, l) % x(1) = real(i + j + l, 8)
        end do
      end do
    end do
    do l = 1, k
      do j = 1, m
        do i = 1, n
          out(i, j, l) = z(i, j, l) % x(1)
        end do
      end do
    end do
  end subroutine kern
end module lib
"""


def test_runtime_local_cartesian_section_arg(tmp_path: Path):
    sdfg = build_sdfg(_SECTION_ARG_SRC, tmp_path / "sdfg", name="kern", entry="lib::kern").build()
    sdfg.validate()
    assert "z_x" in sdfg.arrays, f"missing companion z_x; arrays={sorted(sdfg.arrays)}"

    n, m, k = 3, 2, 4
    out = np.asfortranarray(np.zeros((n, m, k)))
    sdfg(out=out, n=np.int32(n), m=np.int32(m), k=np.int32(k), out_d0=n, out_d1=m, out_d2=k)
    expected = np.fromfunction(lambda i, j, l: (i + 1) + (j + 1) + (l + 1), (n, m, k), dtype=np.float64)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Two routines with a SAME-NAMED local cartesian AoS of DIFFERENT rank, both inlined into
# one driver (ICON-O ``veloc_adv_vert_mimetic_div``/``_rot`` shape). Exercises the
# multi-scope flattened-companion naming path (``flatCompanionName``/``traceToDecl``): ``z_x``
# lands in two inlined scopes and must resolve the same way everywhere.
# NOTE: does not itself trigger the asymmetric-collision KeyError that motivated the fix
# (needs the real ocean's copy_in-kept-alive base) -- that e2e case is validated against
# ``solve_free_sfc``. Kept as a regression guard for the naming path only.
# ---------------------------------------------------------------------------
_MULTISCOPE_COMPANION_SRC = """
module lib
  implicit none
  type cc
    real(8) :: x(3)
  end type cc
contains
  subroutine zero2d(a)
    real(8), intent(out) :: a(:, :)
    a = 0.0d0
  end subroutine zero2d

  subroutine rotk(n, m, acc)
    integer, intent(in) :: n, m
    real(8), intent(inout) :: acc(n, m)
    type(cc) :: z(n, m)
    integer :: i, j
    call zero2d(z(:, :) % x(1))
    do j = 1, m
      do i = 1, n
        z(i, j) % x(1) = real(i + j, 8)
      end do
    end do
    do j = 1, m
      do i = 1, n
        acc(i, j) = acc(i, j) + z(i, j) % x(1)
      end do
    end do
  end subroutine rotk

  subroutine divk(n, m, k, acc)
    integer, intent(in) :: n, m, k
    real(8), intent(inout) :: acc(n, m)
    type(cc) :: z(n, m, k)
    integer :: i, j, l
    do l = 1, k
      do j = 1, m
        do i = 1, n
          z(i, j, l) % x(1) = real(i + j + l, 8)
        end do
      end do
    end do
    do j = 1, m
      do i = 1, n
        acc(i, j) = acc(i, j) + z(i, j, 1) % x(1)
      end do
    end do
  end subroutine divk

  subroutine driver(n, m, k, acc)
    integer, intent(in) :: n, m, k
    real(8), intent(inout) :: acc(n, m)
    call rotk(n, m, acc)
    call divk(n, m, k, acc)
  end subroutine driver
end module lib
"""


def test_multiscope_samename_cartesian_companions(tmp_path: Path):
    sdfg = build_sdfg(_MULTISCOPE_COMPANION_SRC, tmp_path / "sdfg", name="driver", entry="lib::driver").build()
    sdfg.validate()

    n, m, k = 3, 2, 4
    acc = np.asfortranarray(np.zeros((n, m)))
    sdfg(acc=acc, n=np.int32(n), m=np.int32(m), k=np.int32(k), acc_d0=n, acc_d1=m)
    # rotk: acc += (i+j); divk: acc += z(i,j,1)%x(1) = (i+j+1)  ->  acc = 2(i+j)+1
    expected = np.fromfunction(lambda i, j: 2.0 * ((i + 1) + (j + 1)) + 1.0, (n, m), dtype=np.float64)
    np.testing.assert_allclose(acc, expected, rtol=1e-12, atol=1e-12)
