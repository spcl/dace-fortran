"""End-to-end tests for ``splitDoubleBufferMembers`` in
``hlfir-flatten-structs``.

The split fires on a struct DUMMY argument with an alloc-or-pointer-array-
of-records member accessed by stable scalar-symbol indices (the ICON
``s%prog(nnow)`` / ``s%prog(nnew)`` two-time-level pattern).  Each
``(member, idx_sym)`` pair becomes a fresh per-symbol dummy of the
record-element type; the existing scalar-struct flatten then expands the
inner members into plain companion arrays.

For each test, the kernel under test takes the struct as a dummy
argument.  A neighbouring ``wrapper`` subroutine (compiled into the same
module, but exposed to f2py via ``only=``) builds the AoR in Fortran,
calls the kernel, and writes the result back to flat arrays the Python
side can consume.  The SDFG is built for the kernel; its arglist is the
per-time-level companion arrays.  Outputs are compared to the f2py
reference produced by ``wrapper``.

The inner record members use STATIC shape (``real :: w(2, 3)`` instead
of ``real, allocatable :: w(:, :)``) so the companion's shape and offset
are baked into the SDFG.  Dynamic-shape inner members require the
bindings layer to marshal the descriptor's shape/offset symbols at call
time, which is exercised separately by ``external_aos_test.py``.
"""

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_dbuf_split_simple(tmp_path):
    """Minimal end-to-end: single struct dummy, single inner array
    member, two time levels.  The split must produce ``s_prog_nnow_w``
    and ``s_prog_nnew_w`` as plain rank-2 array dummies on the SDFG."""
    src = """
module dbuf_simple_mod
  implicit none
  type t_prog
    real(kind=8) :: w(2, 3)
  end type t_prog
  type t_state
    type(t_prog), allocatable :: prog(:)
  end type t_state
contains
  subroutine kernel(s, nnow, nnew, out)
    type(t_state), intent(inout) :: s
    integer, intent(in)          :: nnow, nnew
    real(kind=8), intent(inout)  :: out(2, 3)
    integer :: i, j
    do j = 1, 3
      do i = 1, 2
        s%prog(nnew)%w(i, j) = s%prog(nnow)%w(i, j) * 2.0d0
        out(i, j) = s%prog(nnew)%w(i, j)
      end do
    end do
  end subroutine
end module

subroutine wrapper(w_now, w_new, out, nnow, nnew)
  use dbuf_simple_mod
  implicit none
  real(kind=8), intent(inout) :: w_now(2, 3), w_new(2, 3)
  real(kind=8), intent(inout) :: out(2, 3)
  integer, intent(in)         :: nnow, nnew
  type(t_state) :: s
  allocate(s%prog(2))
  s%prog(1)%w = w_now
  s%prog(2)%w = w_new
  call kernel(s, nnow, nnew, out)
  w_now = s%prog(1)%w
  w_new = s%prog(2)%w
  deallocate(s%prog)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path / 'sdfg', name='kernel', entry='kernel').build()

    # Split-produced companions must appear; the original struct dummy
    # is gone.
    assert 's_prog_nnow_w' in sdfg.arrays
    assert 's_prog_nnew_w' in sdfg.arrays
    assert 's' not in sdfg.arrays
    assert 's_prog' not in sdfg.arrays

    ref = f2py_compile(src, tmp_path / 'ref', 'dbuf_simple_ref', only=('wrapper', ))

    rng = np.random.default_rng(7)
    w_now_in = np.asfortranarray(rng.standard_normal((2, 3)))
    w_new_in = np.asfortranarray(rng.standard_normal((2, 3)))

    w_now_sdfg = w_now_in.copy(order='F')
    w_new_sdfg = w_new_in.copy(order='F')
    out_sdfg = np.zeros((2, 3), order='F', dtype=np.float64)
    sdfg(s_prog_nnow_w=w_now_sdfg, s_prog_nnew_w=w_new_sdfg, out=out_sdfg)

    w_now_ref = w_now_in.copy(order='F')
    w_new_ref = w_new_in.copy(order='F')
    out_ref = np.zeros((2, 3), order='F', dtype=np.float64)
    ref.wrapper(w_now_ref, w_new_ref, out_ref, np.int32(1), np.int32(2))

    np.testing.assert_allclose(w_now_sdfg, w_now_ref, rtol=0, atol=0)
    np.testing.assert_allclose(w_new_sdfg, w_new_ref, rtol=0, atol=0)
    np.testing.assert_allclose(out_sdfg, out_ref, rtol=0, atol=0)


def test_dbuf_in_kernel_swap_rejected(tmp_path):
    """A double-buffer kernel that ROTATES the time-level indices itself (an
    in-kernel ``swap(nnow, nnew)``) cannot be lane-split: the static per-symbol
    companions (``s_prog_nnow_w`` / ``s_prog_nnew_w``) are bound to one physical
    buffer each at call time and cannot be re-pointed mid-kernel.  The bridge must
    reject it loudly rather than silently miscompile -- the time-level rotation
    belongs in the driver (ICON's ``CALL swap(nnow, nnew)`` lives in
    mo_nh_stepping, outside solve_nonhydro)."""
    src = """
module dbuf_swap_mod
  implicit none
  type t_prog
    real(kind=8) :: w(2, 3)
  end type t_prog
  type t_state
    type(t_prog), allocatable :: prog(:)
  end type t_state
contains
  subroutine kernel(s, nnow, nnew, out)
    type(t_state), intent(inout) :: s
    integer, intent(inout)       :: nnow, nnew
    real(kind=8), intent(inout)  :: out(2, 3)
    integer :: i, j, itmp
    do j = 1, 3
      do i = 1, 2
        s%prog(nnew)%w(i, j) = s%prog(nnow)%w(i, j) * 2.0d0
        out(i, j) = s%prog(nnew)%w(i, j)
      end do
    end do
    ! in-kernel time-level swap -- belongs in the driver; the static lane split
    ! cannot represent it, so the bridge must reject.
    itmp = nnow
    nnow = nnew
    nnew = itmp
  end subroutine
end module
"""
    with pytest.raises(RuntimeError) as exc:
        build_sdfg(src, tmp_path / 'sdfg', name='kernel', entry='kernel').build()
    msg = str(exc.value).lower()
    assert ("pipeline failed" in msg or "swap" in msg or "time-level" in msg
            or "reassigned" in msg), f"expected an in-kernel-swap rejection, got: {msg}"


def test_dbuf_split_multi_member(tmp_path):
    """Struct with two inner array members.  Each (member, idx_sym)
    pair gets its own companion -- four total: ``s_prog_nnow_{w,vn}``
    and ``s_prog_nnew_{w,vn}``."""
    src = """
module dbuf_multi_mod
  implicit none
  type t_prog
    real(kind=8) :: w(2, 3)
    real(kind=8) :: vn(2, 3)
  end type t_prog
  type t_state
    type(t_prog), allocatable :: prog(:)
  end type t_state
contains
  subroutine kernel(s, nnow, nnew, out_w, out_vn)
    type(t_state), intent(inout) :: s
    integer, intent(in)          :: nnow, nnew
    real(kind=8), intent(inout)  :: out_w(2, 3), out_vn(2, 3)
    integer :: i, j
    do j = 1, 3
      do i = 1, 2
        s%prog(nnew)%w(i, j)  = s%prog(nnow)%w(i, j)  + 1.0d0
        s%prog(nnew)%vn(i, j) = s%prog(nnow)%vn(i, j) - 0.5d0
        out_w(i, j)  = s%prog(nnew)%w(i, j)
        out_vn(i, j) = s%prog(nnew)%vn(i, j)
      end do
    end do
  end subroutine
end module

subroutine wrapper(w_now, w_new, vn_now, vn_new, out_w, out_vn, nnow, nnew)
  use dbuf_multi_mod
  implicit none
  real(kind=8), intent(inout) :: w_now(2, 3), w_new(2, 3)
  real(kind=8), intent(inout) :: vn_now(2, 3), vn_new(2, 3)
  real(kind=8), intent(inout) :: out_w(2, 3), out_vn(2, 3)
  integer, intent(in)         :: nnow, nnew
  type(t_state) :: s
  allocate(s%prog(2))
  s%prog(1)%w  = w_now;  s%prog(2)%w  = w_new
  s%prog(1)%vn = vn_now; s%prog(2)%vn = vn_new
  call kernel(s, nnow, nnew, out_w, out_vn)
  w_now  = s%prog(1)%w;  w_new  = s%prog(2)%w
  vn_now = s%prog(1)%vn; vn_new = s%prog(2)%vn
  deallocate(s%prog)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path / 'sdfg', name='kernel', entry='kernel').build()
    for name in ('s_prog_nnow_w', 's_prog_nnew_w', 's_prog_nnow_vn', 's_prog_nnew_vn'):
        assert name in sdfg.arrays, f'missing companion {name!r}'

    ref = f2py_compile(src, tmp_path / 'ref', 'dbuf_multi_ref', only=('wrapper', ))

    rng = np.random.default_rng(11)
    arrs = {k: np.asfortranarray(rng.standard_normal((2, 3))) for k in ('w_now', 'w_new', 'vn_now', 'vn_new')}
    out_w_sdfg = np.zeros((2, 3), order='F', dtype=np.float64)
    out_vn_sdfg = np.zeros((2, 3), order='F', dtype=np.float64)
    sdfg_arrs = {k: v.copy(order='F') for k, v in arrs.items()}
    sdfg(s_prog_nnow_w=sdfg_arrs['w_now'],
         s_prog_nnew_w=sdfg_arrs['w_new'],
         s_prog_nnow_vn=sdfg_arrs['vn_now'],
         s_prog_nnew_vn=sdfg_arrs['vn_new'],
         out_w=out_w_sdfg,
         out_vn=out_vn_sdfg)

    out_w_ref = np.zeros((2, 3), order='F', dtype=np.float64)
    out_vn_ref = np.zeros((2, 3), order='F', dtype=np.float64)
    ref_arrs = {k: v.copy(order='F') for k, v in arrs.items()}
    ref.wrapper(ref_arrs['w_now'], ref_arrs['w_new'], ref_arrs['vn_now'], ref_arrs['vn_new'], out_w_ref, out_vn_ref,
                np.int32(1), np.int32(2))

    for k in sdfg_arrs:
        np.testing.assert_allclose(sdfg_arrs[k], ref_arrs[k], rtol=0, atol=0, err_msg=f'mismatch on {k}')
    np.testing.assert_allclose(out_w_sdfg, out_w_ref, rtol=0, atol=0)
    np.testing.assert_allclose(out_vn_sdfg, out_vn_ref, rtol=0, atol=0)


def test_dbuf_split_three_levels(tmp_path):
    """Three time-level symbols (nnow, nnew, ntemp).  Each becomes its
    own companion -- exercises the ``bySym`` map branching beyond two
    entries."""
    src = """
module dbuf_three_mod
  implicit none
  type t_prog
    real(kind=8) :: w(2, 3)
  end type t_prog
  type t_state
    type(t_prog), allocatable :: prog(:)
  end type t_state
contains
  subroutine kernel(s, nnow, nnew, ntemp, out)
    type(t_state), intent(inout) :: s
    integer, intent(in)          :: nnow, nnew, ntemp
    real(kind=8), intent(inout)  :: out(2, 3)
    integer :: i, j
    do j = 1, 3
      do i = 1, 2
        s%prog(ntemp)%w(i, j) = s%prog(nnow)%w(i, j) + s%prog(nnew)%w(i, j)
        out(i, j) = s%prog(ntemp)%w(i, j)
      end do
    end do
  end subroutine
end module

subroutine wrapper(w_now, w_new, w_temp, out, nnow, nnew, ntemp)
  use dbuf_three_mod
  implicit none
  real(kind=8), intent(inout) :: w_now(2, 3), w_new(2, 3), w_temp(2, 3)
  real(kind=8), intent(inout) :: out(2, 3)
  integer, intent(in)         :: nnow, nnew, ntemp
  type(t_state) :: s
  allocate(s%prog(3))
  s%prog(1)%w = w_now;  s%prog(2)%w = w_new;  s%prog(3)%w = w_temp
  call kernel(s, nnow, nnew, ntemp, out)
  w_now = s%prog(1)%w; w_new = s%prog(2)%w; w_temp = s%prog(3)%w
  deallocate(s%prog)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path / 'sdfg', name='kernel', entry='kernel').build()
    for name in ('s_prog_nnow_w', 's_prog_nnew_w', 's_prog_ntemp_w'):
        assert name in sdfg.arrays, f'missing companion {name!r}'

    ref = f2py_compile(src, tmp_path / 'ref', 'dbuf_three_ref', only=('wrapper', ))

    rng = np.random.default_rng(13)
    w_now = np.asfortranarray(rng.standard_normal((2, 3)))
    w_new = np.asfortranarray(rng.standard_normal((2, 3)))
    w_temp = np.asfortranarray(rng.standard_normal((2, 3)))

    sdfg_now = w_now.copy(order='F')
    sdfg_new = w_new.copy(order='F')
    sdfg_temp = w_temp.copy(order='F')
    out_sdfg = np.zeros((2, 3), order='F', dtype=np.float64)
    sdfg(s_prog_nnow_w=sdfg_now, s_prog_nnew_w=sdfg_new, s_prog_ntemp_w=sdfg_temp, out=out_sdfg)

    ref_now = w_now.copy(order='F')
    ref_new = w_new.copy(order='F')
    ref_temp = w_temp.copy(order='F')
    out_ref = np.zeros((2, 3), order='F', dtype=np.float64)
    ref.wrapper(ref_now, ref_new, ref_temp, out_ref, np.int32(1), np.int32(2), np.int32(3))

    np.testing.assert_allclose(sdfg_now, ref_now, rtol=0, atol=0)
    np.testing.assert_allclose(sdfg_new, ref_new, rtol=0, atol=0)
    np.testing.assert_allclose(sdfg_temp, ref_temp, rtol=0, atol=0)
    np.testing.assert_allclose(out_sdfg, out_ref, rtol=0, atol=0)


def test_dbuf_split_pointer_aor(tmp_path):
    """The split also handles the ``type(t), pointer :: prog(:)``
    flavour of the array-of-records spine (vs ``allocatable``).
    Ensures ``allocOrPtrArrayOfRecordsMember`` recognises both."""
    src = """
module dbuf_ptr_mod
  implicit none
  type t_prog
    real(kind=8) :: w(2, 3)
  end type t_prog
  type t_state
    type(t_prog), pointer :: prog(:)
  end type t_state
contains
  subroutine kernel(s, nnow, nnew, out)
    type(t_state), intent(inout) :: s
    integer, intent(in)          :: nnow, nnew
    real(kind=8), intent(inout)  :: out(2, 3)
    integer :: i, j
    do j = 1, 3
      do i = 1, 2
        s%prog(nnew)%w(i, j) = s%prog(nnow)%w(i, j) + 3.0d0
        out(i, j) = s%prog(nnew)%w(i, j)
      end do
    end do
  end subroutine
end module

subroutine wrapper(w_now, w_new, out, nnow, nnew)
  use dbuf_ptr_mod
  implicit none
  real(kind=8), intent(inout) :: w_now(2, 3), w_new(2, 3)
  real(kind=8), intent(inout) :: out(2, 3)
  integer, intent(in)         :: nnow, nnew
  type(t_state) :: s
  type(t_prog), target, save :: spine(2)
  s%prog => spine
  s%prog(1)%w = w_now
  s%prog(2)%w = w_new
  call kernel(s, nnow, nnew, out)
  w_now = s%prog(1)%w
  w_new = s%prog(2)%w
  nullify(s%prog)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path / 'sdfg', name='kernel', entry='kernel').build()
    assert 's_prog_nnow_w' in sdfg.arrays
    assert 's_prog_nnew_w' in sdfg.arrays

    ref = f2py_compile(src, tmp_path / 'ref', 'dbuf_ptr_ref', only=('wrapper', ))

    rng = np.random.default_rng(17)
    w_now_in = np.asfortranarray(rng.standard_normal((2, 3)))
    w_new_in = np.asfortranarray(rng.standard_normal((2, 3)))

    w_now_sdfg = w_now_in.copy(order='F')
    w_new_sdfg = w_new_in.copy(order='F')
    out_sdfg = np.zeros((2, 3), order='F', dtype=np.float64)
    sdfg(s_prog_nnow_w=w_now_sdfg, s_prog_nnew_w=w_new_sdfg, out=out_sdfg)

    w_now_ref = w_now_in.copy(order='F')
    w_new_ref = w_new_in.copy(order='F')
    out_ref = np.zeros((2, 3), order='F', dtype=np.float64)
    ref.wrapper(w_now_ref, w_new_ref, out_ref, np.int32(1), np.int32(2))

    np.testing.assert_allclose(w_now_sdfg, w_now_ref, rtol=0, atol=0)
    np.testing.assert_allclose(w_new_sdfg, w_new_ref, rtol=0, atol=0)
    np.testing.assert_allclose(out_sdfg, out_ref, rtol=0, atol=0)


def test_dbuf_split_nested_struct(tmp_path):
    """Nested-struct access chain: the AoR member lives below one or more
    plain struct members, e.g. ``s%inner%prog(idx)%w``.  The split must
    walk the full chain to the dummy root and bake the joined member path
    into the companion name: ``s_inner_prog_<sym>_<inner_member>``."""
    src = """
module dbuf_nested_mod
  implicit none
  type t_prog
    real(kind=8) :: w(2, 3)
  end type t_prog
  type t_inner
    type(t_prog), allocatable :: prog(:)
  end type t_inner
  type t_state
    type(t_inner) :: inner
  end type t_state
contains
  subroutine kernel(s, nnow, nnew, out)
    type(t_state), intent(inout) :: s
    integer, intent(in)          :: nnow, nnew
    real(kind=8), intent(inout)  :: out(2, 3)
    integer :: i, j
    do j = 1, 3
      do i = 1, 2
        s%inner%prog(nnew)%w(i, j) = s%inner%prog(nnow)%w(i, j) * 2.0d0
        out(i, j) = s%inner%prog(nnew)%w(i, j)
      end do
    end do
  end subroutine
end module

subroutine wrapper(w_now, w_new, out, nnow, nnew)
  use dbuf_nested_mod
  implicit none
  real(kind=8), intent(inout) :: w_now(2, 3), w_new(2, 3)
  real(kind=8), intent(inout) :: out(2, 3)
  integer, intent(in)         :: nnow, nnew
  type(t_state) :: s
  allocate(s%inner%prog(2))
  s%inner%prog(1)%w = w_now
  s%inner%prog(2)%w = w_new
  call kernel(s, nnow, nnew, out)
  w_now = s%inner%prog(1)%w
  w_new = s%inner%prog(2)%w
  deallocate(s%inner%prog)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path / 'sdfg', name='kernel', entry='kernel').build()

    assert 's_inner_prog_nnow_w' in sdfg.arrays
    assert 's_inner_prog_nnew_w' in sdfg.arrays
    assert 's' not in sdfg.arrays
    assert 's_inner' not in sdfg.arrays

    ref = f2py_compile(src, tmp_path / 'ref', 'dbuf_nested_ref', only=('wrapper', ))

    rng = np.random.default_rng(23)
    w_now_in = np.asfortranarray(rng.standard_normal((2, 3)))
    w_new_in = np.asfortranarray(rng.standard_normal((2, 3)))

    w_now_sdfg = w_now_in.copy(order='F')
    w_new_sdfg = w_new_in.copy(order='F')
    out_sdfg = np.zeros((2, 3), order='F', dtype=np.float64)
    sdfg(s_inner_prog_nnow_w=w_now_sdfg, s_inner_prog_nnew_w=w_new_sdfg, out=out_sdfg)

    w_now_ref = w_now_in.copy(order='F')
    w_new_ref = w_new_in.copy(order='F')
    out_ref = np.zeros((2, 3), order='F', dtype=np.float64)
    ref.wrapper(w_now_ref, w_new_ref, out_ref, np.int32(1), np.int32(2))

    np.testing.assert_allclose(w_now_sdfg, w_now_ref, rtol=0, atol=0)
    np.testing.assert_allclose(w_new_sdfg, w_new_ref, rtol=0, atol=0)
    np.testing.assert_allclose(out_sdfg, out_ref, rtol=0, atol=0)


def test_dbuf_split_direct_aor_dummy(tmp_path):
    """The dummy ITSELF is the alloc-array-of-records (no outer struct).
    Access is ``s(idx)%w`` rather than ``s%X(idx)%w``.  The split must
    treat the dummy as the AoR root with an empty member path; the
    companion name is just ``s_<sym>_<inner_member>``."""
    src = """
module dbuf_direct_mod
  implicit none
  type t_prog
    real(kind=8) :: w(2, 3)
  end type t_prog
contains
  subroutine kernel(s, nnow, nnew, out)
    type(t_prog), allocatable, intent(inout) :: s(:)
    integer, intent(in)                       :: nnow, nnew
    real(kind=8), intent(inout)               :: out(2, 3)
    integer :: i, j
    do j = 1, 3
      do i = 1, 2
        s(nnew)%w(i, j) = s(nnow)%w(i, j) * 2.0d0
        out(i, j) = s(nnew)%w(i, j)
      end do
    end do
  end subroutine
end module

subroutine wrapper(w_now, w_new, out, nnow, nnew)
  use dbuf_direct_mod
  implicit none
  real(kind=8), intent(inout) :: w_now(2, 3), w_new(2, 3)
  real(kind=8), intent(inout) :: out(2, 3)
  integer, intent(in)         :: nnow, nnew
  type(t_prog), allocatable :: s(:)
  allocate(s(2))
  s(1)%w = w_now
  s(2)%w = w_new
  call kernel(s, nnow, nnew, out)
  w_now = s(1)%w
  w_new = s(2)%w
  deallocate(s)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path / 'sdfg', name='kernel', entry='kernel').build()

    assert 's_nnow_w' in sdfg.arrays
    assert 's_nnew_w' in sdfg.arrays
    assert 's' not in sdfg.arrays

    ref = f2py_compile(src, tmp_path / 'ref', 'dbuf_direct_ref', only=('wrapper', ))

    rng = np.random.default_rng(29)
    w_now_in = np.asfortranarray(rng.standard_normal((2, 3)))
    w_new_in = np.asfortranarray(rng.standard_normal((2, 3)))

    w_now_sdfg = w_now_in.copy(order='F')
    w_new_sdfg = w_new_in.copy(order='F')
    out_sdfg = np.zeros((2, 3), order='F', dtype=np.float64)
    sdfg(s_nnow_w=w_now_sdfg, s_nnew_w=w_new_sdfg, out=out_sdfg)

    w_now_ref = w_now_in.copy(order='F')
    w_new_ref = w_new_in.copy(order='F')
    out_ref = np.zeros((2, 3), order='F', dtype=np.float64)
    ref.wrapper(w_now_ref, w_new_ref, out_ref, np.int32(1), np.int32(2))

    np.testing.assert_allclose(w_now_sdfg, w_now_ref, rtol=0, atol=0)
    np.testing.assert_allclose(w_new_sdfg, w_new_ref, rtol=0, atol=0)
    np.testing.assert_allclose(out_sdfg, out_ref, rtol=0, atol=0)


def test_dbuf_in_kernel_toggle_alias_unrolled(tmp_path):
    """ICON ``solve_nh`` time-level toggle: a THIRD index symbol ``nvar`` is
    reassigned in-kernel (``nvar = nnow`` predictor / ``nvar = nnew`` corrector)
    inside the ``DO istep = 1, 2`` loop.  ``nvar`` is a pure alias of the stable
    time levels, so ``hlfir-eliminate-double-buffer-toggle`` fully unrolls that
    constant-trip loop, substitutes ``nvar`` away (``prog(nvar)`` -> ``prog(nnow)``
    in the predictor copy, ``prog(nnew)`` in the corrector copy), and the static
    lane split then handles only the stable ``nnow`` / ``nnew`` -- no runtime-
    indexed companion, no reject."""
    src = """
module dbuf_nvar_mod
  implicit none
  type t_prog
    real(kind=8) :: w(2, 3)
  end type t_prog
  type t_state
    type(t_prog), allocatable :: prog(:)
  end type t_state
contains
  subroutine kernel(s, nnow, nnew, out)
    type(t_state), intent(inout) :: s
    integer, intent(in)          :: nnow, nnew
    real(kind=8), intent(inout)  :: out(2, 3)
    integer :: i, j, istep, nvar
    do istep = 1, 2
      if (istep == 1) then
        nvar = nnow
      else
        nvar = nnew
      end if
      do j = 1, 3
        do i = 1, 2
          out(i, j) = s%prog(nnow)%w(i, j) + s%prog(nvar)%w(i, j)
        end do
      end do
    end do
  end subroutine
end module

subroutine wrapper(w_now, w_new, out, nnow, nnew)
  use dbuf_nvar_mod
  implicit none
  real(kind=8), intent(inout) :: w_now(2, 3), w_new(2, 3)
  real(kind=8), intent(inout) :: out(2, 3)
  integer, intent(in)         :: nnow, nnew
  type(t_state) :: s
  allocate(s%prog(2))
  s%prog(1)%w = w_now
  s%prog(2)%w = w_new
  call kernel(s, nnow, nnew, out)
  deallocate(s%prog)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path / 'sdfg', name='kernel', entry='kernel').build()

    # The toggle is eliminated: only the stable per-time-level lanes remain.
    assert 's_prog_nnow_w' in sdfg.arrays
    assert 's_prog_nnew_w' in sdfg.arrays
    assert not any('nvar' in a for a in sdfg.arrays), \
        f"nvar toggle survived as a companion: {sorted(sdfg.arrays)}"

    ref = f2py_compile(src, tmp_path / 'ref', 'dbuf_nvar_ref', only=('wrapper', ))

    rng = np.random.default_rng(11)
    w_now_in = np.asfortranarray(rng.standard_normal((2, 3)))
    w_new_in = np.asfortranarray(rng.standard_normal((2, 3)))

    out_sdfg = np.zeros((2, 3), order='F', dtype=np.float64)
    sdfg(s_prog_nnow_w=w_now_in.copy(order='F'), s_prog_nnew_w=w_new_in.copy(order='F'), out=out_sdfg)

    out_ref = np.zeros((2, 3), order='F', dtype=np.float64)
    ref.wrapper(w_now_in.copy(order='F'), w_new_in.copy(order='F'), out_ref, np.int32(1), np.int32(2))

    # The corrector overwrites with ``=``: final out = prog(nnow) + prog(nnew).
    np.testing.assert_allclose(out_sdfg, out_ref, rtol=0, atol=0)
    np.testing.assert_allclose(out_sdfg, w_now_in + w_new_in, rtol=0, atol=1e-12)


def test_dbuf_bind_c_shim_reconstructs_lanes(tmp_path):
    """The bind(c) shim + binding marshal a double-buffer AoR with ALLOCATABLE
    inner members (the real ICON ``p_nh%prog(nnow)%rho`` shape) through the C
    ABI.  The bridge split + in-kernel-toggle elimination produce static lanes
    ``s_prog_nnow_rho`` / ``s_prog_nnew_rho``; the binding must alias
    ``c_loc(s%prog(nnow)%rho)`` (NOT the synthetic dummy name) and the shim must
    allocate ``s%prog(max(nnow,nnew))`` and populate each time-level element
    from its own C-ABI lane buffer.  Asserts the whole library (binding + shim)
    links."""
    from dace_fortran.bindings import build_fortran_library
    from dace_fortran.build import make_builder

    src = """
module dbuf_shim_mod
  implicit none
  type t_prog
    real(kind=8), allocatable :: rho(:, :, :)
  end type t_prog
  type t_state
    type(t_prog), allocatable :: prog(:)
  end type t_state
contains
  subroutine kernel(s, nnow, nnew, out, n, nlev, nblk)
    type(t_state), intent(inout) :: s
    integer, intent(in)          :: nnow, nnew, n, nlev, nblk
    real(kind=8), intent(inout)  :: out(n, nlev, nblk)
    integer :: i, j, k, istep, nvar
    do istep = 1, 2
      if (istep == 1) then
        nvar = nnow
      else
        nvar = nnew
      end if
      do k = 1, nblk
        do j = 1, nlev
          do i = 1, n
            out(i, j, k) = s%prog(nnow)%rho(i, j, k) + s%prog(nvar)%rho(i, j, k)
          end do
        end do
      end do
    end do
  end subroutine
end module
"""
    src_f90 = tmp_path / 'dbuf_shim_mod.f90'
    src_f90.write_text(src)
    # Short SDFG name (the bind(c) shim derives ``<name>_dace_finalize`` -- the
    # test-util xdist suffix would blow Fortran's 63-char identifier limit).
    sdfg = make_builder(src, entry='kernel', name='dbufshim', out_dir=str(tmp_path / 'sdfg')).build()
    assert 's_prog_nnow_rho' in sdfg.arrays and 's_prog_nnew_rho' in sdfg.arrays

    lib = build_fortran_library(sdfg, out_dir=str(tmp_path / 'lib'), prelude_sources=[src_f90], bind_c_shim=True)
    shim = __import__('pathlib').Path(lib.bind_c_shim_f90).read_text()
    # The shim reconstructs the AoR sized for both time levels and populates
    # each lane from its own buffer -- not the old generic size-1 path.
    assert 'allocate(s%prog(max(' in shim
    assert 's%prog(nnow)%rho = s_prog_nnow_rho' in shim
    assert 's%prog(nnew)%rho = s_prog_nnew_rho' in shim
    assert __import__('pathlib').Path(lib.so_path).is_file()


@pytest.mark.parametrize("jg, kmatch, fires", [(2, 7, True), (2, 8, False), (1, 5, True)])
def test_dbuf_unroll_spills_tracked_istep_into_nonconstant_if(tmp_path, jg, kmatch, fires):
    """ICON ``solve_nh`` ``exner_dyn_incr`` shape: inside the unrolled
    ``DO istep = 1, 2`` double-buffer loop a NON-CONSTANT ``fir.if`` reads the
    tracked induction slot ``istep`` in its condition
    (``istep == 2 .AND. ndyn(jg) == kmatch`` -- data-dependent via the array
    read).  ``EliminateDoubleBufferToggle`` unrolls + substitutes ``istep`` per
    copy for directly-walked ops, but a data-dependent ``fir.if`` falls back to
    a bulk ``b.clone()`` that copies nested ``fir.load %istep`` verbatim.

    Before commit 32ca718 that read a slot the pass never writes: ``istep``
    surfaced as an UNPOPULATED free symbol passed uninitialised, so the
    ``istep == 2`` guard never fired and the DUT dropped the istep-2 update
    (``+100``).  ``spillTrackedSlots`` now materialises each tracked slot's
    substituted value back into its memref before the structural clone, so the
    bulk-cloned nested loads read the right value.

    Asserts (a) ``istep`` is fully eliminated from the SDFG free symbols (no
    stray unpopulated symbol), and (b) the run is BIT-EXACT vs gfortran with the
    ``istep == 2`` branch firing exactly when ``ndyn(jg) == kmatch``."""
    src = """
module dbuf_istep_mod
  implicit none
  type t_prog
    real(kind=8) :: w(2, 3)
  end type t_prog
  type t_state
    type(t_prog), allocatable :: prog(:)
  end type t_state
contains
  subroutine kernel(s, nnow, nnew, out, ndyn, jg, kmatch)
    type(t_state), intent(inout) :: s
    integer, intent(in)          :: nnow, nnew, jg, kmatch
    integer, intent(in)          :: ndyn(:)
    real(kind=8), intent(inout)  :: out(2, 3)
    integer :: i, j, istep, nvar
    do istep = 1, 2
      if (istep == 1) then
        nvar = nnow
      else
        nvar = nnew
      end if
      do j = 1, 3
        do i = 1, 2
          out(i, j) = s%prog(nnow)%w(i, j) + s%prog(nvar)%w(i, j)
        end do
      end do
      ! Non-constant fir.if (array-read condition) that also reads the tracked
      ! istep slot -- the exner_dyn_incr istep-2 guard.
      if (istep == 2 .and. ndyn(jg) == kmatch) then
        do j = 1, 3
          do i = 1, 2
            out(i, j) = out(i, j) + 100.0d0
          end do
        end do
      end if
    end do
  end subroutine
end module

subroutine wrapper(w_now, w_new, out, nnow, nnew, ndyn, nd, jg, kmatch)
  use dbuf_istep_mod
  implicit none
  integer, intent(in)         :: nnow, nnew, nd, jg, kmatch
  real(kind=8), intent(inout) :: w_now(2, 3), w_new(2, 3)
  real(kind=8), intent(inout) :: out(2, 3)
  integer, intent(in)         :: ndyn(nd)
  type(t_state) :: s
  allocate(s%prog(2))
  s%prog(1)%w = w_now
  s%prog(2)%w = w_new
  call kernel(s, nnow, nnew, out, ndyn, jg, kmatch)
  deallocate(s%prog)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path / 'sdfg', name='kernel', entry='kernel').build()

    # The toggle is eliminated AND the tracked induction slot is fully spilled:
    # no ``istep`` remains as a free (would-be-uninitialised) symbol.
    assert not any('istep' in str(s) for s in sdfg.free_symbols), \
        f"istep survived as an unpopulated free symbol: {sorted(str(s) for s in sdfg.free_symbols)}"

    ref = f2py_compile(src, tmp_path / 'ref', 'dbuf_istep_ref', only=('wrapper', ))

    rng = np.random.default_rng(31)
    w_now = np.asfortranarray(rng.standard_normal((2, 3)))
    w_new = np.asfortranarray(rng.standard_normal((2, 3)))
    ndyn = np.asfortranarray(np.array([5, 7, 9], dtype=np.int32))

    out_sdfg = np.zeros((2, 3), order='F', dtype=np.float64)
    sdfg(s_prog_nnow_w=w_now.copy(order='F'),
         s_prog_nnew_w=w_new.copy(order='F'),
         out=out_sdfg,
         ndyn=ndyn.copy(order='F'),
         jg=np.int32(jg),
         kmatch=np.int32(kmatch),
         nnow=np.int32(1),
         nnew=np.int32(2))

    out_ref = np.zeros((2, 3), order='F', dtype=np.float64)
    # f2py infers ``nd`` from ``ndyn`` (dropped from the positional list).
    ref.wrapper(w_now.copy(order='F'), w_new.copy(order='F'), out_ref, np.int32(1), np.int32(2), ndyn.copy(order='F'),
                np.int32(jg), np.int32(kmatch))

    np.testing.assert_array_equal(out_sdfg, out_ref)
    # The istep-2 branch fires exactly when ndyn(jg) == kmatch: closed form
    # (out overwritten each iter to prog(nnow)+prog(nvar), then +100 if fired).
    expected = w_now + w_new + (100.0 if fires else 0.0)
    np.testing.assert_array_equal(out_sdfg, expected)
