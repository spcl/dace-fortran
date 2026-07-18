"""``hlfir-flatten-structs`` struct-flattening tests: module-level and local ``type(t) :: s``
with array/scalar members, extent recovery via ``fir.SequenceType``. SDFG output checked
against gfortran/f2py; includes a negative loud-failure test.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _build(src: str, tmp: Path, name: str = "main", entry: str | None = None):
    sdfg_dir = tmp / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, sdfg_dir, name=name, entry=entry).build()


def test_local_struct_element_write_and_read(tmp_path: Path):
    """Local ``type(t) :: s`` with array member: single element write+read exercises the
    local-instance flatten + SequenceType-extent fallback in ``extract_vars``."""
    src = """
module lib
  implicit none
  type simple_type
    real :: w(5, 5, 5)
    integer :: a
  end type simple_type
end module lib

subroutine main(d)
  use lib
  implicit none
  real, intent(out) :: d(2)
  type(simple_type) :: s
  s%w(1, 1, 1) = 5.5
  d(1) = s%w(1, 1, 1)
  d(2) = 5.5 + s%w(1, 1, 1)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "local_struct_element_ref")
    d_ref = np.asarray(mod.main(), dtype=np.float32)

    sdfg = _build(src, tmp_path)
    d = np.zeros(2, dtype=np.float32)
    sdfg(d=d)
    np.testing.assert_array_equal(d, d_ref)
    np.testing.assert_array_equal(d, [5.5, 11.0])


def test_local_struct_two_array_members(tmp_path: Path):
    """Two array members of different shapes generate two separate flat per-member arrays."""
    src = """
module lib
  implicit none
  type two_arrays
    real :: u(4)
    real :: v(7)
  end type two_arrays
end module lib

subroutine main(out)
  use lib
  implicit none
  real, intent(out) :: out(2)
  type(two_arrays) :: t
  t%u(2) = 3.0
  t%v(7) = 4.0
  out(1) = t%u(2)
  out(2) = t%v(7)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "local_struct_two_arrays_ref")
    out_ref = np.asarray(mod.main(), dtype=np.float32)

    sdfg = _build(src, tmp_path)
    out = np.zeros(2, dtype=np.float32)
    sdfg(out=out)
    np.testing.assert_array_equal(out, out_ref)
    np.testing.assert_array_equal(out, [3.0, 4.0])


def test_local_struct_member_in_loop(tmp_path: Path):
    """Loop-driven writes to a struct's array member; flat ``s_w`` carries the SequenceType's
    static (5,) extent so the SDFG signature needs no synth shape symbol."""
    src = """
module lib
  implicit none
  type sum_type
    real :: w(5)
  end type sum_type
end module lib

subroutine main(out)
  use lib
  implicit none
  real, intent(out) :: out
  type(sum_type) :: s
  integer :: i
  do i = 1, 5
    s%w(i) = real(i) * 2.0
  end do
  out = s%w(1) + s%w(2) + s%w(3) + s%w(4) + s%w(5)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "local_struct_loop_ref")
    out_ref = float(mod.main())

    sdfg = _build(src, tmp_path)
    out = np.zeros(1, dtype=np.float32)
    sdfg(out=out)
    np.testing.assert_array_equal(out, out_ref)
    assert out[0] == 2.0 + 4.0 + 6.0 + 8.0 + 10.0


def test_icon_style_state_struct(tmp_path: Path):
    """ICON dycore-flavoured state struct: several parallel 3-D arrays in one derived type
    (``p_diag%vn`` etc), indexed elementwise. Verifies Phase 1's standard path at realistic scale."""
    src = """
module lib
  implicit none
  integer, parameter :: nproma = 4, nlev = 3, nblks = 2
  type state_t
    real :: vn(nproma, nlev, nblks)
    real :: w(nproma, nlev, nblks)
    real :: theta_v(nproma, nlev, nblks)
  end type state_t
end module lib

subroutine main(out)
  use lib
  implicit none
  real, intent(out) :: out(nproma, nlev, nblks)
  type(state_t) :: s
  integer :: i, k, b
  do b = 1, nblks
    do k = 1, nlev
      do i = 1, nproma
        s%vn(i, k, b)      = real(i + k * 10 + b * 100)
        s%w(i, k, b)       = real(i)
        s%theta_v(i, k, b) = real(b)
        out(i, k, b) = s%vn(i, k, b) + s%w(i, k, b) * s%theta_v(i, k, b)
      end do
    end do
  end do
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "icon_state_ref")
    out_ref = mod.main()

    sdfg = _build(src, tmp_path)
    out = np.zeros((4, 3, 2), order='F', dtype=np.float32)
    sdfg(out=out)
    np.testing.assert_array_equal(out, out_ref)


def test_qe_style_pdf_sampler_struct(tmp_path: Path):
    """QE-flavoured ``pdf_sampler_type`` (npbench ``usxx.py``): mixed scalar/fixed-shape members.
    Drops the original's ``ALLOCATABLE val(:,:)`` (Phase 3 territory), keeps the rest."""
    src = """
module lib
  implicit none
  integer, parameter :: ncdf = 4, nfsd = 3
  type pdf_sampler_type
    integer :: ncdf_n, nfsd_n
    real(8) :: fsd1, inv_fsd_interval
    real(8) :: val(ncdf, nfsd)
  end type pdf_sampler_type
end module lib

subroutine main(out)
  use lib
  implicit none
  real(8), intent(out) :: out
  type(pdf_sampler_type) :: s
  integer :: i, j
  s%ncdf_n = ncdf
  s%nfsd_n = nfsd
  s%fsd1 = 1.5d0
  s%inv_fsd_interval = 2.5d0
  do j = 1, nfsd
    do i = 1, ncdf
      s%val(i, j) = real(i * j, 8)
    end do
  end do
  out = s%val(2, 2) * s%fsd1 + s%inv_fsd_interval &
        + real(s%ncdf_n + s%nfsd_n, 8)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "qe_pdf_sampler_ref")
    out_ref = float(mod.main())

    sdfg = _build(src, tmp_path)
    out = np.zeros(1, dtype=np.float64)
    sdfg(out=out)
    np.testing.assert_allclose(float(out[0]), out_ref, rtol=1e-12)


def test_batched_csr_fixed_capacity(tmp_path: Path):
    """Batched CSR (Phase 1.5): ``A(N)`` of struct with array members flattens to per-member SoA
    ``A_rowptr(N, ROW_CAP)`` / ``A_colidx(N, NNZ_CAP)`` / ``A_val(N, NNZ_CAP)``. Runtime-jagged
    sizes are Phase 3 -- see ``test_batched_csr_allocatable_jagged``.
    """
    src = """
module lib
  implicit none
  integer, parameter :: ROWS    = 3
  integer, parameter :: ROW_CAP = ROWS + 1
  integer, parameter :: NNZ_CAP = 4
  type csr_t
    integer :: rowptr(ROW_CAP)
    integer :: colidx(NNZ_CAP)
    real(8) :: val(NNZ_CAP)
  end type csr_t
end module lib

subroutine main(out)
  ! Run a tiny SpMV on each batched CSR matrix, accumulating into ``out``.
  ! Per-element init avoids whole-array component access ``A(1)%rowptr =
  ! (/ ... /)``, which the bridge rewriter doesn't yet collapse to a
  ! flat slice ``A_rowptr(1, :)`` (Phase 2.1 follow-up).
  use lib
  implicit none
  integer, parameter :: BATCH = 2
  real(8), intent(out) :: out(BATCH, ROWS)
  type(csr_t) :: A(BATCH)
  real(8) :: x(ROWS)
  integer :: b, r, k

  ! ---- Init batch 1: identity (3x3) ----
  A(1)%rowptr(1) = 1
  A(1)%rowptr(2) = 2
  A(1)%rowptr(3) = 3
  A(1)%rowptr(4) = 4
  A(1)%colidx(1) = 1
  A(1)%colidx(2) = 2
  A(1)%colidx(3) = 3
  A(1)%colidx(4) = 0
  A(1)%val(1) = 1.0d0
  A(1)%val(2) = 1.0d0
  A(1)%val(3) = 1.0d0
  A(1)%val(4) = 0.0d0

  ! ---- Init batch 2: tridiagonal-flavoured ----
  A(2)%rowptr(1) = 1
  A(2)%rowptr(2) = 3
  A(2)%rowptr(3) = 4
  A(2)%rowptr(4) = 5
  A(2)%colidx(1) = 1
  A(2)%colidx(2) = 2
  A(2)%colidx(3) = 1
  A(2)%colidx(4) = 3
  A(2)%val(1) = 2.0d0
  A(2)%val(2) = -1.0d0
  A(2)%val(3) = 1.0d0
  A(2)%val(4) = 1.0d0

  x(1) = 10.0d0
  x(2) = 20.0d0
  x(3) = 30.0d0

  ! ---- SpMV per batch ----
  do b = 1, BATCH
    do r = 1, ROWS
      out(b, r) = 0.0d0
      do k = A(b)%rowptr(r), A(b)%rowptr(r + 1) - 1
        out(b, r) = out(b, r) + A(b)%val(k) * x(A(b)%colidx(k))
      end do
    end do
  end do
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "batched_csr_ref")
    out_ref = mod.main()

    sdfg = _build(src, tmp_path)
    out = np.zeros((2, 3), order='F', dtype=np.float64)
    sdfg(out=out)
    np.testing.assert_allclose(out, out_ref, rtol=1e-12)


def test_batched_csr_fixed_capacity_cross_boundary(tmp_path: Path):
    """Same fixed-capacity CSR as the local test, but the SpMV runs in a CALLEE -- caller hands
    the batched CSR struct array across the subroutine boundary, exercising the bindings
    layer's pack/unpack (``A_rowptr(BATCH, ROW_CAP)`` etc, copied per-instance both ways).
    """
    src = """
module lib
  implicit none
  integer, parameter :: ROWS    = 3
  integer, parameter :: ROW_CAP = ROWS + 1
  integer, parameter :: NNZ_CAP = 4
  type csr_t
    integer :: rowptr(ROW_CAP)
    integer :: colidx(NNZ_CAP)
    real(8) :: val(NNZ_CAP)
  end type csr_t
end module lib

subroutine main(out)
  use lib
  implicit none
  integer, parameter :: BATCH = 2
  real(8), intent(out) :: out(BATCH, ROWS)
  type(csr_t) :: A(BATCH)
  real(8) :: x(ROWS)
  integer :: i

  ! Init batches and x (per-element so the pass is exercised on the
  ! local-allocation path; the test then passes A across the boundary).
  A(1)%rowptr(1) = 1
  A(1)%rowptr(2) = 2
  A(1)%rowptr(3) = 3
  A(1)%rowptr(4) = 4
  do i = 1, NNZ_CAP
    A(1)%colidx(i) = i
    A(1)%val(i) = 1.0d0
  end do

  A(2)%rowptr(1) = 1
  A(2)%rowptr(2) = 3
  A(2)%rowptr(3) = 4
  A(2)%rowptr(4) = 5
  A(2)%colidx(1) = 1
  A(2)%colidx(2) = 2
  A(2)%colidx(3) = 1
  A(2)%colidx(4) = 3
  A(2)%val(1) = 2.0d0
  A(2)%val(2) = -1.0d0
  A(2)%val(3) = 1.0d0
  A(2)%val(4) = 1.0d0

  x(1) = 10.0d0
  x(2) = 20.0d0
  x(3) = 30.0d0

  call spmv_batched(A, x, out)
end subroutine main

subroutine spmv_batched(A, x, out)
  use lib
  implicit none
  integer, parameter :: BATCH = 2
  type(csr_t), intent(in) :: A(BATCH)
  real(8), intent(in)     :: x(ROWS)
  real(8), intent(out)    :: out(BATCH, ROWS)
  integer :: b, r, k

  do b = 1, BATCH
    do r = 1, ROWS
      out(b, r) = 0.0d0
      do k = A(b)%rowptr(r), A(b)%rowptr(r + 1) - 1
        out(b, r) = out(b, r) + A(b)%val(k) * x(A(b)%colidx(k))
      end do
    end do
  end do
end subroutine spmv_batched
"""
    sdfg = _build(src, tmp_path, name='main', entry='main')
    out = np.zeros((2, 3), order='F', dtype=np.float64)
    sdfg(out=out)


def test_local_struct_allocatable_member_element_writes(tmp_path: Path):
    """Phase 5a: local struct with one allocatable array member; allocate, per-element write,
    read back two + sum. Flatten pass replaces ``s%w`` with flat allocatable ``s_w`` and renames
    the ``fir.allocmem`` op so ``collectAllocSites`` finds it under ``s_w.alloc``.
    """
    src = """
module lib
  implicit none
  type t
    real, allocatable :: w(:)
  end type t
end module lib

subroutine main(n, res)
  use lib
  implicit none
  integer, intent(in) :: n
  real, intent(out)   :: res
  type(t) :: s
  allocate(s%w(n))
  s%w(1) = 1.0
  s%w(2) = 2.0
  s%w(3) = 3.0
  s%w(4) = 4.0
  res = s%w(2) + s%w(4)
  deallocate(s%w)
end subroutine main
"""
    sdfg = _build(src, tmp_path)
    res = np.zeros(1, dtype=np.float32)
    sdfg(n=4, res=res)
    assert res[0] == 6.0


def test_local_struct_allocatable_with_scalar_sibling_member(tmp_path: Path):
    """Phase 5a: struct with a scalar field and an allocatable array field. Pins both flat
    declares co-existing, and the scalar field usable as the allocate's extent."""
    src = """
module lib
  implicit none
  type t
    integer :: n
    real, allocatable :: w(:)
  end type t
end module lib

subroutine main(extent, res)
  use lib
  implicit none
  integer, intent(in) :: extent
  real, intent(out)   :: res
  type(t) :: s
  s%n = extent
  allocate(s%w(s%n))
  s%w(1) = 10.0
  s%w(s%n) = 20.0
  res = s%w(1) + s%w(s%n)
  deallocate(s%w)
end subroutine main
"""
    sdfg = _build(src, tmp_path)
    res = np.zeros(1, dtype=np.float32)
    sdfg(extent=5, res=res)
    assert res[0] == 30.0


def test_local_struct_allocatable_whole_array_assign(tmp_path: Path):
    """Phase 5a: allocate-then-loop element writes/reads against the rewritten ``s_w`` flat declare."""
    src = """
module lib
  implicit none
  type t
    real, allocatable :: w(:)
  end type t
end module lib

subroutine main(n, src, res)
  use lib
  implicit none
  integer, intent(in)    :: n
  real, intent(in)       :: src(n)
  real, intent(out)      :: res(n)
  type(t) :: s
  integer :: i
  allocate(s%w(n))
  do i = 1, n
    s%w(i) = src(i)
  end do
  do i = 1, n
    res(i) = s%w(i)
  end do
  deallocate(s%w)
end subroutine main
"""
    sdfg = _build(src, tmp_path)
    n = 6
    src_arr = np.arange(1.0, n + 1.0, dtype=np.float32)
    res = np.zeros(n, dtype=np.float32)
    sdfg(n=n, src=src_arr, res=res)
    np.testing.assert_array_equal(res, src_arr)


def test_dummy_struct_with_allocatable_member_top_level_call(tmp_path: Path):
    """Phase 5b: struct with an allocatable member passed across a top-level (non
    module-contained) call; inlined alias traces back to caller's decl. Bindings wrapper
    marshals the descriptor (nullptr if unallocated, packed copy if allocated) -- no runtime
    ``ALLOCATED()`` check, program tracks allocation state itself.
    """
    src = """
module lib
  implicit none
  type t
    integer :: n
    real, allocatable :: w(:)
  end type t
end module lib

subroutine main(out)
  use lib
  implicit none
  real, intent(out) :: out
  type(t) :: s
  s%n = 4
  allocate(s%w(s%n))
  s%w(1) = 10.0
  s%w(2) = 20.0
  s%w(3) = 30.0
  s%w(4) = 40.0
  call accumulate(s, out)
  deallocate(s%w)
end subroutine main

subroutine accumulate(s, out)
  use lib
  implicit none
  type(t), intent(in) :: s
  real, intent(out) :: out
  integer :: i
  out = 0.0
  do i = 1, s%n
    out = out + s%w(i)
  end do
end subroutine accumulate
"""
    sdfg = _build(src, tmp_path, entry='main')
    out = np.zeros(1, dtype=np.float32)
    sdfg(out=out)
    assert out[0] == 100.0


def test_struct_pointer_member_slice_rebind(tmp_path: Path):
    """Phase 5b: pointer array member rebound to a TARGET'd section (``s%w => src(1:n)``).
    Flatten treats pointer members like allocatable (Phase 5a); ``hlfir-rewrite-pointer-assigns``
    forwards the rebound section to every load, so ``s%w(i)`` reads ``src(i)`` directly.
    Strict-no-aliasing assumption -- unsafe if the program relies on aliasing; read-only direction only.
    """
    src = """
module lib
  implicit none
  type t
    real, pointer :: w(:)
  end type t
end module lib

subroutine main(n, src, res)
  use lib
  implicit none
  integer, intent(in) :: n
  real, intent(in), target :: src(n)
  real, intent(out) :: res
  type(t) :: s
  s%w => src(1:n)
  res = s%w(2) + s%w(4)
end subroutine main
"""
    sdfg = _build(src, tmp_path, entry='main')
    n = 5
    src_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    res = np.zeros(1, dtype=np.float32)
    sdfg(n=n, src=src_arr, res=res)
    assert res[0] == 6.0  # src[1] + src[3] = 2 + 4


def test_local_struct_allocatable_via_inlined_subprogram(tmp_path: Path):
    """Phase 5b: allocate happens inside a module-contained subroutine taking the struct as
    ``intent(inout)``. ``renameMemberAllocmems`` must follow the inlined alias chain
    (``hlfir.declare`` -> ``fir.embox``/``fir.convert`` -> caller's declare) to find and rename
    the allocate site, else the SDFG gets an unbound ``s_w_d0`` symbol.
    NOTE: ``entry='main'`` is REQUIRED -- without it the bridge walks the first public function
    in module order (``fill_and_set``), whose un-inlined body is unsupported.
    """
    src = """
module lib
  implicit none
  type t
    real, allocatable :: w(:)
  end type t
contains
  subroutine fill_and_set(s, n)
    type(t), intent(inout) :: s
    integer, intent(in) :: n
    integer :: i
    allocate(s%w(n))
    do i = 1, n
      s%w(i) = real(i * 10)
    end do
  end subroutine fill_and_set
end module lib

subroutine main(n, res)
  use lib
  implicit none
  integer, intent(in) :: n
  real, intent(out) :: res(3)
  type(t) :: s
  call fill_and_set(s, n)
  res(1) = s%w(1)
  res(2) = s%w(n / 2)
  res(3) = s%w(n)
  deallocate(s%w)
end subroutine main
"""
    sdfg = _build(src, tmp_path, entry='main')
    res = np.zeros(3, dtype=np.float32)
    sdfg(n=6, res=res)
    np.testing.assert_array_equal(res, [10.0, 30.0, 60.0])


def test_parametric_dim_from_struct_field(tmp_path: Path):
    """Phase 6: local array whose extent is a struct field's runtime value (``bob(st%a)``).
    ``st%a`` flattens to SDFG symbol ``st_a``; ``bob`` gets shape ``(st_a,)`` as a runtime-extent transient."""
    src = """
module lib
  implicit none
  type t
    integer :: a
  end type t
end module lib

subroutine main(n, res)
  use lib
  implicit none
  integer, intent(in) :: n
  real, intent(out) :: res(10)
  type(t) :: st
  st%a = n
  block
    real :: bob(st%a)
    bob(:) = 0.0
    bob(1) = 5.5
    res(1:st%a) = bob + 1.0
  end block
end subroutine main
"""
    sdfg = _build(src, tmp_path, entry='main')
    res = np.zeros(10, dtype=np.float32)
    sdfg(n=4, res=res)
    np.testing.assert_array_equal(res, [6.5, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0])


def test_parametric_dim_two_locals_one_struct(tmp_path: Path):
    """Phase 6: two parametric locals from sibling struct fields get independent SDFG symbols (``st_a``/``st_b``) that don't shadow each other."""
    src = """
module lib
  implicit none
  type t
    integer :: a
    integer :: b
  end type t
end module lib

subroutine main(av, bv, res)
  use lib
  implicit none
  integer, intent(in) :: av, bv
  real, intent(out) :: res(20)
  type(t) :: st
  st%a = av
  st%b = bv
  block
    real :: outer(st%a)
    real :: inner(st%b)
    integer :: i
    outer = 1.5
    inner = 2.5
    do i = 1, st%a
      res(i) = outer(i)
    end do
    do i = 1, st%b
      res(st%a + i) = inner(i)
    end do
  end block
end subroutine main
"""
    sdfg = _build(src, tmp_path, entry='main')
    res = np.zeros(20, dtype=np.float32)
    sdfg(av=3, bv=4, res=res)
    np.testing.assert_array_equal(res, [1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 2.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def test_parametric_dim_via_inlined_subprogram(tmp_path: Path):
    """Phase 6 + cross-subprogram: parametric local inside an inlined subroutine; pins that the
    runtime-extent symbol binds at the right scope after inlining."""
    src = """
module lib
  implicit none
  type t
    integer :: a
  end type t
contains
  subroutine fill(d, st)
    type(t), intent(in) :: st
    real, intent(out) :: d(10)
    real :: bob(st%a)
    integer :: i
    do i = 1, st%a
      bob(i) = real(i)
    end do
    do i = 1, st%a
      d(i) = bob(i) * 2.0
    end do
  end subroutine fill
end module lib

subroutine main(n, res)
  use lib
  implicit none
  integer, intent(in) :: n
  real, intent(out) :: res(10)
  type(t) :: st
  st%a = n
  call fill(res, st)
end subroutine main
"""
    sdfg = _build(src, tmp_path, entry='main')
    res = np.zeros(10, dtype=np.float32)
    sdfg(n=4, res=res)
    np.testing.assert_array_equal(res, [2, 4, 6, 8, 0, 0, 0, 0, 0, 0])


def test_local_struct_allocatable_member_reallocate(tmp_path: Path):
    """Phase 5a: allocate/deallocate/re-allocate cycle on the same member. Pins the
    multiple-allocate-site path: the second ``fir.allocmem`` gets its own renamed transient
    (``s_w_alloc1``) via ``allocAliasName``."""
    src = """
module lib
  implicit none
  type t
    real, allocatable :: w(:)
  end type t
end module lib

subroutine main(n1, n2, res)
  use lib
  implicit none
  integer, intent(in) :: n1, n2
  real, intent(out)   :: res
  type(t) :: s
  allocate(s%w(n1))
  s%w(1) = 7.0
  deallocate(s%w)
  allocate(s%w(n2))
  s%w(1) = 11.0
  res = s%w(1)
  deallocate(s%w)
end subroutine main
"""
    sdfg = _build(src, tmp_path)
    res = np.zeros(1, dtype=np.float32)
    sdfg(n1=3, n2=4, res=res)
    assert res[0] == 11.0


def test_aos_allocatable_uniform_const_size(tmp_path: Path):
    """Phase 5c-A: AoS + allocatable member with uniform compile-time-constant sizes flattens
    to a fully static ``A_w(N, M)`` companion; per-instance allocate/deallocate become no-ops.
    """
    src = """
module lib
  implicit none
  type t
    real, allocatable :: w(:)
  end type t
end module lib

subroutine main(out)
  use lib
  implicit none
  real, intent(out) :: out
  type(t) :: A(2)
  integer :: i, j
  do i = 1, 2
    allocate(A(i)%w(3))
    do j = 1, 3
      A(i)%w(j) = real(i * j)
    end do
  end do
  out = A(1)%w(1) + A(2)%w(2)
  do i = 1, 2
    deallocate(A(i)%w)
  end do
end subroutine main
"""
    sdfg = _build(src, tmp_path, entry='main')
    out = np.zeros(1, dtype=np.float32)
    sdfg(out=out)
    # A(1)%w(1) = 1*1 = 1, A(2)%w(2) = 2*2 = 4, sum = 5
    assert out[0] == 5.0


def test_aos_allocatable_via_inlined_kernel(tmp_path: Path):
    """Phase 5c-B: AoS+allocatable struct passed ``intent(inout)`` to an inlined kernel.
    ``collapseAosAllocReads`` must follow the inlined alias chain to find every
    member-designate read; without it, reads stay 1-D against the loaded 2D companion and
    silently produce wrong indices.
    """
    src = """
module lib
  implicit none
  type t
    real, allocatable :: w(:)
  end type t
contains
  subroutine kernel(A, n, m, out)
    type(t), intent(inout) :: A(2)
    integer, intent(in) :: n, m
    real, intent(out) :: out
    integer :: i, j
    do i = 1, n
      do j = 1, m
        A(i)%w(j) = A(i)%w(j) * 2.0
      end do
    end do
    out = A(1)%w(1) + A(2)%w(2)
  end subroutine kernel
end module lib

subroutine main(out)
  use lib
  implicit none
  real, intent(out) :: out
  type(t) :: A(2)
  integer :: i, j
  do i = 1, 2
    allocate(A(i)%w(3))
    do j = 1, 3
      A(i)%w(j) = real(i + j)
    end do
  end do
  call kernel(A, 2, 3, out)
  do i = 1, 2
    deallocate(A(i)%w)
  end do
end subroutine main
"""
    sdfg = _build(src, tmp_path, entry='main')
    out = np.zeros(1, dtype=np.float32)
    sdfg(out=out)
    # A(1)%w -> [4,6,8], A(2)%w -> [6,8,10] after doubling; out = A(1)%w(1)+A(2)%w(2) = 4+8 = 12.
    assert out[0] == 12.0


def test_aos_allocatable_whole_array_assign(tmp_path: Path):
    """Phase 5c-A: ``A(i)%w = scalar`` must lower to a row-section assign ``A_w(i, 1:M:1) =
    scalar``, NOT a whole-2D broadcast that would splat and corrupt earlier rows.
    Pinned by ``rewriteAosWholeMemberAssign`` in FlattenStructs.cpp.
    """
    src = """
module lib
  implicit none
  type t
    real, allocatable :: w(:)
  end type t
end module lib

subroutine main(out)
  use lib
  implicit none
  real, intent(out) :: out
  type(t) :: A(2)
  integer :: i
  do i = 1, 2
    allocate(A(i)%w(3))
    A(i)%w = real(i)         ! whole-array assign of scalar
  end do
  out = A(1)%w(1) + A(2)%w(2)
  do i = 1, 2
    deallocate(A(i)%w)
  end do
end subroutine main
"""
    sdfg = _build(src, tmp_path, entry='main')
    out = np.zeros(1, dtype=np.float32)
    sdfg(out=out)
    # A(1)%w = [1, 1, 1], A(2)%w = [2, 2, 2]; sum = 1 + 2 = 3.
    assert out[0] == 3.0


def test_batched_csr_allocatable_jagged(tmp_path: Path):
    """Genuinely jagged batched CSR: each instance's arrays are allocated to different constant
    sizes. ``aosAllocMaxConstSize`` sizes the companion to ``max_i(N_i)``;
    ``rewriteAosWholeMemberAssign`` resolves each instance's section to its own (smaller) size,
    not the global cap. SDFG output checked against gfortran/f2py at rtol=1e-12.
    """
    src = """
module lib
  implicit none
  type csr_t
    integer, allocatable :: rowptr(:)
    integer, allocatable :: colidx(:)
    real(8), allocatable :: val(:)
  end type csr_t
end module lib

subroutine main(out)
  use lib
  implicit none
  integer, parameter :: BATCH = 2, ROWS = 3
  real(8), intent(out) :: out(BATCH, ROWS)
  type(csr_t) :: A(BATCH)
  real(8) :: x(ROWS)
  integer :: b, r, k

  allocate(A(1)%rowptr(ROWS + 1), A(1)%colidx(3), A(1)%val(3))
  A(1)%rowptr = (/ 1, 2, 3, 4 /)
  A(1)%colidx = (/ 1, 2, 3 /)
  A(1)%val    = (/ 1.0d0, 1.0d0, 1.0d0 /)

  allocate(A(2)%rowptr(ROWS + 1), A(2)%colidx(4), A(2)%val(4))
  A(2)%rowptr = (/ 1, 3, 4, 5 /)
  A(2)%colidx = (/ 1, 2, 1, 3 /)
  A(2)%val    = (/ 2.0d0, -1.0d0, 1.0d0, 1.0d0 /)

  x = (/ 10.0d0, 20.0d0, 30.0d0 /)

  do b = 1, BATCH
    do r = 1, ROWS
      out(b, r) = 0.0d0
      do k = A(b)%rowptr(r), A(b)%rowptr(r + 1) - 1
        out(b, r) = out(b, r) + A(b)%val(k) * x(A(b)%colidx(k))
      end do
    end do
  end do
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "batched_csr_jagged_ref")
    out_ref = mod.main()

    sdfg = _build(src, tmp_path)
    out = np.zeros((2, 3), order='F', dtype=np.float64)
    sdfg(out=out)
    np.testing.assert_allclose(out, out_ref, rtol=1e-12)


def test_static_polymorphism_devirtualised(tmp_path: Path):
    """CLASS dispatch (``c%area()``) statically devirtualised by flang's ``fir-polymorphic-op``
    to ``circle_area``/``rect_area`` -- receiver type is always statically known, so the SDFG
    sees plain flat scalars. Pairs with the bail-out test in ``noncontig_unsupported_test.py``
    for when polymorphic-op can't resolve everything.
    """
    src = """
module shapes
  implicit none
  type :: circle_t
    real(8) :: r
  contains
    procedure :: area => circle_area
  end type circle_t

  type :: rect_t
    real(8) :: w, h
  contains
    procedure :: area => rect_area
  end type rect_t

contains
  function circle_area(self) result(a)
    class(circle_t), intent(in) :: self
    real(8) :: a
    a = 3.141592653589793d0 * self%r * self%r
  end function
  function rect_area(self) result(a)
    class(rect_t), intent(in) :: self
    real(8) :: a
    a = self%w * self%h
  end function
end module shapes

subroutine main(r, w, h, out)
  use shapes
  implicit none
  real(8), intent(in)  :: r, w, h
  real(8), intent(out) :: out(2)
  type(circle_t) :: c
  type(rect_t)   :: rect
  c%r = r
  rect%w = w
  rect%h = h
  ! Type-bound procedure call - flang devirtualises to the concrete
  ! ``circle_area`` / ``rect_area`` at compile time because the
  ! receiver's type is statically known.
  out(1) = c%area()
  out(2) = rect%area()
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "static_poly_ref")
    out_ref = mod.main(2.0, 3.0, 4.0)

    sdfg = _build(src, tmp_path, name='main', entry='main')
    out = np.zeros(2, dtype=np.float64)
    sdfg(r=2.0, w=3.0, h=4.0, out=out)
    np.testing.assert_allclose(out, out_ref, rtol=1e-12)


@pytest.mark.parametrize("call_arg,kwarg_for_sdfg", [("x", True), ("0.5d0", False)],
                         ids=["runtime_arg", "literal_constant"])
def test_class_as_monomorphic_box(tmp_path: Path, call_arg, kwarg_for_sdfg):
    """``CLASS(t) :: this`` as a non-polymorphic box (ECRAD/ICON: declared ``class(...)`` but
    every call site uses a concrete subtype). FlattenStructs treats ``fir.class<T>`` like
    ``fir.box<T>`` (both ``fir::BaseBoxType``). Parametrised runtime-arg vs literal-constant:
    the latter exercises ``hlfir-expand-vector-subscript-gather``'s associate-to-alloca rewrite.
    """
    src = f"""
module lib
  implicit none
  type :: pdf_sampler_t
    integer :: ncdf, nfsd
    real(8) :: fsd1, inv_fsd_interval
  end type pdf_sampler_t
contains
  subroutine evaluate(this, x, out)
    class(pdf_sampler_t), intent(in) :: this
    real(8),              intent(in)  :: x
    real(8),              intent(out) :: out
    out = x * this%fsd1 + this%inv_fsd_interval &
        + real(this%ncdf + this%nfsd, 8)
  end subroutine evaluate
end module lib

subroutine main(x, out)
  use lib
  implicit none
  real(8), intent(in)  :: x
  real(8), intent(out) :: out
  type(pdf_sampler_t) :: s
  s%ncdf = 3
  s%nfsd = 5
  s%fsd1 = 1.5d0
  s%inv_fsd_interval = 2.5d0
  call evaluate(s, {call_arg}, out)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "class_box_ref")
    out_ref = float(mod.main(0.5))

    sdfg = _build(src, tmp_path, name='main', entry='main')
    out = np.zeros(1, dtype=np.float64)
    if kwarg_for_sdfg:
        sdfg(x=0.5, out=out)
    else:
        # literal-constant case: x is unused (flang folds the constant inline) but the SDFG signature still binds it -- pass a placeholder.
        sdfg(x=0.0, out=out)
    np.testing.assert_allclose(float(out[0]), out_ref, rtol=1e-12)


def test_three_level_nested_struct(tmp_path: Path):
    """Three levels of pure-record nesting (Function->BasicBlock->Instruction shape); exercises
    the Phase 2 path walker at depth 3 and the flattened name ``f_bb_inst_pc``."""
    src = """
module lib
  implicit none
  type instr_t
    integer :: pc(8)
  end type instr_t
  type bb_t
    type(instr_t) :: inst
  end type bb_t
  type func_t
    type(bb_t) :: bb
  end type func_t
end module lib

subroutine main(d)
  use lib
  implicit none
  integer, intent(out) :: d(8)
  type(func_t) :: f
  integer :: i
  do i = 1, 8
    f%bb%inst%pc(i) = i * 7
  end do
  d(:) = f%bb%inst%pc(:)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "three_level_ref")
    d_ref = mod.main()

    sdfg = _build(src, tmp_path)
    d = np.zeros(8, dtype=np.int32)
    sdfg(d=d)
    np.testing.assert_array_equal(d, d_ref)


def test_local_struct_used_as_2d_assignment_target(tmp_path: Path):
    """Slice assignment ``s%w(:, k) = arr(:)`` into a struct's 2-D array member exercises the section-to-section path onto a flat per-field array."""
    src = """
module lib
  implicit none
  type two_d
    real :: w(3, 4)
  end type two_d
end module lib

subroutine main(arr, out)
  use lib
  implicit none
  real, intent(in)  :: arr(3)
  real, intent(out) :: out(3)
  type(two_d) :: t
  integer :: i
  do i = 1, 3
    t%w(i, 2) = arr(i) * 10.0
  end do
  out(:) = t%w(:, 2)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "local_struct_2d_ref")
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32, order="F")
    out_ref = np.asarray(mod.main(arr), dtype=np.float32)

    sdfg = _build(src, tmp_path)
    out = np.zeros(3, dtype=np.float32)
    sdfg(arr=arr, out=out)
    np.testing.assert_array_equal(out, out_ref)
    np.testing.assert_array_equal(out, [10.0, 20.0, 30.0])


def test_nested_struct_lowered_via_phase2(tmp_path: Path):
    """Phase 2: nested struct (member is itself a struct). ``hlfir-flatten-structs`` synthesises
    one declare per leaf named ``<base>_<m1>_..._<leaf>`` (here ``o_inner_x``); designates rewrite accordingly."""
    src = """
module lib
  implicit none
  type inner_t
    real :: x(5)
  end type inner_t
  type outer_t
    type(inner_t) :: inner
  end type outer_t
end module lib

subroutine main(d)
  use lib
  implicit none
  real, intent(out) :: d(1)
  type(outer_t) :: o
  o%inner%x(1) = 1.0
  d(1) = o%inner%x(1)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "nested_struct_ref")
    d_ref = np.asarray(mod.main(), dtype=np.float32)

    sdfg = _build(src, tmp_path)
    d = np.zeros(1, dtype=np.float32)
    sdfg(d=d)
    np.testing.assert_array_equal(d, d_ref)


def test_array_of_nested_struct_member(tmp_path: Path):
    """Phase 2 extension: struct member is an ARRAY of another struct. Flat companion folds the
    array dim into the leaf's shape (``p_prog%pprog(i)%w(j,k)`` -> ``p_prog_pprog_w(i,j,k)``);
    exercises ``collectFlatLeaves``'s ``array<N x RecordType>`` branch.
    """
    src = """
module lib
  implicit none
  type simple_type
    real :: w(5, 5)
  end type simple_type
  type simple_type2
    type(simple_type) :: pprog(10)
  end type simple_type2
end module lib

subroutine main(d)
  use lib
  implicit none
  real, intent(out) :: d(5, 5)
  type(simple_type2) :: p_prog
  p_prog%pprog(1)%w(1, 1) = 47.0
  d(1, 1) = p_prog%pprog(1)%w(1, 1)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "array_of_nested_struct_ref")
    d_ref = np.asarray(mod.main(), dtype=np.float32)

    sdfg = _build(src, tmp_path)
    d = np.zeros((5, 5), order="F", dtype=np.float32)
    sdfg(d=d)
    np.testing.assert_array_equal(d, d_ref)
    assert d[0][0] == 47.0


def test_outer_array_of_nested_struct(tmp_path: Path):
    """Phase 2 extension: outer-array of a nested struct (``s(3)`` over ``simple_type{inner{w(5,5)}}``)
    collapses to ``s_inner_w`` shape ``(3,5,5)``; exercises ``isLocallyFlattenable`` +
    ``splitLocal`` with ``outerIsArray = true``.
    """
    src = """
module lib
  implicit none
  type inner_t
    real :: w(5, 5)
  end type inner_t
  type simple_t
    type(inner_t) :: inner
  end type simple_t
end module lib

subroutine main(d)
  use lib
  implicit none
  real, intent(out) :: d(2)
  type(simple_t) :: s(3)
  s(1)%inner%w(1, 1) = 11.0
  s(3)%inner%w(2, 2) = 33.0
  d(1) = s(1)%inner%w(1, 1)
  d(2) = s(3)%inner%w(2, 2)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "outer_array_nested_ref")
    d_ref = np.asarray(mod.main(), dtype=np.float32)

    sdfg = _build(src, tmp_path)
    d = np.zeros(2, dtype=np.float32)
    sdfg(d=d)
    np.testing.assert_array_equal(d, d_ref)
    np.testing.assert_array_equal(d, [11.0, 33.0])


def test_aos_member_to_member_array_copy(tmp_path: Path):
    """AoS pattern ``a(i)%b = a(j)%c`` (array members) flattens to a whole-row section copy
    ``a_b(i, :) = a_c(j, :)``. Exercises AoS index-merging (``rewriteDesignate`` concat case)
    together with the whole-member triplet section path (no inner indices on the designate).
    """
    src = """
module lib
  implicit none
  type pair
    integer :: b(4)
    integer :: c(4)
  end type pair
end module lib

subroutine main(out)
  use lib
  implicit none
  integer, intent(out) :: out(4, 2)
  type(pair) :: a(3)
  integer :: i, k

  ! Initialise: a(j)%c(k) = j*10 + k
  do i = 1, 3
    do k = 1, 4
      a(i)%c(k) = i * 10 + k
      a(i)%b(k) = 0
    end do
  end do

  ! Whole-row copy: a(1)%b <- a(3)%c  (copy row 3 of c into row 1 of b)
  a(1)%b = a(3)%c

  ! Read back
  do k = 1, 4
    out(k, 1) = a(1)%b(k)
    out(k, 2) = a(3)%c(k)
  end do
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "aos_row_copy_ref")
    # ``intent(out)`` dummy: f2py returns it.
    out_ref = np.asarray(mod.main(), dtype=np.int32)

    sdfg = _build(src, tmp_path)
    out = np.zeros((4, 2), order="F", dtype=np.int32)
    sdfg(out=out)
    np.testing.assert_array_equal(out, out_ref)
    # both columns hold row [31, 32, 33, 34] (a(3)%c, copied into a(1)%b)
    np.testing.assert_array_equal(out[:, 0], [31, 32, 33, 34])
    np.testing.assert_array_equal(out[:, 1], [31, 32, 33, 34])


def test_intent_out_aos_alloc_member_raises_loudly(tmp_path, capfd):
    """``intent(out)`` AoS dummy with an allocatable member: F2003 auto-deallocates on entry,
    so there's no caller data to size the padded companion. ``hlfir-flatten-structs`` must fail
    loudly, not emit a degenerate cap-1 buffer that silently truncates writes.
    """
    src = """
module mo_ki
  use iso_c_binding
  implicit none
  integer, parameter :: N = 3
  type :: bag
     real(c_double), allocatable :: w(:)
  end type bag
end module mo_ki
subroutine kern_ki(a, m)
  use mo_ki
  implicit none
  type(bag), intent(out) :: a(N)
  integer, intent(in) :: m
  integer :: i, j
  do i = 1, N
     allocate(a(i)%w(m))
     do j = 1, m
        a(i)%w(j) = real(j, c_double)
     end do
  end do
end subroutine kern_ki
"""
    with pytest.raises(Exception):
        _build(src, tmp_path, name="kern_ki", entry="kern_ki")
    assert "intent(out)" in capfd.readouterr().err
