"""Verbatim port of f2dace/dev:tests/fortran/allocate_test.py."""

import numpy as np
import pytest

from tests._util import build_sdfg


def test_fortran_frontend_basic_allocate(tmp_path):
    src = """
subroutine main(d)
  double precision, allocatable, intent(out) :: d(:, :)
  allocate (d(4, 5))
  d(2, 1) = 5.5
end subroutine main"""
    sdfg = build_sdfg(src, tmp_path, name="main").build()
    a = np.full([4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert a[0, 0] == 42
    assert a[1, 0] == 5.5
    assert a[2, 0] == 42


def test_allocating_a_member_of_one_array_element_raises(tmp_path):
    """``ALLOCATE(send(i) % msg(i, ...))`` sizes one element's member (QE ``initialize_local_to_exact_map``), while
    the flattened ``send_msg`` companion has one extent per member dimension for every element: the build names the
    construct and points to ``keep_external`` instead of resizing all elements to this one."""
    src = """
module comm
  implicit none
  type packet
    complex(8), allocatable :: msg(:, :)
  end type
  type(packet), allocatable :: send(:)
contains
  subroutine setup(n, total)
    integer, intent(in) :: n
    integer, intent(out) :: total
    allocate(send(n))
    allocate(send(n) % msg(n, 3))
    total = size(send(n) % msg, 2)
  end subroutine
end module
"""
    with pytest.raises(
        NotImplementedError, match=r"ALLOCATE of the component 'msg' of an array element.*keep_external"
    ):
        build_sdfg(src, tmp_path, name="setup", entry="comm::setup").build()


@pytest.mark.parametrize("flag", [0, 1])
def test_allocate_extent_reads_an_element_written_before(tmp_path, flag):
    """QE ``fwfft_orbital_gamma``'s ``ALLOCATE(psio(ngk(1), ...))``: the extent is the element's value AT the
    ALLOCATE, not at entry."""
    src = """
module m
  integer :: ngk(2)
end module
subroutine top(flag, s)
  use m
  logical, intent(in) :: flag
  real(8), intent(out) :: s
  real(8), allocatable :: w(:)
  if (flag) ngk = 4
  allocate (w(ngk(1)))
  w = 1.0d0
  s = sum(w)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="top", entry="top").build()
    assert "__sym_ngk_1" not in sdfg.arrays["w"].free_symbols
    s, ngk = np.zeros(1), np.array([2, 2], np.int32)
    sdfg(flag=np.int32(flag), s=s, ngk=ngk)
    assert s[0] == (4.0 if flag else 2.0)


def test_allocate_with_source_copies_the_source(tmp_path):
    """QE vexx ``ALLOCATE(psi_d, SOURCE = psi)``: the allocatable takes the source's shape and values (it stayed
    unsized and empty before)."""
    src = """
subroutine top(n, m, a, b)
  integer, intent(in) :: n, m
  complex(8), intent(in) :: a(n, m)
  complex(8), intent(out) :: b(n, m)
  complex(8), allocatable :: t(:, :)
  allocate(t, source=a)
  t(1, 1) = t(1, 1) + 1
  b = 2 * t
  deallocate(t)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="top", entry="top").build()
    rng = np.random.default_rng(0)
    a = np.asfortranarray(rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2)))
    b = np.zeros_like(a, order="F")
    sdfg(n=3, m=2, a=a, b=b)
    want = 2 * a
    want[0, 0] += 2
    np.testing.assert_allclose(b, want, rtol=1e-14)


# Reproducer: ICON velocity and solve_nh stopped building: their sources hold a CLASS(*) ``ALLOCATE(.., SOURCE=..)``
# (mo_key_value_store) the kernel never calls, and the pass failed the whole module on it
def test_an_unsupported_allocate_with_source_the_kernel_never_calls_builds(tmp_path):
    src = """
module store
  implicit none
contains
  subroutine put(val)
    class(*), intent(in) :: val
    class(*), pointer :: obj
    allocate(obj, source=val)
  end subroutine
end module

subroutine top(n, a, b)
  integer, intent(in) :: n
  double precision, intent(in) :: a(n)
  double precision, intent(out) :: b(n)
  b = 2 * a
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="top", entry="top").build()
    a = np.arange(4, dtype=np.float64)
    b = np.zeros(4)
    sdfg(n=4, a=a, b=b)
    np.testing.assert_array_equal(b, 2 * a)


def test_a_scalar_allocate_with_source_the_kernel_runs_is_refused(tmp_path):
    """The copy a scalar ``ALLOCATE(.., SOURCE=..)`` performs has no lowering: dropping it would leave ``t`` empty."""
    src = """
subroutine top(x, y)
  double precision, intent(in) :: x
  double precision, intent(out) :: y
  double precision, allocatable :: t
  allocate(t, source=x)
  y = 2 * t
  deallocate(t)
end subroutine
"""
    with pytest.raises(Exception, match=r"SOURCE=\.\.\.\) of a scalar or polymorphic allocatable"):
        build_sdfg(src, tmp_path, name="top", entry="top").build()


# Reproducer: QE h_psi stopped building: ``ALLOCATE(iexx_istart_d, SOURCE = iexx_istart)`` copies an allocatable whose
# lower bound flang reads off its descriptor, which the pass left as a runtime call
@pytest.mark.parametrize("low", [0, 1])
def test_allocate_with_source_takes_the_lower_bound_of_an_allocatable_source(tmp_path, low):
    src = f"""
subroutine top(n, b, lo)
  integer, intent(in) :: n
  integer, intent(out) :: b(n), lo
  integer, allocatable :: s(:), t(:)
  integer :: i
  allocate(s({low}:n + {low} - 1))
  do i = {low}, n + {low} - 1
    s(i) = 10 * i
  end do
  allocate(t, source=s)
  lo = lbound(t, 1)
  do i = 1, n
    b(i) = t(i + {low} - 1) + 1
  end do
  deallocate(s, t)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="top", entry="top").build()
    b, lo = np.zeros(5, np.int32), np.zeros(1, np.int32)
    sdfg(n=5, b=b, lo=lo)
    assert lo[0] == low
    np.testing.assert_array_equal(b, 10 * np.arange(low, low + 5) + 1)


def test_allocate_with_source_of_a_run_time_lower_bound_is_refused(tmp_path):
    """The copy's lower bound would be ``k``, which the frontend cannot name for it: refused, never taken as 1."""
    src = """
subroutine top(n, k, b)
  integer, intent(in) :: n, k
  integer, intent(out) :: b(n)
  integer, allocatable :: s(:), t(:)
  allocate(s(k:k + n - 1))
  s = 1
  allocate(t, source=s)
  b = t
  deallocate(s, t)
end subroutine
"""
    with pytest.raises(Exception, match=r"SOURCE=\.\.\.\)"):
        build_sdfg(src, tmp_path, name="top", entry="top").build()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
