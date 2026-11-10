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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
