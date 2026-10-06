"""ALLOCATABLE members of an array of records allocated in the kernel with runtime or multi-dimensional extents:
flattened ELLPACK-style into a companion padded to the largest extent seen, sized before the first ALLOCATE."""

import numpy as np

from tests._util import build_sdfg


def _run(tmp_path, source: str, **args) -> float:
    sdfg = build_sdfg(source, tmp_path, name="ragged", entry="m::k").build()
    out = np.zeros(1)
    sdfg(out=out, **args)
    return float(out[0])


def test_runtime_extent_per_element(tmp_path):
    """``ALLOCATE(p(i) % w(i))``: element ``i`` holds ``i`` values; whole-member reads and ``SIZE`` see each live row."""
    src = """
module m
  implicit none
  type packet
    real(8), allocatable :: w(:)
  end type
contains
  subroutine k(out)
    real(8), intent(out) :: out
    type(packet) :: p(4)
    integer :: i
    do i = 1, 4
      allocate(p(i) % w(i))
      p(i) % w = i
    end do
    out = 0
    do i = 1, 4
      out = out + sum(p(i) % w) + 100 * size(p(i) % w)
    end do
  end subroutine
end module
"""
    assert _run(tmp_path, src) == sum(i * i + 100 * i for i in range(1, 5))


def test_counting_loop_bounds_the_extent(tmp_path):
    """``cnt = 0; DO j = 1, n; IF (src(j) == i) cnt = cnt + 1; END DO; ALLOCATE(p(i) % w(cnt))`` (QE
    ``initialize_local_to_exact_map``): the companion is bounded by the counting loop's trip count ``n``."""
    src = """
module m
  implicit none
  type packet
    integer, allocatable :: idx(:)
  end type
contains
  subroutine k(n, src, out)
    integer, intent(in) :: n
    integer, intent(in) :: src(n)
    real(8), intent(out) :: out
    type(packet) :: p(3)
    integer :: i, j, cnt
    do i = 1, 3
      cnt = 0
      do j = 1, n
        if (src(j) == i) cnt = cnt + 1
      end do
      allocate(p(i) % idx(cnt))
      cnt = 0
      do j = 1, n
        if (src(j) == i) then
          cnt = cnt + 1
          p(i) % idx(cnt) = j
        end if
      end do
    end do
    out = 0
    do i = 1, 3
      out = out + 1000 * size(p(i) % idx) + sum(p(i) % idx * i)
    end do
  end subroutine
end module
"""
    src_vals = np.array([2, 1, 2, 3, 2, 1], dtype=np.int32)
    expected = sum(
        1000 * int((src_vals == i).sum()) + i * sum(j + 1 for j in range(len(src_vals)) if src_vals[j] == i)
        for i in range(1, 4)
    )
    assert _run(tmp_path, src, n=len(src_vals), src=src_vals) == expected


def test_constant_rank_two_member(tmp_path):
    """``ALLOCATE(p(i) % w(3, 2))``: the companion keeps both member dimensions."""
    src = """
module m
  implicit none
  type packet
    real(8), allocatable :: w(:, :)
  end type
contains
  subroutine k(out)
    real(8), intent(out) :: out
    type(packet) :: p(4)
    integer :: i, j, l
    do i = 1, 4
      allocate(p(i) % w(3, 2))
      do l = 1, 2
        do j = 1, 3
          p(i) % w(j, l) = i * j + 10 * l
        end do
      end do
    end do
    out = sum(p(2) % w) + size(p(3) % w, 1) * 1000 + size(p(3) % w, 2) * 10000
  end subroutine
end module
"""
    expected = sum(2 * j + 10 * k for j in range(1, 4) for k in range(1, 3)) + 3000 + 20000
    assert _run(tmp_path, src) == expected


def test_allocatable_array_of_records(tmp_path):
    """``ALLOCATE(p(n))`` then ``ALLOCATE(p(i) % w(i))``: the outer extent of the companion is the array's own."""
    src = """
module m
  implicit none
  type packet
    real(8), allocatable :: w(:)
  end type
contains
  subroutine k(n, out)
    integer, intent(in) :: n
    real(8), intent(out) :: out
    type(packet), allocatable :: p(:)
    integer :: i
    allocate(p(n))
    do i = 1, n
      allocate(p(i) % w(i + 1))
      p(i) % w = i
    end do
    out = 0
    do i = 1, n
      out = out + sum(p(i) % w)
    end do
  end subroutine
end module
"""
    n = 5
    assert _run(tmp_path, src, n=n) == sum(i * (i + 1) for i in range(1, n + 1))


def test_extent_read_from_an_array_element(tmp_path):
    """``ALLOCATE(p(i) % w(cnt(i), 2))``: the cap loop reads ``cnt(i)`` itself, subscript included."""
    src = """
module m
  implicit none
  type packet
    real(8), allocatable :: w(:, :)
  end type
contains
  subroutine k(n, cnt, out)
    integer, intent(in) :: n
    integer, intent(in) :: cnt(n)
    real(8), intent(out) :: out
    type(packet) :: p(4)
    integer :: i, j
    do i = 1, n
      allocate(p(i) % w(cnt(i), 2))
      do j = 1, cnt(i)
        p(i) % w(j, :) = i + 10 * j
      end do
    end do
    out = 0
    do i = 1, n
      out = out + sum(p(i) % w)
    end do
  end subroutine
end module
"""
    cnt = np.array([3, 1, 4, 2], dtype=np.int32)
    expected = sum(2 * (i + 10 * j) for i in range(1, cnt.size + 1) for j in range(1, cnt[i - 1] + 1))
    assert _run(tmp_path, src, n=cnt.size, cnt=cnt) == expected


def test_host_allocated_module_array_of_records(tmp_path):
    """A module ``type(packet), allocatable :: p(:)`` the host allocated (QE exx_bp_utils' comm packets): the kernel
    never ALLOCATEs ``p``, so the companion's outer extent is read from ``p``'s descriptor, a free symbol."""
    src = """
module packets
  implicit none
  type packet
    real(8), allocatable :: w(:, :)
  end type
  type(packet), allocatable :: p(:)
end module

module m
  implicit none
contains
  subroutine k(n, cnt, out)
    use packets, only: p
    integer, intent(in) :: n
    integer, intent(in) :: cnt(n)
    real(8), intent(out) :: out
    integer :: i, j
    do i = 1, n
      allocate(p(i) % w(cnt(i), 2))
      do j = 1, cnt(i)
        p(i) % w(j, :) = i + 10 * j
      end do
    end do
    out = 0
    do i = 1, n
      out = out + sum(p(i) % w)
    end do
  end subroutine
end module
"""
    cnt = np.array([3, 1, 4, 2], dtype=np.int32)
    n = cnt.size
    expected = sum(2 * (i + 10 * j) for i in range(1, n + 1) for j in range(1, cnt[i - 1] + 1))
    assert _run(tmp_path, src, n=n, cnt=cnt, p_d0=n) == expected


if __name__ == "__main__":
    import pathlib
    import tempfile

    for test in (
        test_runtime_extent_per_element,
        test_counting_loop_bounds_the_extent,
        test_constant_rank_two_member,
        test_allocatable_array_of_records,
        test_extent_read_from_an_array_element,
        test_host_allocated_module_array_of_records,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            test(pathlib.Path(tmp))
