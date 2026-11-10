"""One regression per bridge fix: each kernel runs through the SDFG and a gfortran reference, or is refused loudly."""

import ctypes
import re
from pathlib import Path

import numpy as np
import pytest

from tests._util import build_sdfg, gfortran_compile_so

I4 = np.int32
I8 = np.int64
F4 = np.float32
F8 = np.float64
C16 = np.complex128
FALSE = I4(0)
ONE_TO_NINE = np.arange(1.0, 10.0)
THREE_BY_FOUR = np.arange(1.0, 13.0).reshape(3, 4, order="F")
SQUARE = np.arange(1.0, 10.0).reshape(3, 3, order="F")


def as_buffer(value):
    """A Fortran-ordered private copy a callee may write through."""
    return np.array(value, order="F", copy=True) if isinstance(value, np.ndarray) else np.array([value])


def run_reference(src: str, entry: str, args: dict, tmp_path: Path) -> dict:
    source = tmp_path / "reference.f90"
    source.write_text(src)
    shared = tmp_path / "reference.so"
    gfortran_compile_so(shared, source, mod_dir=tmp_path)
    buffers = {name: as_buffer(value) for name, value in args.items()}
    dummies = re.search(rf"subroutine\s+{entry}\s*\(([^)]*)\)", src).group(1)
    pointers = [buffers[d.strip()].ctypes.data_as(ctypes.c_void_p) for d in dummies.split(",")]
    getattr(ctypes.CDLL(str(shared)), f"{entry}_")(*pointers)
    return buffers


def run_sdfg(src: str, entry: str, args: dict, tmp_path: Path):
    sdfg = build_sdfg(src, tmp_path, name=entry, entry=entry).build()
    sdfg.validate()
    arguments, buffers = {}, {}
    for name, value in args.items():
        buffer = as_buffer(value)
        desc = sdfg.arrays.get(name)
        if desc is None or desc.transient or type(desc).__name__ == "Scalar":
            arguments[name] = buffer.flat[0].item()
            continue
        buffers[name] = arguments[name] = buffer.astype(desc.dtype.as_numpy_dtype(), order="F")
    sdfg(**arguments)
    return sdfg, buffers


# (kernel with @ for its name, inputs, name of the transient family the fix must add or None)
CASES = {
    "intent_out_index": (
        """subroutine @(n, a, idx)
  integer, intent(in) :: n
  real(8), intent(in) :: a(n)
  integer, intent(out) :: idx
  integer :: i
  idx = 1
  do i = 2, n
    if (a(i) > a(idx)) idx = i
  end do
end subroutine""",
        {"n": I4(5), "a": np.array([1.0, 5.0, 3.0, 7.0, 2.0]), "idx": I4(0)},
        None,
    ),
    "intent_out_do_variable": (
        """subroutine @(n, last, a)
  integer, intent(in) :: n
  integer, intent(out) :: last
  real(8), intent(inout) :: a(n)
  do last = 1, n
    a(last) = last
  end do
end subroutine""",
        {"n": I4(5), "last": I4(0), "a": np.zeros(5)},
        None,
    ),
    "strided_dummy_write": (
        """subroutine @(a, n)
  integer, intent(in) :: n
  real(8), intent(inout) :: a(n, 4)
  call inner(a(1:n:2, 2))
contains
  subroutine inner(x)
    real(8), intent(inout) :: x(:)
    integer :: i
    do i = 1, size(x)
      x(i) = -1
    end do
  end subroutine
end subroutine""",
        {"a": THREE_BY_FOUR, "n": I4(3)},
        None,
    ),
    "unit_section_dummy": (
        """subroutine @(a, n)
  integer, intent(in) :: n
  real(8), intent(inout) :: a(n + 2)
  call inner(a(3:n))
contains
  subroutine inner(x)
    real(8), intent(inout) :: x(:)
    integer :: i
    do i = 1, size(x)
      x(i) = -1
    end do
  end subroutine
end subroutine""",
        {"a": ONE_TO_NINE, "n": I4(7)},
        None,
    ),
    "section_reduce_stride": (
        """subroutine @(a, s)
  real(8), intent(in) :: a(9)
  real(8), intent(out) :: s
  s = sum(a(9:1:-2))
end subroutine""",
        {"a": ONE_TO_NINE, "s": F8(0)},
        None,
    ),
    "section_fill_stride": (
        """subroutine @(a)
  real(8), intent(inout) :: a(9)
  a(9:1:-2) = 5.0d0
end subroutine""",
        {"a": ONE_TO_NINE},
        None,
    ),
    "section_zero_fill_stride": (
        """subroutine @(a)
  real(8), intent(inout) :: a(9)
  a(1:9:2) = 0.0d0
end subroutine""",
        {"a": ONE_TO_NINE},
        None,
    ),
    "overlapping_section": (
        """subroutine @(a, n)
  integer, intent(in) :: n
  real(8), intent(inout) :: a(n)
  a(2:n) = a(1:n-1)
end subroutine""",
        {"a": np.arange(1.0, 6.0), "n": I4(5)},
        "_rhs_buffer",
    ),
    "overlapping_elemental": (
        """subroutine @(a, n)
  integer, intent(in) :: n
  real(8), intent(inout) :: a(n)
  a(2:n) = a(1:n-1) * 2.0d0
end subroutine""",
        {"a": np.arange(1.0, 6.0), "n": I4(5)},
        "_rhs_buffer",
    ),
    "condition_lower_bound": (
        """subroutine @(a, n, k)
  integer, intent(in) :: n
  real(8), intent(in) :: a(0:n)
  integer, intent(out) :: k
  integer :: i
  k = -1
  do i = 0, n
    if (a(i) > 1) exit
    k = i
  end do
end subroutine""",
        {"a": np.array([0.0, 0.5, 2.0, 0.0]), "n": I4(3), "k": I4(0)},
        None,
    ),
    "condition_symbolic_lower_bound": (
        """subroutine @(a, k, n, r)
  integer, intent(in) :: k, n
  real(8), intent(in) :: a(k:n)
  integer, intent(out) :: r
  integer :: i
  r = 0
  do i = k, n
    if (a(i) > 1.5d0) r = i
  end do
end subroutine""",
        {"a": np.array([0.0, 1.0, 2.0, 1.0]), "k": I4(3), "n": I4(6), "r": I4(0)},
        None,
    ),
    "merge_read_order": (
        """subroutine @(a, r)
  real(8), intent(in) :: a(3)
  real(8), intent(out) :: r
  r = merge(a(1), a(2), a(3) > 0.0d0)
end subroutine""",
        {"a": np.array([-1.0, 5.0, 7.0]), "r": F8(0)},
        None,
    ),
    "conjugate_read_count": (
        """subroutine @(z, r)
  complex(8), intent(in) :: z(2)
  complex(8), intent(out) :: r
  r = conjg(z(1)) * z(2)
end subroutine""",
        {"z": np.array([1 + 2j, 3 - 4j]), "r": C16(0)},
        None,
    ),
    "elemental_max_read_count": (
        """subroutine @(x, r, n)
  integer, intent(in) :: n
  real(8), intent(in) :: x(n)
  real(8), intent(inout) :: r(n)
  r(2:n-1) = max(x(1:n-2), x(2:n-1), x(3:n))
end subroutine""",
        {"x": np.array([1.0, 5.0, 2.0, 8.0, 3.0, 4.0]), "r": np.zeros(6), "n": I4(6)},
        None,
    ),
    "condition_real_cast": (
        """subroutine @(i, j, r)
  integer, intent(in) :: i, j
  integer, intent(out) :: r
  r = 0
  if (real(i) / real(j) > 0.4) r = 1
end subroutine""",
        {"i": I4(1), "j": I4(2), "r": I4(0)},
        None,
    ),
    "condition_int_truncation": (
        """subroutine @(x, r)
  real(8), intent(in) :: x
  integer, intent(out) :: r
  logical :: flag
  flag = int(x) > 0
  r = 0
  if (flag) r = 1
end subroutine""",
        {"x": F8(0.5), "r": I4(9)},
        None,
    ),
    "condition_wide_product": (
        """subroutine @(a, b, r)
  integer, intent(in) :: a, b
  integer, intent(out) :: r
  r = 0
  if (int(a, 8) * int(b, 8) > 5000000000_8) r = 1
end subroutine""",
        {"a": I4(100000), "b": I4(100000), "r": I4(0)},
        None,
    ),
    "condition_real4_literal": (
        """subroutine @(x, r)
  real(4), intent(in) :: x
  integer, intent(out) :: r
  r = 0
  if (x >= 0.7) r = 1
end subroutine""",
        {"x": F4(0.7), "r": I4(0)},
        None,
    ),
    "maxloc_back_false": (
        """subroutine @(a, i)
  real(8), intent(in) :: a(3)
  integer, intent(out) :: i
  i = maxloc(a, dim=1, back=.false.)
end subroutine""",
        {"a": np.array([1.0, 3.0, 3.0]), "i": I4(0)},
        None,
    ),
    "maxloc_dim_mask": (
        """subroutine @(a, i)
  real(8), intent(in) :: a(3)
  integer, intent(out) :: i
  i = maxloc(a, dim=1, mask=a < 2.0d0)
end subroutine""",
        {"a": np.array([1.0, 3.0, 0.0]), "i": I4(0)},
        None,
    ),
    "integer_power_int8": (
        """subroutine @(b, e, r)
  integer(8), intent(in) :: b, e
  integer(8), intent(out) :: r
  r = b ** e
end subroutine""",
        {"b": I8(3), "e": I8(39), "r": I8(0)},
        None,
    ),
    "integer_power_truncating_quotient": (
        """subroutine @(i, j, k)
  integer, intent(in) :: i, j
  integer, intent(out) :: k
  k = (i ** j) / 2 * 2
end subroutine""",
        {"i": I4(3), "j": I4(1), "k": I4(0)},
        None,
    ),
    "integer_power_negative_exponent": (
        """subroutine @(j, k)
  integer, intent(in) :: j
  integer, intent(out) :: k
  k = 4 * 2 ** j
end subroutine""",
        {"j": I4(-1), "k": I4(9)},
        None,
    ),
    "real_narrowing": (
        """subroutine @(x8, r4)
  real(8), intent(in) :: x8
  real(4), intent(out) :: r4
  r4 = real(x8, 4) - 1.0
end subroutine""",
        {"x8": F8(1.0 + 1e-10), "r4": F4(9)},
        None,
    ),
    "do_bound_int_of_real": (
        """subroutine @(x, c)
  real(8), intent(in) :: x
  integer, intent(out) :: c
  integer :: i
  c = 0
  do i = 1, int(x)
    c = c + 1
  end do
end subroutine""",
        {"x": F8(2.5), "c": I4(0)},
        None,
    ),
    "do_bound_merge": (
        """subroutine @(n, flag, c)
  integer, intent(in) :: n
  logical, intent(in) :: flag
  integer, intent(out) :: c
  integer :: i
  c = 0
  do i = 1, merge(n, 0, flag)
    c = c + 1
  end do
end subroutine""",
        {"n": I4(5), "flag": FALSE, "c": I4(9)},
        None,
    ),
    "do_bound_merge_constant": (
        """subroutine @(flag, c)
  logical, intent(in) :: flag
  integer, intent(out) :: c
  integer :: i
  c = 0
  do i = 1, merge(5, 0, flag)
    c = c + 1
  end do
end subroutine""",
        {"flag": FALSE, "c": I4(9)},
        None,
    ),
    "reduction_of_bit_op": (
        """subroutine @(a, b, s)
  integer, intent(in) :: a(2), b(2)
  integer, intent(out) :: s
  s = maxval(ior(a, b))
end subroutine""",
        {"a": np.array([6, 1], dtype=I4), "b": np.array([3, 8], dtype=I4), "s": I4(0)},
        None,
    ),
    "ieee_rint_ties_to_even": (
        """subroutine @(x, y)
  use, intrinsic :: ieee_arithmetic
  real(8), intent(in) :: x
  real(8), intent(out) :: y
  y = ieee_rint(x)
end subroutine""",
        {"x": F8(2.5), "y": F8(0)},
        None,
    ),
    "matmul_overwrites_operand": (
        """subroutine @(c, d)
  real(8), intent(inout) :: c(3, 3)
  real(8), intent(in) :: d(3, 3)
  c = matmul(c, d)
end subroutine""",
        {"c": SQUARE, "d": SQUARE.T.copy()},
        "c_rhs",
    ),
    "transpose_overwrites_operand": (
        """subroutine @(c)
  real(8), intent(inout) :: c(3, 3)
  c = transpose(c)
end subroutine""",
        {"c": SQUARE},
        "c_rhs",
    ),
    "section_alias_copy": (
        """subroutine @(a, b, n, k)
  integer, intent(in) :: n, k
  real(8), intent(inout) :: a(n, 4), b(n)
  call inner(a(:, k), b, n)
contains
  subroutine inner(x, y, l)
    integer, intent(in) :: l
    real(8), intent(inout) :: x(l), y(l)
    y = x
  end subroutine
end subroutine""",
        {"a": THREE_BY_FOUR, "b": np.zeros(3), "n": I4(3), "k": I4(2)},
        None,
    ),
    "section_alias_reduce": (
        """subroutine @(a, n, k, d)
  integer, intent(in) :: n, k
  real(8), intent(inout) :: a(n, 4)
  real(8), intent(out) :: d
  call inner(a(:, k), n, d)
contains
  subroutine inner(x, l, s)
    integer, intent(in) :: l
    real(8), intent(inout) :: x(l)
    real(8), intent(out) :: s
    s = sum(x)
  end subroutine
end subroutine""",
        {"a": THREE_BY_FOUR, "n": I4(3), "k": I4(2), "d": F8(0)},
        None,
    ),
    "section_alias_dot_product": (
        """subroutine @(a, b, n, k, d)
  integer, intent(in) :: n, k
  real(8), intent(inout) :: a(n, 4), b(n)
  real(8), intent(out) :: d
  call inner(a(:, k), b, n, d)
contains
  subroutine inner(x, y, l, r)
    integer, intent(in) :: l
    real(8), intent(inout) :: x(l), y(l)
    real(8), intent(out) :: r
    r = dot_product(x, y)
  end subroutine
end subroutine""",
        {"a": THREE_BY_FOUR, "b": np.arange(7.0, 10.0), "n": I4(3), "k": I4(2), "d": F8(0)},
        None,
    ),
    "section_alias_fill_lower_bound": (
        """subroutine @(a, n, k)
  integer, intent(in) :: n, k
  real(8), intent(inout) :: a(n, 0:3)
  call inner(a(:, k), n)
contains
  subroutine inner(x, l)
    integer, intent(in) :: l
    real(8), intent(inout) :: x(l)
    x = 0
  end subroutine
end subroutine""",
        {"a": THREE_BY_FOUR, "n": I4(3), "k": I4(2)},
        None,
    ),
    "exit_after_increment": (
        """subroutine @(a, r)
  real(8), intent(in) :: a(3)
  integer, intent(out) :: r
  integer :: i, j
  r = 0
  i = 0
  do j = 1, 3
    i = i + 1
    if (a(i) > 1.5d0) exit
    r = j
  end do
end subroutine""",
        {"a": np.array([1.0, 1.0, 2.0]), "r": I4(0)},
        None,
    ),
    "exit_test_before_later_write": (
        """subroutine @(a, r)
  real(8), intent(in) :: a(4)
  integer, intent(out) :: r
  integer :: i, j
  r = 0
  i = 1
  do j = 1, 3
    if (a(i) > 1.5d0) exit
    i = i + 1
    r = j
  end do
end subroutine""",
        {"a": np.array([1.0, 1.0, 2.0, 2.0]), "r": I4(0)},
        None,
    ),
    "any_of_section_in_condition": (
        """subroutine @(l, n, s)
  integer, intent(in) :: n
  logical, intent(in) :: l(n, 4)
  integer, intent(out) :: s
  s = 0
  if (any(l(:, 2))) s = 1
end subroutine""",
        {"l": np.array([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=I4, order="F"), "n": I4(3), "s": I4(9)},
        None,
    ),
    "fill_automatic_array_element_extent": (
        """subroutine @(sz, sel, s)
  integer, intent(in) :: sz(3), sel
  real(8), intent(out) :: s
  real(8) :: work(sz(sel))
  work = 1d0
  s = sum(work)
end subroutine""",
        {"sz": np.array([2, 4, 6], dtype=I4), "sel": I4(2), "s": F8(0)},
        None,
    ),
    "empty_section_maxval": (
        """subroutine @(a, k, m)
  real(8), intent(in) :: a(3)
  integer, intent(in) :: k
  real(8), intent(out) :: m
  m = maxval(a(2:k))
end subroutine""",
        {"a": np.arange(1.0, 4.0), "k": I4(1), "m": F8(0)},
        None,
    ),
}


@pytest.mark.parametrize("name", CASES)
def test_matches_gfortran(name: str, tmp_path: Path):
    template, args, buffer_family = CASES[name]
    src = template.replace("@", name)
    reference = run_reference(src, name, args, tmp_path)
    sdfg, buffers = run_sdfg(src, name, args, tmp_path / "sdfg")
    written = [n for n, value in reference.items() if not np.array_equal(value, as_buffer(args[n]))]
    assert written, "the kernel must write something a dropped output would lose"
    for arg in written:
        assert arg in buffers, f"{arg} has no output buffer"
        assert np.array_equal(reference[arg], buffers[arg].astype(reference[arg].dtype)), arg
    if buffer_family is not None:
        assert any(data.startswith(buffer_family) for data in sdfg.arrays)


def test_slice_scaled_by_another_slice_stays_in_place(tmp_path: Path):
    """Subscripts differing only outside the section name the written element or a slice left alone: no buffer."""
    src = """subroutine scale_row(a, n, j, k)
  integer, intent(in) :: n, j, k
  real(8), intent(inout) :: a(n, 3, 3)
  a(1:n, j, k) = a(1:n, j, k) / a(1:n, k, k)
end subroutine"""
    a = np.arange(2.0, 38.0).reshape(4, 3, 3, order="F")
    for j in (1, 3):  # another slice, then the written one
        args = {"a": a, "n": I4(4), "j": I4(j), "k": I4(3)}
        (tmp_path / f"ref{j}").mkdir()
        reference = run_reference(src, "scale_row", args, tmp_path / f"ref{j}")
        sdfg, buffers = run_sdfg(src, "scale_row", args, tmp_path / f"sdfg{j}")
        assert np.array_equal(reference["a"], buffers["a"])
        assert not any(data.startswith("_rhs_buffer") for data in sdfg.arrays)


def test_array_actual_of_a_scalar_dummy_is_its_first_element(tmp_path: Path):
    """QE ``gk_sort(..., ngk, ...)`` passes the array ``ngk`` to a scalar dummy it counts with and indexes by:
    sequence association with ``ngk(1)``, the only element written."""
    src = """subroutine sortk(n, ngk, igk)
  integer, intent(in) :: n
  integer, intent(inout) :: ngk
  integer, intent(inout) :: igk(n)
  integer :: i
  ngk = 0
  do i = 1, n
    if (mod(i, 2) == 0) then
      ngk = ngk + 1
      igk(ngk) = i
    end if
  end do
end subroutine
subroutine top(cnt, idx)
  integer, intent(inout) :: cnt(2), idx(6)
  call sortk(6, cnt, idx)
end subroutine"""
    sdfg = build_sdfg(src, tmp_path, name="top", entry="top").build()
    cnt, idx = np.array([7, 9], np.int32), np.zeros(6, np.int32)
    sdfg(cnt=cnt, idx=idx)
    assert cnt.tolist() == [3, 9] and idx.tolist() == [2, 4, 6, 0, 0, 0]


@pytest.mark.parametrize("flag", [0, 1])
def test_section_bound_read_from_an_element_written_before_the_call(tmp_path: Path, flag):
    """QE ``tgwave_g2r(orbital(1 : ngk(1), ibnd : last), ...)`` after ``ngk`` is assigned: the section ends at the
    element's value where the call happens, not at entry."""
    src = """module m
  integer :: ngk(2)
contains
  subroutine colsum(f_in, s)
    real(8) :: f_in(:, :)
    real(8), intent(out) :: s
    s = sum(f_in)
  end subroutine
end module
subroutine top(flag, a, j, s)
  use m
  logical, intent(in) :: flag
  integer, intent(in) :: j
  real(8), intent(inout) :: a(4, 3)
  real(8), intent(out) :: s
  if (flag) ngk = 3
  call colsum(a(1:ngk(1), j:3), s)
end subroutine"""
    sdfg = build_sdfg(src, tmp_path, name="top", entry="top").build()
    a, s = np.array(np.arange(1.0, 13.0).reshape(4, 3, order="F"), order="F"), np.zeros(1)
    sdfg(flag=I4(flag), a=a, j=I4(2), s=s, ngk=np.array([2, 2], np.int32))
    assert s[0] == a[: 3 if flag else 2, 1:3].sum()


def test_branch_on_the_size_of_a_one_column_section_keeps_only_the_live_side(tmp_path: Path):
    """QE ``wave_g2r(orbital(:, ibnd : ibnd), ...)``: ``dim2 = SIZE(f_in(1, :))`` is 1, so the ``dim2 == 2`` branch
    reading ``f_in(:, 2)`` is dead and must not reach the SDFG, whose view has one column."""
    src = """module m
contains
  subroutine g2r(f_in, s)
    real(8), intent(in) :: f_in(:, :)
    real(8), intent(out) :: s
    integer :: dim2
    dim2 = size(f_in(1, :))
    if (dim2 /= 2) s = sum(f_in(:, 1))
    if (dim2 == 2) s = sum(f_in(:, 1)) + 10 * sum(f_in(:, 2))
  end subroutine
end module
subroutine top(a, i, s)
  use m
  integer, intent(in) :: i
  real(8), intent(inout) :: a(4, 3)
  real(8), intent(out) :: s
  call g2r(a(:, i:i), s)
end subroutine"""
    sdfg = build_sdfg(src, tmp_path, name="top", entry="top").build()
    a, s = np.array(np.arange(1.0, 13.0).reshape(4, 3, order="F"), order="F"), np.zeros(1)
    sdfg(a=a, i=I4(2), s=s)
    assert s[0] == a[:, 1].sum()


REFUSED = {
    "reduction_mask": (
        """subroutine @(a, s)
  real(8), intent(in) :: a(3)
  real(8), intent(out) :: s
  s = sum(a, mask=a > 0.0d0)
end subroutine""",
        "MASK=",
    ),
    "runtime_back": (
        """subroutine @(a, i, flag)
  real(8), intent(in) :: a(3)
  integer, intent(out) :: i(1)
  logical, intent(in) :: flag
  i = maxloc(a, back=flag)
end subroutine""",
        "non-constant BACK",
    ),
    "complex_parameter_array": (
        """subroutine @(i, oc)
  integer, intent(in) :: i
  complex(8), intent(out) :: oc
  complex(8), parameter :: z(2) = [(1d0, 2d0), (3d0, 4d0)]
  oc = z(i)
end subroutine""",
        "COMPLEX initial data",
    ),
    "merge_in_subscript": (
        """subroutine @(a, j, k, b)
  real(8), intent(in) :: a(3)
  integer, intent(in) :: j, k
  real(8), intent(out) :: b
  b = a(merge(1, j, k == 0))
end subroutine""",
        "unlowered",
    ),
    "strided_section_any": (
        """subroutine @(l, n, s)
  integer, intent(in) :: n
  logical, intent(in) :: l(n, 4)
  integer, intent(out) :: s
  s = 0
  if (any(l(1:n:2, 2))) s = 1
end subroutine""",
        "strided section",
    ),
    "unresolved_external": (
        """subroutine @(x)
  real(8), intent(inout) :: x
  external :: not_in_this_source
  call not_in_this_source(x)
  x = x + 1
end subroutine""",
        "not_in_this_source",
    ),
    "fills_with_random": (
        """subroutine @(y)
  real(8), intent(out) :: y
  call random_number(y)
end subroutine""",
        "RandomNumber",
    ),
}


@pytest.mark.parametrize("name", REFUSED)
def test_refused_loudly(name: str, tmp_path: Path):
    template, message = REFUSED[name]
    with pytest.raises((RuntimeError, NotImplementedError), match=message):
        build_sdfg(template.replace("@", name), tmp_path, name=name, entry=name).build()


def test_getrf_info_read_by_a_branch(tmp_path: Path):
    """``info`` is a symbol through the branch on it, yet stays an output of the factorization."""
    src = """subroutine getrf_branch(a, n, info, ok)
  integer, intent(in) :: n
  real(8), intent(inout) :: a(n, n)
  integer, intent(out) :: info, ok
  integer :: ipiv(n)
  external dgetrf
  ok = 1
  call dgetrf(n, n, a, n, ipiv, info)
  if (info /= 0) ok = 0
end subroutine"""
    sdfg = build_sdfg(src, tmp_path, name="getrf_branch", entry="getrf_branch").build()
    sdfg.validate()
    assert not sdfg.arrays["info"].transient


MPI_DECLS = """
  integer :: ierr
  integer, parameter :: MPI_COMM_WORLD = 0
  integer, parameter :: MPI_DOUBLE_PRECISION = 17
  external :: MPI_Send, MPI_Bcast
"""


def test_mpi_send_count_narrows_the_buffer(tmp_path: Path):
    src = f"""subroutine send_three(buf, dst, tag)
  integer, intent(in) :: dst, tag
  real(8), intent(inout) :: buf(10)
  {MPI_DECLS}
  call MPI_Send(buf, 3, MPI_DOUBLE_PRECISION, dst, tag, MPI_COMM_WORLD, ierr)
end subroutine"""
    sdfg = build_sdfg(src, tmp_path, name="send_three", entry="send_three").build()
    sdfg.validate()
    sends = [
        edge.data.subset
        for state in sdfg.states()
        for edge in state.edges()
        if edge.data.data == "buf" and type(edge.dst).__name__ == "Send"
    ]
    assert [str(subset) for subset in sends] == ["0:3"]


def test_mpi_bcast_root_read_by_a_branch(tmp_path: Path):
    src = f"""subroutine bcast_root(buf, rank, root)
  integer, intent(in) :: rank, root
  real(8), intent(inout) :: buf(10)
  {MPI_DECLS}
  if (rank == root) buf(1) = 1
  call MPI_Bcast(buf, 10, MPI_DOUBLE_PRECISION, root, MPI_COMM_WORLD, ierr)
end subroutine"""
    sdfg = build_sdfg(src, tmp_path, name="bcast_root", entry="bcast_root").build()
    sdfg.validate()
    assert "root" in sdfg.symbols
