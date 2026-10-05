"""Fortran KIND -> SDFG dtype mapping invariants.

Two rules that must hold globally (the binding/codegen depend on them):

* ``LOGICAL(KIND=k)`` -> the unsigned integer of its width (``uint8`` ..
  ``uint64``): the SDFG works on the caller's LOGICAL storage.
* ``INTEGER(1/2/4/8)`` -> ``int8/16/32/64`` (width preserved, never
  widened or conflated with ``bool``).
"""

from pathlib import Path

import dace

from tests._util import build_sdfg


def _dtypes(src: str, tmp_path: Path, entry: str) -> dict:
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="k", entry=entry).build()
    return {n: str(d.dtype) for n, d in sdfg.arrays.items()}


def test_logical_kind_is_its_unsigned_width(tmp_path):
    # Every LOGICAL kind arg gets a write so the post-build prune
    # (``prune_unused_arrays``) keeps each in ``sdfg.arrays`` -- the
    # dtype-mapping invariant the test is here to enforce is about
    # *kernel-arg* args, so a dead-code-eliminated arg never reached
    # the SDFG to begin with.
    src = """
subroutine klog(n, ld, l1, l4, l8, lcb)
  use iso_c_binding
  implicit none
  integer(c_int), intent(in) :: n
  logical,        intent(inout) :: ld(n)
  logical(1),     intent(inout) :: l1(n)
  logical(4),     intent(inout) :: l4(n)
  logical(8),     intent(inout) :: l8(n)
  logical(c_bool),intent(inout) :: lcb(n)
  integer :: i
  do i = 1, n
     ld(i)  = .not. ld(i)
     l1(i)  = .not. l1(i)
     l4(i)  = .not. l4(i)
     l8(i)  = .not. l8(i)
     lcb(i) = .not. lcb(i)
  end do
end subroutine klog
"""
    d = _dtypes(src, tmp_path, "klog")
    expected = {"ld": dace.uint32, "l1": dace.uint8, "l4": dace.uint32, "l8": dace.uint64, "lcb": dace.uint8}
    assert {nm: d[nm] for nm in expected} == {nm: str(dt) for nm, dt in expected.items()}


def test_integer_kinds_preserve_width(tmp_path):
    # Each INTEGER width arg gets a write so the post-build prune
    # keeps every descriptor; otherwise the unused widths drop out
    # before the dtype assertions run.
    src = """
subroutine kint(n, a1, a2, a4, a8)
  use iso_c_binding
  implicit none
  integer(c_int), intent(in) :: n
  integer(1), intent(inout) :: a1(n)
  integer(2), intent(inout) :: a2(n)
  integer(4), intent(inout) :: a4(n)
  integer(8), intent(inout) :: a8(n)
  integer :: i
  do i = 1, n
     a1(i) = a1(i) + 1_1
     a2(i) = a2(i) + 1_2
     a4(i) = a4(i) + 1_4
     a8(i) = a8(i) + 1_8
  end do
end subroutine kint
"""
    d = _dtypes(src, tmp_path, "kint")
    assert d["a1"] == "int8_t"
    assert d["a2"] == "int16_t"
    assert d["a4"] == "int"
    assert d["a8"] == "int64_t"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
