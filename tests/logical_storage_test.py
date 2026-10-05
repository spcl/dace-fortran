"""The SDFG works on the caller's LOGICAL storage.

``LOGICAL(KIND=k)`` is the unsigned integer of its width (``uint8`` .. ``uint64``); a read is ``!= 0`` and a write
stores 0 / 1 (flang's and gfortran's .FALSE. / .TRUE.).  Every test passes values another producer may store for
.TRUE. (-1, i.e. all bits set, and the largest signed value), and checks the result lands in the caller's own buffer.
"""

from pathlib import Path

import numpy as np
import pytest

from tests._util import build_sdfg

_STORAGE: dict[int, type[np.unsignedinteger]] = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


def _foreign_true(kind: int) -> np.ndarray:
    """``[.FALSE., .TRUE., -1, HUGE, 2]`` in LOGICAL(KIND=kind) storage: every value but the first is .TRUE."""
    dtype = _STORAGE[kind]
    return np.array([0, 1, np.iinfo(dtype).max, np.iinfo(dtype).max >> 1, 2], dtype=dtype)


@pytest.mark.parametrize("kind", sorted(_STORAGE))
def test_not_in_place_on_every_kind(tmp_path: Path, kind: int) -> None:
    src = f"""
subroutine invert(mask, n)
  implicit none
  integer, intent(in) :: n
  logical({kind}), intent(inout) :: mask(n)
  integer :: i
  do i = 1, n
    mask(i) = .not. mask(i)
  end do
end subroutine invert
"""
    sdfg = build_sdfg(src, tmp_path, name=f"invert{kind}", entry="invert").build()
    mask = _foreign_true(kind)
    sdfg(mask=mask, n=mask.size)
    np.testing.assert_array_equal(mask, np.array([1, 0, 0, 0, 0], dtype=_STORAGE[kind]))


def test_mixed_kinds_and_eqv(tmp_path: Path) -> None:
    src = """
subroutine mix(a, b, both, same, differ, n)
  implicit none
  integer, intent(in) :: n
  logical(1), intent(in) :: a(n)
  logical(8), intent(in) :: b(n)
  logical(4), intent(out) :: both(n)
  logical(2), intent(out) :: same(n), differ(n)
  integer :: i
  do i = 1, n
    both(i) = a(i) .and. b(i)
    same(i) = a(i) .eqv. b(i)
    differ(i) = a(i) .neqv. b(i)
  end do
end subroutine mix
"""
    sdfg = build_sdfg(src, tmp_path, name="mix", entry="mix").build()
    a = _foreign_true(1)
    b = np.array([0, np.iinfo(np.uint64).max, 0, 1, 7], dtype=np.uint64)
    both, same, differ = np.full(5, 9, np.uint32), np.full(5, 9, np.uint16), np.full(5, 9, np.uint16)
    sdfg(a=a, b=b, both=both, same=same, differ=differ, n=a.size)
    np.testing.assert_array_equal(both, [0, 1, 0, 1, 1])
    np.testing.assert_array_equal(same, [1, 1, 0, 1, 1])
    np.testing.assert_array_equal(differ, [0, 0, 1, 0, 0])


def test_if_count_any_all_on_foreign_true(tmp_path: Path) -> None:
    src = """
subroutine reduce(mask, picked, counts, anyall, n)
  implicit none
  integer, intent(in) :: n
  logical, intent(in) :: mask(n)
  integer, intent(out) :: picked(n), counts(1)
  logical, intent(out) :: anyall(2)
  integer :: i
  do i = 1, n
    if (mask(i)) then
      picked(i) = i
    else
      picked(i) = 0
    end if
  end do
  counts(1) = count(mask)
  anyall(1) = any(mask(1:1))
  anyall(2) = all(mask(2:n))
end subroutine reduce
"""
    sdfg = build_sdfg(src, tmp_path, name="reduce", entry="reduce").build()
    mask = _foreign_true(4)
    picked, counts, anyall = np.zeros(5, np.int32), np.zeros(1, np.int32), np.full(2, 9, np.uint32)
    sdfg(mask=mask, picked=picked, counts=counts, anyall=anyall, n=mask.size)
    np.testing.assert_array_equal(picked, [0, 2, 3, 4, 5])
    assert counts[0] == 4
    np.testing.assert_array_equal(anyall, [0, 1])


if __name__ == "__main__":
    import tempfile

    for kind in sorted(_STORAGE):
        with tempfile.TemporaryDirectory() as tmp:
            test_not_in_place_on_every_kind(Path(tmp), kind)
    with tempfile.TemporaryDirectory() as tmp:
        test_mixed_kinds_and_eqv(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_if_count_any_all_on_foreign_true(Path(tmp))
