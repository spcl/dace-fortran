"""An array constructor assigned to a section (QE ``ggen``: ``mill(:, ng) = (/i, j, k/)``)."""

from pathlib import Path

import numpy as np

import dace_fortran
from tests._util import f2py_compile

SRC = """
subroutine fill_columns(mill, n)
  implicit none
  integer, intent(in)    :: n
  integer, intent(inout) :: mill(3, n)
  integer :: ng
  do ng = 1, n
    mill(:, ng) = (/ng, 2 * ng, 3 * ng/)
  end do
end subroutine fill_columns
"""


def test_array_constructor_fills_one_column(tmp_path: Path):
    """Each element of the constructor lands in its row of column ``ng``; the write is rank 2, not rank 1."""
    mod = f2py_compile(SRC, tmp_path / "ref", "fill_columns")
    sdfg = dace_fortran.build_sdfg(SRC, out_dir=str(tmp_path / "sdfg"), entry="fill_columns", name="fill_columns")
    ref = np.zeros((3, 4), order="F", dtype=np.int32)
    mod.fill_columns(ref, 4)
    got = np.zeros((3, 4), order="F", dtype=np.int32)
    sdfg(mill=got, n=4)
    np.testing.assert_array_equal(got, ref)


STRIDED_SRC = """
subroutine fill_strided(out, a)
  implicit none
  integer, intent(in)    :: a
  integer, intent(inout) :: out(7)
  out(2:6:2) = (/a, a + 1, a + 2/)
end subroutine fill_strided
"""


def test_array_constructor_fills_a_strided_section(tmp_path: Path):
    """Element ``k`` lands at ``lo + (k - 1) * stride`` of the section, not at ``out(k)``. The values are not
    constants: Flang folds an all-constant constructor into a read-only global, which takes the section-copy path."""
    sdfg = dace_fortran.build_sdfg(STRIDED_SRC, out_dir=str(tmp_path), entry="fill_strided", name="fill_strided")
    got = np.zeros(7, dtype=np.int32)
    sdfg(out=got, a=7)
    np.testing.assert_array_equal(got, [0, 7, 0, 8, 0, 9, 0])


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_array_constructor_fills_one_column(Path(d) / "column")
        test_array_constructor_fills_a_strided_section(Path(d) / "strided")
