"""Unrecognised *result-producing* HLFIR compute ops (``hlfir.where``, ``hlfir.forall``,
bare ``hlfir.region_assign``) must raise, not be silently skipped by ``buildAST`` --
skipping used to drop the computation and produce a wrong result with no error.
"""
from pathlib import Path

import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# WHERE lowers to hlfir.where; no pipeline pass rewrites it, so it reaches buildAST and must trigger the compute-drop guard.
_WHERE_KERNEL = """
subroutine where_masked_assign(a, b, n)
  implicit none
  integer, intent(in) :: n
  real, intent(inout) :: a(n)
  real, intent(in) :: b(n)
  where (b > 0.0)
    a = b * 2.0
  end where
end subroutine where_masked_assign
"""


def test_where_masked_assign_raises(tmp_path: Path):
    """Unlowered ``hlfir.where`` raises a located diagnostic naming the op, not a silent drop."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError) as excinfo:
        build_sdfg(_WHERE_KERNEL, sdfg_dir, name="where_masked_assign", entry="where_masked_assign").build()
    msg = str(excinfo.value)
    assert "hlfir.where" in msg, f"diagnostic should name the unhandled op, got: {msg}"
    assert "unhandled compute statement" in msg, f"diagnostic should flag the compute drop, got: {msg}"
