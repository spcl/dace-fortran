"""A result-producing HLFIR compute statement the bridge does not lower
must FAIL LOUD, not be silently skipped.

``buildAST`` walks the ops in a block and dispatches each to a handler.
Ops it does not recognise fall through and are skipped -- correct for the
many genuinely-benign ops (pure-value producers, declares, ``hlfir.destroy``
temp cleanups, terminators, ops already consumed by an earlier handler).
But for a *result-producing* compute op it does NOT lower -- ``hlfir.where``
(masked WHERE assignment), ``hlfir.forall`` (FORALL), or a bare
``hlfir.region_assign`` -- silently skipping means the kernel drops that
computation and produces a WRONG numerical result with no error.

A bare ``WHERE (mask) a = b`` lowers to ``hlfir.where`` and survives the
default pipeline (no where/forall lowering pass runs before AST extraction).
Before the fix, building this kernel produced an SDFG with zero tasklets --
the assignment was silently dropped.  The bridge now throws a located
diagnostic naming the unhandled op instead.
"""
from pathlib import Path

import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# Elemental masked assignment -> ``hlfir.where`` wrapping an
# ``hlfir.region_assign``.  No pipeline pass lowers WHERE, so the op reaches
# ``buildAST`` and must trigger the compute-drop guard.
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
    """An unlowered ``hlfir.where`` must raise a located diagnostic naming the
    op, not silently drop the masked assignment."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError) as excinfo:
        build_sdfg(_WHERE_KERNEL, sdfg_dir, name="where_masked_assign", entry="where_masked_assign").build()
    msg = str(excinfo.value)
    assert "hlfir.where" in msg, f"diagnostic should name the unhandled op, got: {msg}"
    assert "unhandled compute statement" in msg, f"diagnostic should flag the compute drop, got: {msg}"
