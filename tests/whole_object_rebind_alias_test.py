"""Regression: a WHOLE derived-type-OBJECT pointer rebind (obj_ptr => src_obj) that the
bridge lowered as a plain scalar assign.

The target is a TYPE(...), POINTER object, not numeric data, so it misses every array/view/
section tag -- absent from arrays/scalars/symbols, RHS a bare reference. Emitting the store
fabricated a descriptor-less AccessNode that crashed read_and_write_sets/prune_unused_arrays
with a KeyError. This blocked the ICON ocean dycore (solve_free_sfc_ab_mimetic) from lowering:
params_oce => v_params (whose params_oce % a_veloc_v read-modify-write must land on the real
source, not be dropped) and the dead store free_sfc_solver_lhs % patch_3d => patch_3d.

Fix: the builder now collects object-alias edges (descriptors.scan_object_aliases). A member
access on an aliased object resolves to the source object's real flattened storage
(resolve_object_member), so a live RMW updates the SOURCE (no lost update); the data-less
rebind store itself is dropped at emit (emit_scalar_assign early return).
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# Module-global TARGET derived-type object (mirrors ICON's mo_ocean_physics_types::v_params)
# with a POINTER array member, rebound by a local pointer. out0 = g%arr(1) reads directly (so
# flatten materialises g_arr); the loop updates the member THROUGH alias p -- ICON's shape.
_LIVE_SRC = """
module mo_objalias
  implicit none
  type t_box
    real(8), pointer :: arr(:) => null()
  end type
  type(t_box), target :: g
contains
  subroutine run(n, out0)
    integer, intent(in) :: n
    real(8), intent(out) :: out0
    type(t_box), pointer :: p
    integer :: i
    out0 = g%arr(1)
    p => g
    do i = 1, n
      p%arr(i) = p%arr(i) * 2.0d0 + 1.0d0
    end do
  end subroutine run
end module mo_objalias
"""

# Local OPAQUE struct whose OBJECT-pointer member is rebound to a module global -- a dead
# store (this is never read as data), mirroring ICON's free_sfc_solver_lhs % patch_3d => patch_3d.
_DEAD_SRC = """
module mo_op
  implicit none
  type t_inner
    integer :: tag = 0
  end type
  type t_outer
    type(t_inner), pointer :: p => null()
  end type
  type(t_inner), target :: gi
contains
  subroutine run(y, n)
    real(8), intent(inout) :: y(:)
    integer, intent(in) :: n
    type(t_outer) :: this
    integer :: i
    this%p => gi
    do i = 1, n
      y(i) = y(i) * 3.0d0
    end do
  end subroutine run
end module mo_op
"""


def test_whole_object_rebind_live_update_hits_source(tmp_path: Path):
    """``p => g; p%arr(i) = f(p%arr(i))`` -- the read-modify-write must land on
    the SOURCE member companion ``g_arr`` (a lost update would leave it
    unchanged), and the data-less ``p => g`` store must not dangle the build."""
    b = build_sdfg(_LIVE_SRC, tmp_path / "sdfg", name="objalias", entry="mo_objalias::run")
    sdfg = b.build()
    sdfg.validate()
    # p => g must resolve away before emit so no descriptor-less p companion dangles. Two
    # resolutions are valid: object-alias tracking, or collapsing p%arr onto g_arr directly.
    resolved_via_alias = b.object_aliases.get("p") == "g" and "p" in b.object_alias_defs
    resolved_via_collapse = "p" not in sdfg.arrays and "p_arr" not in sdfg.arrays
    assert resolved_via_alias or resolved_via_collapse, "p => g rebind left a dangling companion"

    n = 5
    arr = np.asfortranarray(np.arange(1, n + 1, dtype=np.float64))
    out0 = np.zeros(1, dtype=np.float64)
    original = arr.copy()
    sdfg(g_arr=arr, n=np.int32(n), out0=out0, g_arr_d0=n)

    # The SOURCE member array was updated through the alias (catches lost update).
    np.testing.assert_allclose(arr, original * 2.0 + 1.0, rtol=1e-12, atol=1e-12)
    # The direct read saw the pre-update value.
    np.testing.assert_allclose(out0[0], original[0], rtol=1e-12, atol=1e-12)


def test_opaque_struct_dead_rebind_prunes_clean(tmp_path: Path):
    """``this % p => gi`` into an opaque local struct is a dead store -- it must
    lower + prune cleanly (no descriptor-less AccessNode reaching prune) while
    the real work on ``y`` is preserved."""
    b = build_sdfg(_DEAD_SRC, tmp_path / "sdfg", name="deadobj", entry="mo_op::run")
    sdfg = b.build()
    sdfg.validate()
    assert "this_p" in b.object_alias_defs
    # The dead rebind target was pruned -- no dangling descriptor survives.
    assert "this_p" not in sdfg.arrays

    n = 4
    y = np.asfortranarray(np.arange(1, n + 1, dtype=np.float64))
    original = y.copy()
    sdfg(y=y, n=np.int32(n), y_d0=n)
    np.testing.assert_allclose(y, original * 3.0, rtol=1e-12, atol=1e-12)
