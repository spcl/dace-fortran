"""Regression: a WHOLE derived-type-OBJECT pointer rebind (``obj_ptr => src_obj``)
the bridge lowered as a plain scalar ``assign``.

The target of such a rebind is a ``TYPE(...), POINTER`` object -- not numeric
data -- so it misses every array / view / section tag: it is absent from the
builder's ``arrays`` / ``scalars`` / ``symbols`` and its RHS is a bare reference.
Emitting the store fabricates a descriptor-less AccessNode that later crashes
``read_and_write_sets`` / ``prune_unused_arrays`` with a ``KeyError`` on the
target name.  This is the concrete blocker that stopped the ICON ocean dynamical
core (``solve_free_sfc_ab_mimetic``) from lowering -- e.g. ``params_oce =>
v_params`` (whose ``params_oce % a_veloc_v`` read-modify-write must land on the
real source descriptor, NOT be silently dropped) and the dead solver-struct
stores ``free_sfc_solver_lhs % patch_3d => patch_3d``.

The builder now collects the object-alias edges (``descriptors.scan_object_aliases``)
and:

* a member access on an aliased object (``params_oce % a_veloc_v``) resolves to
  the real flattened storage of the source object (``resolve_object_member``), so
  a live read-modify-write updates the SOURCE member (no lost update), and
* the data-less rebind store itself is dropped at emit (``emit_scalar_assign``
  early return), so no dangling AccessNode reaches ``prune_unused_arrays``.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# A module-global ``TARGET`` derived-type object (mirrors ICON's
# ``mo_ocean_physics_types :: v_params``) with a POINTER array member, rebound to
# by a local pointer.  ``out0 = g%arr(1)`` reads the member DIRECTLY so the
# flatten pass materialises the real companion ``g_arr``; the loop then updates
# the member THROUGH the alias ``p`` -- exactly ICON's
# ``params_oce % a_veloc_v(...) = ... params_oce % a_veloc_v(...) ...``.
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

# A local OPAQUE struct whose OBJECT-pointer member is rebound to a module global
# -- a dead store (``this`` is never read as data), mirroring ICON's
# ``free_sfc_solver_lhs % patch_3d => patch_3d``.  The routine does real work on
# ``y``; the dead rebind must lower + prune cleanly (no dangling AccessNode).
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
    # The rebind is registered as an object alias (collected during ``build``),
    # and its data-less store target is dropped at emit.
    assert b.object_aliases.get("p") == "g"
    assert "p" in b.object_alias_defs

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
