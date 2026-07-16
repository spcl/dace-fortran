"""Velocity-tendencies nested-struct indirect-access pattern.

Distilled minimal repro of the failing memlet seen when running the
full ``mo_velocity_advection.velocity_tendencies`` through the bridge:

    p_prog%w(p_patch%edges%cell_idx(je, jb, 1), jk,
             p_patch%edges%cell_blk(je, jb, 1))

The struct ``t_patch`` carries a nested ``t_edges`` whose member
arrays (``cell_idx`` / ``cell_blk``) are themselves used as the OUTER
dim 0 / dim 2 indices into another array (``w``).  The bridge's
struct-flatten pass currently bails on dummy-arg structs whose
members include a nested record (``allMembersFlattenable`` is false)
 --  the nested designate chain survives into the AST extractor, and
``build_memlet_index`` produces a memlet string with a raw
sub-subscript ``arr[other[i]]``.  ``Memlet._parse_from_subexpr``
splits on the outermost ``[`` and crashes with
``ValueError: too many values to unpack (expected 2)``.

Test sizes / data shape (per user spec):
    * ``nproma = nlev = nblks = 32``
    * indirection arrays carry values in ``[1, 31]`` so every
      ``w(idx_arr(...), jk, jb)`` access is in-bounds.
"""

from pathlib import Path

import dace
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
module mo_test_types
  implicit none
  integer, parameter :: nproma = 32, nblks = 32
  type :: t_edges
    integer :: cell_idx(nproma, nblks, 2)
    integer :: cell_blk(nproma, nblks, 2)
  end type
  type :: t_patch
    type(t_edges) :: edges
  end type
end module

subroutine kernel(p_patch, w, out, nlev)
  use mo_test_types
  implicit none
  integer, intent(in) :: nlev
  type(t_patch), intent(in) :: p_patch
  real(8), intent(in) :: w(nproma, nlev, nblks)
  real(8), intent(out) :: out(nproma, nlev, nblks)
  integer :: je, jk, jb
  do jb = 1, nblks
    do jk = 1, nlev
      do je = 1, nproma
        out(je, jk, jb) = w(p_patch % edges % cell_idx(je, jb, 1), jk, &
                            p_patch % edges % cell_blk(je, jb, 1))
      end do
    end do
  end do
end subroutine kernel
"""


def test_velocity_nested_struct_indirection(tmp_path: Path):
    """End-to-end numerical check on the velocity-tendencies indirect
    pattern.  Sizes match the user's request: ``nproma = nlev = nblks
    = 32``; indirection arrays carry values in ``[1, 31]``."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC, sdfg_dir, name='kernel').build()

    nproma, nblks, nlev = 32, 32, 32
    rng = np.random.default_rng(0)
    w = np.asfortranarray(rng.standard_normal((nproma, nlev, nblks)))
    cell_idx = np.asfortranarray(rng.integers(1, 32, size=(nproma, nblks, 2), dtype=np.int32))
    cell_blk = np.asfortranarray(rng.integers(1, 32, size=(nproma, nblks, 2), dtype=np.int32))
    out_sdfg = np.zeros((nproma, nlev, nblks), dtype=np.float64, order='F')

    # Reference: NumPy gather (Fortran 1-based -> 0-based on the indirect
    # axes).  This is the exact arithmetic ``kernel`` performs for every
    # ``(je, jk, jb)``.
    cell_i = cell_idx[..., 0] - 1
    cell_b = cell_blk[..., 0] - 1
    out_ref = np.empty_like(out_sdfg)
    for jb in range(nblks):
        for jk in range(nlev):
            for je in range(nproma):
                out_ref[je, jk, jb] = w[cell_i[je, jb], jk, cell_b[je, jb]]

    # Pack the nested-struct dummy.  numpy doesn't have a native
    # Fortran-derived-type binding; for the bridge call the struct
    # arg is materialised through the flattened companions
    # (``p_patch_edges_cell_idx`` / ``_cell_blk``) once flatten-structs
    # learns the nested-dummy path.
    sdfg(p_patch_edges_cell_idx=cell_idx, p_patch_edges_cell_blk=cell_blk, w=w, out=out_sdfg, nlev=nlev)

    np.testing.assert_allclose(out_sdfg, out_ref, rtol=0, atol=0)


# A POINTER local rebound onto a TARGET struct-member dummy, then read ONLY as an
# inline indirect index -- the exact shape of ICON's inlined
# ``cells2verts_scalar_ri_lib`` / ``rot_vertex_ri_lib`` (``iidx => vert_cell_idx``
# then ``p_cell_in(iidx(jv, jb, 1), jk, iblk(jv, jb, 1))``).  The whole-array
# rebind of a rank-3 POINTER onto the inlined TARGET dummy is tagged
# ``pointer_view`` by ``RewritePointerAssigns`` and ``iidx`` / ``iblk`` survive to
# ``extract_vars`` as ``role='view_alias'`` Views.  Read only through inline
# indirection (both are read, like the real kernel, so neither is a dead
# orphan), their
# source->view link (installed lazily by ``acc`` only when the view is touched
# FROM a state) was never installed -- the indirection reads become interstate
# ``sym_*`` edge assignments -- so the View reached codegen ORPHANED (zero
# AccessNodes, no ``views`` edge) and framecode's ``get_view_edge`` raised
# ``KeyError`` AT COMPILE.  Guards ``materialize_indirect_view_sources``.
_REBIND_SRC = """
module mo_rebind_indirect
  implicit none
  integer, parameter :: nverts = 16, nblks = 8, ncells = 64, ncblks = 4
  type :: t_edges
    integer :: cell_idx(nverts, nblks, 6)
    integer :: cell_blk(nverts, nblks, 6)
  end type
  type :: t_patch
    type(t_edges) :: edges
  end type
contains
  subroutine cells2verts_lib(p_cell_in, vert_cell_idx, vert_cell_blk, p_vert_out, nlev)
    integer, intent(in) :: nlev
    real(kind=8), intent(in) :: p_cell_in(ncells, nlev, ncblks)
    integer, target, intent(in) :: vert_cell_idx(:, :, :)
    integer, target, intent(in) :: vert_cell_blk(:, :, :)
    real(kind=8), intent(inout) :: p_vert_out(nverts, nlev, nblks)
    integer, dimension(:, :, :), pointer :: iidx, iblk
    integer :: jv, jk, jb
    iidx => vert_cell_idx
    iblk => vert_cell_blk
    do jb = 1, nblks
      do jk = 1, nlev
        do jv = 1, nverts
          p_vert_out(jv, jk, jb) = p_cell_in(iidx(jv, jb, 1), jk, iblk(jv, jb, 1))
        end do
      end do
    end do
  end subroutine cells2verts_lib

  subroutine cells2verts_entry(p_patch, p_cell_in, p_vert_out, nlev)
    integer, intent(in) :: nlev
    type(t_patch), intent(in) :: p_patch
    real(kind=8), intent(in) :: p_cell_in(ncells, nlev, ncblks)
    real(kind=8), intent(inout) :: p_vert_out(nverts, nlev, nblks)
    call cells2verts_lib(p_cell_in, p_patch % edges % cell_idx, p_patch % edges % cell_blk, p_vert_out, nlev)
  end subroutine cells2verts_entry
end module mo_rebind_indirect
"""


def views_with_source_link(sdfg):
    """Every ``dace.data.View`` descriptor -> whether some AccessNode of it has an
    incoming ``'views'`` (source->view) linking edge.  An orphaned view_alias --
    the pre-fix bug -- has a View descriptor but no such AccessNode/edge."""
    views = {n: False for n, d in sdfg.arrays.items() if isinstance(d, dace.data.View)}
    for state in sdfg.all_states():
        for node in state.data_nodes():
            if node.data in views and any(e.dst_conn == 'views' for e in state.in_edges(node)):
                views[node.data] = True
    return views


def test_rebound_pointer_indirect_index_view_compiles(tmp_path: Path):
    """A rebound POINTER read only as an inline indirect index must reach codegen
    as a properly-LINKED view_alias, not an orphaned View.  Pre-fix this raised
    ``KeyError`` in DaCe's ``get_view_edge`` at compile time; the fix
    (``materialize_indirect_view_sources``) installs the source->view link in the
    state preceding the ``sym_*`` indirection edges."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_REBIND_SRC, sdfg_dir, name="cells2verts_entry",
                      entry="mo_rebind_indirect::cells2verts_entry").build()
    sdfg.validate()

    # Precondition: the rebind must actually lower as a view_alias (View).  If the
    # reduction collapsed it instead, the test would pass vacuously -- fail loudly
    # here so a wrong assumption surfaces rather than a false green.
    views = views_with_source_link(sdfg)
    assert views, "expected the rebound pointer to lower as a view_alias (dace.data.View)"

    # The fix must give every such View an AccessNode carrying a source->view
    # 'views' edge; pre-fix the indirect-only read left it orphaned.
    for name, has_link in views.items():
        assert has_link, f"view_alias {name!r} has no source->view 'views' edge -- orphaned view (the pre-fix bug)"

    # Load-bearing: the crash was at codegen, not build/validate.  Compiling is
    # what actually exercises get_view_edge.
    sdfg.compile()
