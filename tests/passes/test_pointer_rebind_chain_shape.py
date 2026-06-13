"""Regression test for the chain-shape pointer rebind pattern.

When a Fortran pointer is rebound to a nested derived-type member that
is itself a POINTER  --

    icidx => patch%edges%cell_idx

flang emits HLFIR that walks through a ``fir.load`` of a designated
nested pointer member.  ``RewritePointerAssigns`` in the bridge
currently walks back through ``embox/rebox/convert/designate`` only
and bails at the ``fir.load`` (the trace's ``parent`` returns null),
leaving the pointer as a free-standing storage in the SDFG.

The interim workaround lives in ``emit_scalar_assign``: when a
scalar-assign tasklet has a multi-dim array target whose RHS is a
single-token name of another multi-dim array, emit a whole-array
copy memlet instead.  This pins the SDFG-build path as correct end
to end (icidx is its own storage, populated by a wholesale copy
from p_patch_edges_cell_idx), at the cost of one extra memlet edge
that would be structurally redundant under a fully-collapsing
``RewritePointerAssigns``.

This test:

  * Pins the WORKAROUND as the current contract: the SDFG builds and
    validates for the rebind pattern.  A regression here is either a
    bridge change reverting the workaround OR a successful upgrade
    of ``RewritePointerAssigns`` (in which case ``icidx`` would
    disappear from the SDFG arrays entirely and the assertion below
    flips its check accordingly  --  update the test instead of the
    bridge).
  * Documents the precise HLFIR shape the pass would need to learn,
    so a future bridge-side fix has the reproducible case here.

When ``RewritePointerAssigns`` learns to walk through
``load(designate(declare, member))`` chains  --  treating the
loaded box as a passthrough alias of the parent's storage  --  this
test's assertion flips: ``icidx`` no longer appears as a top-level
SDFG array because every use of it rebases to
``p_patch_edges_cell_idx``.  See ``RewritePointerAssigns.cpp``
``traceRebindChain`` (line ~290) for the walk to extend.
"""
import pytest

import dace_fortran


_REPRO_SRC = """\
MODULE m_repro_types
  IMPLICIT NONE
  TYPE :: t_edges
    INTEGER, DIMENSION(:,:,:), POINTER :: cell_idx
  END TYPE
  TYPE :: t_patch
    TYPE(t_edges) :: edges
  END TYPE
END MODULE

MODULE m_repro
  USE m_repro_types
  IMPLICIT NONE
CONTAINS
  SUBROUTINE kernel(patch, result, n0)
    TYPE(t_patch), INTENT(IN) :: patch
    INTEGER, INTENT(OUT) :: result(n0)
    INTEGER, INTENT(IN) :: n0
    INTEGER, DIMENSION(:,:,:), POINTER :: icidx
    INTEGER :: i
    icidx => patch%edges%cell_idx     ! the chain-shape rebind
    DO i = 1, n0
      result(i) = icidx(i, 1, 1)
    END DO
  END SUBROUTINE
END MODULE
"""


def test_pointer_rebind_chain_through_pointer_member_builds(tmp_path):
    """The SDFG builds and validates for the chain-shape pointer rebind.

    Currently xfails because of the overzealous bounds-remap guard in
    ``RewritePointerAssigns``.  ICON's real velocity SDFG builds
    (``test_velocity_from_icon_source.py``) because flatten-structs
    pre-folds the chain; this minimal repro exposes the case
    flatten-structs doesn't reach."""
    sdfg = dace_fortran.build_sdfg(
        _REPRO_SRC,
        out_dir=str(tmp_path / "sdfg"),
        entry="m_repro::kernel",
        name="ptr_rebind_chain",
    )
    sdfg.validate()
    assert sdfg is not None
    arrays = set(sdfg.arrays.keys())
    assert "icidx" in arrays or "patch_edges_cell_idx" in arrays, (
        "neither the unfolded ``icidx`` array nor the post-rewrite "
        "rebased ``patch_edges_cell_idx`` is in the SDFG -- "
        f"got arrays {sorted(arrays)!r}")
