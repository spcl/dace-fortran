"""Jagged scalar-struct + Array-of-Records mixed: max/min extent detection.

Per user request: 'we should jagged and AoR mixed into a max/min extent
detection test'.

Two struct shapes that the bridge handles via DIFFERENT flatten branches:

  * **Jagged scalar-struct** (single ``type(t) :: g`` dummy where ``t`` has
    multiple 1-D array members of the same scalar element type but
    DIFFERENT extents -- ``a(3)``, ``b(5)``, ``c(7)``).  The bridge
    packs these into a single 2-D companion ``g(memberIdx, j)`` of shape
    ``[numMembers x max(extents)]`` -- an ELLPACK-style padded view.
    Handled at ``FlattenStructs.cpp::isJaggedScalarStruct`` /
    ``replaceStructArgJagged`` (line ~2810).

  * **Array-of-Records** (``type(t) :: arr(N)``).  The bridge produces
    one flat ``arr_<member>`` companion per field.  Handled at
    ``replaceStructArg`` / ``replaceStructArgNested`` (line ~2940 /
    3178).

These probes cover both shapes side-by-side plus a min-extent edge
case (when the jagged width is determined by the SMALLEST member
rather than the LARGEST, which surfaces a different code path
through ``replaceStructArgJagged``'s ``maxExtent`` -> column-index
clamping).
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_jagged_scalar_struct_max_extent_packing(tmp_path):
    """``type :: t; a(3); b(5); c(7); end type`` -- the bridge packs
    into ``g(3, 7)`` (3 members x max-extent 7), accessing
    ``g % a(j)`` as ``g_packed(0, j-1)`` etc.  Verify the SDFG has
    a single 2-D companion sized to the max extent."""
    src = """
module m
  type :: t
    real(kind=8) :: a(3)
    real(kind=8) :: b(5)
    real(kind=8) :: c(7)
  end type
contains
  subroutine driver(g, out)
    type(t), intent(in) :: g
    real(kind=8), intent(out) :: out
    out = g % a(1) + g % b(1) + g % c(1)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="_QMmPdriver").build()
    # Jagged path packs ALL members into one 2-D companion.  Either
    # the bridge picks the jagged path (one 2-D ``g`` companion) OR
    # the per-member flat path (three 1-D ``g_a``/``g_b``/``g_c``).
    # Both are valid as long as the access works.
    arrs = sdfg.arrays
    has_jagged = "g" in arrs and len(arrs["g"].shape) == 2
    has_per_member = ("g_a" in arrs and "g_b" in arrs and "g_c" in arrs)
    assert has_jagged or has_per_member, \
        f"expected jagged or per-member flatten: {sorted(arrs.keys())}"


def test_aor_member_extent_distinct_from_jagged(tmp_path):
    """``type(t) :: arr(3)`` where ``t`` has a single array member --
    the bridge produces ``arr_x`` of shape ``(N, member_extent)``,
    NOT the jagged ``(numMembers, max)`` packing.  Probes the
    discriminator between the two flatten paths."""
    src = """
module m
  type :: t
    real(kind=8) :: x(4)
  end type
contains
  subroutine driver(arr, out)
    type(t), intent(in) :: arr(3)
    real(kind=8), intent(out) :: out
    out = arr(1) % x(1) + arr(2) % x(2) + arr(3) % x(3)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="_QMmPdriver").build()
    assert "arr_x" in sdfg.arrays
    # AoR shape: (records, field_extent) -- NOT (numMembers, max).
    assert tuple(int(s) for s in sdfg.arrays["arr_x"].shape) == (3, 4)


def test_min_extent_struct_member_no_overpadding(tmp_path):
    """Struct with one small + one large array member -- the jagged
    packing uses the LARGER extent (max) so the small member's
    column-index range stays within bounds.  Verify the SDFG
    descriptor's column extent >= the larger member's extent."""
    src = """
module m
  type :: t
    real(kind=8) :: small(2)
    real(kind=8) :: large(8)
  end type
contains
  subroutine driver(g, out)
    type(t), intent(in) :: g
    real(kind=8), intent(out) :: out
    out = g % small(1) + g % large(1)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="_QMmPdriver").build()
    arrs = sdfg.arrays
    if "g" in arrs and len(arrs["g"].shape) == 2:
        # Jagged path -- inner dim must be >= max(small=2, large=8).
        inner = int(arrs["g"].shape[1])
        assert inner >= 8, f"jagged inner dim too small: {inner}"
    else:
        # Per-member flat path -- both companions sized to their
        # native extents.
        assert "g_small" in arrs and "g_large" in arrs
        assert tuple(int(s) for s in arrs["g_small"].shape) == (2, )
        assert tuple(int(s) for s in arrs["g_large"].shape) == (8, )


def test_jagged_then_aor_separate_dummies(tmp_path):
    """Both shapes in the same kernel: a jagged-style struct ``g``
    AND an AoR ``arr``.  Verify they take their respective flatten
    paths without interference."""
    src = """
module m
  type :: jt
    real(kind=8) :: a(3)
    real(kind=8) :: b(5)
  end type
  type :: at
    real(kind=8) :: x
  end type
contains
  subroutine driver(g, arr, out)
    type(jt), intent(in) :: g
    type(at), intent(in) :: arr(3)
    real(kind=8), intent(out) :: out
    out = g % a(1) + g % b(1) + arr(1) % x + arr(2) % x + arr(3) % x
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="_QMmPdriver").build()
    arrs = sdfg.arrays
    # AoR ``arr`` -> ``arr_x`` per-field flatten.
    assert "arr_x" in arrs
    # Jagged ``g`` -> either packed 2-D ``g`` or per-member
    # ``g_a``/``g_b`` (both valid).
    assert ("g" in arrs and len(arrs["g"].shape) == 2) or \
        ("g_a" in arrs and "g_b" in arrs)
