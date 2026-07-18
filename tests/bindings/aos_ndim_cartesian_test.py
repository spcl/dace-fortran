"""N-dim AoS gather + static value-member marshalling (binding gate #12d).

``_render_aos_copy_in``/``_render_aos_copy_out`` flatten a derived-type component
over an N-D record array into SoA; for a fixed-shape VALUE member (e.g. ICON-O's
``p_vn_dual(:,:,:)%x``) all record indices are used and no presence guard is
emitted. Text-only checks on emitted Fortran -- no flang/gfortran needed.
"""
from dace_fortran.bindings.block_builders import _render_aos_copy_in, _render_aos_copy_out
from dace_fortran.bindings.frozen_signature import FrozenArg


def _cartesian_member_arg(is_written=False):
    """``p_diag % p_vn_dual(:,:,:) % x(3)`` -- rank-3 pointer record array, a
    fixed-shape rank-1 VALUE member (companion rank 4, member dim literal 3)."""
    return FrozenArg(
        fortran_name="p_diag_p_vn_dual_x",
        sdfg_name="p_diag_p_vn_dual_x",
        kind="array",
        dtype="float64",
        rank=4,
        shape=("p_diag_p_vn_dual_x_d0", "p_diag_p_vn_dual_x_d1", "p_diag_p_vn_dual_x_d2", "3"),
        is_written=is_written,
        aos_origin_struct="p_diag % p_vn_dual",
        aos_member_path="x",
        aos_outer_rank=3,
        aos_struct_pointer=True,
        aos_member_pointer=False,
    )


def _becxx_member_arg():
    """QE ``becxx(:) % k(:, :)`` -- 1-D allocatable record array, a rank-2
    ALLOCATABLE member (companion rank 3, symbolic member dims)."""
    return FrozenArg(
        fortran_name="becxx_k",
        sdfg_name="becxx_k",
        kind="array",
        dtype="float64",
        rank=3,
        shape=("becxx_k_d0", "becxx_k_d1", "becxx_k_d2"),
        aos_origin_struct="becxx",
        aos_member_path="k",
        aos_outer_rank=1,
        aos_struct_pointer=False,
        aos_member_pointer=False,
    )


def test_ndim_cartesian_copy_in_indexes_all_outer_dims():
    """The rank-3 record array is indexed by all three element loops, and the
    rank-4 companion is allocated over the three outer extents + literal 3."""
    src = "\n".join(_render_aos_copy_in(_cartesian_member_arg()))
    # Three nested element loops over the three record-array dims.
    assert "do aos_p_diag_p_vn_dual_x_i0 = 1, size(p_diag % p_vn_dual, 1)" in src
    assert "do aos_p_diag_p_vn_dual_x_i1 = 1, size(p_diag % p_vn_dual, 2)" in src
    assert "do aos_p_diag_p_vn_dual_x_i2 = 1, size(p_diag % p_vn_dual, 3)" in src
    # The element accessor uses all three indices (not a single-index 1/3 ref).
    assert ("p_diag % p_vn_dual(aos_p_diag_p_vn_dual_x_i0, aos_p_diag_p_vn_dual_x_i1, "
            "aos_p_diag_p_vn_dual_x_i2)%x") in src
    # Rank-4 allocate: three outer extents + the literal member extent.
    assert ("allocate(p_diag_p_vn_dual_x(size(p_diag % p_vn_dual, 1), "
            "size(p_diag % p_vn_dual, 2), size(p_diag % p_vn_dual, 3), 3)") in src


def test_static_value_member_emits_no_presence_guard():
    """``x(3)`` is fixed-shape: no allocated/associated guard or cap-max scan on
    it, only the struct-present (``associated``) guard remains."""
    src = "\n".join(_render_aos_copy_in(_cartesian_member_arg()))
    assert "allocated(p_diag % p_vn_dual(" not in src  # member is not allocatable
    assert "aos_p_diag_p_vn_dual_x_c0" not in src  # no cap-max scan var
    assert "associated(p_diag % p_vn_dual)" in src  # struct-pointer guard kept


def test_static_value_member_copy_out_scatters_all_dims():
    """When written, copy-out scatters back through all three element loops."""
    src = "\n".join(_render_aos_copy_out(_cartesian_member_arg(is_written=True)))
    assert "do aos_p_diag_p_vn_dual_x_i2 = 1, size(p_diag % p_vn_dual, 3)" in src
    assert ("p_diag % p_vn_dual(aos_p_diag_p_vn_dual_x_i0, aos_p_diag_p_vn_dual_x_i1, "
            "aos_p_diag_p_vn_dual_x_i2)%x = p_diag_p_vn_dual_x(aos_p_diag_p_vn_dual_x_i0, "
            "aos_p_diag_p_vn_dual_x_i1, aos_p_diag_p_vn_dual_x_i2, 1:3)") in src


def test_1d_allocatable_member_unchanged():
    """Regression: QE ``becxx(:)%k`` 1-D allocatable path keeps its single element
    loop, cap-max scan, and member ``allocated`` guard after the N-dim generalisation."""
    src = "\n".join(_render_aos_copy_in(_becxx_member_arg()))
    assert "do aos_becxx_k_i0 = 1, size(becxx, 1)" in src
    assert "allocated(becxx(aos_becxx_k_i0)%k)" in src  # allocatable member guard
    assert "aos_becxx_k_c0 = max(aos_becxx_k_c0, size(becxx(aos_becxx_k_i0)%k, 1))" in src
    assert "aos_becxx_k_c1 = max(aos_becxx_k_c1, size(becxx(aos_becxx_k_i0)%k, 2))" in src
