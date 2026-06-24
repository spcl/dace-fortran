MODULE mo_fortran_tools
  USE iso_c_binding, ONLY: c_ptr, c_f_pointer, c_loc, c_null_ptr
  IMPLICIT NONE
  CONTAINS
  PURE SUBROUTINE set_acc_host_or_device(lzacc, lacc)
    LOGICAL, INTENT(OUT) :: lzacc
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    lzacc = .FALSE.
  END SUBROUTINE set_acc_host_or_device
END MODULE mo_fortran_tools
MODULE mo_math_types
  USE iso_c_binding, ONLY: c_int64_t
  IMPLICIT NONE
  TYPE :: t_cartesian_coordinates
    REAL(KIND = 8) :: x(3)
  END TYPE t_cartesian_coordinates
  CONTAINS
END MODULE mo_math_types
MODULE mo_model_domain
  IMPLICIT NONE
  TYPE :: t_subset_range
    INTEGER :: start_block
    INTEGER :: start_index
    INTEGER :: end_block
    INTEGER :: end_index
    INTEGER :: block_size
  END TYPE t_subset_range
  TYPE :: t_grid_edges
    INTEGER, ALLOCATABLE :: vertex_idx(:, :, :)
    INTEGER, ALLOCATABLE :: vertex_blk(:, :, :)
    REAL(KIND = 8), POINTER :: primal_edge_length(:, :) => NULL()
    TYPE(t_subset_range) :: in_domain
  END TYPE t_grid_edges
  TYPE :: t_grid_vertices
    INTEGER, ALLOCATABLE :: edge_idx(:, :, :)
    INTEGER, ALLOCATABLE :: edge_blk(:, :, :)
    INTEGER, ALLOCATABLE :: num_edges(:, :)
    REAL(KIND = 8), ALLOCATABLE :: f_v(:, :)
    TYPE(t_subset_range) :: in_domain
  END TYPE t_grid_vertices
  TYPE :: t_patch
    INTEGER :: nblks_e
    INTEGER :: nblks_v
    TYPE(t_grid_edges) :: edges
    TYPE(t_grid_vertices) :: verts
  END TYPE t_patch
  TYPE :: t_patch_vert
    INTEGER, POINTER :: dolic_e(:, :) => NULL()
    INTEGER, POINTER :: vertex_bottomlevel(:, :) => NULL()
    REAL(KIND = 8), POINTER :: prism_thick_e(:, :, :)
  END TYPE t_patch_vert
  TYPE :: t_patch_3d
    TYPE(t_patch), POINTER :: p_patch_2d(:) => NULL()
    TYPE(t_patch_vert), POINTER :: p_patch_1d(:) => NULL()
  END TYPE t_patch_3d
  CONTAINS
END MODULE mo_model_domain
MODULE mo_grid_subset
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE get_index_range(subset_range, current_block, start_index, end_index)
    USE mo_model_domain, ONLY: t_subset_range
    TYPE(t_subset_range), INTENT(IN) :: subset_range
    INTEGER, INTENT(IN) :: current_block
    INTEGER, INTENT(OUT) :: start_index, end_index
    IF (current_block < subset_range % start_block .OR. current_block > subset_range % end_block) THEN
      start_index = 1
      end_index = 0
    ELSE
      start_index = 1
      end_index = subset_range % block_size
      IF (current_block == subset_range % start_block) start_index = subset_range % start_index
      IF (current_block == subset_range % end_block) end_index = subset_range % end_index
    END IF
  END SUBROUTINE get_index_range
END MODULE mo_grid_subset
MODULE mo_ocean_nml
  IMPLICIT NONE
  INTEGER :: n_zlev
  INTEGER :: i_bc_veloc_lateral = 0
  LOGICAL :: l_anticipated_vorticity = .FALSE.
  NAMELIST /ocean_dynamics_nml/ i_bc_veloc_lateral, n_zlev
  CONTAINS
END MODULE mo_ocean_nml
MODULE mo_ocean_types
  USE mo_math_types, ONLY: t_cartesian_coordinates
  TYPE :: t_operator_coeff
    REAL(KIND = 8), POINTER, DIMENSION(:, :, :, :) :: rot_coeff
    INTEGER, POINTER, DIMENSION(:, :, :) :: bnd_edges_per_vertex
    INTEGER, POINTER, DIMENSION(:, :, :, :) :: vertex_bnd_edge_idx
    INTEGER, POINTER, DIMENSION(:, :, :, :) :: vertex_bnd_edge_blk
    INTEGER, POINTER, DIMENSION(:, :, :, :) :: boundaryedge_coefficient_index
    REAL(KIND = 8), POINTER, DIMENSION(:, :, :, :) :: edge2edge_viavert_coeff
    TYPE(t_cartesian_coordinates), POINTER, DIMENSION(:, :, :, :) :: edge2vert_coeff_cc_t
  END TYPE t_operator_coeff
END MODULE mo_ocean_types
MODULE mo_operator_ocean_coeff_3d
  IMPLICIT NONE
  INTEGER, PUBLIC :: no_dual_edges
  CONTAINS
END MODULE mo_operator_ocean_coeff_3d
MODULE mo_parallel_config
  IMPLICIT NONE
  INTEGER :: nproma = 0
  CONTAINS
END MODULE mo_parallel_config
MODULE mo_ocean_math_operators
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE rot_vertex_ocean_3d(patch_3d, vn, vn_dual, p_op_coeff, rot_vec_v, lacc)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_math_types, ONLY: t_cartesian_coordinates
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_nml, ONLY: i_bc_veloc_lateral, n_zlev
    USE mo_ocean_types, ONLY: t_operator_coeff
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_grid_subset, ONLY: get_index_range
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    REAL(KIND = 8), INTENT(IN) :: vn(:, :, :)
    TYPE(t_cartesian_coordinates), INTENT(IN) :: vn_dual(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_v)
    TYPE(t_operator_coeff), TARGET, INTENT(IN) :: p_op_coeff
    REAL(KIND = 8), INTENT(INOUT) :: rot_vec_v(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_v)
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    REAL(KIND = 8) :: z_vort_internal(n_zlev)
    REAL(KIND = 8) :: z_vort_boundary(n_zlev)
    REAL(KIND = 8) :: z_vt(4)
    INTEGER :: start_level, end_level
    INTEGER :: vertexindex, level, blockno, vertexconnect
    INTEGER :: edge_index, edge_block, boundaryedge_index, boundaryedge_block, boundaryedge_invertex
    INTEGER :: il_v1, il_v2, ib_v1, ib_v2
    INTEGER :: start_index_v, end_index_v
    LOGICAL :: lzacc
    TYPE(t_subset_range), POINTER :: verts_in_domain
    TYPE(t_patch), POINTER :: patch_2d
    patch_2d => patch_3d % p_patch_2d(1)
    verts_in_domain => patch_2d % verts % in_domain
    start_level = 1
    CALL set_acc_host_or_device(lzacc, lacc)
    DO blockno = verts_in_domain % start_block, verts_in_domain % end_block
      CALL get_index_range(patch_2d % verts % in_domain, blockno, start_index_v, end_index_v)
      rot_vec_v(:, :, blockno) = 0.0D0
      DO vertexindex = start_index_v, end_index_v
        end_level = patch_3d % p_patch_1d(1) % vertex_bottomlevel(vertexindex, blockno)
        z_vort_internal(:) = 0.0D0
        DO vertexconnect = 1, patch_2d % verts % num_edges(vertexindex, blockno)
          edge_index = patch_2d % verts % edge_idx(vertexindex, blockno, vertexconnect)
          edge_block = patch_2d % verts % edge_blk(vertexindex, blockno, vertexconnect)
          DO level = start_level, end_level
            z_vort_internal(level) = z_vort_internal(level) + vn(edge_index, level, edge_block) * p_op_coeff % rot_coeff(vertexindex, level, blockno, vertexconnect)
          END DO
        END DO
        IF (i_bc_veloc_lateral /= 0) THEN
          z_vort_boundary(1 : end_level) = 0.0D0
          z_vt(:) = 0.0D0
          DO level = start_level, end_level
            DO boundaryedge_invertex = 1, p_op_coeff % bnd_edges_per_vertex(vertexindex, level, blockno)
              boundaryedge_index = p_op_coeff % vertex_bnd_edge_idx(vertexindex, level, blockno, boundaryedge_invertex)
              boundaryedge_block = p_op_coeff % vertex_bnd_edge_blk(vertexindex, level, blockno, boundaryedge_invertex)
              il_v1 = patch_2d % edges % vertex_idx(boundaryedge_index, boundaryedge_block, 1)
              ib_v1 = patch_2d % edges % vertex_blk(boundaryedge_index, boundaryedge_block, 1)
              il_v2 = patch_2d % edges % vertex_idx(boundaryedge_index, boundaryedge_block, 2)
              ib_v2 = patch_2d % edges % vertex_blk(boundaryedge_index, boundaryedge_block, 2)
              z_vt(boundaryedge_invertex) = - DOT_PRODUCT(vn_dual(il_v1, level, ib_v1) % x, p_op_coeff % edge2vert_coeff_cc_t(boundaryedge_index, level, boundaryedge_block, 1) % x) + DOT_PRODUCT(vn_dual(il_v2, level, ib_v2) % x, p_op_coeff % edge2vert_coeff_cc_t(boundaryedge_index, level, boundaryedge_block, 2) % x)
            END DO
            DO boundaryedge_invertex = 1, p_op_coeff % bnd_edges_per_vertex(vertexindex, level, blockno)
              z_vort_boundary(level) = z_vort_boundary(level) + z_vt(boundaryedge_invertex) * p_op_coeff % rot_coeff(vertexindex, level, blockno, p_op_coeff % boundaryedge_coefficient_index(vertexindex, level, blockno, boundaryedge_invertex))
            END DO
          END DO
          DO level = start_level, end_level
            rot_vec_v(vertexindex, level, blockno) = z_vort_internal(level) + z_vort_boundary(level)
          END DO
        ELSE IF (i_bc_veloc_lateral == 0) THEN
          DO level = start_level, end_level
            rot_vec_v(vertexindex, level, blockno) = z_vort_internal(level)
          END DO
        END IF
      END DO
    END DO
  END SUBROUTINE rot_vertex_ocean_3d
END MODULE mo_ocean_math_operators
MODULE mo_sync
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE sync_patch_array_3d_dp(typ, p_patch, arr, lacc, opt_varname)
    USE mo_model_domain, ONLY: t_patch
    INTEGER, INTENT(IN) :: typ
    TYPE(t_patch), TARGET, INTENT(IN) :: p_patch
    REAL(KIND = 8), INTENT(INOUT) :: arr(:, :, :)
    LOGICAL, INTENT(IN) :: lacc
    CHARACTER(LEN = *), TARGET, INTENT(IN), OPTIONAL :: opt_varname
  END SUBROUTINE sync_patch_array_3d_dp
END MODULE mo_sync
MODULE mo_scalar_product
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE nonlinear_coriolis_3d_fast_scalar(patch_3d, vn, p_vn_dual, vort_v, operators_coefficients, vort_flux, lacc)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_nml, ONLY: l_anticipated_vorticity, n_zlev
    USE mo_math_types, ONLY: t_cartesian_coordinates
    USE mo_ocean_types, ONLY: t_operator_coeff
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_ocean_math_operators, ONLY: rot_vertex_ocean_3d
    USE mo_sync, ONLY: sync_patch_array_3d_dp_deconiface_71 => sync_patch_array_3d_dp
    USE mo_grid_subset, ONLY: get_index_range
    USE mo_operator_ocean_coeff_3d, ONLY: no_dual_edges
    TYPE(t_patch_3d), POINTER, INTENT(IN) :: patch_3d
    REAL(KIND = 8), INTENT(INOUT) :: vn(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_e)
    TYPE(t_cartesian_coordinates), INTENT(INOUT) :: p_vn_dual(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_v)
    REAL(KIND = 8), INTENT(INOUT) :: vort_v(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_v)
    TYPE(t_operator_coeff), INTENT(IN), TARGET :: operators_coefficients
    REAL(KIND = 8), INTENT(INOUT) :: vort_flux(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_e)
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: startlevel
    INTEGER :: je, level, blockno, jv
    INTEGER :: start_edge_index, end_edge_index
    INTEGER :: ictr, vertex_edge
    INTEGER :: vertex1_idx, vertex1_blk, vertex2_idx, vertex2_blk
    INTEGER :: edgeofvertex_index, edgeofvertex_block
    LOGICAL :: lzacc
    REAL(KIND = 8) :: this_vort_flux(n_zlev, 2)
    REAL(KIND = 8) :: thick_edge(n_zlev, 2), thick_vert(n_zlev, 2)
    REAL(KIND = 8) :: numofedges(n_zlev, 2)
    TYPE(t_subset_range), POINTER :: edges_in_domain
    TYPE(t_patch), POINTER :: patch_2d
    REAL(KIND = 8) :: vort_flux_old(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_e)
    patch_2d => patch_3d % p_patch_2d(1)
    edges_in_domain => patch_2d % edges % in_domain
    startlevel = 1
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL rot_vertex_ocean_3d(patch_3d, vn, p_vn_dual, operators_coefficients, vort_v, lacc = lzacc)
    CALL sync_patch_array_3d_dp_deconiface_71(3, patch_3d % p_patch_2d(1), vort_v, lacc = lzacc)
    IF (.NOT. l_anticipated_vorticity) THEN
      DO blockno = edges_in_domain % start_block, edges_in_domain % end_block
        CALL get_index_range(patch_2d % edges % in_domain, blockno, start_edge_index, end_edge_index)
        DO je = start_edge_index, end_edge_index
          vertex1_idx = patch_2d % edges % vertex_idx(je, blockno, 1)
          vertex1_blk = patch_2d % edges % vertex_blk(je, blockno, 1)
          vertex2_idx = patch_2d % edges % vertex_idx(je, blockno, 2)
          vertex2_blk = patch_2d % edges % vertex_blk(je, blockno, 2)
          DO jv = 1, 2
            DO level = 1, n_zlev
              this_vort_flux(level, jv) = 0.0D0
              numofedges(level, jv) = 0.0D0
            END DO
          END DO
          DO vertex_edge = 1, patch_2d % verts % num_edges(vertex1_idx, vertex1_blk)
            edgeofvertex_index = patch_2d % verts % edge_idx(vertex1_idx, vertex1_blk, vertex_edge)
            edgeofvertex_block = patch_2d % verts % edge_blk(vertex1_idx, vertex1_blk, vertex_edge)
            DO level = startlevel, MIN(patch_3d % p_patch_1d(1) % dolic_e(je, blockno), patch_3d % p_patch_1d(1) % dolic_e(edgeofvertex_index, edgeofvertex_block))
              numofedges(level, 1) = numofedges(level, 1) + 1.0D0
              this_vort_flux(level, 1) = this_vort_flux(level, 1) + vn(edgeofvertex_index, level, edgeofvertex_block) * operators_coefficients % edge2edge_viavert_coeff(je, level, blockno, vertex_edge)
            END DO
          END DO
          DO vertex_edge = 1, patch_2d % verts % num_edges(vertex2_idx, vertex2_blk)
            edgeofvertex_index = patch_2d % verts % edge_idx(vertex2_idx, vertex2_blk, vertex_edge)
            edgeofvertex_block = patch_2d % verts % edge_blk(vertex2_idx, vertex2_blk, vertex_edge)
            DO level = startlevel, MIN(patch_3d % p_patch_1d(1) % dolic_e(je, blockno), patch_3d % p_patch_1d(1) % dolic_e(edgeofvertex_index, edgeofvertex_block))
              numofedges(level, 2) = numofedges(level, 2) + 1.0D0
              this_vort_flux(level, 2) = this_vort_flux(level, 2) + vn(edgeofvertex_index, level, edgeofvertex_block) * operators_coefficients % edge2edge_viavert_coeff(je, level, blockno, no_dual_edges + vertex_edge)
            END DO
          END DO
          DO level = startlevel, patch_3d % p_patch_1d(1) % dolic_e(je, blockno)
            vort_flux(je, level, blockno) = this_vort_flux(level, 1) * (vort_v(vertex1_idx, level, vertex1_blk) + patch_2d % verts % f_v(vertex1_idx, vertex1_blk)) + this_vort_flux(level, 2) * (vort_v(vertex2_idx, level, vertex2_blk) + patch_2d % verts % f_v(vertex2_idx, vertex2_blk))
          END DO
        END DO
      END DO
    ELSE IF (l_anticipated_vorticity) THEN
      vort_flux_old(:, :, :) = 0.0D0
      DO blockno = edges_in_domain % start_block, edges_in_domain % end_block
        CALL get_index_range(patch_2d % edges % in_domain, blockno, start_edge_index, end_edge_index)
        DO je = start_edge_index, end_edge_index
          this_vort_flux(:, :) = 0.0D0
          vertex1_idx = patch_2d % edges % vertex_idx(je, blockno, 1)
          vertex1_blk = patch_2d % edges % vertex_blk(je, blockno, 1)
          vertex2_idx = patch_2d % edges % vertex_idx(je, blockno, 2)
          vertex2_blk = patch_2d % edges % vertex_blk(je, blockno, 2)
          ictr = 0
          thick_vert(1 : n_zlev, 1) = 0.0D0
          thick_vert(1 : n_zlev, 2) = 0.0D0
          numofedges(1 : n_zlev, 1) = 0.0D0
          numofedges(1 : n_zlev, 2) = 0.0D0
          DO vertex_edge = 1, patch_2d % verts % num_edges(vertex1_idx, vertex1_blk)
            ictr = ictr + 1
            edgeofvertex_index = patch_2d % verts % edge_idx(vertex1_idx, vertex1_blk, vertex_edge)
            edgeofvertex_block = patch_2d % verts % edge_blk(vertex1_idx, vertex1_blk, vertex_edge)
            DO level = startlevel, MIN(patch_3d % p_patch_1d(1) % dolic_e(je, blockno), patch_3d % p_patch_1d(1) % dolic_e(edgeofvertex_index, edgeofvertex_block))
              numofedges(level, 1) = numofedges(level, 1) + 1.0D0
              thick_edge(level, 1) = patch_3d % p_patch_1d(1) % prism_thick_e(edgeofvertex_index, level, edgeofvertex_block)
              thick_vert(level, 1) = thick_vert(level, 1) + thick_edge(level, 1)
              this_vort_flux(level, 1) = this_vort_flux(level, 1) + vn(edgeofvertex_index, level, edgeofvertex_block) * operators_coefficients % edge2edge_viavert_coeff(je, level, blockno, ictr) * thick_edge(level, 1)
            END DO
          END DO
          ictr = no_dual_edges
          DO vertex_edge = 1, patch_2d % verts % num_edges(vertex2_idx, vertex2_blk)
            ictr = ictr + 1
            edgeofvertex_index = patch_2d % verts % edge_idx(vertex2_idx, vertex2_blk, vertex_edge)
            edgeofvertex_block = patch_2d % verts % edge_blk(vertex2_idx, vertex2_blk, vertex_edge)
            DO level = startlevel, MIN(patch_3d % p_patch_1d(1) % dolic_e(je, blockno), patch_3d % p_patch_1d(1) % dolic_e(edgeofvertex_index, edgeofvertex_block))
              numofedges(level, 2) = numofedges(level, 2) + 1.0D0
              thick_edge(level, 2) = patch_3d % p_patch_1d(1) % prism_thick_e(edgeofvertex_index, level, edgeofvertex_block)
              thick_vert(level, 2) = thick_vert(level, 2) + thick_edge(level, 2)
              this_vort_flux(level, 2) = this_vort_flux(level, 2) + vn(edgeofvertex_index, level, edgeofvertex_block) * operators_coefficients % edge2edge_viavert_coeff(je, level, blockno, ictr) * thick_edge(level, 2)
            END DO
          END DO
          DO level = startlevel, patch_3d % p_patch_1d(1) % dolic_e(je, blockno)
            vort_flux(je, level, blockno) = this_vort_flux(level, 1) * numofedges(level, 1) / thick_vert(level, 1) * (vort_v(vertex1_idx, level, vertex1_blk) + patch_2d % verts % f_v(vertex1_idx, vertex1_blk)) + this_vort_flux(level, 2) * numofedges(level, 2) / thick_vert(level, 2) * (vort_v(vertex2_idx, level, vertex2_blk) + patch_2d % verts % f_v(vertex2_idx, vertex2_blk))
            vort_flux_old(je, level, blockno) = vort_flux(je, level, blockno)
            vort_flux(je, level, blockno) = vort_flux(je, level, blockno) - (this_vort_flux(level, 1) * numofedges(level, 1) / thick_vert(level, 1) + this_vort_flux(level, 2) * numofedges(level, 2) / thick_vert(level, 2)) * 0.5D0 * (vort_v(vertex2_idx, level, vertex2_blk) - vort_v(vertex1_idx, level, vertex1_blk)) / patch_2d % edges % primal_edge_length(je, blockno)
          END DO
        END DO
      END DO
    END IF
  END SUBROUTINE nonlinear_coriolis_3d_fast_scalar
END MODULE mo_scalar_product