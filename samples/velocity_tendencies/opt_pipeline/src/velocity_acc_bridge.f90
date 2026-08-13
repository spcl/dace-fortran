! Calls the OpenACC Fortran velocity_tendencies from the serde-based C++ runner, so the ACC
! reference and the DaCe GPU variants consume byte-identical inputs from the same dump.
!
! The C++ side hands over one pointer per deserialised struct; the BIND(C) mirrors in
! velocity_acc_mirror (generated from the same header) give the component pointers plus their
! f2dace SA/SOA extents, which are the Fortran shape and lower bounds.
!
! Two residency rules, and they are what keep this honest:
!   * POINTER components (p_prog, p_metrics, p_diag, z_*) are BOUND to the C++ allocations, so the
!     kernel reads and writes the same memory the runner will serialise -- no copy, no drift.
!   * ALLOCATABLE components (p_patch grids, p_int) cannot alias C memory, so they are copied once
!     in setup. All read-only, all outside the timed region.
!
! l_vert_nested is FALSE here and asserted, not assumed: the DaCe frontend constant-folded that
! branch when it built the SDFGs (neither p_diag%vn_ie_ubc nor p_patch%nshift survives into the
! generated signature), so a TRUE run would time a different program than the one being compared.
MODULE velocity_acc_bridge
  USE iso_c_binding, ONLY: c_int, c_double, c_ptr, c_f_pointer, c_associated
  USE iso_fortran_env, ONLY: error_unit, int64
  USE velocity_acc_mirror
  USE mo_model_domain, ONLY: t_patch
  USE mo_intp_data_strc, ONLY: t_int_state
  USE mo_nonhydro_types, ONLY: t_nh_diag, t_nh_metrics, t_nh_prog
  USE mo_velocity_advection, ONLY: velocity_tendencies
  USE mo_parallel_config, ONLY: nproma
  USE mo_run_config, ONLY: lvert_nest, timers_level
  USE mo_vertical_grid, ONLY: nrdmax
  USE mo_init_vgrid, ONLY: nflatlev
  USE mo_nonhydrostatic_config, ONLY: lextra_diffu
  IMPLICIT NONE
  PRIVATE

  PUBLIC :: velocity_acc_setup, velocity_acc_run, velocity_acc_max_vcfl, velocity_acc_teardown

  INTEGER, PARAMETER :: wp = c_double

  TYPE(t_patch), SAVE, TARGET :: p_patch
  TYPE(t_int_state), SAVE, TARGET :: p_int
  TYPE(t_nh_prog), SAVE :: p_prog
  TYPE(t_nh_metrics), SAVE :: p_metrics
  TYPE(t_nh_diag), SAVE :: p_diag
  REAL(wp), POINTER, CONTIGUOUS, SAVE :: z_kin_hor_e(:, :, :) => NULL()
  REAL(wp), POINTER, CONTIGUOUS, SAVE :: z_vt_ie(:, :, :) => NULL()
  REAL(wp), POINTER, CONTIGUOUS, SAVE :: z_w_concorr_me(:, :, :) => NULL()

  ! Targets for the two components the specialised SDFGs dropped. Both sit in branches that are
  ! dead under this configuration (l_vert_nested, ddt_vn_*_is_associated), but the kernel's
  ! !$ACC DATA still names them, so they must exist and be on the device.
  REAL(wp), ALLOCATABLE, TARGET, SAVE :: dead_vn_ie_ubc(:, :, :)
  REAL(wp), ALLOCATABLE, TARGET, SAVE :: dead_ddt_vn_cor_pc(:, :, :, :)

  INTEGER, SAVE :: g_ntnd = 0, g_istep = 0
  LOGICAL, SAVE :: g_lvn_only = .FALSE., g_ldeepatmo = .FALSE.
  REAL(wp), SAVE :: g_dtime = 0.0_wp, g_dt_linintp_ubc = 0.0_wp
  LOGICAL, SAVE :: g_ready = .FALSE.

CONTAINS

  SUBROUTINE die(msg)
    CHARACTER(LEN=*), INTENT(IN) :: msg
    WRITE (error_unit, '(A)') 'velocity_acc: '//msg
    STOP 1
  END SUBROUTINE die

  ! ─── binding helpers: POINTER components alias the C++ allocation ────────────────────────
  ! Bounds-remapping (p(l:u,...) => raw) is what applies the SOA lower bounds; c_f_pointer alone
  ! would always produce 1-based bounds and silently shift every refin_ctrl lookup.

  SUBROUTINE bind_r1(cp, m, p)
    TYPE(c_ptr), INTENT(IN) :: cp
    TYPE(cm_meta), INTENT(IN) :: m
    REAL(wp), POINTER, CONTIGUOUS, INTENT(OUT) :: p(:)
    REAL(wp), POINTER, CONTIGUOUS :: raw(:)
    IF (.NOT. c_associated(cp)) CALL die('bind_r1 on a null pointer')
    CALL c_f_pointer(cp, raw, [m%dims(1)])
    p(m%lbs(1):m%lbs(1) + m%dims(1) - 1) => raw
  END SUBROUTINE bind_r1

  SUBROUTINE bind_r2_np(cp, m, p)
    TYPE(c_ptr), INTENT(IN) :: cp
    TYPE(cm_meta), INTENT(IN) :: m
    REAL(wp), POINTER, INTENT(OUT) :: p(:, :)
    REAL(wp), POINTER, CONTIGUOUS :: raw(:, :)
    IF (.NOT. c_associated(cp)) CALL die('bind_r2_np on a null pointer')
    CALL c_f_pointer(cp, raw, [m%dims(1), m%dims(2)])
    p(m%lbs(1):m%lbs(1) + m%dims(1) - 1, m%lbs(2):m%lbs(2) + m%dims(2) - 1) => raw
  END SUBROUTINE bind_r2_np

  SUBROUTINE bind_r3(cp, m, p)
    TYPE(c_ptr), INTENT(IN) :: cp
    TYPE(cm_meta), INTENT(IN) :: m
    REAL(wp), POINTER, CONTIGUOUS, INTENT(OUT) :: p(:, :, :)
    REAL(wp), POINTER, CONTIGUOUS :: raw(:, :, :)
    IF (.NOT. c_associated(cp)) CALL die('bind_r3 on a null pointer')
    CALL c_f_pointer(cp, raw, [m%dims(1), m%dims(2), m%dims(3)])
    p(m%lbs(1):m%lbs(1) + m%dims(1) - 1, m%lbs(2):m%lbs(2) + m%dims(2) - 1, &
      m%lbs(3):m%lbs(3) + m%dims(3) - 1) => raw
  END SUBROUTINE bind_r3

  SUBROUTINE bind_r4(cp, m, p)
    TYPE(c_ptr), INTENT(IN) :: cp
    TYPE(cm_meta), INTENT(IN) :: m
    REAL(wp), POINTER, CONTIGUOUS, INTENT(OUT) :: p(:, :, :, :)
    REAL(wp), POINTER, CONTIGUOUS :: raw(:, :, :, :)
    IF (.NOT. c_associated(cp)) CALL die('bind_r4 on a null pointer')
    CALL c_f_pointer(cp, raw, [m%dims(1), m%dims(2), m%dims(3), m%dims(4)])
    p(m%lbs(1):m%lbs(1) + m%dims(1) - 1, m%lbs(2):m%lbs(2) + m%dims(2) - 1, &
      m%lbs(3):m%lbs(3) + m%dims(3) - 1, m%lbs(4):m%lbs(4) + m%dims(4) - 1) => raw
  END SUBROUTINE bind_r4

  ! ─── copy helpers: ALLOCATABLE components cannot alias C memory ──────────────────────────

  SUBROUTINE copy_i1(cp, m, a)
    TYPE(c_ptr), INTENT(IN) :: cp
    TYPE(cm_meta), INTENT(IN) :: m
    INTEGER, ALLOCATABLE, INTENT(OUT) :: a(:)
    INTEGER(c_int), POINTER :: raw(:)
    IF (.NOT. c_associated(cp)) CALL die('copy_i1 on a null pointer')
    CALL c_f_pointer(cp, raw, [m%dims(1)])
    ALLOCATE (a(m%lbs(1):m%lbs(1) + m%dims(1) - 1))
    a = raw
  END SUBROUTINE copy_i1

  SUBROUTINE copy_i3(cp, m, a)
    TYPE(c_ptr), INTENT(IN) :: cp
    TYPE(cm_meta), INTENT(IN) :: m
    INTEGER, ALLOCATABLE, INTENT(OUT) :: a(:, :, :)
    INTEGER(c_int), POINTER :: raw(:, :, :)
    IF (.NOT. c_associated(cp)) CALL die('copy_i3 on a null pointer')
    CALL c_f_pointer(cp, raw, [m%dims(1), m%dims(2), m%dims(3)])
    ALLOCATE (a(m%lbs(1):m%lbs(1) + m%dims(1) - 1, m%lbs(2):m%lbs(2) + m%dims(2) - 1, &
                m%lbs(3):m%lbs(3) + m%dims(3) - 1))
    a = raw
  END SUBROUTINE copy_i3

  SUBROUTINE copy_r2(cp, m, a)
    TYPE(c_ptr), INTENT(IN) :: cp
    TYPE(cm_meta), INTENT(IN) :: m
    REAL(wp), ALLOCATABLE, INTENT(OUT) :: a(:, :)
    REAL(wp), POINTER :: raw(:, :)
    IF (.NOT. c_associated(cp)) CALL die('copy_r2 on a null pointer')
    CALL c_f_pointer(cp, raw, [m%dims(1), m%dims(2)])
    ALLOCATE (a(m%lbs(1):m%lbs(1) + m%dims(1) - 1, m%lbs(2):m%lbs(2) + m%dims(2) - 1))
    a = raw
  END SUBROUTINE copy_r2

  SUBROUTINE copy_r3(cp, m, a)
    TYPE(c_ptr), INTENT(IN) :: cp
    TYPE(cm_meta), INTENT(IN) :: m
    REAL(wp), ALLOCATABLE, INTENT(OUT) :: a(:, :, :)
    REAL(wp), POINTER :: raw(:, :, :)
    IF (.NOT. c_associated(cp)) CALL die('copy_r3 on a null pointer')
    CALL c_f_pointer(cp, raw, [m%dims(1), m%dims(2), m%dims(3)])
    ALLOCATE (a(m%lbs(1):m%lbs(1) + m%dims(1) - 1, m%lbs(2):m%lbs(2) + m%dims(2) - 1, &
                m%lbs(3):m%lbs(3) + m%dims(3) - 1))
    a = raw
  END SUBROUTINE copy_r3

  !> owner_mask is LOGICAL in the kernel but int32 in the serde struct.
  SUBROUTINE copy_l2(cp, m, a)
    TYPE(c_ptr), INTENT(IN) :: cp
    TYPE(cm_meta), INTENT(IN) :: m
    LOGICAL, ALLOCATABLE, INTENT(OUT) :: a(:, :)
    INTEGER(c_int), POINTER :: raw(:, :)
    IF (.NOT. c_associated(cp)) CALL die('copy_l2 on a null pointer')
    CALL c_f_pointer(cp, raw, [m%dims(1), m%dims(2)])
    ALLOCATE (a(m%lbs(1):m%lbs(1) + m%dims(1) - 1, m%lbs(2):m%lbs(2) + m%dims(2) - 1))
    a = (raw /= 0)
  END SUBROUTINE copy_l2

  ! ─── entry points ────────────────────────────────────────────────────────────────────────

  SUBROUTINE velocity_acc_setup(c_gd, c_diag, c_int_state, c_metrics, c_patch, c_prog, &
                                c_zkin, c_zvt, c_zwcc, zdims, zlbs, &
                                ntnd, istep, lvn_only, ldeepatmo, dtime, dt_linintp_ubc) &
    BIND(C, name='velocity_acc_setup')
    TYPE(c_ptr), VALUE :: c_gd, c_diag, c_int_state, c_metrics, c_patch, c_prog
    TYPE(c_ptr), VALUE :: c_zkin, c_zvt, c_zwcc
    INTEGER(c_int), INTENT(IN) :: zdims(9), zlbs(9)
    INTEGER(c_int), VALUE :: ntnd, istep, lvn_only, ldeepatmo
    REAL(c_double), VALUE :: dtime, dt_linintp_ubc

    TYPE(cm_global_data_type), POINTER :: gd
    TYPE(cm_t_nh_diag), POINTER :: diag
    TYPE(cm_t_int_state), POINTER :: intp
    TYPE(cm_t_nh_metrics), POINTER :: met
    TYPE(cm_t_patch), POINTER :: pat
    TYPE(cm_t_nh_prog), POINTER :: prog
    TYPE(cm_t_grid_cells), POINTER :: cel
    TYPE(cm_t_grid_edges), POINTER :: edg
    TYPE(cm_t_grid_vertices), POINTER :: vrt
    TYPE(cm_t_grid_domain_decomp_info), POINTER :: dci
    INTEGER(c_int), POINTER :: nflat_p(:), nrd_p(:)
    TYPE(cm_meta) :: m
    INTEGER :: nlev, nlevp1, ntnd_max

    CALL c_f_pointer(c_gd, gd)
    CALL c_f_pointer(c_diag, diag)
    CALL c_f_pointer(c_int_state, intp)
    CALL c_f_pointer(c_metrics, met)
    CALL c_f_pointer(c_patch, pat)
    CALL c_f_pointer(c_prog, prog)
    CALL c_f_pointer(pat%f_cells, cel)
    CALL c_f_pointer(pat%f_edges, edg)
    CALL c_f_pointer(pat%f_verts, vrt)
    CALL c_f_pointer(cel%f_decomp_info, dci)

    ! --- module-level configuration the kernel reads straight off its USE'd modules ---------
    nproma = gd%f_nproma
    lextra_diffu = gd%f_lextra_diffu /= 0
    timers_level = 0
    lvert_nest = .FALSE.
    CALL c_f_pointer(gd%f_nflatlev, nflat_p, [1])
    CALL c_f_pointer(gd%f_nrdmax, nrd_p, [1])
    nflatlev = nflat_p(1)
    nrdmax = nrd_p(1)

    IF (.NOT. lextra_diffu) CALL die('lextra_diffu is false; the compared SDFGs assume true')

    ! --- p_patch scalars: nlev/nlevp1/nshift are not in the serde struct because the frontend
    ! folded them, so derive nlev from vn and cross-check nlevp1 against vn_ie. ---------------
    m = meta_t_nh_prog_vn(prog)
    nlev = m%dims(2)
    p_patch%id = 1
    p_patch%n_childdom = 0
    p_patch%nshift = 0
    p_patch%nshift_total = 0
    p_patch%nshift_child = 0
    p_patch%nlev = nlev
    p_patch%nlevp1 = nlev + 1
    p_patch%nblks_c = pat%f_nblks_c
    p_patch%nblks_e = pat%f_nblks_e
    p_patch%nblks_v = pat%f_nblks_v
    p_patch%geometry_info%mean_cell_area = 0.0_wp
    nlevp1 = p_patch%nlevp1

    m = meta_t_nh_diag_vn_ie(diag)
    IF (m%dims(2) /= nlevp1) CALL die('vn_ie second extent does not equal nlev+1')

    ! --- p_patch grids and p_int: ALLOCATABLE, so copied once ------------------------------
    CALL copy_i3(cel%f_neighbor_idx, meta_t_grid_cells_neighbor_idx(cel), p_patch%cells%neighbor_idx)
    CALL copy_i3(cel%f_neighbor_blk, meta_t_grid_cells_neighbor_blk(cel), p_patch%cells%neighbor_blk)
    CALL copy_i3(cel%f_edge_idx, meta_t_grid_cells_edge_idx(cel), p_patch%cells%edge_idx)
    CALL copy_i3(cel%f_edge_blk, meta_t_grid_cells_edge_blk(cel), p_patch%cells%edge_blk)
    CALL copy_i1(cel%f_start_index, meta_t_grid_cells_start_index(cel), p_patch%cells%start_index)
    CALL copy_i1(cel%f_end_index, meta_t_grid_cells_end_index(cel), p_patch%cells%end_index)
    CALL copy_i1(cel%f_start_block, meta_t_grid_cells_start_block(cel), p_patch%cells%start_block)
    CALL copy_i1(cel%f_end_block, meta_t_grid_cells_end_block(cel), p_patch%cells%end_block)
    CALL copy_l2(dci%f_owner_mask, meta_t_grid_domain_decomp_info_owner_mask(dci), &
                 p_patch%cells%decomp_info%owner_mask)
    CALL bind_r2_np(cel%f_area, meta_t_grid_cells_area(cel), p_patch%cells%area)

    CALL copy_i3(edg%f_cell_idx, meta_t_grid_edges_cell_idx(edg), p_patch%edges%cell_idx)
    CALL copy_i3(edg%f_cell_blk, meta_t_grid_edges_cell_blk(edg), p_patch%edges%cell_blk)
    CALL copy_i3(edg%f_vertex_idx, meta_t_grid_edges_vertex_idx(edg), p_patch%edges%vertex_idx)
    CALL copy_i3(edg%f_vertex_blk, meta_t_grid_edges_vertex_blk(edg), p_patch%edges%vertex_blk)
    CALL copy_i3(edg%f_quad_idx, meta_t_grid_edges_quad_idx(edg), p_patch%edges%quad_idx)
    CALL copy_i3(edg%f_quad_blk, meta_t_grid_edges_quad_blk(edg), p_patch%edges%quad_blk)
    CALL copy_r2(edg%f_tangent_orientation, meta_t_grid_edges_tangent_orientation(edg), &
                 p_patch%edges%tangent_orientation)
    CALL copy_r2(edg%f_inv_primal_edge_length, meta_t_grid_edges_inv_primal_edge_length(edg), &
                 p_patch%edges%inv_primal_edge_length)
    CALL copy_r2(edg%f_inv_dual_edge_length, meta_t_grid_edges_inv_dual_edge_length(edg), &
                 p_patch%edges%inv_dual_edge_length)
    CALL copy_r2(edg%f_area_edge, meta_t_grid_edges_area_edge(edg), p_patch%edges%area_edge)
    CALL copy_r2(edg%f_f_e, meta_t_grid_edges_f_e(edg), p_patch%edges%f_e)
    CALL copy_r2(edg%f_fn_e, meta_t_grid_edges_fn_e(edg), p_patch%edges%fn_e)
    CALL copy_r2(edg%f_ft_e, meta_t_grid_edges_ft_e(edg), p_patch%edges%ft_e)
    CALL copy_i1(edg%f_start_index, meta_t_grid_edges_start_index(edg), p_patch%edges%start_index)
    CALL copy_i1(edg%f_end_index, meta_t_grid_edges_end_index(edg), p_patch%edges%end_index)
    CALL copy_i1(edg%f_start_block, meta_t_grid_edges_start_block(edg), p_patch%edges%start_block)
    CALL copy_i1(edg%f_end_block, meta_t_grid_edges_end_block(edg), p_patch%edges%end_block)

    CALL copy_i3(vrt%f_cell_idx, meta_t_grid_vertices_cell_idx(vrt), p_patch%verts%cell_idx)
    CALL copy_i3(vrt%f_cell_blk, meta_t_grid_vertices_cell_blk(vrt), p_patch%verts%cell_blk)
    CALL copy_i3(vrt%f_edge_idx, meta_t_grid_vertices_edge_idx(vrt), p_patch%verts%edge_idx)
    CALL copy_i3(vrt%f_edge_blk, meta_t_grid_vertices_edge_blk(vrt), p_patch%verts%edge_blk)
    CALL copy_i1(vrt%f_start_index, meta_t_grid_vertices_start_index(vrt), p_patch%verts%start_index)
    CALL copy_i1(vrt%f_end_index, meta_t_grid_vertices_end_index(vrt), p_patch%verts%end_index)
    CALL copy_i1(vrt%f_start_block, meta_t_grid_vertices_start_block(vrt), p_patch%verts%start_block)
    CALL copy_i1(vrt%f_end_block, meta_t_grid_vertices_end_block(vrt), p_patch%verts%end_block)

    CALL copy_r3(intp%f_c_lin_e, meta_t_int_state_c_lin_e(intp), p_int%c_lin_e)
    CALL copy_r3(intp%f_cells_aw_verts, meta_t_int_state_cells_aw_verts(intp), p_int%cells_aw_verts)
    CALL copy_r3(intp%f_e_bln_c_s, meta_t_int_state_e_bln_c_s(intp), p_int%e_bln_c_s)
    CALL copy_r3(intp%f_geofac_grdiv, meta_t_int_state_geofac_grdiv(intp), p_int%geofac_grdiv)
    CALL copy_r3(intp%f_geofac_n2s, meta_t_int_state_geofac_n2s(intp), p_int%geofac_n2s)
    CALL copy_r3(intp%f_geofac_rot, meta_t_int_state_geofac_rot(intp), p_int%geofac_rot)
    CALL copy_r3(intp%f_rbf_vec_coeff_e, meta_t_int_state_rbf_vec_coeff_e(intp), p_int%rbf_vec_coeff_e)

    ! --- POINTER components alias the runner's buffers -------------------------------------
    CALL bind_r3(prog%f_vn, meta_t_nh_prog_vn(prog), p_prog%vn)
    CALL bind_r3(prog%f_w, meta_t_nh_prog_w(prog), p_prog%w)

    CALL bind_r3(met%f_coeff1_dwdz, meta_t_nh_metrics_coeff1_dwdz(met), p_metrics%coeff1_dwdz)
    CALL bind_r3(met%f_coeff2_dwdz, meta_t_nh_metrics_coeff2_dwdz(met), p_metrics%coeff2_dwdz)
    CALL bind_r3(met%f_coeff_gradekin, meta_t_nh_metrics_coeff_gradekin(met), p_metrics%coeff_gradekin)
    CALL bind_r3(met%f_ddqz_z_full_e, meta_t_nh_metrics_ddqz_z_full_e(met), p_metrics%ddqz_z_full_e)
    CALL bind_r3(met%f_ddqz_z_half, meta_t_nh_metrics_ddqz_z_half(met), p_metrics%ddqz_z_half)
    CALL bind_r3(met%f_ddxn_z_full, meta_t_nh_metrics_ddxn_z_full(met), p_metrics%ddxn_z_full)
    CALL bind_r3(met%f_ddxt_z_full, meta_t_nh_metrics_ddxt_z_full(met), p_metrics%ddxt_z_full)
    CALL bind_r3(met%f_wgtfac_c, meta_t_nh_metrics_wgtfac_c(met), p_metrics%wgtfac_c)
    CALL bind_r3(met%f_wgtfac_e, meta_t_nh_metrics_wgtfac_e(met), p_metrics%wgtfac_e)
    CALL bind_r3(met%f_wgtfacq_e, meta_t_nh_metrics_wgtfacq_e(met), p_metrics%wgtfacq_e)
    CALL bind_r1(met%f_deepatmo_gradh_ifc, meta_t_nh_metrics_deepatmo_gradh_ifc(met), &
                 p_metrics%deepatmo_gradh_ifc)
    CALL bind_r1(met%f_deepatmo_gradh_mc, meta_t_nh_metrics_deepatmo_gradh_mc(met), &
                 p_metrics%deepatmo_gradh_mc)
    CALL bind_r1(met%f_deepatmo_invr_ifc, meta_t_nh_metrics_deepatmo_invr_ifc(met), &
                 p_metrics%deepatmo_invr_ifc)
    CALL bind_r1(met%f_deepatmo_invr_mc, meta_t_nh_metrics_deepatmo_invr_mc(met), &
                 p_metrics%deepatmo_invr_mc)

    CALL bind_r3(diag%f_vt, meta_t_nh_diag_vt(diag), p_diag%vt)
    CALL bind_r3(diag%f_vn_ie, meta_t_nh_diag_vn_ie(diag), p_diag%vn_ie)
    CALL bind_r3(diag%f_w_concorr_c, meta_t_nh_diag_w_concorr_c(diag), p_diag%w_concorr_c)
    CALL bind_r4(diag%f_ddt_vn_apc_pc, meta_t_nh_diag_ddt_vn_apc_pc(diag), p_diag%ddt_vn_apc_pc)
    CALL bind_r4(diag%f_ddt_w_adv_pc, meta_t_nh_diag_ddt_w_adv_pc(diag), p_diag%ddt_w_adv_pc)
    p_diag%max_vcfl_dyn = diag%f_max_vcfl_dyn
    p_diag%ddt_vn_adv_is_associated = .FALSE.
    p_diag%ddt_vn_cor_is_associated = .FALSE.

    m = meta_t_nh_diag_ddt_vn_apc_pc(diag)
    ntnd_max = m%dims(4)
    ! The runner calls setup once per timestep, so these two must survive re-entry; every other
    ! component is either an INTENT(OUT) allocatable (auto-deallocated) or a rebindable pointer.
    IF (ALLOCATED(dead_ddt_vn_cor_pc)) DEALLOCATE (dead_ddt_vn_cor_pc)
    IF (ALLOCATED(dead_vn_ie_ubc)) DEALLOCATE (dead_vn_ie_ubc)
    ALLOCATE (dead_ddt_vn_cor_pc(nproma, nlev, p_patch%nblks_e, ntnd_max))
    ALLOCATE (dead_vn_ie_ubc(nproma, 2, p_patch%nblks_e))
    dead_ddt_vn_cor_pc = 0.0_wp
    dead_vn_ie_ubc = 0.0_wp
    p_diag%ddt_vn_cor_pc => dead_ddt_vn_cor_pc
    p_diag%vn_ie_ubc => dead_vn_ie_ubc

    CALL bind_zarray(c_zkin, zdims(1:3), zlbs(1:3), z_kin_hor_e)
    CALL bind_zarray(c_zvt, zdims(4:6), zlbs(4:6), z_vt_ie)
    CALL bind_zarray(c_zwcc, zdims(7:9), zlbs(7:9), z_w_concorr_me)

    g_ntnd = ntnd
    g_istep = istep
    g_lvn_only = lvn_only /= 0
    g_ldeepatmo = ldeepatmo /= 0
    g_dtime = dtime
    g_dt_linintp_ubc = dt_linintp_ubc
    IF (g_ldeepatmo) CALL die('ldeepatmo is true; the compared SDFGs assume false')

    CALL stage_to_device()
    g_ready = .TRUE.
  END SUBROUTINE velocity_acc_setup

  SUBROUTINE bind_zarray(cp, dims, lbs, p)
    TYPE(c_ptr), INTENT(IN) :: cp
    INTEGER(c_int), INTENT(IN) :: dims(3), lbs(3)
    REAL(wp), POINTER, CONTIGUOUS, INTENT(OUT) :: p(:, :, :)
    TYPE(cm_meta) :: m
    m%rank = 3
    m%dims(1:3) = dims
    m%lbs(1:3) = lbs
    CALL bind_r3(cp, m, p)
  END SUBROUTINE bind_zarray

  !> Manual deep copy, no managed memory: the kernel's !$ACC DATA says PRESENT(...) because in ICON
  !> solve_nh has already staged everything. Shallow COPYIN carries the scalars and the unattached
  !> descriptors, then each component array is staged in turn and its device address attached.
  SUBROUTINE stage_to_device()
    !$ACC ENTER DATA COPYIN(p_patch, p_int, p_prog, p_metrics, p_diag)
    !$ACC ENTER DATA COPYIN(p_patch%cells%neighbor_idx, p_patch%cells%neighbor_blk, &
    !$ACC   p_patch%cells%edge_idx, p_patch%cells%edge_blk, p_patch%cells%area, &
    !$ACC   p_patch%cells%start_index, p_patch%cells%end_index, p_patch%cells%start_block, &
    !$ACC   p_patch%cells%end_block, p_patch%cells%decomp_info%owner_mask)
    !$ACC ENTER DATA COPYIN(p_patch%edges%cell_idx, p_patch%edges%cell_blk, &
    !$ACC   p_patch%edges%vertex_idx, p_patch%edges%vertex_blk, p_patch%edges%quad_idx, &
    !$ACC   p_patch%edges%quad_blk, p_patch%edges%tangent_orientation, &
    !$ACC   p_patch%edges%inv_primal_edge_length, p_patch%edges%inv_dual_edge_length, &
    !$ACC   p_patch%edges%area_edge, p_patch%edges%f_e, p_patch%edges%fn_e, p_patch%edges%ft_e, &
    !$ACC   p_patch%edges%start_index, p_patch%edges%end_index, p_patch%edges%start_block, &
    !$ACC   p_patch%edges%end_block)
    !$ACC ENTER DATA COPYIN(p_patch%verts%cell_idx, p_patch%verts%cell_blk, &
    !$ACC   p_patch%verts%edge_idx, p_patch%verts%edge_blk, p_patch%verts%start_index, &
    !$ACC   p_patch%verts%end_index, p_patch%verts%start_block, p_patch%verts%end_block)
    !$ACC ENTER DATA COPYIN(p_int%c_lin_e, p_int%cells_aw_verts, p_int%e_bln_c_s, &
    !$ACC   p_int%geofac_grdiv, p_int%geofac_n2s, p_int%geofac_rot, p_int%rbf_vec_coeff_e)
    !$ACC ENTER DATA COPYIN(p_prog%vn, p_prog%w)
    !$ACC ENTER DATA COPYIN(p_metrics%coeff1_dwdz, p_metrics%coeff2_dwdz, p_metrics%coeff_gradekin, &
    !$ACC   p_metrics%ddqz_z_full_e, p_metrics%ddqz_z_half, p_metrics%ddxn_z_full, &
    !$ACC   p_metrics%ddxt_z_full, p_metrics%wgtfac_c, p_metrics%wgtfac_e, p_metrics%wgtfacq_e, &
    !$ACC   p_metrics%deepatmo_gradh_ifc, p_metrics%deepatmo_gradh_mc, &
    !$ACC   p_metrics%deepatmo_invr_ifc, p_metrics%deepatmo_invr_mc)
    ! COPYIN, not CREATE: the runner's buffers hold the dumped t0 state and the kernel reads
    ! several of these (vn_ie, vt) before writing them.
    !$ACC ENTER DATA COPYIN(p_diag%ddt_vn_apc_pc, p_diag%ddt_vn_cor_pc, p_diag%ddt_w_adv_pc, &
    !$ACC   p_diag%vt, p_diag%vn_ie, p_diag%vn_ie_ubc, p_diag%w_concorr_c) &
    !$ACC   COPYIN(z_kin_hor_e, z_vt_ie, z_w_concorr_me)
  END SUBROUTINE stage_to_device

  !> One timed call. The WAIT pair makes the bracket local: the kernel's regions are ASYNC(1) and
  !> the timing must not depend on the callee happening to end on a WAIT.
  FUNCTION velocity_acc_run() RESULT(ms) BIND(C, name='velocity_acc_run')
    REAL(c_double) :: ms
    INTEGER(int64) :: t0, t1, rate
    IF (.NOT. g_ready) CALL die('run before setup')
    CALL SYSTEM_CLOCK(COUNT_RATE=rate)
    !$ACC WAIT
    CALL SYSTEM_CLOCK(t0)
    CALL velocity_tendencies(p_prog, p_patch, p_int, p_metrics, p_diag, z_w_concorr_me, &
                             z_kin_hor_e, z_vt_ie, g_ntnd, g_istep, g_lvn_only, g_dtime, &
                             g_dt_linintp_ubc, g_ldeepatmo)
    !$ACC WAIT
    CALL SYSTEM_CLOCK(t1)
    ms = 1000.0_wp*REAL(t1 - t0, wp)/REAL(rate, wp)
  END FUNCTION velocity_acc_run

  FUNCTION velocity_acc_max_vcfl() RESULT(v) BIND(C, name='velocity_acc_max_vcfl')
    REAL(c_double) :: v
    v = p_diag%max_vcfl_dyn
  END FUNCTION velocity_acc_max_vcfl

  !> Write set back into the runner's buffers, then drop the rest. DELETE (never COPYOUT) for the
  !> shallow parents: their device image holds device addresses that would clobber the host
  !> descriptors, so members must detach while the parent is still resident.
  SUBROUTINE velocity_acc_teardown() BIND(C, name='velocity_acc_teardown')
    IF (.NOT. g_ready) RETURN
    !$ACC EXIT DATA COPYOUT(p_diag%ddt_vn_apc_pc, p_diag%ddt_vn_cor_pc, p_diag%ddt_w_adv_pc, &
    !$ACC   p_diag%vt, p_diag%vn_ie, p_diag%vn_ie_ubc, p_diag%w_concorr_c) &
    !$ACC   COPYOUT(z_kin_hor_e, z_vt_ie, z_w_concorr_me)
    !$ACC EXIT DATA DELETE(p_patch%cells%neighbor_idx, p_patch%cells%neighbor_blk, &
    !$ACC   p_patch%cells%edge_idx, p_patch%cells%edge_blk, p_patch%cells%area, &
    !$ACC   p_patch%cells%start_index, p_patch%cells%end_index, p_patch%cells%start_block, &
    !$ACC   p_patch%cells%end_block, p_patch%cells%decomp_info%owner_mask)
    !$ACC EXIT DATA DELETE(p_patch%edges%cell_idx, p_patch%edges%cell_blk, &
    !$ACC   p_patch%edges%vertex_idx, p_patch%edges%vertex_blk, p_patch%edges%quad_idx, &
    !$ACC   p_patch%edges%quad_blk, p_patch%edges%tangent_orientation, &
    !$ACC   p_patch%edges%inv_primal_edge_length, p_patch%edges%inv_dual_edge_length, &
    !$ACC   p_patch%edges%area_edge, p_patch%edges%f_e, p_patch%edges%fn_e, p_patch%edges%ft_e, &
    !$ACC   p_patch%edges%start_index, p_patch%edges%end_index, p_patch%edges%start_block, &
    !$ACC   p_patch%edges%end_block)
    !$ACC EXIT DATA DELETE(p_patch%verts%cell_idx, p_patch%verts%cell_blk, &
    !$ACC   p_patch%verts%edge_idx, p_patch%verts%edge_blk, p_patch%verts%start_index, &
    !$ACC   p_patch%verts%end_index, p_patch%verts%start_block, p_patch%verts%end_block)
    !$ACC EXIT DATA DELETE(p_int%c_lin_e, p_int%cells_aw_verts, p_int%e_bln_c_s, &
    !$ACC   p_int%geofac_grdiv, p_int%geofac_n2s, p_int%geofac_rot, p_int%rbf_vec_coeff_e)
    !$ACC EXIT DATA DELETE(p_prog%vn, p_prog%w)
    !$ACC EXIT DATA DELETE(p_metrics%coeff1_dwdz, p_metrics%coeff2_dwdz, p_metrics%coeff_gradekin, &
    !$ACC   p_metrics%ddqz_z_full_e, p_metrics%ddqz_z_half, p_metrics%ddxn_z_full, &
    !$ACC   p_metrics%ddxt_z_full, p_metrics%wgtfac_c, p_metrics%wgtfac_e, p_metrics%wgtfacq_e, &
    !$ACC   p_metrics%deepatmo_gradh_ifc, p_metrics%deepatmo_gradh_mc, &
    !$ACC   p_metrics%deepatmo_invr_ifc, p_metrics%deepatmo_invr_mc)
    !$ACC EXIT DATA DELETE(p_patch, p_int, p_prog, p_metrics, p_diag)

    ! Checked after the copyout above brought them back, not via a separate UPDATE HOST: these were
    ! staged under their p_diag pointer-component names, so naming the targets directly risks a
    ! "not present" lookup. Every dump records ddt_vn_cor_associated=0 and l_vert_nested=0, so both
    ! stay untouched; should a configuration ever turn either on, the DaCe side needs regenerating
    ! (the frontend folded both out of the signature) -- fail loudly rather than silently time an
    ! ACC run that computed more than the variants it is compared against.
    IF (ANY(dead_ddt_vn_cor_pc /= 0.0_wp)) CALL die('ddt_vn_cor_pc was written; the SDFGs omit it')
    IF (ANY(dead_vn_ie_ubc /= 0.0_wp)) CALL die('vn_ie_ubc was written; the SDFGs omit it')
    g_ready = .FALSE.
  END SUBROUTINE velocity_acc_teardown

END MODULE velocity_acc_bridge
