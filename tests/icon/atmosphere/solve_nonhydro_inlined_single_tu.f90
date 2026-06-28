MODULE mo_decomposition_tools
  IMPLICIT NONE
  TYPE :: t_grid_domain_decomp_info
    INTEGER, ALLOCATABLE :: glb_index(:)
    INTEGER, POINTER :: decomp_domain(:, :)
  END TYPE
  CONTAINS
END MODULE mo_decomposition_tools
MODULE mo_dynamics_config
  IMPLICIT NONE
  LOGICAL :: ldeepatmo
  CONTAINS
END MODULE mo_dynamics_config
MODULE mo_exception
  IMPLICIT NONE
  INTERFACE
    SUBROUTINE callback_function
      IMPLICIT NONE
    END SUBROUTINE callback_function
  END INTERFACE
  INTERFACE
    FUNCTION prefix_function() RESULT(pre)
      IMPLICIT NONE
      CHARACTER(LEN = :), ALLOCATABLE :: pre
      CHARACTER(LEN = 100) :: tmp
    END FUNCTION prefix_function
  END INTERFACE
  CONTAINS
  SUBROUTINE finish(name, text)
    CHARACTER(LEN = *), INTENT(IN) :: name
    CHARACTER(LEN = *), INTENT(IN), OPTIONAL :: text
  END SUBROUTINE finish
END MODULE mo_exception
MODULE mo_fortran_tools
  USE iso_c_binding, ONLY: c_ptr, c_f_pointer, c_loc, c_null_ptr
  IMPLICIT NONE
  TYPE :: t_ptr_3d_dp
    REAL(KIND = 8), POINTER :: p(:, :, :) => NULL()
  END TYPE t_ptr_3d_dp
  TYPE :: t_ptr_3d_sp
    REAL(KIND = 4), POINTER :: p(:, :, :) => NULL()
  END TYPE t_ptr_3d_sp
  CONTAINS
  SUBROUTINE acc_wait_if_requested(acc_async_queue, opt_acc_async)
    INTEGER, INTENT(IN) :: acc_async_queue
    LOGICAL, INTENT(IN), OPTIONAL :: opt_acc_async
  END SUBROUTINE acc_wait_if_requested
  SUBROUTINE init_contiguous_dp(var, n, v, lacc, opt_acc_async)
    INTEGER, INTENT(IN) :: n
    REAL(KIND = 8), INTENT(OUT) :: var(n)
    REAL(KIND = 8), INTENT(IN) :: v
    LOGICAL, INTENT(IN) :: lacc
    LOGICAL, INTENT(IN), OPTIONAL :: opt_acc_async
    INTEGER :: i
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, .TRUE.)
    DO i = 1, n
      var(i) = 0.0D0
    END DO
    CALL acc_wait_if_requested(1, .TRUE.)
  END SUBROUTINE init_contiguous_dp
  SUBROUTINE init_zero_contiguous_dp(var, n, lacc, opt_acc_async)
    INTEGER, INTENT(IN) :: n
    REAL(KIND = 8), INTENT(OUT) :: var(n)
    LOGICAL, INTENT(IN) :: lacc
    LOGICAL, INTENT(IN), OPTIONAL :: opt_acc_async
    CALL init_contiguous_dp(var, n, 0.0D0, .TRUE., .TRUE.)
  END SUBROUTINE init_zero_contiguous_dp
  SUBROUTINE assert_acc_device_only(routine_name, lacc)
    CHARACTER(LEN = *), INTENT(IN) :: routine_name
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
  END SUBROUTINE assert_acc_device_only
  PURE SUBROUTINE set_acc_host_or_device(lzacc, lacc)
    LOGICAL, INTENT(OUT) :: lzacc
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    lzacc = .FALSE.
  END SUBROUTINE set_acc_host_or_device
END MODULE mo_fortran_tools
MODULE mo_grid_config
  IMPLICIT NONE
  LOGICAL :: l_limited_area
  CONTAINS
END MODULE mo_grid_config
MODULE mo_gridref_config
  IMPLICIT NONE
  INTEGER :: grf_intmethod_e
  CONTAINS
END MODULE mo_gridref_config
MODULE mo_init_vgrid
  IMPLICIT NONE
  INTEGER :: nflatlev(10)
  CONTAINS
END MODULE mo_init_vgrid
MODULE mo_initicon_config
  IMPLICIT NONE
  LOGICAL :: is_iau_active = .FALSE.
  REAL(KIND = 8) :: iau_wgt_dyn = 0.0D0
  CONTAINS
END MODULE mo_initicon_config
MODULE mo_interpol_config
  IMPLICIT NONE
  REAL(KIND = 8) :: nudge_max_coeff
  CONTAINS
END MODULE mo_interpol_config
MODULE mo_intp_data_strc
  IMPLICIT NONE
  TYPE :: t_int_state
    REAL(KIND = 8), ALLOCATABLE :: c_lin_e(:, :, :)
    REAL(KIND = 8), ALLOCATABLE :: e_bln_c_s(:, :, :)
    REAL(KIND = 8), ALLOCATABLE :: e_flx_avg(:, :, :)
    REAL(KIND = 8), ALLOCATABLE :: cells_aw_verts(:, :, :)
    REAL(KIND = 8), ALLOCATABLE :: rbf_vec_coeff_e(:, :, :)
    REAL(KIND = 8), ALLOCATABLE :: geofac_div(:, :, :)
    REAL(KIND = 8), ALLOCATABLE :: geofac_grdiv(:, :, :)
    REAL(KIND = 8), ALLOCATABLE :: geofac_grg(:, :, :, :)
    REAL(KIND = 8), ALLOCATABLE :: pos_on_tplane_e(:, :, :, :)
    REAL(KIND = 8), ALLOCATABLE :: nudgecoeff_e(:, :)
  END TYPE t_int_state
END MODULE mo_intp_data_strc
MODULE mo_io_units
  IMPLICIT NONE
  INTEGER, PARAMETER :: filename_max = 1024
  CONTAINS
  FUNCTION find_next_free_unit(istart, istop) RESULT(iunit)
    INTEGER :: iunit
    INTEGER, INTENT(IN) :: istart, istop
    INTEGER :: kstart, kstop
    LOGICAL :: lfound, lopened
    INTEGER :: i
    lfound = .FALSE.
    kstart = 10
    kstop = 99
    DO i = kstart, kstop
      INQUIRE(UNIT = i, OPENED = lopened)
      IF (.NOT. lopened) THEN
        iunit = i
        lfound = .TRUE.
        EXIT
      END IF
    END DO
    IF (.NOT. lfound) THEN
      iunit = (- 1)
    END IF
  END FUNCTION find_next_free_unit
END MODULE mo_io_units
MODULE mo_lib_grid_geometry_info
  IMPLICIT NONE
  TYPE :: t_grid_geometry_info
    REAL(KIND = 8) :: mean_cell_area
  END TYPE t_grid_geometry_info
END MODULE mo_lib_grid_geometry_info
MODULE mo_lib_loopindices
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE get_indices_c_lib(i_startidx_in, i_endidx_in, nproma, i_blk, i_startblk, i_endblk, i_startidx_out, i_endidx_out)
    INTEGER, INTENT(IN) :: i_startidx_in
    INTEGER, INTENT(IN) :: i_endidx_in
    INTEGER, INTENT(IN) :: nproma
    INTEGER, INTENT(IN) :: i_blk
    INTEGER, INTENT(IN) :: i_startblk
    INTEGER, INTENT(IN) :: i_endblk
    INTEGER, INTENT(OUT) :: i_startidx_out, i_endidx_out
    IF (i_blk == i_startblk) THEN
      i_startidx_out = MAX(1, i_startidx_in)
      i_endidx_out = nproma
      IF (i_blk == i_endblk) i_endidx_out = i_endidx_in
    ELSE IF (i_blk == i_endblk) THEN
      i_startidx_out = 1
      i_endidx_out = i_endidx_in
    ELSE
      i_startidx_out = 1
      i_endidx_out = nproma
    END IF
  END SUBROUTINE get_indices_c_lib
  SUBROUTINE get_indices_e_lib(i_startidx_in, i_endidx_in, nproma, i_blk, i_startblk, i_endblk, i_startidx_out, i_endidx_out)
    INTEGER, INTENT(IN) :: i_startidx_in
    INTEGER, INTENT(IN) :: i_endidx_in
    INTEGER, INTENT(IN) :: nproma
    INTEGER, INTENT(IN) :: i_blk
    INTEGER, INTENT(IN) :: i_startblk
    INTEGER, INTENT(IN) :: i_endblk
    INTEGER, INTENT(OUT) :: i_startidx_out, i_endidx_out
    i_startidx_out = MERGE(1, MAX(1, i_startidx_in), i_blk /= i_startblk)
    i_endidx_out = MERGE(nproma, i_endidx_in, i_blk /= i_endblk)
  END SUBROUTINE get_indices_e_lib
  SUBROUTINE get_indices_v_lib(i_startidx_in, i_endidx_in, nproma, i_blk, i_startblk, i_endblk, i_startidx_out, i_endidx_out)
    INTEGER, INTENT(IN) :: i_startidx_in
    INTEGER, INTENT(IN) :: i_endidx_in
    INTEGER, INTENT(IN) :: nproma
    INTEGER, INTENT(IN) :: i_blk
    INTEGER, INTENT(IN) :: i_startblk
    INTEGER, INTENT(IN) :: i_endblk
    INTEGER, INTENT(OUT) :: i_startidx_out, i_endidx_out
    IF (i_blk == i_startblk) THEN
      i_startidx_out = i_startidx_in
      i_endidx_out = nproma
      IF (i_blk == i_endblk) i_endidx_out = i_endidx_in
    ELSE IF (i_blk == i_endblk) THEN
      i_startidx_out = 1
      i_endidx_out = i_endidx_in
    ELSE
      i_startidx_out = 1
      i_endidx_out = nproma
    END IF
  END SUBROUTINE get_indices_v_lib
END MODULE mo_lib_loopindices
MODULE mo_lib_gradients
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE grad_green_gauss_cell_dycore_lib(p_ccpr, cell_neighbor_idx, cell_neighbor_blk, geofac_grg, p_grad, i_startblk, i_endblk, i_startidx_in, i_endidx_in, slev, elev, nproma, lacc, acc_async)
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_lib_loopindices, ONLY: get_indices_c_lib
    REAL(KIND = 8), INTENT(IN) :: p_ccpr(:, :, :, :)
    INTEGER, TARGET, INTENT(IN) :: cell_neighbor_idx(:, :, :)
    INTEGER, TARGET, INTENT(IN) :: cell_neighbor_blk(:, :, :)
    REAL(KIND = 8), INTENT(IN) :: geofac_grg(:, :, :, :)
    REAL(KIND = 8), INTENT(INOUT) :: p_grad(:, :, :, :)
    INTEGER, INTENT(IN) :: i_startblk
    INTEGER, INTENT(IN) :: i_endblk
    INTEGER, INTENT(IN) :: i_startidx_in
    INTEGER, INTENT(IN) :: i_endidx_in
    INTEGER, INTENT(IN) :: slev
    INTEGER, INTENT(IN) :: elev
    INTEGER, INTENT(IN) :: nproma
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    LOGICAL, INTENT(IN), OPTIONAL :: acc_async
    INTEGER :: jc, jk, jb
    INTEGER :: i_startidx, i_endidx
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, .TRUE.)
    DO jb = i_startblk, i_endblk
      CALL get_indices_c_lib(i_startidx_in, i_endidx_in, nproma, jb, i_startblk, i_endblk, i_startidx, i_endidx)
      DO jc = i_startidx, i_endidx
        DO jk = slev, elev
          p_grad(1, jc, jk, jb) = geofac_grg(jc, 1, jb, 1) * p_ccpr(1, jc, jk, jb) + geofac_grg(jc, 2, jb, 1) * p_ccpr(1, cell_neighbor_idx(jc, jb, 1), jk, cell_neighbor_blk(jc, jb, 1)) + geofac_grg(jc, 3, jb, 1) * p_ccpr(1, cell_neighbor_idx(jc, jb, 2), jk, cell_neighbor_blk(jc, jb, 2)) + geofac_grg(jc, 4, jb, 1) * p_ccpr(1, cell_neighbor_idx(jc, jb, 3), jk, cell_neighbor_blk(jc, jb, 3))
          p_grad(2, jc, jk, jb) = geofac_grg(jc, 1, jb, 2) * p_ccpr(1, jc, jk, jb) + geofac_grg(jc, 2, jb, 2) * p_ccpr(1, cell_neighbor_idx(jc, jb, 1), jk, cell_neighbor_blk(jc, jb, 1)) + geofac_grg(jc, 3, jb, 2) * p_ccpr(1, cell_neighbor_idx(jc, jb, 2), jk, cell_neighbor_blk(jc, jb, 2)) + geofac_grg(jc, 4, jb, 2) * p_ccpr(1, cell_neighbor_idx(jc, jb, 3), jk, cell_neighbor_blk(jc, jb, 3))
          p_grad(3, jc, jk, jb) = geofac_grg(jc, 1, jb, 1) * p_ccpr(2, jc, jk, jb) + geofac_grg(jc, 2, jb, 1) * p_ccpr(2, cell_neighbor_idx(jc, jb, 1), jk, cell_neighbor_blk(jc, jb, 1)) + geofac_grg(jc, 3, jb, 1) * p_ccpr(2, cell_neighbor_idx(jc, jb, 2), jk, cell_neighbor_blk(jc, jb, 2)) + geofac_grg(jc, 4, jb, 1) * p_ccpr(2, cell_neighbor_idx(jc, jb, 3), jk, cell_neighbor_blk(jc, jb, 3))
          p_grad(4, jc, jk, jb) = geofac_grg(jc, 1, jb, 2) * p_ccpr(2, jc, jk, jb) + geofac_grg(jc, 2, jb, 2) * p_ccpr(2, cell_neighbor_idx(jc, jb, 1), jk, cell_neighbor_blk(jc, jb, 1)) + geofac_grg(jc, 3, jb, 2) * p_ccpr(2, cell_neighbor_idx(jc, jb, 2), jk, cell_neighbor_blk(jc, jb, 2)) + geofac_grg(jc, 4, jb, 2) * p_ccpr(2, cell_neighbor_idx(jc, jb, 3), jk, cell_neighbor_blk(jc, jb, 3))
        END DO
      END DO
    END DO
  END SUBROUTINE grad_green_gauss_cell_dycore_lib
END MODULE mo_lib_gradients
MODULE mo_lib_interpolation_scalar
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE cells2verts_scalar_dp_lib(p_cell_in, vert_cell_idx, vert_cell_blk, coeff_int, p_vert_out, i_startblk, i_endblk, i_startidx_in, i_endidx_in, slev, elev, nproma, lacc, acc_async)
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_lib_loopindices, ONLY: get_indices_v_lib
    REAL(KIND = 8), INTENT(IN) :: p_cell_in(:, :, :)
    INTEGER, TARGET, INTENT(IN) :: vert_cell_idx(:, :, :)
    INTEGER, TARGET, INTENT(IN) :: vert_cell_blk(:, :, :)
    REAL(KIND = 8), INTENT(IN) :: coeff_int(:, :, :)
    REAL(KIND = 8), INTENT(INOUT) :: p_vert_out(:, :, :)
    INTEGER, INTENT(IN) :: i_startblk
    INTEGER, INTENT(IN) :: i_endblk
    INTEGER, INTENT(IN) :: i_startidx_in
    INTEGER, INTENT(IN) :: i_endidx_in
    INTEGER, INTENT(IN) :: slev
    INTEGER, INTENT(IN) :: elev
    INTEGER, INTENT(IN) :: nproma
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    LOGICAL, INTENT(IN), OPTIONAL :: acc_async
    INTEGER :: jv, jk, jb
    INTEGER :: i_startidx, i_endidx
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, .TRUE.)
    DO jb = i_startblk, i_endblk
      CALL get_indices_v_lib(i_startidx_in, i_endidx_in, nproma, jb, i_startblk, i_endblk, i_startidx, i_endidx)
      DO jv = i_startidx, i_endidx
        DO jk = slev, elev
          p_vert_out(jv, jk, jb) = coeff_int(jv, 1, jb) * p_cell_in(vert_cell_idx(jv, jb, 1), jk, vert_cell_blk(jv, jb, 1)) + coeff_int(jv, 2, jb) * p_cell_in(vert_cell_idx(jv, jb, 2), jk, vert_cell_blk(jv, jb, 2)) + coeff_int(jv, 3, jb) * p_cell_in(vert_cell_idx(jv, jb, 3), jk, vert_cell_blk(jv, jb, 3)) + coeff_int(jv, 4, jb) * p_cell_in(vert_cell_idx(jv, jb, 4), jk, vert_cell_blk(jv, jb, 4)) + coeff_int(jv, 5, jb) * p_cell_in(vert_cell_idx(jv, jb, 5), jk, vert_cell_blk(jv, jb, 5)) + coeff_int(jv, 6, jb) * p_cell_in(vert_cell_idx(jv, jb, 6), jk, vert_cell_blk(jv, jb, 6))
        END DO
      END DO
    END DO
    IF (.NOT. acc_async) THEN
    END IF
  END SUBROUTINE cells2verts_scalar_dp_lib
END MODULE mo_lib_interpolation_scalar
MODULE mo_math_types
  USE iso_c_binding, ONLY: c_int64_t
  IMPLICIT NONE
  TYPE :: t_tangent_vectors
    REAL(KIND = 8) :: v1
    REAL(KIND = 8) :: v2
  END TYPE t_tangent_vectors
  CONTAINS
END MODULE mo_math_types
MODULE mo_nonhydro_types
  IMPLICIT NONE
  TYPE :: t_nh_prog
    REAL(KIND = 8), POINTER, CONTIGUOUS :: w(:, :, :), vn(:, :, :), rho(:, :, :), exner(:, :, :), theta_v(:, :, :)
  END TYPE t_nh_prog
  TYPE :: t_nh_diag
    REAL(KIND = 8), POINTER, CONTIGUOUS :: exner_pr(:, :, :), mass_fl_e(:, :, :), rho_ic(:, :, :), theta_v_ic(:, :, :), grf_tend_vn(:, :, :), grf_tend_w(:, :, :), grf_tend_rho(:, :, :), grf_tend_mflx(:, :, :), grf_bdy_mflx(:, :, :), grf_tend_thv(:, :, :), vn_ie_int(:, :, :), vn_ie_ubc(:, :, :), w_int(:, :, :), w_ubc(:, :, :), theta_v_ic_int(:, :, :), theta_v_ic_ubc(:, :, :), rho_ic_int(:, :, :), rho_ic_ubc(:, :, :), mflx_ic_int(:, :, :), mflx_ic_ubc(:, :, :)
    REAL(KIND = 8), POINTER, CONTIGUOUS :: vn_incr(:, :, :), exner_incr(:, :, :), rho_incr(:, :, :), vt(:, :, :), ddt_exner_phy(:, :, :), ddt_vn_phy(:, :, :), exner_dyn_incr(:, :, :), vn_ie(:, :, :), w_concorr_c(:, :, :), mass_fl_e_sv(:, :, :), ddt_vn_apc_pc(:, :, :, :), ddt_vn_cor_pc(:, :, :, :), ddt_w_adv_pc(:, :, :, :)
    REAL(KIND = 8), POINTER, CONTIGUOUS :: ddt_vn_dyn(:, :, :) => NULL(), ddt_vn_dmp(:, :, :) => NULL(), ddt_vn_adv(:, :, :) => NULL(), ddt_vn_cor(:, :, :) => NULL(), ddt_vn_pgr(:, :, :) => NULL(), ddt_vn_phd(:, :, :) => NULL(), ddt_vn_iau(:, :, :) => NULL(), ddt_vn_ray(:, :, :) => NULL(), ddt_vn_grf(:, :, :) => NULL()
    LOGICAL :: ddt_vn_dyn_is_associated = .FALSE., ddt_vn_dmp_is_associated = .FALSE., ddt_vn_adv_is_associated = .FALSE., ddt_vn_cor_is_associated = .FALSE., ddt_vn_pgr_is_associated = .FALSE., ddt_vn_phd_is_associated = .FALSE., ddt_vn_iau_is_associated = .FALSE., ddt_vn_ray_is_associated = .FALSE., ddt_vn_grf_is_associated = .FALSE.
  END TYPE t_nh_diag
  TYPE :: t_nh_ref
    REAL(KIND = 8), POINTER :: vn_ref(:, :, :), w_ref(:, :, :) => NULL()
  END TYPE t_nh_ref
  TYPE :: t_nh_metrics
    REAL(KIND = 8), POINTER, CONTIGUOUS :: rayleigh_w(:), rayleigh_vn(:), scalfac_dd3d(:), hmask_dd3d(:, :), vwind_expl_wgt(:, :), vwind_impl_wgt(:, :)
    REAL(KIND = 8), POINTER, CONTIGUOUS :: ddxn_z_full(:, :, :), ddxt_z_full(:, :, :), ddqz_z_full_e(:, :, :), ddqz_z_half(:, :, :), inv_ddqz_z_full(:, :, :), wgtfac_c(:, :, :), wgtfac_e(:, :, :), wgtfacq_c(:, :, :), wgtfacq_e(:, :, :), wgtfacq1_c(:, :, :), zdiff_gradp(:, :, :, :), coeff_gradp(:, :, :, :), exner_exfac(:, :, :), theta_ref_mc(:, :, :), theta_ref_me(:, :, :), theta_ref_ic(:, :, :), exner_ref_mc(:, :, :), rho_ref_mc(:, :, :), rho_ref_me(:, :, :), d_exner_dz_ref_ic(:, :, :), d2dexdz2_fac1_mc(:, :, :), d2dexdz2_fac2_mc(:, :, :), pg_exdist(:) => NULL()
    INTEGER, POINTER, CONTIGUOUS :: vertidx_gradp(:, :, :, :), pg_edgeidx(:), pg_edgeblk(:), pg_vertidx(:), bdy_halo_c_idx(:), bdy_halo_c_blk(:), bdy_mflx_e_idx(:), bdy_mflx_e_blk(:) => NULL()
    REAL(KIND = 8), POINTER, CONTIGUOUS :: deepatmo_gradh_mc(:), deepatmo_divh_mc(:), deepatmo_divzu_mc(:), deepatmo_divzl_mc(:)
    INTEGER :: pg_listdim
    INTEGER :: bdy_halo_c_dim
    INTEGER :: bdy_mflx_e_dim
    LOGICAL, POINTER :: mask_prog_halo_c(:, :) => NULL()
  END TYPE t_nh_metrics
  TYPE :: t_nh_state
    TYPE(t_nh_prog), ALLOCATABLE :: prog(:)
    TYPE(t_nh_diag) :: diag
    TYPE(t_nh_ref) :: ref
    TYPE(t_nh_metrics) :: metrics
  END TYPE t_nh_state
END MODULE mo_nonhydro_types
MODULE mo_nonhydrostatic_config
  IMPLICIT NONE
  INTEGER :: itime_scheme
  INTEGER :: ndyn_substeps_var(10)
  REAL(KIND = 8) :: divdamp_fac
  REAL(KIND = 8) :: divdamp_fac2
  REAL(KIND = 8) :: divdamp_fac3
  REAL(KIND = 8) :: divdamp_fac4
  REAL(KIND = 8) :: divdamp_z
  REAL(KIND = 8) :: divdamp_z2
  REAL(KIND = 8) :: divdamp_z3
  REAL(KIND = 8) :: divdamp_z4
  REAL(KIND = 8) :: divdamp_fac_o2
  INTEGER :: divdamp_order
  INTEGER :: divdamp_type
  INTEGER :: rayleigh_type
  REAL(KIND = 8) :: rhotheta_offctr
  REAL(KIND = 8) :: veladv_offctr
  INTEGER :: iadv_rhotheta
  INTEGER :: igradp_method
  INTEGER :: kstart_dd3d(10)
  INTEGER :: kstart_moist(10)
  CONTAINS
END MODULE mo_nonhydrostatic_config
MODULE mo_parallel_config
  IMPLICIT NONE
  INTEGER :: nproma = 0
  INTEGER :: n_ghost_rows = 1
  LOGICAL :: l_log_checks = .FALSE.
  LOGICAL :: p_test_run = .FALSE.
  LOGICAL :: use_dycore_barrier = .FALSE.
  INTEGER :: itype_exch_barrier = 0
  INTEGER :: iorder_sendrecv = 1
  CONTAINS
  FUNCTION cpu_min_nproma(nproma, min_nproma) RESULT(new_nproma)
    INTEGER, INTENT(IN) :: nproma, min_nproma
    INTEGER :: new_nproma
    new_nproma = MIN(nproma, 256)
  END FUNCTION cpu_min_nproma
  ELEMENTAL INTEGER FUNCTION blk_no(j)
    INTEGER, INTENT(IN) :: j
    blk_no = MAX((ABS(j) - 1) / nproma + 1, 1)
  END FUNCTION blk_no
  ELEMENTAL INTEGER FUNCTION idx_no(j)
    INTEGER, INTENT(IN) :: j
    IF (j == 0) THEN
      idx_no = 0
    ELSE
      idx_no = SIGN(MOD(ABS(j) - 1, nproma) + 1, j)
    END IF
  END FUNCTION idx_no
END MODULE mo_parallel_config
MODULE mo_prepadv_types
  IMPLICIT NONE
  TYPE :: t_prepare_adv
    REAL(KIND = 8), POINTER, CONTIGUOUS :: mass_flx_me(:, :, :), mass_flx_ic(:, :, :), vol_flx_ic(:, :, :), vn_traj(:, :, :)
  END TYPE t_prepare_adv
END MODULE mo_prepadv_types
MODULE mo_real_timer
  USE iso_c_binding, ONLY: c_loc
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE timer_start(it)
    INTEGER, INTENT(IN) :: it
  END SUBROUTINE timer_start
  SUBROUTINE timer_stop(it)
    INTEGER, INTENT(IN) :: it
  END SUBROUTINE timer_stop
END MODULE mo_real_timer
MODULE mo_run_config
  IMPLICIT NONE
  LOGICAL :: lvert_nest
  LOGICAL :: ltimer
  INTEGER :: timers_level
  LOGICAL :: activate_sync_timers
  CONTAINS
END MODULE mo_run_config
MODULE mo_timer
  IMPLICIT NONE
  INTEGER :: timer_exch_data, timer_exch_data_wait
  INTEGER :: timer_barrier
  INTEGER :: timer_solve_nh, timer_solve_nh_cellcomp, timer_solve_nh_edgecomp, timer_solve_nh_vnupd, timer_solve_nh_vimpl, timer_solve_nh_exch
  INTEGER :: timer_intp
  CONTAINS
END MODULE mo_timer
MODULE mo_util_system
  USE iso_c_binding, ONLY: c_int, c_char, c_null_char
  IMPLICIT NONE
  INTERFACE
    SUBROUTINE util_exit(exit_no) BIND(C)
      IMPORT :: c_int
      IMPLICIT NONE
      INTEGER(KIND = c_int), VALUE :: exit_no
    END SUBROUTINE util_exit
    SUBROUTINE util_abort() BIND(C)
      IMPLICIT NONE
    END SUBROUTINE util_abort
  END INTERFACE
  CONTAINS
END MODULE mo_util_system
MODULE mo_mpi
  USE, INTRINSIC :: iso_c_binding, ONLY: c_char, c_signed_char, c_int, c_bool
  USE mo_util_system, ONLY: util_exit
  IMPLICIT NONE
  INTEGER :: p_error
  INTEGER :: p_status(6)
  INTEGER, ALLOCATABLE, SAVE :: p_request(:)
  INTEGER :: p_irequest
  INTEGER :: p_mrequest
  INTEGER :: process_mpi_all_comm
  INTEGER :: process_mpi_all_size
  INTEGER :: my_process_mpi_all_id
  INTEGER :: process_mpi_all_test_id
  LOGICAL :: process_is_mpi_parallel
  INTEGER :: my_mpi_function
  INTEGER :: num_test_procs
  INTEGER :: p_work_pe0
  INTEGER :: p_pe_work
  INTEGER :: p_comm_work_test
  INTEGER :: p_pe = 0
  INTEGER :: p_real_dp = 0
  INTEGER :: p_real_sp = 0
  INTEGER, PUBLIC :: comm_lev = 0, glob_comm(0 : 10), comm_proc0(0 : 10)
  CONTAINS
  FUNCTION get_comm_acc_queue()
    INTEGER :: get_comm_acc_queue
    get_comm_acc_queue = 1
  END FUNCTION
  SUBROUTINE acc_wait_comms(queue)
    INTEGER, INTENT(IN) :: queue
  END SUBROUTINE
  LOGICAL FUNCTION my_process_is_mpi_test()
    my_process_is_mpi_test = (my_mpi_function == 1)
  END FUNCTION my_process_is_mpi_test
  LOGICAL FUNCTION my_process_is_mpi_parallel()
    my_process_is_mpi_parallel = process_is_mpi_parallel
  END FUNCTION my_process_is_mpi_parallel
  LOGICAL FUNCTION my_process_is_mpi_all_seq()
    my_process_is_mpi_all_seq = (process_mpi_all_size <= 1)
  END FUNCTION my_process_is_mpi_all_seq
  LOGICAL FUNCTION my_process_is_mpi_seq()
    my_process_is_mpi_seq = .FALSE.
  END FUNCTION my_process_is_mpi_seq
  SUBROUTINE abort_mpi
    CALL mpi_abort(0, 1, p_error)
    IF (p_error /= 0) THEN
      WRITE(0, '(a)') ' MPI_ABORT failed.'
      WRITE(0, '(a,i4)') ' Error =  ', p_error
    END IF
    CALL util_exit(1)
  END SUBROUTINE abort_mpi
  SUBROUTINE p_send_dp(t_buffer, p_destination, p_tag, p_count, comm, use_g2g)
    REAL(KIND = 8), INTENT(IN) :: t_buffer
    INTEGER, INTENT(IN) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(IN) :: p_count, comm
    LOGICAL, OPTIONAL, INTENT(IN) :: use_g2g
    LOGICAL :: loc_use_g2g
    INTEGER :: p_comm, icount
    p_comm = comm
    icount = p_count
    loc_use_g2g = .FALSE.
    CALL mpi_send(t_buffer, icount, p_real_dp, p_destination, 1, p_comm, p_error)
  END SUBROUTINE p_send_dp
  SUBROUTINE p_send_sp(t_buffer, p_destination, p_tag, p_count, comm, use_g2g)
    REAL(KIND = 4), INTENT(IN) :: t_buffer
    INTEGER, INTENT(IN) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(IN) :: p_count, comm
    LOGICAL, OPTIONAL, INTENT(IN) :: use_g2g
    LOGICAL :: loc_use_g2g
    INTEGER :: p_comm, icount
    p_comm = comm
    icount = p_count
    loc_use_g2g = .FALSE.
    CALL mpi_send(t_buffer, icount, p_real_sp, p_destination, 1, p_comm, p_error)
  END SUBROUTINE p_send_sp
  SUBROUTINE p_send_dp_3d(t_buffer, p_destination, p_tag, p_count, comm)
    REAL(KIND = 8), INTENT(IN) :: t_buffer(:, :, :)
    INTEGER, INTENT(IN) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(IN) :: p_count, comm
    INTEGER :: p_comm, icount
    p_comm = process_mpi_all_comm
    icount = SIZE(t_buffer)
    CALL mpi_send(t_buffer, icount, p_real_dp, p_destination, 1, p_comm, p_error)
  END SUBROUTINE p_send_dp_3d
  SUBROUTINE p_inc_request
    INTEGER, ALLOCATABLE :: tmp(:)
    p_irequest = p_irequest + 1
    IF (p_irequest > p_mrequest) THEN
      ALLOCATE(tmp(p_mrequest))
      tmp(:) = p_request(:)
      DEALLOCATE(p_request)
      ALLOCATE(p_request(p_mrequest + 4096))
      p_request(1 : p_mrequest) = tmp(:)
      p_mrequest = p_mrequest + 4096
      DEALLOCATE(tmp)
    END IF
  END SUBROUTINE p_inc_request
  SUBROUTINE p_isend_dp(t_buffer, p_destination, p_tag, p_count, comm, request, use_g2g)
    REAL(KIND = 8), INTENT(INOUT) :: t_buffer
    INTEGER, INTENT(IN) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(IN) :: p_count, comm
    INTEGER, INTENT(INOUT), OPTIONAL :: request
    LOGICAL, OPTIONAL, INTENT(IN) :: use_g2g
    LOGICAL :: loc_use_g2g
    INTEGER :: p_comm, icount, out_request
    p_comm = comm
    icount = p_count
    loc_use_g2g = .FALSE.
    CALL mpi_isend(t_buffer, icount, p_real_dp, p_destination, 1, p_comm, out_request, p_error)
    CALL p_inc_request
    p_request(p_irequest) = out_request
  END SUBROUTINE p_isend_dp
  SUBROUTINE p_isend_sp(t_buffer, p_destination, p_tag, p_count, comm, request, use_g2g)
    REAL(KIND = 4), INTENT(INOUT) :: t_buffer
    INTEGER, INTENT(IN) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(IN) :: p_count, comm
    INTEGER, INTENT(INOUT), OPTIONAL :: request
    LOGICAL, OPTIONAL, INTENT(IN) :: use_g2g
    LOGICAL :: loc_use_g2g
    INTEGER :: p_comm, out_request, icount
    p_comm = comm
    icount = p_count
    loc_use_g2g = .FALSE.
    CALL mpi_isend(t_buffer, icount, p_real_sp, p_destination, 1, p_comm, out_request, p_error)
    CALL p_inc_request
    p_request(p_irequest) = out_request
  END SUBROUTINE p_isend_sp
  SUBROUTINE p_recv_dp(t_buffer, p_source, p_tag, p_count, comm, use_g2g)
    REAL(KIND = 8), INTENT(OUT) :: t_buffer
    INTEGER, INTENT(IN) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(IN) :: p_count, comm
    LOGICAL, OPTIONAL, INTENT(IN) :: use_g2g
    LOGICAL :: loc_use_g2g
    INTEGER :: p_comm, icount
    p_comm = comm
    icount = p_count
    loc_use_g2g = .FALSE.
    CALL mpi_recv(t_buffer, icount, p_real_dp, p_source, 1, p_comm, p_status, p_error)
  END SUBROUTINE p_recv_dp
  SUBROUTINE p_recv_sp(t_buffer, p_source, p_tag, p_count, comm, use_g2g)
    REAL(KIND = 4), INTENT(OUT) :: t_buffer
    INTEGER, INTENT(IN) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(IN) :: p_count, comm
    LOGICAL, OPTIONAL, INTENT(IN) :: use_g2g
    LOGICAL :: loc_use_g2g
    INTEGER :: p_comm, icount
    p_comm = comm
    icount = p_count
    loc_use_g2g = .FALSE.
    CALL mpi_recv(t_buffer, icount, p_real_sp, p_source, 1, p_comm, p_status, p_error)
  END SUBROUTINE p_recv_sp
  SUBROUTINE p_recv_dp_3d(t_buffer, p_source, p_tag, p_count, comm)
    REAL(KIND = 8), INTENT(OUT) :: t_buffer(:, :, :)
    INTEGER, INTENT(IN) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(IN) :: p_count, comm
    INTEGER :: p_comm, icount
    p_comm = process_mpi_all_comm
    icount = SIZE(t_buffer)
    CALL mpi_recv(t_buffer, icount, p_real_dp, p_source, 1, p_comm, p_status, p_error)
  END SUBROUTINE p_recv_dp_3d
  SUBROUTINE p_irecv_dp(t_buffer, p_source, p_tag, p_count, comm, use_g2g)
    REAL(KIND = 8), INTENT(INOUT) :: t_buffer
    INTEGER, INTENT(IN) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(IN) :: p_count, comm
    LOGICAL, OPTIONAL, INTENT(IN) :: use_g2g
    LOGICAL :: loc_use_g2g
    INTEGER :: p_comm, icount
    p_comm = comm
    icount = p_count
    loc_use_g2g = .FALSE.
    CALL p_inc_request
    CALL mpi_irecv(t_buffer, icount, p_real_dp, p_source, 1, p_comm, p_request(p_irequest), p_error)
  END SUBROUTINE p_irecv_dp
  SUBROUTINE p_irecv_sp(t_buffer, p_source, p_tag, p_count, comm, use_g2g)
    REAL(KIND = 4), INTENT(INOUT) :: t_buffer
    INTEGER, INTENT(IN) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(IN) :: p_count, comm
    LOGICAL, OPTIONAL, INTENT(IN) :: use_g2g
    LOGICAL :: loc_use_g2g
    INTEGER :: p_comm, icount
    p_comm = comm
    icount = p_count
    loc_use_g2g = .FALSE.
    CALL p_inc_request
    CALL mpi_irecv(t_buffer, icount, p_real_sp, p_source, 1, p_comm, p_request(p_irequest), p_error)
  END SUBROUTINE p_irecv_sp
  SUBROUTINE p_bcast_dp_3d(t_buffer, p_source, comm)
    REAL(KIND = 8) :: t_buffer(:, :, :)
    INTEGER, INTENT(IN) :: p_source
    INTEGER, OPTIONAL, INTENT(IN) :: comm
    INTEGER :: p_comm
    p_comm = comm
    IF (process_mpi_all_size == 1) THEN
      RETURN
    ELSE
      CALL mpi_bcast(t_buffer, SIZE(t_buffer), p_real_dp, p_source, p_comm, p_error)
    END IF
  END SUBROUTINE p_bcast_dp_3d
  SUBROUTINE p_wait_noarg
    IF (p_irequest > 0) CALL mpi_waitall(p_irequest, p_request, 1, p_error)
    p_irequest = 0
  END SUBROUTINE p_wait_noarg
  SUBROUTINE p_barrier(comm)
    INTEGER, INTENT(IN), OPTIONAL :: comm
    INTEGER :: com
    com = 0
    com = comm
    CALL mpi_barrier(com, p_error)
    IF (p_error /= 0) THEN
      WRITE(0, '(a,i4,a)') ' MPI_BARRIER on ', my_process_mpi_all_id, ' failed.'
      WRITE(0, '(a,i4)') ' Error = ', p_error
      CALL abort_mpi
    END IF
  END SUBROUTINE p_barrier
  SUBROUTINE work_mpi_barrier
  END SUBROUTINE work_mpi_barrier
END MODULE mo_mpi
MODULE mo_communication_types
  IMPLICIT NONE
  TYPE, ABSTRACT :: t_comm_pattern
  END TYPE t_comm_pattern
  TYPE, EXTENDS(t_comm_pattern) :: t_comm_pattern_orig
    INTEGER :: n_recv
    INTEGER :: n_pnts
    INTEGER :: n_send
    INTEGER :: np_recv
    INTEGER :: np_send
    INTEGER :: comm
    INTEGER, ALLOCATABLE :: recv_limits(:)
    INTEGER, ALLOCATABLE :: recv_src(:)
    INTEGER, ALLOCATABLE :: recv_dst_blk(:)
    INTEGER, ALLOCATABLE :: recv_dst_idx(:)
    INTEGER, ALLOCATABLE :: send_limits(:)
    INTEGER, ALLOCATABLE :: send_src_blk(:)
    INTEGER, ALLOCATABLE :: send_src_idx(:)
    INTEGER, ALLOCATABLE :: pelist_send(:)
    INTEGER, ALLOCATABLE :: pelist_recv(:)
    INTEGER, ALLOCATABLE :: send_startidx(:)
    INTEGER, ALLOCATABLE :: send_count(:)
    INTEGER, ALLOCATABLE :: recv_startidx(:)
    INTEGER, ALLOCATABLE :: recv_count(:)
  END TYPE t_comm_pattern_orig
  TYPE :: t_p_comm_pattern
    TYPE(t_comm_pattern_orig), POINTER :: p
  END TYPE t_p_comm_pattern
  CHARACTER(LEN = *), PARAMETER :: modname = "mo_communication_orig"
  CONTAINS
  SUBROUTINE exchange_data_r3d(p_pat, lacc, recv, send, add)
    USE mo_mpi, ONLY: acc_wait_comms, get_comm_acc_queue, my_process_is_mpi_seq, p_barrier, p_irecv_dp_deconiface_10 => p_irecv_dp, p_isend_dp_deconiface_12 => p_isend_dp, p_isend_dp_deconiface_14 => p_isend_dp, p_recv_dp_deconiface_13 => p_recv_dp, p_send_dp_deconiface_11 => p_send_dp, p_wait_noarg_deconiface_15 => p_wait_noarg
    USE mo_parallel_config, ONLY: iorder_sendrecv, itype_exch_barrier, nproma
    USE mo_run_config, ONLY: activate_sync_timers
    USE mo_real_timer, ONLY: timer_start, timer_stop
    USE mo_timer, ONLY: timer_barrier, timer_exch_data, timer_exch_data_wait
    USE mo_exception, ONLY: finish
    CLASS(t_comm_pattern_orig), TARGET, INTENT(INOUT) :: p_pat
    LOGICAL, INTENT(IN) :: lacc
    REAL(KIND = 8), INTENT(INOUT), TARGET :: recv(:, :, :)
    REAL(KIND = 8), INTENT(IN), OPTIONAL, TARGET :: send(:, :, :)
    REAL(KIND = 8), INTENT(IN), OPTIONAL, TARGET :: add(:, :, :)
    CHARACTER(LEN = *), PARAMETER :: routine = modname // "::exchange_data_r3d"
    REAL(KIND = 8) :: send_buf(SIZE(recv, 2), p_pat % n_send), recv_buf(SIZE(recv, 2), p_pat % n_recv)
    INTEGER :: i, k, np, irs, iss, pid, icount, ndim2
    IF (my_process_is_mpi_seq()) THEN
      CALL exchange_data_r3d_seq(p_pat, lacc, recv, send, add)
      RETURN
    END IF
    IF (itype_exch_barrier == 1 .OR. itype_exch_barrier == 3) THEN
      IF (activate_sync_timers) CALL timer_start(timer_barrier)
      CALL p_barrier(p_pat % comm)
      IF (activate_sync_timers) CALL timer_stop(timer_barrier)
    END IF
    IF (activate_sync_timers) CALL timer_start(timer_exch_data)
    IF (SIZE(recv, 1) /= nproma) THEN
      CALL finish(routine, 'Illegal first dimension of data array')
    END IF
    ndim2 = SIZE(recv, 2)
    IF (iorder_sendrecv == 1 .OR. iorder_sendrecv == 3) THEN
      DO np = 1, p_pat % np_recv
        pid = p_pat % pelist_recv(np)
        irs = p_pat % recv_startidx(np)
        icount = p_pat % recv_count(np) * ndim2
        CALL p_irecv_dp_deconiface_10(recv_buf(1, irs), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
      END DO
    END IF
    IF (ndim2 == 1) THEN
      DO i = 1, p_pat % n_send
        send_buf(1, i) = send(p_pat % send_src_idx(i), 1, p_pat % send_src_blk(i))
      END DO
    ELSE
      DO i = 1, p_pat % n_send
        send_buf(1 : ndim2, i) = send(p_pat % send_src_idx(i), 1 : ndim2, p_pat % send_src_blk(i))
      END DO
    END IF
    IF (iorder_sendrecv == 1) THEN
      DO np = 1, p_pat % np_send
        pid = p_pat % pelist_send(np)
        iss = p_pat % send_startidx(np)
        icount = p_pat % send_count(np) * ndim2
        CALL p_send_dp_deconiface_11(send_buf(1, iss), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
      END DO
    ELSE IF (iorder_sendrecv == 2) THEN
      DO np = 1, p_pat % np_send
        pid = p_pat % pelist_send(np)
        iss = p_pat % send_startidx(np)
        icount = p_pat % send_count(np) * ndim2
        CALL p_isend_dp_deconiface_12(send_buf(1, iss), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
      END DO
      DO np = 1, p_pat % np_recv
        pid = p_pat % pelist_recv(np)
        irs = p_pat % recv_startidx(np)
        icount = p_pat % recv_count(np) * ndim2
        CALL p_recv_dp_deconiface_13(recv_buf(1, irs), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
      END DO
    ELSE IF (iorder_sendrecv == 3) THEN
      DO np = 1, p_pat % np_send
        pid = p_pat % pelist_send(np)
        iss = p_pat % send_startidx(np)
        icount = p_pat % send_count(np) * ndim2
        CALL p_isend_dp_deconiface_14(send_buf(1, iss), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
      END DO
    END IF
    IF (activate_sync_timers) CALL timer_start(timer_exch_data_wait)
    CALL p_wait_noarg_deconiface_15
    IF (activate_sync_timers) CALL timer_stop(timer_exch_data_wait)
    IF (itype_exch_barrier == 2 .OR. itype_exch_barrier == 3) THEN
      IF (activate_sync_timers) CALL timer_start(timer_barrier)
      CALL p_barrier(p_pat % comm)
      IF (activate_sync_timers) CALL timer_stop(timer_barrier)
    END IF
    IF (ndim2 == 1) THEN
      k = 1
      DO i = 1, p_pat % n_pnts
        recv(p_pat % recv_dst_idx(i), 1, p_pat % recv_dst_blk(i)) = recv_buf(1, p_pat % recv_src(i)) + add(p_pat % recv_dst_idx(i), 1, p_pat % recv_dst_blk(i))
      END DO
    ELSE
      DO i = 1, p_pat % n_pnts
        recv(p_pat % recv_dst_idx(i), :, p_pat % recv_dst_blk(i)) = recv_buf(:, p_pat % recv_src(i)) + add(p_pat % recv_dst_idx(i), 1 : ndim2, p_pat % recv_dst_blk(i))
      END DO
    END IF
    CALL acc_wait_comms(get_comm_acc_queue())
    IF (activate_sync_timers) CALL timer_stop(timer_exch_data)
  END SUBROUTINE exchange_data_r3d
  SUBROUTINE exchange_data_r3d_seq(p_pat, lacc, recv, send, add)
    USE mo_mpi, ONLY: my_process_is_mpi_seq
    USE mo_exception, ONLY: finish
    CLASS(t_comm_pattern_orig), INTENT(IN), TARGET :: p_pat
    LOGICAL, INTENT(IN) :: lacc
    REAL(KIND = 8), INTENT(INOUT), TARGET :: recv(:, :, :)
    REAL(KIND = 8), INTENT(IN), OPTIONAL, TARGET :: send(:, :, :)
    REAL(KIND = 8), INTENT(IN), OPTIONAL, TARGET :: add(:, :, :)
    CHARACTER(LEN = *), PARAMETER :: routine = modname // ":exchange_data_r3d_seq"
    INTEGER :: i, k, ndim2
    INTEGER, POINTER :: recv_src(:)
    INTEGER, POINTER :: recv_dst_blk(:)
    INTEGER, POINTER :: recv_dst_idx(:)
    INTEGER, POINTER :: send_src_blk(:)
    INTEGER, POINTER :: send_src_idx(:)
    recv_src => p_pat % recv_src(:)
    recv_dst_blk => p_pat % recv_dst_blk(:)
    recv_dst_idx => p_pat % recv_dst_idx(:)
    send_src_blk => p_pat % send_src_blk(:)
    send_src_idx => p_pat % send_src_idx(:)
    IF (.NOT. my_process_is_mpi_seq()) THEN
      CALL finish(routine, "Internal error: sequential routine called in parallel run!")
    END IF
    IF ((p_pat % np_recv /= 1) .OR. (p_pat % np_send /= 1)) THEN
      CALL finish(routine, "Internal error: inconsistent no. send/receive peers!")
    END IF
    IF ((p_pat % recv_limits(1) - p_pat % recv_limits(0)) /= (p_pat % send_limits(1) - p_pat % send_limits(0))) THEN
      CALL finish(routine, "Internal error: inconsistent sender/receiver size!")
    END IF
    IF ((p_pat % recv_limits(0) /= 0) .OR. (p_pat % send_limits(0) /= 0)) THEN
      CALL finish(routine, "Internal error: inconsistent sender/receiver start position!")
    END IF
    IF ((p_pat % recv_limits(1) /= p_pat % n_recv) .OR. (p_pat % n_recv /= p_pat % n_send)) THEN
      CALL finish(routine, "Internal error: inconsistent counts for sender/receiver!")
    END IF
    ndim2 = SIZE(recv, 2)
    IF (PRESENT(add)) THEN
      DO k = 1, ndim2
        DO i = 1, p_pat % n_pnts
          recv(recv_dst_idx(i), k, recv_dst_blk(i)) = add(recv_dst_idx(i), k, recv_dst_blk(i)) + send(send_src_idx(recv_src(i)), k, send_src_blk(recv_src(i)))
        END DO
      END DO
    ELSE
      DO k = 1, ndim2
        DO i = 1, p_pat % n_pnts
          recv(recv_dst_idx(i), k, recv_dst_blk(i)) = send(send_src_idx(recv_src(i)), k, send_src_blk(recv_src(i)))
        END DO
      END DO
    END IF
  END SUBROUTINE exchange_data_r3d_seq
  SUBROUTINE exchange_data_s3d_seq(p_pat, lacc, recv, send, add)
    USE mo_mpi, ONLY: my_process_is_mpi_seq
    USE mo_exception, ONLY: finish
    CLASS(t_comm_pattern_orig), INTENT(IN), TARGET :: p_pat
    LOGICAL, INTENT(IN) :: lacc
    REAL(KIND = 4), INTENT(INOUT), TARGET :: recv(:, :, :)
    REAL(KIND = 4), INTENT(IN), OPTIONAL, TARGET :: send(:, :, :)
    REAL(KIND = 4), INTENT(IN), OPTIONAL, TARGET :: add(:, :, :)
    CHARACTER(LEN = *), PARAMETER :: routine = modname // ":exchange_data_s3d_seq"
    INTEGER :: i, k, ndim2
    INTEGER, POINTER :: recv_src(:)
    INTEGER, POINTER :: recv_dst_blk(:)
    INTEGER, POINTER :: recv_dst_idx(:)
    INTEGER, POINTER :: send_src_blk(:)
    INTEGER, POINTER :: send_src_idx(:)
    recv_src => p_pat % recv_src(:)
    recv_dst_blk => p_pat % recv_dst_blk(:)
    recv_dst_idx => p_pat % recv_dst_idx(:)
    send_src_blk => p_pat % send_src_blk(:)
    send_src_idx => p_pat % send_src_idx(:)
    IF (.NOT. my_process_is_mpi_seq()) THEN
      CALL finish(routine, "Internal error: sequential routine called in parallel run!")
    END IF
    IF ((p_pat % np_recv /= 1) .OR. (p_pat % np_send /= 1)) THEN
      CALL finish(routine, "Internal error: inconsistent no. send/receive peers!")
    END IF
    IF ((p_pat % recv_limits(1) - p_pat % recv_limits(0)) /= (p_pat % send_limits(1) - p_pat % send_limits(0))) THEN
      CALL finish(routine, "Internal error: inconsistent sender/receiver size!")
    END IF
    IF ((p_pat % recv_limits(0) /= 0) .OR. (p_pat % send_limits(0) /= 0)) THEN
      CALL finish(routine, "Internal error: inconsistent sender/receiver start position!")
    END IF
    IF ((p_pat % recv_limits(1) /= p_pat % n_recv) .OR. (p_pat % n_recv /= p_pat % n_send)) THEN
      CALL finish(routine, "Internal error: inconsistent counts for sender/receiver!")
    END IF
    ndim2 = SIZE(recv, 2)
    DO k = 1, ndim2
      DO i = 1, p_pat % n_pnts
        recv(recv_dst_idx(i), k, recv_dst_blk(i)) = send(send_src_idx(recv_src(i)), k, send_src_blk(recv_src(i)))
      END DO
    END DO
  END SUBROUTINE exchange_data_s3d_seq
  SUBROUTINE exchange_data_mult_mixprec(p_pat, lacc, nfields_dp, ndim2tot_dp, nfields_sp, ndim2tot_sp, recv_dp, send_dp, recv_sp, send_sp, nshift)
    USE mo_fortran_tools, ONLY: t_ptr_3d_dp, t_ptr_3d_sp
    USE mo_parallel_config, ONLY: iorder_sendrecv, itype_exch_barrier
    USE mo_run_config, ONLY: activate_sync_timers
    USE mo_real_timer, ONLY: timer_start, timer_stop
    USE mo_timer, ONLY: timer_barrier, timer_exch_data, timer_exch_data_wait
    USE mo_mpi, ONLY: acc_wait_comms, get_comm_acc_queue, my_process_is_mpi_seq, p_barrier, p_irecv_dp_deconiface_34 => p_irecv_dp, p_irecv_sp_deconiface_35 => p_irecv_sp, p_isend_dp_deconiface_38 => p_isend_dp, p_isend_dp_deconiface_42 => p_isend_dp, p_isend_sp_deconiface_39 => p_isend_sp, p_isend_sp_deconiface_43 => p_isend_sp, p_recv_dp_deconiface_40 => p_recv_dp, p_recv_sp_deconiface_41 => p_recv_sp, p_send_dp_deconiface_36 => p_send_dp, p_send_sp_deconiface_37 => p_send_sp, p_wait_noarg_deconiface_44 => p_wait_noarg
    CLASS(t_comm_pattern_orig), TARGET, INTENT(INOUT) :: p_pat
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN) :: nfields_dp, ndim2tot_dp, nfields_sp, ndim2tot_sp
    TYPE(t_ptr_3d_dp), INTENT(IN), OPTIONAL :: recv_dp(:)
    TYPE(t_ptr_3d_dp), INTENT(IN), OPTIONAL :: send_dp(:)
    TYPE(t_ptr_3d_sp), INTENT(IN), OPTIONAL :: recv_sp(:)
    TYPE(t_ptr_3d_sp), INTENT(IN), OPTIONAL :: send_sp(:)
    INTEGER, OPTIONAL, INTENT(IN) :: nshift
    INTEGER :: ndim2_dp(nfields_dp), noffset_dp(nfields_dp), ndim2_sp(nfields_sp), noffset_sp(nfields_sp)
    REAL(KIND = 8) :: send_buf_dp(ndim2tot_dp, p_pat % n_send), recv_buf_dp(ndim2tot_dp, p_pat % n_recv)
    REAL(KIND = 4) :: send_buf_sp(ndim2tot_sp, p_pat % n_send), recv_buf_sp(ndim2tot_sp, p_pat % n_recv)
    INTEGER :: i, k, kshift_dp(nfields_dp), kshift_sp(nfields_sp), jb, ik, jl, n, np, irs, iss, pid, icount, accum
    LOGICAL :: lsend
    INTEGER, POINTER :: recv_src(:)
    INTEGER, POINTER :: recv_dst_blk(:)
    INTEGER, POINTER :: recv_dst_idx(:)
    INTEGER, POINTER :: send_src_blk(:)
    INTEGER, POINTER :: send_src_idx(:)
    INTEGER :: n_send, n_pnts
    recv_src => p_pat % recv_src(:)
    recv_dst_blk => p_pat % recv_dst_blk(:)
    recv_dst_idx => p_pat % recv_dst_idx(:)
    send_src_blk => p_pat % send_src_blk(:)
    send_src_idx => p_pat % send_src_idx(:)
    n_send = p_pat % n_send
    n_pnts = p_pat % n_pnts
    IF (itype_exch_barrier == 1 .OR. itype_exch_barrier == 3) THEN
      IF (activate_sync_timers) CALL timer_start(timer_barrier)
      CALL p_barrier(p_pat % comm)
      IF (activate_sync_timers) CALL timer_stop(timer_barrier)
    END IF
    IF (activate_sync_timers) CALL timer_start(timer_exch_data)
    lsend = .FALSE.
    kshift_dp = nshift
    kshift_sp = nshift
    IF (my_process_is_mpi_seq()) THEN
      DO n = 1, nfields_dp
        CALL exchange_data_r3d_seq(p_pat, .TRUE., recv_dp(n) % p)
      END DO
      DO n = 1, 0
        CALL exchange_data_s3d_seq(p_pat, .TRUE., recv_sp(n) % p)
      END DO
      IF (activate_sync_timers) CALL timer_stop(timer_exch_data)
      RETURN
    END IF
    IF ((iorder_sendrecv == 1 .OR. iorder_sendrecv == 3) .AND. .NOT. my_process_is_mpi_seq()) THEN
      DO np = 1, p_pat % np_recv
        pid = p_pat % pelist_recv(np)
        irs = p_pat % recv_startidx(np)
        icount = p_pat % recv_count(np) * ndim2tot_dp
        IF (icount > 0) CALL p_irecv_dp_deconiface_34(recv_buf_dp(1, irs), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
        icount = p_pat % recv_count(np) * 0
        IF (icount > 0) CALL p_irecv_sp_deconiface_35(recv_buf_sp(1, irs), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
      END DO
    END IF
    DO n = 1, nfields_dp
      IF (SIZE(recv_dp(n) % p, 2) == 1) kshift_dp(n) = 0
    END DO
    DO n = 1, 0
      IF (SIZE(recv_sp(n) % p, 2) == 1) kshift_sp(n) = 0
    END DO
    accum = 0
    DO n = 1, nfields_dp
      noffset_dp(n) = accum
      ndim2_dp(n) = SIZE(recv_dp(n) % p, 2) - kshift_dp(n)
      accum = accum + ndim2_dp(n)
    END DO
    accum = 0
    DO n = 1, 0
      noffset_sp(n) = accum
      ndim2_sp(n) = SIZE(recv_sp(n) % p, 2) - kshift_sp(n)
      accum = accum + ndim2_sp(n)
    END DO
    DO i = 1, n_send
      jb = send_src_blk(i)
      jl = send_src_idx(i)
      DO n = 1, nfields_dp
        DO k = 1, ndim2_dp(n)
          send_buf_dp(k + noffset_dp(n), i) = recv_dp(n) % p(jl, k + kshift_dp(n), jb)
        END DO
      END DO
      DO n = 1, 0
        DO k = 1, ndim2_sp(n)
          send_buf_sp(k + noffset_sp(n), i) = recv_sp(n) % p(jl, k + kshift_sp(n), jb)
        END DO
      END DO
    END DO
    IF (iorder_sendrecv == 1) THEN
      DO np = 1, p_pat % np_send
        pid = p_pat % pelist_send(np)
        iss = p_pat % send_startidx(np)
        icount = p_pat % send_count(np) * ndim2tot_dp
        IF (icount > 0) CALL p_send_dp_deconiface_36(send_buf_dp(1, iss), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
        icount = p_pat % send_count(np) * 0
        IF (icount > 0) CALL p_send_sp_deconiface_37(send_buf_sp(1, iss), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
      END DO
    ELSE IF (iorder_sendrecv == 2) THEN
      DO np = 1, p_pat % np_send
        pid = p_pat % pelist_send(np)
        iss = p_pat % send_startidx(np)
        icount = p_pat % send_count(np) * ndim2tot_dp
        IF (icount > 0) CALL p_isend_dp_deconiface_38(send_buf_dp(1, iss), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
        icount = p_pat % send_count(np) * 0
        IF (icount > 0) CALL p_isend_sp_deconiface_39(send_buf_sp(1, iss), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
      END DO
      DO np = 1, p_pat % np_recv
        pid = p_pat % pelist_recv(np)
        irs = p_pat % recv_startidx(np)
        icount = p_pat % recv_count(np) * ndim2tot_dp
        IF (icount > 0) CALL p_recv_dp_deconiface_40(recv_buf_dp(1, irs), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
        icount = p_pat % recv_count(np) * 0
        IF (icount > 0) CALL p_recv_sp_deconiface_41(recv_buf_sp(1, irs), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
      END DO
    ELSE IF (iorder_sendrecv == 3) THEN
      DO np = 1, p_pat % np_send
        pid = p_pat % pelist_send(np)
        iss = p_pat % send_startidx(np)
        icount = p_pat % send_count(np) * ndim2tot_dp
        IF (icount > 0) CALL p_isend_dp_deconiface_42(send_buf_dp(1, iss), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
        icount = p_pat % send_count(np) * 0
        IF (icount > 0) CALL p_isend_sp_deconiface_43(send_buf_sp(1, iss), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
      END DO
    END IF
    IF (activate_sync_timers) CALL timer_start(timer_exch_data_wait)
    CALL p_wait_noarg_deconiface_44
    IF (activate_sync_timers) CALL timer_stop(timer_exch_data_wait)
    IF (itype_exch_barrier == 2 .OR. itype_exch_barrier == 3) THEN
      IF (activate_sync_timers) CALL timer_start(timer_barrier)
      CALL p_barrier(p_pat % comm)
      IF (activate_sync_timers) CALL timer_stop(timer_barrier)
    END IF
    DO i = 1, n_pnts
      jb = recv_dst_blk(i)
      jl = recv_dst_idx(i)
      ik = recv_src(i)
      DO n = 1, nfields_dp
        DO k = 1, ndim2_dp(n)
          recv_dp(n) % p(jl, k + kshift_dp(n), jb) = recv_buf_dp(k + noffset_dp(n), ik)
        END DO
      END DO
      DO n = 1, 0
        DO k = 1, ndim2_sp(n)
          recv_sp(n) % p(jl, k + kshift_sp(n), jb) = recv_buf_sp(k + noffset_sp(n), ik)
        END DO
      END DO
    END DO
    CALL acc_wait_comms(get_comm_acc_queue())
    IF (activate_sync_timers) CALL timer_stop(timer_exch_data)
  END SUBROUTINE exchange_data_mult_mixprec
END MODULE mo_communication_types
MODULE mo_communication
  IMPLICIT NONE
  CHARACTER(LEN = *), PARAMETER :: modname = "mo_communication"
  CONTAINS
  SUBROUTINE exchange_data_r3d(p_pat, lacc, recv, send, add)
    USE mo_communication_types, ONLY: exchange_data_r3d_deconproc_0 => exchange_data_r3d, t_comm_pattern_orig
    TYPE(t_comm_pattern_orig), POINTER :: p_pat
    LOGICAL, INTENT(IN) :: lacc
    REAL(KIND = 8), INTENT(INOUT), TARGET :: recv(:, :, :)
    REAL(KIND = 8), INTENT(IN), OPTIONAL, TARGET :: send(:, :, :)
    REAL(KIND = 8), INTENT(IN), OPTIONAL, TARGET :: add(:, :, :)
    CALL exchange_data_r3d_deconproc_0(p_pat, lacc, recv, send, add)
  END SUBROUTINE exchange_data_r3d
  SUBROUTINE exchange_data_mult_mixprec(p_pat, lacc, nfields_dp, ndim2tot_dp, nfields_sp, ndim2tot_sp, recv1_dp, send1_dp, recv2_dp, send2_dp, recv3_dp, send3_dp, recv4_dp, send4_dp, recv5_dp, send5_dp, recv1_sp, send1_sp, recv2_sp, send2_sp, recv3_sp, send3_sp, recv4_sp, send4_sp, recv5_sp, send5_sp, recv4d_dp, send4d_dp, recv4d_sp, send4d_sp, recv3d_arr_dp, recv3d_arr_sp, nshift)
    USE mo_communication_types, ONLY: exchange_data_mult_mixprec_deconproc_15 => exchange_data_mult_mixprec, t_comm_pattern_orig
    USE mo_fortran_tools, ONLY: t_ptr_3d_dp, t_ptr_3d_sp
    USE mo_exception, ONLY: finish
    TYPE(t_comm_pattern_orig), INTENT(INOUT) :: p_pat
    LOGICAL, INTENT(IN) :: lacc
    REAL(KIND = 8), INTENT(INOUT), TARGET, OPTIONAL :: recv1_dp(:, :, :), recv2_dp(:, :, :), recv3_dp(:, :, :), recv4_dp(:, :, :), recv5_dp(:, :, :), recv4d_dp(:, :, :, :)
    REAL(KIND = 8), INTENT(IN), TARGET, OPTIONAL :: send1_dp(:, :, :), send2_dp(:, :, :), send3_dp(:, :, :), send4_dp(:, :, :), send5_dp(:, :, :), send4d_dp(:, :, :, :)
    REAL(KIND = 4), INTENT(INOUT), TARGET, OPTIONAL :: recv1_sp(:, :, :), recv2_sp(:, :, :), recv3_sp(:, :, :), recv4_sp(:, :, :), recv5_sp(:, :, :), recv4d_sp(:, :, :, :)
    REAL(KIND = 4), INTENT(IN), TARGET, OPTIONAL :: send1_sp(:, :, :), send2_sp(:, :, :), send3_sp(:, :, :), send4_sp(:, :, :), send5_sp(:, :, :), send4d_sp(:, :, :, :)
    TYPE(t_ptr_3d_dp), INTENT(INOUT), OPTIONAL :: recv3d_arr_dp(:)
    TYPE(t_ptr_3d_sp), INTENT(INOUT), OPTIONAL :: recv3d_arr_sp(:)
    INTEGER, INTENT(IN) :: nfields_dp, ndim2tot_dp, nfields_sp, ndim2tot_sp
    INTEGER, OPTIONAL, INTENT(IN) :: nshift
    TYPE(t_ptr_3d_dp) :: recv_dp(nfields_dp)
    TYPE(t_ptr_3d_sp) :: recv_sp(nfields_sp)
    INTEGER :: i, i_dp, i_sp
    LOGICAL :: lsend
    CHARACTER(LEN = *), PARAMETER :: routine = modname // "::exchange_data_mult_mixprec"
    lsend = .FALSE.
    i_dp = 0
    i_sp = 0
    DO i = 1, SIZE(recv4d_dp, 4)
      i_dp = i_dp + 1
      recv_dp(i_dp) % p => recv4d_dp(:, :, :, i)
    END DO
    i_dp = i_dp + 1
    recv_dp(i_dp) % p => recv1_dp
    i_dp = i_dp + 1
    recv_dp(i_dp) % p => recv2_dp
    i_dp = i_dp + 1
    recv_dp(i_dp) % p => recv3_dp
    i_dp = i_dp + 1
    recv_dp(i_dp) % p => recv4_dp
    i_dp = i_dp + 1
    recv_dp(i_dp) % p => recv5_dp
    DO i = 1, SIZE(recv3d_arr_dp)
      i_dp = i_dp + 1
      recv_dp(i_dp) % p => recv3d_arr_dp(i) % p
    END DO
    DO i = 1, SIZE(recv4d_sp, 4)
      i_sp = i_sp + 1
      recv_sp(i_sp) % p => recv4d_sp(:, :, :, i)
    END DO
    i_sp = i_sp + 1
    recv_sp(i_sp) % p => recv1_sp
    i_sp = i_sp + 1
    recv_sp(i_sp) % p => recv2_sp
    i_sp = i_sp + 1
    recv_sp(i_sp) % p => recv3_sp
    i_sp = i_sp + 1
    recv_sp(i_sp) % p => recv4_sp
    i_sp = i_sp + 1
    recv_sp(i_sp) % p => recv5_sp
    DO i = 1, SIZE(recv3d_arr_sp)
      i_sp = i_sp + 1
      recv_sp(i_sp) % p => recv3d_arr_sp(i) % p
    END DO
    IF (i_dp /= nfields_dp) CALL finish(routine, "internal error nfields_dp")
    IF (i_sp /= 0) CALL finish(routine, "internal error nfields_sp")
    CALL exchange_data_mult_mixprec_deconproc_15(p_pat, .TRUE., nfields_dp, ndim2tot_dp, 0, 0, recv_dp = recv_dp, recv_sp = recv_sp, nshift = nshift)
  END SUBROUTINE exchange_data_mult_mixprec
END MODULE mo_communication
MODULE mo_model_domain
  USE mo_decomposition_tools, ONLY: t_grid_domain_decomp_info
  USE mo_math_types, ONLY: t_tangent_vectors
  USE mo_lib_grid_geometry_info, ONLY: t_grid_geometry_info
  USE mo_communication_types, ONLY: t_comm_pattern_orig, t_p_comm_pattern
  IMPLICIT NONE
  TYPE :: t_grid_cells
    INTEGER, ALLOCATABLE :: neighbor_idx(:, :, :)
    INTEGER, ALLOCATABLE :: neighbor_blk(:, :, :)
    INTEGER, ALLOCATABLE :: edge_idx(:, :, :)
    INTEGER, ALLOCATABLE :: edge_blk(:, :, :)
    INTEGER, ALLOCATABLE :: start_index(:)
    INTEGER, ALLOCATABLE :: end_index(:)
    INTEGER, ALLOCATABLE :: start_block(:)
    INTEGER, ALLOCATABLE :: end_block(:)
    TYPE(t_grid_domain_decomp_info) :: decomp_info
  END TYPE t_grid_cells
  TYPE :: t_grid_edges
    INTEGER, ALLOCATABLE :: cell_idx(:, :, :)
    INTEGER, ALLOCATABLE :: cell_blk(:, :, :)
    INTEGER, ALLOCATABLE :: vertex_idx(:, :, :)
    INTEGER, ALLOCATABLE :: vertex_blk(:, :, :)
    REAL(KIND = 8), ALLOCATABLE :: tangent_orientation(:, :)
    INTEGER, ALLOCATABLE :: quad_idx(:, :, :)
    INTEGER, ALLOCATABLE :: quad_blk(:, :, :)
    TYPE(t_tangent_vectors), ALLOCATABLE :: primal_normal_cell(:, :, :)
    TYPE(t_tangent_vectors), ALLOCATABLE :: dual_normal_cell(:, :, :)
    REAL(KIND = 8), ALLOCATABLE :: inv_primal_edge_length(:, :)
    REAL(KIND = 8), ALLOCATABLE :: inv_dual_edge_length(:, :)
    INTEGER, ALLOCATABLE :: refin_ctrl(:, :)
    INTEGER, ALLOCATABLE :: start_index(:)
    INTEGER, ALLOCATABLE :: end_index(:)
    INTEGER, ALLOCATABLE :: start_block(:)
    INTEGER, ALLOCATABLE :: end_block(:)
    TYPE(t_grid_domain_decomp_info) :: decomp_info
  END TYPE t_grid_edges
  TYPE :: t_grid_vertices
    INTEGER, ALLOCATABLE :: cell_idx(:, :, :)
    INTEGER, ALLOCATABLE :: cell_blk(:, :, :)
    INTEGER, ALLOCATABLE :: start_index(:)
    INTEGER, ALLOCATABLE :: end_index(:)
    INTEGER, ALLOCATABLE :: start_block(:)
    INTEGER, ALLOCATABLE :: end_block(:)
    TYPE(t_grid_domain_decomp_info) :: decomp_info
  END TYPE t_grid_vertices
  TYPE :: t_patch
    INTEGER :: id
    TYPE(t_grid_geometry_info) :: geometry_info
    INTEGER :: n_childdom
    INTEGER :: n_patch_cells
    INTEGER :: n_patch_edges
    INTEGER :: n_patch_verts
    INTEGER :: n_patch_cells_g
    INTEGER :: n_patch_edges_g
    INTEGER :: n_patch_verts_g
    INTEGER :: nblks_c
    INTEGER :: nblks_e
    INTEGER :: nblks_v
    INTEGER :: nlev
    INTEGER :: nlevp1
    INTEGER :: nshift_total
    INTEGER :: nshift_child
    TYPE(t_grid_cells) :: cells
    TYPE(t_grid_edges) :: edges
    TYPE(t_grid_vertices) :: verts
    TYPE(t_comm_pattern_orig), POINTER :: comm_pat_c
    TYPE(t_comm_pattern_orig), POINTER :: comm_pat_c1
    TYPE(t_comm_pattern_orig), POINTER :: comm_pat_e
    TYPE(t_comm_pattern_orig), POINTER :: comm_pat_v
    TYPE(t_p_comm_pattern) :: comm_pat_work2test(3)
  END TYPE t_patch
  CONTAINS
END MODULE mo_model_domain
MODULE mo_icon_interpolation_scalar
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE cells2verts_scalar_dp(p_cell_in, ptr_patch, c_int, p_vert_out, lacc, opt_slev, opt_elev, opt_rlstart, opt_rlend, opt_acc_async)
    USE mo_model_domain, ONLY: t_patch
    USE mo_run_config, ONLY: timers_level
    USE mo_real_timer, ONLY: timer_start, timer_stop
    USE mo_timer, ONLY: timer_intp
    USE mo_lib_interpolation_scalar, ONLY: cells2verts_scalar_dp_lib_deconiface_71 => cells2verts_scalar_dp_lib
    USE mo_parallel_config, ONLY: nproma
    TYPE(t_patch), TARGET, INTENT(IN) :: ptr_patch
    REAL(KIND = 8), INTENT(IN) :: p_cell_in(:, :, :)
    REAL(KIND = 8), INTENT(IN) :: c_int(:, :, :)
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_slev
    INTEGER, INTENT(IN), OPTIONAL :: opt_elev
    INTEGER, INTENT(IN), OPTIONAL :: opt_rlstart, opt_rlend
    LOGICAL, INTENT(IN), OPTIONAL :: opt_acc_async
    REAL(KIND = 8), INTENT(INOUT) :: p_vert_out(:, :, :)
    INTEGER :: slev, elev
    INTEGER :: rl_start, rl_end
    INTEGER :: i_startblk, i_endblk, i_startidx_in, i_endidx_in
    slev = 1
    elev = UBOUND(p_cell_in, 2)
    rl_start = 2
    rl_end = opt_rlend
    i_startblk = ptr_patch % verts % start_block(2)
    i_endblk = ptr_patch % verts % end_block(rl_end)
    i_startidx_in = ptr_patch % verts % start_index(2)
    i_endidx_in = ptr_patch % verts % end_index(rl_end)
    IF (timers_level > 10) CALL timer_start(timer_intp)
    CALL cells2verts_scalar_dp_lib_deconiface_71(p_cell_in, ptr_patch % verts % cell_idx, ptr_patch % verts % cell_blk, c_int, p_vert_out, i_startblk, i_endblk, i_startidx_in, i_endidx_in, slev, elev, nproma, lacc = .TRUE., acc_async = opt_acc_async)
    IF (timers_level > 10) CALL timer_stop(timer_intp)
  END SUBROUTINE cells2verts_scalar_dp
END MODULE mo_icon_interpolation_scalar
MODULE mo_loopindices
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE get_indices_c(p_patch, i_blk, i_startblk, i_endblk, i_startidx, i_endidx, irl_start, opt_rl_end)
    USE mo_model_domain, ONLY: t_patch
    USE mo_lib_loopindices, ONLY: get_indices_c_lib
    USE mo_parallel_config, ONLY: nproma
    TYPE(t_patch), INTENT(IN) :: p_patch
    INTEGER, INTENT(IN) :: i_blk
    INTEGER, INTENT(IN) :: i_startblk
    INTEGER, INTENT(IN) :: i_endblk
    INTEGER, INTENT(IN) :: irl_start
    INTEGER, OPTIONAL, INTENT(IN) :: opt_rl_end
    INTEGER, INTENT(OUT) :: i_startidx, i_endidx
    INTEGER :: irl_end, i_startidx_in, i_endidx_in
    i_startidx_in = p_patch % cells % start_index(irl_start)
    irl_end = opt_rl_end
    i_endidx_in = p_patch % cells % end_index(irl_end)
    CALL get_indices_c_lib(i_startidx_in, i_endidx_in, nproma, i_blk, i_startblk, i_endblk, i_startidx, i_endidx)
  END SUBROUTINE get_indices_c
  SUBROUTINE get_indices_e(p_patch, i_blk, i_startblk, i_endblk, i_startidx, i_endidx, irl_start, opt_rl_end)
    USE mo_model_domain, ONLY: t_patch
    USE mo_lib_loopindices, ONLY: get_indices_e_lib
    USE mo_parallel_config, ONLY: nproma
    TYPE(t_patch), INTENT(IN) :: p_patch
    INTEGER, INTENT(IN) :: i_blk
    INTEGER, INTENT(IN) :: i_startblk
    INTEGER, INTENT(IN) :: i_endblk
    INTEGER, INTENT(IN) :: irl_start
    INTEGER, OPTIONAL, INTENT(IN) :: opt_rl_end
    INTEGER, INTENT(OUT) :: i_startidx, i_endidx
    INTEGER :: irl_end, i_startidx_in, i_endidx_in
    i_startidx_in = p_patch % edges % start_index(irl_start)
    irl_end = opt_rl_end
    i_endidx_in = p_patch % edges % end_index(irl_end)
    CALL get_indices_e_lib(i_startidx_in, i_endidx_in, nproma, i_blk, i_startblk, i_endblk, i_startidx, i_endidx)
  END SUBROUTINE get_indices_e
END MODULE mo_loopindices
MODULE mo_math_gradients
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE grad_green_gauss_cell_dycore(p_ccpr, ptr_patch, ptr_int, p_grad, lacc, opt_slev, opt_elev, opt_rlstart, opt_rlend, opt_acc_async)
    USE mo_model_domain, ONLY: t_patch
    USE mo_intp_data_strc, ONLY: t_int_state
    USE mo_lib_gradients, ONLY: grad_green_gauss_cell_dycore_lib_deconiface_76 => grad_green_gauss_cell_dycore_lib
    USE mo_parallel_config, ONLY: nproma
    TYPE(t_patch), TARGET, INTENT(IN) :: ptr_patch
    TYPE(t_int_state), TARGET, INTENT(IN) :: ptr_int
    REAL(KIND = 8), INTENT(IN) :: p_ccpr(:, :, :, :)
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_slev
    INTEGER, INTENT(IN), OPTIONAL :: opt_elev
    INTEGER, INTENT(IN), OPTIONAL :: opt_rlstart, opt_rlend
    LOGICAL, INTENT(IN), OPTIONAL :: opt_acc_async
    REAL(KIND = 8), INTENT(INOUT) :: p_grad(:, :, :, :)
    INTEGER :: slev, elev
    INTEGER :: rl_start, rl_end
    INTEGER :: i_startblk, i_endblk, i_startidx_in, i_endidx_in
    slev = 1
    elev = UBOUND(p_ccpr, 3)
    rl_start = 3
    rl_end = opt_rlend
    i_startblk = ptr_patch % cells % start_block(3)
    i_endblk = ptr_patch % cells % end_block(rl_end)
    i_startidx_in = ptr_patch % cells % start_index(3)
    i_endidx_in = ptr_patch % cells % end_index(rl_end)
    CALL grad_green_gauss_cell_dycore_lib_deconiface_76(p_ccpr, ptr_patch % cells % neighbor_idx, ptr_patch % cells % neighbor_blk, ptr_int % geofac_grg, p_grad, i_startblk, i_endblk, i_startidx_in, i_endidx_in, slev, elev, nproma, lacc = .TRUE., acc_async = .TRUE.)
  END SUBROUTINE grad_green_gauss_cell_dycore
END MODULE mo_math_gradients
MODULE mo_sync
  IMPLICIT NONE
  INTEGER, SAVE :: log_unit = - 1
  LOGICAL, SAVE :: do_sync_checks = .TRUE.
  CONTAINS
  FUNCTION comm_pat_of_type(p_patch, typ) RESULT(p_pat)
    USE mo_model_domain, ONLY: t_patch
    USE mo_communication_types, ONLY: t_comm_pattern_orig
    USE mo_exception, ONLY: finish
    INTEGER, INTENT(IN) :: typ
    TYPE(t_patch), TARGET, INTENT(IN) :: p_patch
    TYPE(t_comm_pattern_orig), POINTER :: p_pat
    IF (typ == 1) THEN
      p_pat => p_patch % comm_pat_c
    ELSE IF (typ == 2) THEN
      p_pat => p_patch % comm_pat_e
    ELSE IF (typ == 3) THEN
      p_pat => p_patch % comm_pat_v
    ELSE IF (typ == 4) THEN
      p_pat => p_patch % comm_pat_c1
    ELSE
      CALL finish('comm_pat_of_type', 'Illegal type parameter')
    END IF
  END FUNCTION comm_pat_of_type
  SUBROUTINE sync_patch_array_3d_dp(typ, p_patch, arr, lacc, opt_varname)
    USE mo_model_domain, ONLY: t_patch
    USE mo_parallel_config, ONLY: p_test_run
    USE mo_mpi, ONLY: my_process_is_mpi_parallel
    USE mo_communication, ONLY: exchange_data_r3d_deconiface_77 => exchange_data_r3d
    INTEGER, INTENT(IN) :: typ
    TYPE(t_patch), TARGET, INTENT(IN) :: p_patch
    REAL(KIND = 8), INTENT(INOUT) :: arr(:, :, :)
    LOGICAL, INTENT(IN) :: lacc
    CHARACTER(LEN = *), TARGET, INTENT(IN), OPTIONAL :: opt_varname
    IF (p_test_run .AND. do_sync_checks) CALL check_patch_array_3d_dp(typ, p_patch, arr, lacc = .TRUE., opt_varname = opt_varname)
    IF (my_process_is_mpi_parallel()) THEN
      CALL exchange_data_r3d_deconiface_77(p_pat = comm_pat_of_type(p_patch, typ), lacc = .TRUE., recv = arr)
    END IF
  END SUBROUTINE sync_patch_array_3d_dp
  SUBROUTINE sync_patch_array_mult_f3din_dp(typ, p_patch, nfields, lacc, f3din1, f3din2, f3din3, f3din4, f3din5, opt_varname)
    USE mo_model_domain, ONLY: t_patch
    INTEGER, INTENT(IN) :: typ
    TYPE(t_patch), INTENT(IN), TARGET :: p_patch
    INTEGER, INTENT(IN) :: nfields
    LOGICAL, INTENT(IN) :: lacc
    REAL(KIND = 8), TARGET, INTENT(INOUT) :: f3din1(:, :, :)
    REAL(KIND = 8), TARGET, OPTIONAL, INTENT(INOUT) :: f3din2(:, :, :), f3din3(:, :, :), f3din4(:, :, :), f3din5(:, :, :)
    CHARACTER(LEN = *), TARGET, INTENT(IN), OPTIONAL :: opt_varname
    CALL sync_patch_array_mult_mixprec(typ = typ, p_patch = p_patch, nfields_sp = 0, nfields_dp = nfields, lacc = .TRUE., f3din1_dp = f3din1, f3din2_dp = f3din2, f3din3_dp = f3din3, f3din4_dp = f3din4, f3din5_dp = f3din5, opt_varname = opt_varname)
  END SUBROUTINE sync_patch_array_mult_f3din_dp
  SUBROUTINE sync_patch_array_mult_mixprec(typ, p_patch, nfields_dp, nfields_sp, lacc, f3din1_dp, f3din2_dp, f3din3_dp, f3din4_dp, f3din5_dp, f3din1_sp, f3din2_sp, f3din3_sp, f3din4_sp, f3din5_sp, f4din_dp, f4din_sp, f3din_arr_sp, f3din_arr_dp, opt_varname)
    USE mo_model_domain, ONLY: t_patch
    USE mo_fortran_tools, ONLY: t_ptr_3d_dp, t_ptr_3d_sp
    USE mo_communication_types, ONLY: t_comm_pattern_orig
    USE mo_parallel_config, ONLY: p_test_run
    USE mo_mpi, ONLY: my_process_is_mpi_parallel
    USE mo_communication, ONLY: exchange_data_mult_mixprec
    INTEGER, INTENT(IN) :: typ
    TYPE(t_patch), INTENT(IN), TARGET :: p_patch
    INTEGER, INTENT(IN) :: nfields_dp, nfields_sp
    LOGICAL, INTENT(IN) :: lacc
    REAL(KIND = 8), OPTIONAL, INTENT(INOUT) :: f3din1_dp(:, :, :), f3din2_dp(:, :, :), f3din3_dp(:, :, :), f3din4_dp(:, :, :), f3din5_dp(:, :, :), f4din_dp(:, :, :, :)
    REAL(KIND = 4), OPTIONAL, INTENT(INOUT) :: f3din1_sp(:, :, :), f3din2_sp(:, :, :), f3din3_sp(:, :, :), f3din4_sp(:, :, :), f3din5_sp(:, :, :), f4din_sp(:, :, :, :)
    TYPE(t_ptr_3d_dp), INTENT(INOUT), OPTIONAL :: f3din_arr_dp(:)
    TYPE(t_ptr_3d_sp), INTENT(INOUT), OPTIONAL :: f3din_arr_sp(:)
    TYPE(t_comm_pattern_orig), POINTER :: p_pat
    CHARACTER(LEN = *), TARGET, INTENT(IN), OPTIONAL :: opt_varname
    INTEGER :: ndim2tot_dp, ndim2tot_sp
    IF (typ == 1) THEN
      p_pat => p_patch % comm_pat_c
    ELSE IF (typ == 2) THEN
      p_pat => p_patch % comm_pat_e
    ELSE IF (typ == 3) THEN
      p_pat => p_patch % comm_pat_v
    ELSE IF (typ == 4) THEN
      p_pat => p_patch % comm_pat_c1
    END IF
    IF (p_test_run .AND. do_sync_checks) THEN
      CALL check_patch_array_3d_dp(typ, p_patch, f3din1_dp, lacc = .TRUE., opt_varname = opt_varname)
      CALL check_patch_array_3d_dp(typ, p_patch, f3din2_dp, lacc = .TRUE., opt_varname = opt_varname)
      CALL check_patch_array_3d_dp(typ, p_patch, f3din3_dp, lacc = .TRUE., opt_varname = opt_varname)
      CALL check_patch_array_3d_dp(typ, p_patch, f3din4_dp, lacc = .TRUE., opt_varname = opt_varname)
      CALL check_patch_array_3d_dp(typ, p_patch, f3din5_dp, lacc = .TRUE., opt_varname = opt_varname)
    END IF
    IF (my_process_is_mpi_parallel()) THEN
      ndim2tot_dp = 0
      ndim2tot_dp = 0 + SIZE(f3din1_dp, 2)
      ndim2tot_dp = ndim2tot_dp + SIZE(f3din2_dp, 2)
      ndim2tot_dp = ndim2tot_dp + SIZE(f3din3_dp, 2)
      ndim2tot_dp = ndim2tot_dp + SIZE(f3din4_dp, 2)
      ndim2tot_dp = ndim2tot_dp + SIZE(f3din5_dp, 2)
      ndim2tot_sp = 0
      CALL exchange_data_mult_mixprec(p_pat = p_pat, lacc = .TRUE., nfields_dp = nfields_dp, ndim2tot_dp = ndim2tot_dp, nfields_sp = 0, ndim2tot_sp = 0, recv1_dp = f3din1_dp, recv2_dp = f3din2_dp, recv3_dp = f3din3_dp, recv4_dp = f3din4_dp, recv5_dp = f3din5_dp, recv1_sp = f3din1_sp, recv2_sp = f3din2_sp, recv3_sp = f3din3_sp, recv4_sp = f3din4_sp, recv5_sp = f3din5_sp, recv4d_dp = f4din_dp, recv4d_sp = f4din_sp, recv3d_arr_dp = f3din_arr_dp, recv3d_arr_sp = f3din_arr_sp)
    END IF
  END SUBROUTINE sync_patch_array_mult_mixprec
  SUBROUTINE check_patch_array_3d_dp(typ, p_patch, arr, lacc, opt_varname)
    USE mo_model_domain, ONLY: t_patch
    USE mo_parallel_config, ONLY: blk_no, idx_no, l_log_checks, n_ghost_rows, nproma, p_test_run
    USE mo_communication_types, ONLY: t_comm_pattern_orig
    USE mo_io_units, ONLY: filename_max, find_next_free_unit
    USE mo_exception, ONLY: finish
    USE mo_mpi, ONLY: comm_lev, comm_proc0, glob_comm, my_process_is_mpi_test, num_test_procs, p_bcast_dp_3d_deconiface_92 => p_bcast_dp_3d, p_bcast_dp_3d_deconiface_94 => p_bcast_dp_3d, p_bcast_dp_3d_deconiface_96 => p_bcast_dp_3d, p_comm_work_test, p_pe, p_pe_work, p_recv_dp_3d_deconiface_95 => p_recv_dp_3d, p_send_dp_3d_deconiface_93 => p_send_dp_3d, p_work_pe0, process_mpi_all_test_id
    USE mo_communication, ONLY: exchange_data_r3d_deconiface_91 => exchange_data_r3d
    INTEGER, INTENT(IN) :: typ
    TYPE(t_patch), INTENT(IN), TARGET :: p_patch
    REAL(KIND = 8), INTENT(IN) :: arr(:, :, :)
    LOGICAL, INTENT(IN) :: lacc
    CHARACTER(LEN = *), INTENT(IN), OPTIONAL :: opt_varname
    REAL(KIND = 8), ALLOCATABLE :: arr_g(:, :, :)
    INTEGER :: j, jb, jl, jb_g, jl_g, n, ndim2, ndim3, nblks_g, flag, jk
    INTEGER :: ityp, ndim, ndim_g, jk_min_err
    INTEGER :: nerr(0 : n_ghost_rows), shape_recv(3)
    INTEGER, POINTER :: p_glb_index(:), p_decomp_domain(:, :)
    TYPE(t_comm_pattern_orig), POINTER :: p_pat_work2test
    LOGICAL :: l_my_process_is_mpi_test
    CHARACTER(LEN = 256) :: varname, cfmt
    INTEGER :: varname_tlen
    CHARACTER(LEN = filename_max) :: log_file
    REAL(KIND = 8) :: absmax
    LOGICAL :: sync_error
    ityp = (- 1)
    ndim = (- 1)
    ndim_g = (- 1)
    sync_error = .FALSE.
    NULLIFY(p_glb_index, p_decomp_domain)
    IF (.NOT. p_test_run) RETURN
    varname = opt_varname
    varname_tlen = LEN(opt_varname)
    IF (UBOUND(arr, 1) /= nproma) THEN
      CALL finish('sync_patch_array', 'first dimension /= nproma')
    END IF
    ndim2 = UBOUND(arr, 2)
    ndim3 = UBOUND(arr, 3)
    IF (typ == 1 .OR. typ == 4) THEN
      ndim = p_patch % n_patch_cells
      ndim_g = p_patch % n_patch_cells_g
      p_glb_index => p_patch % cells % decomp_info % glb_index
      p_decomp_domain => p_patch % cells % decomp_info % decomp_domain
      ityp = typ
      p_pat_work2test => p_patch % comm_pat_work2test(1) % p
    ELSE IF (typ == 2) THEN
      ndim = p_patch % n_patch_edges
      ndim_g = p_patch % n_patch_edges_g
      p_glb_index => p_patch % edges % decomp_info % glb_index
      p_decomp_domain => p_patch % edges % decomp_info % decomp_domain
      ityp = typ
      p_pat_work2test => p_patch % comm_pat_work2test(3) % p
    ELSE IF (typ == 3) THEN
      ndim = p_patch % n_patch_verts
      ndim_g = p_patch % n_patch_verts_g
      p_glb_index => p_patch % verts % decomp_info % glb_index
      p_decomp_domain => p_patch % verts % decomp_info % decomp_domain
      ityp = typ
      p_pat_work2test => p_patch % comm_pat_work2test(2) % p
    ELSE IF (typ == 0) THEN
      IF (ndim3 == p_patch % nblks_c) THEN
        ndim = p_patch % n_patch_cells
        ndim_g = p_patch % n_patch_cells_g
        p_glb_index => p_patch % cells % decomp_info % glb_index
        p_decomp_domain => p_patch % cells % decomp_info % decomp_domain
        ityp = 1
        p_pat_work2test => p_patch % comm_pat_work2test(1) % p
      ELSE IF (ndim3 == p_patch % nblks_e) THEN
        ndim = p_patch % n_patch_edges
        ndim_g = p_patch % n_patch_edges_g
        p_glb_index => p_patch % edges % decomp_info % glb_index
        p_decomp_domain => p_patch % edges % decomp_info % decomp_domain
        ityp = 2
        p_pat_work2test => p_patch % comm_pat_work2test(3) % p
      ELSE IF (ndim3 == p_patch % nblks_v) THEN
        ndim = p_patch % n_patch_verts
        ndim_g = p_patch % n_patch_verts_g
        p_glb_index => p_patch % verts % decomp_info % glb_index
        p_decomp_domain => p_patch % verts % decomp_info % decomp_domain
        ityp = 3
        p_pat_work2test => p_patch % comm_pat_work2test(2) % p
      ELSE
        CALL finish('check_patch_array', 'typ==0 but unknown blocksize of array')
      END IF
    ELSE
      CALL finish('sync_patch_array', 'Illegal type parameter')
    END IF
    nblks_g = (ndim_g - 1) / nproma + 1
    l_my_process_is_mpi_test = my_process_is_mpi_test()
    IF (num_test_procs > 1) THEN
      shape_recv = SHAPE(arr)
      ALLOCATE(arr_g(shape_recv(1), shape_recv(2), shape_recv(3)))
      CALL exchange_data_r3d_deconiface_91(p_pat = p_pat_work2test, lacc = .FALSE., recv = arr_g, send = arr)
      IF (l_my_process_is_mpi_test) THEN
        jk_min_err = HUGE(jk_min_err)
        DO jb = 1, ndim3
          DO jk = 1, ndim2
            DO jl = 1, nproma
              IF (p_decomp_domain(jl, jb) == 0) THEN
                sync_error = sync_error .OR. arr(jl, jk, jb) /= arr_g(jl, jk, jb)
                jk_min_err = MIN(jk_min_err, MERGE(jk, jk_min_err, arr(jl, jk, jb) /= arr_g(jl, jk, jb)))
              END IF
            END DO
          END DO
        END DO
      END IF
    ELSE IF (l_my_process_is_mpi_test) THEN
      ALLOCATE(arr_g(nproma, ndim2, nblks_g))
      DO j = 1, ndim
        jb = blk_no(j)
        jl = idx_no(j)
        jb_g = blk_no(p_glb_index(j))
        jl_g = idx_no(p_glb_index(j))
        arr_g(jl_g, 1 : ndim2, jb_g) = arr(jl, 1 : ndim2, jb)
      END DO
      IF (comm_lev == 0) THEN
        CALL p_bcast_dp_3d_deconiface_92(arr_g(:, :, 1 : nblks_g), process_mpi_all_test_id, comm = p_comm_work_test)
      ELSE
        CALL p_send_dp_3d_deconiface_93(arr_g(:, :, 1 : nblks_g), comm_proc0(comm_lev) + p_work_pe0, 1)
      END IF
      DEALLOCATE(arr_g)
    ELSE
      ALLOCATE(arr_g(nproma, ndim2, nblks_g))
      IF (comm_lev == 0) THEN
        CALL p_bcast_dp_3d_deconiface_94(arr_g(:, :, 1 : nblks_g), process_mpi_all_test_id, comm = p_comm_work_test)
      ELSE
        IF (p_pe_work == comm_proc0(comm_lev)) CALL p_recv_dp_3d_deconiface_95(arr_g(:, :, 1 : nblks_g), process_mpi_all_test_id, 1)
        CALL p_bcast_dp_3d_deconiface_96(arr_g(:, :, 1 : nblks_g), 0, comm = glob_comm(comm_lev))
      END IF
      nerr(:) = 0
      absmax = 0.0D0
      DO j = 1, ndim
        jb = blk_no(j)
        jl = idx_no(j)
        jb_g = blk_no(p_glb_index(j))
        jl_g = idx_no(p_glb_index(j))
        flag = p_decomp_domain(jl, jb)
        flag = MAX(flag, 0)
        flag = MIN(flag, UBOUND(nerr, 1))
        DO n = 1, ndim2
          IF (arr(jl, n, jb) /= arr_g(jl_g, n, jb_g)) THEN
            nerr(flag) = nerr(flag) + 1
            IF (flag == 0) THEN
              sync_error = .TRUE.
              absmax = MAX(absmax, ABS(arr(jl, n, jb) - arr_g(jl_g, n, jb_g)))
              IF (l_log_checks) THEN
                WRITE(log_unit, '(2a,5i7,3e18.10)') varname, 'sync error location:', jb, jl, jb_g, jl_g, n, arr(jl, n, jb), arr_g(jl_g, n, jb_g), ABS(arr(jl, n, jb) - arr_g(jl_g, n, jb_g))
              END IF
            END IF
          END IF
        END DO
      END DO
      IF (l_log_checks) THEN
        IF (log_unit < 0) THEN
          WRITE(log_file, '(''log'',i4.4,''.txt'')') p_pe
          log_unit = find_next_free_unit(10, 99)
          OPEN(UNIT = log_unit, FILE = log_file)
        END IF
        n = n_ghost_rows
        WRITE(cfmt, '(a,i3,a)') '(', n + 1, 'i8,'' '',2a)'
        IF (ALL(arr == 0.0D0)) THEN
          WRITE(log_unit, cfmt) nerr(0 : n), varname(1 : varname_tlen), ': ALL 0 !!!'
        ELSE
          WRITE(log_unit, cfmt) nerr(0 : n), varname(1 : varname_tlen)
        END IF
        IF (absmax > 0.0D0) WRITE(log_unit, *) 'Max abs inner err:', absmax
      END IF
      DEALLOCATE(arr_g)
    END IF
    IF (sync_error) THEN
      IF (num_test_procs > 1) WRITE(0, '(2a,i0)') varname(1 : varname_tlen), ' sync error in level jk = ', jk_min_err
      IF (l_log_checks) THEN
        CLOSE(UNIT = log_unit)
      END IF
      CALL finish('sync_patch_array', 'Out of sync detected!')
    END IF
  END SUBROUTINE check_patch_array_3d_dp
END MODULE mo_sync
MODULE mo_velocity_advection
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE velocity_tendencies(p_prog, p_patch, p_int, p_metrics, p_diag, z_w_concorr_me, z_kin_hor_e, z_vt_ie, ntnd, istep, lvn_only, dtime, dt_linintp_ubc, ldeepatmo)
    USE mo_model_domain, ONLY: t_patch
    USE mo_intp_data_strc, ONLY: t_int_state
    USE mo_nonhydro_types, ONLY: t_nh_diag, t_nh_metrics, t_nh_prog
    TYPE(t_patch), TARGET, INTENT(IN) :: p_patch
    TYPE(t_int_state), TARGET, INTENT(IN) :: p_int
    TYPE(t_nh_prog), INTENT(INOUT) :: p_prog
    TYPE(t_nh_metrics), INTENT(INOUT) :: p_metrics
    TYPE(t_nh_diag), INTENT(INOUT) :: p_diag
    REAL(KIND = 8), DIMENSION(:, :, :), INTENT(INOUT) :: z_w_concorr_me, z_kin_hor_e, z_vt_ie
    INTEGER, INTENT(IN) :: ntnd
    INTEGER, INTENT(IN) :: istep
    LOGICAL, INTENT(IN) :: lvn_only
    REAL(KIND = 8), INTENT(IN) :: dtime
    REAL(KIND = 8), INTENT(IN) :: dt_linintp_ubc
    LOGICAL, INTENT(IN) :: ldeepatmo
  END SUBROUTINE velocity_tendencies
END MODULE mo_velocity_advection
MODULE mo_vertical_coord_table
  IMPLICIT NONE
  REAL(KIND = 8), ALLOCATABLE :: vct_a(:)
  CONTAINS
END MODULE mo_vertical_coord_table
MODULE mo_vertical_grid
  IMPLICIT NONE
  INTEGER :: nrdmax(10), nflat_gradp(10)
  CONTAINS
END MODULE mo_vertical_grid
MODULE mo_solve_nonhydro
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE solve_nh(p_nh, p_patch, p_int, prep_adv, nnow, nnew, l_init, l_recompute, lsave_mflx, lprep_adv, lclean_mflx, idyn_timestep, jstep, dtime, lacc)
    USE mo_nonhydro_types, ONLY: t_nh_state
    USE mo_intp_data_strc, ONLY: t_int_state
    USE mo_model_domain, ONLY: t_patch
    USE mo_prepadv_types, ONLY: t_prepare_adv
    USE mo_parallel_config, ONLY: cpu_min_nproma, nproma, p_test_run, use_dycore_barrier
    USE mo_vertical_grid, ONLY: nflat_gradp, nrdmax
    USE mo_nonhydrostatic_config, ONLY: divdamp_fac, divdamp_fac2, divdamp_fac3, divdamp_fac4, divdamp_fac_o2, divdamp_order, divdamp_type, divdamp_z, divdamp_z2, divdamp_z3, divdamp_z4, iadv_rhotheta, igradp_method, itime_scheme, kstart_dd3d, kstart_moist, ndyn_substeps_var, rayleigh_type, rhotheta_offctr, veladv_offctr
    USE mo_fortran_tools, ONLY: assert_acc_device_only, init_zero_contiguous_dp
    USE mo_real_timer, ONLY: timer_start, timer_stop
    USE mo_timer, ONLY: timer_barrier, timer_solve_nh, timer_solve_nh_cellcomp, timer_solve_nh_edgecomp, timer_solve_nh_exch, timer_solve_nh_vimpl, timer_solve_nh_vnupd
    USE mo_mpi, ONLY: my_process_is_mpi_all_seq, work_mpi_barrier
    USE mo_run_config, ONLY: ltimer, lvert_nest, timers_level
    USE mo_vertical_coord_table, ONLY: vct_a
    USE mo_interpol_config, ONLY: nudge_max_coeff
    USE mo_velocity_advection, ONLY: velocity_tendencies
    USE mo_dynamics_config, ONLY: ldeepatmo
    USE mo_grid_config, ONLY: l_limited_area
    USE mo_loopindices, ONLY: get_indices_c, get_indices_e
    USE mo_init_vgrid, ONLY: nflatlev
    USE mo_icon_interpolation_scalar, ONLY: cells2verts_scalar_dp_deconiface_133 => cells2verts_scalar_dp, cells2verts_scalar_dp_deconiface_134 => cells2verts_scalar_dp
    USE mo_math_gradients, ONLY: grad_green_gauss_cell_dycore_deconiface_135 => grad_green_gauss_cell_dycore
    USE mo_initicon_config, ONLY: iau_wgt_dyn, is_iau_active
    USE mo_gridref_config, ONLY: grf_intmethod_e
    USE mo_sync, ONLY: sync_patch_array_3d_dp_deconiface_137 => sync_patch_array_3d_dp, sync_patch_array_3d_dp_deconiface_139 => sync_patch_array_3d_dp, sync_patch_array_mult_f3din_dp_deconiface_136 => sync_patch_array_mult_f3din_dp, sync_patch_array_mult_f3din_dp_deconiface_138 => sync_patch_array_mult_f3din_dp, sync_patch_array_mult_f3din_dp_deconiface_140 => sync_patch_array_mult_f3din_dp
    TYPE(t_nh_state), TARGET, INTENT(INOUT) :: p_nh
    TYPE(t_int_state), TARGET, INTENT(IN) :: p_int
    TYPE(t_patch), TARGET, INTENT(INOUT) :: p_patch
    TYPE(t_prepare_adv), TARGET, INTENT(INOUT) :: prep_adv
    LOGICAL, INTENT(IN) :: l_init
    LOGICAL, INTENT(IN) :: l_recompute
    LOGICAL, INTENT(IN) :: lsave_mflx
    LOGICAL, INTENT(IN) :: lprep_adv
    LOGICAL, INTENT(IN) :: lclean_mflx
    INTEGER, INTENT(IN) :: idyn_timestep
    INTEGER, INTENT(IN) :: jstep
    INTEGER, INTENT(IN) :: nnow, nnew
    REAL(KIND = 8), INTENT(IN) :: dtime
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: jb, jk, jc, je, jks, jg
    INTEGER :: nlev, nlevp1
    INTEGER :: i_startblk, i_endblk, i_startidx, i_endidx, ishift
    INTEGER :: rl_start, rl_end, istep, nvar, nshift, nshift_total
    INTEGER :: ic, ie, ilc0, ibc0, ikp1, ikp2
    REAL(KIND = 8) :: z_theta_v_fl_e(nproma, p_patch % nlev, p_patch % nblks_e), z_theta_v_e(nproma, p_patch % nlev, p_patch % nblks_e), z_rho_e(nproma, p_patch % nlev, p_patch % nblks_e), z_theta_v_v(nproma, p_patch % nlev, p_patch % nblks_v), z_rho_v(nproma, p_patch % nlev, p_patch % nblks_v)
    REAL(KIND = 8) :: z_th_ddz_exner_c(nproma, p_patch % nlev, p_patch % nblks_c), z_dexner_dz_c(2, nproma, p_patch % nlev, p_patch % nblks_c), z_vt_ie(nproma, p_patch % nlev, p_patch % nblks_e), z_kin_hor_e(nproma, p_patch % nlev, p_patch % nblks_e), z_exner_ex_pr(nproma, p_patch % nlevp1, p_patch % nblks_c), z_gradh_exner(nproma, p_patch % nlev, p_patch % nblks_e), z_rth_pr(2, nproma, p_patch % nlev, p_patch % nblks_c), z_grad_rth(4, nproma, p_patch % nlev, p_patch % nblks_c), z_w_concorr_me(nproma, p_patch % nlev, p_patch % nblks_e)
    REAL(KIND = 8) :: z_graddiv_vn(p_patch % nlev, nproma, p_patch % nblks_e)
    REAL(KIND = 8) :: z_w_expl(nproma, p_patch % nlevp1), z_vn_avg(nproma, p_patch % nlev), z_mflx_top(nproma, p_patch % nblks_c), z_contr_w_fl_l(nproma, p_patch % nlevp1), z_rho_expl(nproma, p_patch % nlev), z_exner_expl(nproma, p_patch % nlev)
    REAL(KIND = 8) :: z_theta_tavg_m1, z_theta_tavg, z_rho_tavg_m1, z_rho_tavg
    REAL(KIND = 8) :: z_alpha(nproma, p_patch % nlevp1), z_beta(nproma, p_patch % nlev), z_q(nproma, p_patch % nlev), z_graddiv2_vn(nproma, p_patch % nlev), z_theta_v_pr_ic(nproma, p_patch % nlevp1), z_exner_ic(nproma, p_patch % nlevp1), z_w_concorr_mc(nproma, p_patch % nlev), z_flxdiv_mass(nproma, p_patch % nlev), z_flxdiv_theta(nproma, p_patch % nlev), z_hydro_corr(nproma, p_patch % nblks_e)
    REAL(KIND = 8) :: z_a, z_b, z_c, z_g, z_gamma, z_w_backtraj, z_theta_v_pr_mc_m1, z_theta_v_pr_mc
    REAL(KIND = 8) :: z_theta1, z_theta2, wgt_nnow_vel, wgt_nnew_vel, dt_shift, wgt_nnow_rth, wgt_nnew_rth, dthalf, r_nsubsteps, r_dtimensubsteps, scal_divdamp_o2, alin, dz32, df32, dz42, df42, bqdr, aqdr, zf, dzlin, dzqdr
    REAL(KIND = 8) :: dt_linintp_ubc, dt_linintp_ubc_nnow, dt_linintp_ubc_nnew
    REAL(KIND = 8) :: z_raylfac(nrdmax(p_patch % id))
    REAL(KIND = 8) :: z_ntdistv_bary_1, distv_bary_1, z_ntdistv_bary_2, distv_bary_2
    REAL(KIND = 8), DIMENSION(p_patch % nlev) :: scal_divdamp, bdy_divdamp, enh_divdamp_fac
    REAL(KIND = 8) :: z_dwdz_dd(nproma, kstart_dd3d(p_patch % id) : p_patch % nlev, p_patch % nblks_c)
    REAL(KIND = 8) :: z_ddt_vn_dyn, z_ddt_vn_apc, z_ddt_vn_cor, z_ddt_vn_pgr, z_ddt_vn_ray, z_d_vn_dmp, z_d_vn_iau
    INTEGER :: nproma_gradp, nblks_gradp, npromz_gradp, nlen_gradp, jk_start
    LOGICAL :: lvn_only, lvn_pos
    LOGICAL :: l_vert_nested, l_child_vertnest
    CALL assert_acc_device_only("solve_nh", lacc)
    IF (use_dycore_barrier) THEN
      CALL timer_start(timer_barrier)
      CALL work_mpi_barrier
      CALL timer_stop(timer_barrier)
    END IF
    jg = p_patch % id
    IF (lvert_nest .AND. (p_patch % nshift_total > 0)) THEN
      l_vert_nested = .TRUE.
      nshift_total = p_patch % nshift_total
    ELSE
      l_vert_nested = .FALSE.
      nshift_total = 0
    END IF
    IF (lvert_nest .AND. p_patch % n_childdom > 0 .AND. (p_patch % nshift_child > 0 .OR. p_patch % nshift_total > 0)) THEN
      l_child_vertnest = .TRUE.
      nshift = p_patch % nshift_child + 1
    ELSE
      l_child_vertnest = .FALSE.
      nshift = 0
    END IF
    dthalf = 0.5D0 * dtime
    IF (ltimer) CALL timer_start(timer_solve_nh)
    r_nsubsteps = 1.0D0 / REAL(ndyn_substeps_var(jg), 8)
    r_dtimensubsteps = 1.0D0 / (dtime * REAL(ndyn_substeps_var(jg), 8))
    nlev = p_patch % nlev
    nlevp1 = p_patch % nlevp1
    DO jk = 2, nrdmax(jg)
      z_raylfac(jk) = 1.0D0 / (1.0D0 + dtime * p_nh % metrics % rayleigh_w(jk))
    END DO
    alin = (divdamp_fac2 - divdamp_fac) / (divdamp_z2 - divdamp_z)
    df32 = divdamp_fac3 - divdamp_fac2
    dz32 = divdamp_z3 - divdamp_z2
    df42 = divdamp_fac4 - divdamp_fac2
    dz42 = divdamp_z4 - divdamp_z2
    bqdr = (df42 * dz32 - df32 * dz42) / (dz32 * dz42 * (dz42 - dz32))
    aqdr = df32 / dz32 - bqdr * dz32
    DO jk = 1, nlev
      jks = jk + nshift_total
      zf = 0.5D0 * (vct_a(jks) + vct_a(jks + 1))
      dzlin = MIN(divdamp_z2 - divdamp_z, MAX(0.0D0, zf - divdamp_z))
      dzqdr = MIN(divdamp_z4 - divdamp_z2, MAX(0.0D0, zf - divdamp_z2))
      IF (divdamp_order == 24) THEN
        enh_divdamp_fac(jk) = MAX(0.0D0, divdamp_fac + dzlin * alin + dzqdr * (aqdr + dzqdr * bqdr) - 0.25D0 * divdamp_fac_o2)
      ELSE
        enh_divdamp_fac(jk) = divdamp_fac + dzlin * alin + dzqdr * (aqdr + dzqdr * bqdr)
      END IF
    END DO
    scal_divdamp(:) = - enh_divdamp_fac(:) * p_patch % geometry_info % mean_cell_area ** 2
    dt_shift = dtime * REAL(2 * ndyn_substeps_var(jg) - 1, 8) / 2.0D0
    dt_linintp_ubc = jstep * dtime - dt_shift
    dt_linintp_ubc_nnow = dt_linintp_ubc - 0.5D0 * dtime
    dt_linintp_ubc_nnew = dt_linintp_ubc + 0.5D0 * dtime
    bdy_divdamp(:) = 0.75D0 / (nudge_max_coeff + 2.220446049250313D-16) * ABS(scal_divdamp(:))
    scal_divdamp_o2 = divdamp_fac_o2 * p_patch % geometry_info % mean_cell_area
    IF (p_test_run) THEN
      z_rho_e = 0.0D0
      z_theta_v_e = 0.0D0
      z_dwdz_dd = 0.0D0
      z_graddiv_vn = 0.0D0
    END IF
    wgt_nnow_vel = 0.5D0 - veladv_offctr
    wgt_nnew_vel = 0.5D0 + veladv_offctr
    wgt_nnew_rth = 0.5D0 + rhotheta_offctr
    wgt_nnow_rth = 1.0D0 - wgt_nnew_rth
    DO istep = 1, 2
      IF (istep == 1) THEN
        IF (itime_scheme >= 6 .OR. l_init .OR. l_recompute) THEN
          IF (itime_scheme < 6 .AND. .NOT. l_init) THEN
            lvn_only = .TRUE.
          ELSE
            lvn_only = .FALSE.
          END IF
          CALL velocity_tendencies(p_nh % prog(nnow), p_patch, p_int, p_nh % metrics, p_nh % diag, z_w_concorr_me, z_kin_hor_e, z_vt_ie, nnow, istep, lvn_only, dtime, dt_linintp_ubc_nnow, ldeepatmo)
        END IF
        nvar = nnow
      ELSE
        lvn_only = .FALSE.
        CALL velocity_tendencies(p_nh % prog(nnew), p_patch, p_int, p_nh % metrics, p_nh % diag, z_w_concorr_me, z_kin_hor_e, z_vt_ie, nnew, istep, lvn_only, dtime, dt_linintp_ubc_nnew, ldeepatmo)
        nvar = nnew
      END IF
      IF (istep == 1 .AND. (igradp_method == 3 .OR. igradp_method == 5)) THEN
        nproma_gradp = cpu_min_nproma(nproma, 256)
        nblks_gradp = INT(p_nh % metrics % pg_listdim / nproma_gradp)
        npromz_gradp = MOD(p_nh % metrics % pg_listdim, nproma_gradp)
        IF (npromz_gradp > 0) THEN
          nblks_gradp = nblks_gradp + 1
        ELSE
          npromz_gradp = nproma_gradp
        END IF
      END IF
      IF (timers_level > 5) CALL timer_start(timer_solve_nh_cellcomp)
      rl_start = 3
      IF (istep == 1) THEN
        rl_end = (- 5)
      ELSE
        rl_end = (- 4)
      END IF
      i_startblk = p_patch % cells % start_block(rl_start)
      i_endblk = p_patch % cells % end_block(rl_end)
      IF (istep == 1 .AND. (jg > 1 .OR. l_limited_area)) THEN
        CALL init_zero_contiguous_dp(z_rth_pr(1, 1, 1, 1), 2 * nproma * nlev * i_startblk, opt_acc_async = .TRUE., lacc = .TRUE.)
      END IF
      DO jb = i_startblk, i_endblk
        CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, i_startidx, i_endidx, rl_start, rl_end)
        IF (istep == 1) THEN
          DO jk = 1, nlev
            DO jc = i_startidx, i_endidx
              z_exner_ex_pr(jc, jk, jb) = (1.0D0 + p_nh % metrics % exner_exfac(jc, jk, jb)) * (p_nh % prog(nnow) % exner(jc, jk, jb) - p_nh % metrics % exner_ref_mc(jc, jk, jb)) - p_nh % metrics % exner_exfac(jc, jk, jb) * p_nh % diag % exner_pr(jc, jk, jb)
              p_nh % diag % exner_pr(jc, jk, jb) = p_nh % prog(nnow) % exner(jc, jk, jb) - p_nh % metrics % exner_ref_mc(jc, jk, jb)
            END DO
          END DO
          z_exner_ex_pr(:, nlevp1, jb) = 0.0D0
          IF (igradp_method <= 3) THEN
            DO jc = i_startidx, i_endidx
              z_exner_ic(jc, nlevp1) = p_nh % metrics % wgtfacq_c(jc, 1, jb) * z_exner_ex_pr(jc, nlev, jb) + p_nh % metrics % wgtfacq_c(jc, 2, jb) * z_exner_ex_pr(jc, nlev - 1, jb) + p_nh % metrics % wgtfacq_c(jc, 3, jb) * z_exner_ex_pr(jc, nlev - 2, jb)
            END DO
            DO jk = nlev, MAX(2, nflatlev(jg)), - 1
              DO jc = i_startidx, i_endidx
                z_exner_ic(jc, jk) = p_nh % metrics % wgtfac_c(jc, jk, jb) * z_exner_ex_pr(jc, jk, jb) + (1.0D0 - p_nh % metrics % wgtfac_c(jc, jk, jb)) * z_exner_ex_pr(jc, jk - 1, jb)
              END DO
            END DO
            DO jk = nlev, MAX(2, nflatlev(jg)), - 1
              DO jc = i_startidx, i_endidx
                z_dexner_dz_c(1, jc, jk, jb) = (z_exner_ic(jc, jk) - z_exner_ic(jc, jk + 1)) * p_nh % metrics % inv_ddqz_z_full(jc, jk, jb)
              END DO
            END DO
            IF (nflatlev(jg) == 1) THEN
              DO jc = i_startidx, i_endidx
                z_exner_ic(jc, 1) = p_nh % metrics % wgtfacq1_c(jc, 1, jb) * z_exner_ex_pr(jc, 1, jb) + p_nh % metrics % wgtfacq1_c(jc, 2, jb) * z_exner_ex_pr(jc, 2, jb) + p_nh % metrics % wgtfacq1_c(jc, 3, jb) * z_exner_ex_pr(jc, 3, jb)
                z_dexner_dz_c(1, jc, 1, jb) = (z_exner_ic(jc, 1) - z_exner_ic(jc, 2)) * p_nh % metrics % inv_ddqz_z_full(jc, 1, jb)
              END DO
            END IF
          END IF
          DO jc = i_startidx, i_endidx
            z_rth_pr(1, jc, 1, jb) = p_nh % prog(nnow) % rho(jc, 1, jb) - p_nh % metrics % rho_ref_mc(jc, 1, jb)
            z_rth_pr(2, jc, 1, jb) = p_nh % prog(nnow) % theta_v(jc, 1, jb) - p_nh % metrics % theta_ref_mc(jc, 1, jb)
          END DO
          DO jk = 2, nlev
            DO jc = i_startidx, i_endidx
              p_nh % diag % rho_ic(jc, jk, jb) = p_nh % metrics % wgtfac_c(jc, jk, jb) * p_nh % prog(nnow) % rho(jc, jk, jb) + (1.0D0 - p_nh % metrics % wgtfac_c(jc, jk, jb)) * p_nh % prog(nnow) % rho(jc, jk - 1, jb)
              z_rth_pr(1, jc, jk, jb) = p_nh % prog(nnow) % rho(jc, jk, jb) - p_nh % metrics % rho_ref_mc(jc, jk, jb)
              z_rth_pr(2, jc, jk, jb) = p_nh % prog(nnow) % theta_v(jc, jk, jb) - p_nh % metrics % theta_ref_mc(jc, jk, jb)
              z_theta_v_pr_ic(jc, jk) = p_nh % metrics % wgtfac_c(jc, jk, jb) * z_rth_pr(2, jc, jk, jb) + (1.0D0 - p_nh % metrics % wgtfac_c(jc, jk, jb)) * z_rth_pr(2, jc, jk - 1, jb)
              p_nh % diag % theta_v_ic(jc, jk, jb) = p_nh % metrics % wgtfac_c(jc, jk, jb) * p_nh % prog(nnow) % theta_v(jc, jk, jb) + (1.0D0 - p_nh % metrics % wgtfac_c(jc, jk, jb)) * p_nh % prog(nnow) % theta_v(jc, jk - 1, jb)
              z_th_ddz_exner_c(jc, jk, jb) = p_nh % metrics % vwind_expl_wgt(jc, jb) * p_nh % diag % theta_v_ic(jc, jk, jb) * (p_nh % diag % exner_pr(jc, jk - 1, jb) - p_nh % diag % exner_pr(jc, jk, jb)) / p_nh % metrics % ddqz_z_half(jc, jk, jb) + z_theta_v_pr_ic(jc, jk) * p_nh % metrics % d_exner_dz_ref_ic(jc, jk, jb)
            END DO
          END DO
        ELSE
          DO jk = 2, nlev
            DO jc = i_startidx, i_endidx
              z_w_backtraj = - (p_nh % prog(nnew) % w(jc, jk, jb) - p_nh % diag % w_concorr_c(jc, jk, jb)) * dtime * 0.5D0 / p_nh % metrics % ddqz_z_half(jc, jk, jb)
              z_rho_tavg_m1 = wgt_nnow_rth * p_nh % prog(nnow) % rho(jc, jk - 1, jb) + wgt_nnew_rth * p_nh % prog(nvar) % rho(jc, jk - 1, jb)
              z_theta_tavg_m1 = wgt_nnow_rth * p_nh % prog(nnow) % theta_v(jc, jk - 1, jb) + wgt_nnew_rth * p_nh % prog(nvar) % theta_v(jc, jk - 1, jb)
              z_rho_tavg = wgt_nnow_rth * p_nh % prog(nnow) % rho(jc, jk, jb) + wgt_nnew_rth * p_nh % prog(nvar) % rho(jc, jk, jb)
              z_theta_tavg = wgt_nnow_rth * p_nh % prog(nnow) % theta_v(jc, jk, jb) + wgt_nnew_rth * p_nh % prog(nvar) % theta_v(jc, jk, jb)
              p_nh % diag % rho_ic(jc, jk, jb) = p_nh % metrics % wgtfac_c(jc, jk, jb) * z_rho_tavg + (1.0D0 - p_nh % metrics % wgtfac_c(jc, jk, jb)) * z_rho_tavg_m1 + z_w_backtraj * (z_rho_tavg_m1 - z_rho_tavg)
              z_theta_v_pr_mc_m1 = z_theta_tavg_m1 - p_nh % metrics % theta_ref_mc(jc, jk - 1, jb)
              z_theta_v_pr_mc = z_theta_tavg - p_nh % metrics % theta_ref_mc(jc, jk, jb)
              z_theta_v_pr_ic(jc, jk) = p_nh % metrics % wgtfac_c(jc, jk, jb) * z_theta_v_pr_mc + (1.0D0 - p_nh % metrics % wgtfac_c(jc, jk, jb)) * z_theta_v_pr_mc_m1
              p_nh % diag % theta_v_ic(jc, jk, jb) = p_nh % metrics % wgtfac_c(jc, jk, jb) * z_theta_tavg + (1.0D0 - p_nh % metrics % wgtfac_c(jc, jk, jb)) * z_theta_tavg_m1 + z_w_backtraj * (z_theta_tavg_m1 - z_theta_tavg)
              z_th_ddz_exner_c(jc, jk, jb) = p_nh % metrics % vwind_expl_wgt(jc, jb) * p_nh % diag % theta_v_ic(jc, jk, jb) * (p_nh % diag % exner_pr(jc, jk - 1, jb) - p_nh % diag % exner_pr(jc, jk, jb)) / p_nh % metrics % ddqz_z_half(jc, jk, jb) + z_theta_v_pr_ic(jc, jk) * p_nh % metrics % d_exner_dz_ref_ic(jc, jk, jb)
            END DO
          END DO
        END IF
        IF (istep == 1) THEN
          DO jc = i_startidx, i_endidx
            z_theta_v_pr_ic(jc, 1) = 0.0D0
            z_theta_v_pr_ic(jc, nlevp1) = p_nh % metrics % wgtfacq_c(jc, 1, jb) * z_rth_pr(2, jc, nlev, jb) + p_nh % metrics % wgtfacq_c(jc, 2, jb) * z_rth_pr(2, jc, nlev - 1, jb) + p_nh % metrics % wgtfacq_c(jc, 3, jb) * z_rth_pr(2, jc, nlev - 2, jb)
            p_nh % diag % theta_v_ic(jc, nlevp1, jb) = p_nh % metrics % theta_ref_ic(jc, nlevp1, jb) + z_theta_v_pr_ic(jc, nlevp1)
          END DO
          IF (igradp_method <= 3) THEN
            DO jk = nflat_gradp(jg), nlev
              DO jc = i_startidx, i_endidx
                z_dexner_dz_c(2, jc, jk, jb) = - 0.5D0 * ((z_theta_v_pr_ic(jc, jk) - z_theta_v_pr_ic(jc, jk + 1)) * p_nh % metrics % d2dexdz2_fac1_mc(jc, jk, jb) + z_rth_pr(2, jc, jk, jb) * p_nh % metrics % d2dexdz2_fac2_mc(jc, jk, jb))
              END DO
            END DO
          END IF
        END IF
      END DO
      IF (istep == 1) THEN
        rl_start = (- 6)
        rl_end = (- 6)
        i_startblk = p_patch % cells % start_block((- 6))
        i_endblk = p_patch % cells % end_block((- 6))
        DO jb = i_startblk, i_endblk
          CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, i_startidx, i_endidx, rl_start, rl_end)
          DO jk = 1, nlev
            DO jc = i_startidx, i_endidx
              z_rth_pr(1, jc, jk, jb) = p_nh % prog(nnow) % rho(jc, jk, jb) - p_nh % metrics % rho_ref_mc(jc, jk, jb)
              z_rth_pr(2, jc, jk, jb) = p_nh % prog(nnow) % theta_v(jc, jk, jb) - p_nh % metrics % theta_ref_mc(jc, jk, jb)
            END DO
          END DO
        END DO
      END IF
      IF (timers_level > 5) THEN
        CALL timer_stop(timer_solve_nh_cellcomp)
        CALL timer_start(timer_solve_nh_vnupd)
      END IF
      IF (istep == 1) THEN
        IF (iadv_rhotheta == 1) THEN
          CALL cells2verts_scalar_dp_deconiface_133(p_nh % prog(nnow) % rho, p_patch, p_int % cells_aw_verts, z_rho_v, lacc = .TRUE., opt_rlend = (- 5))
          CALL cells2verts_scalar_dp_deconiface_134(p_nh % prog(nnow) % theta_v, p_patch, p_int % cells_aw_verts, z_theta_v_v, lacc = .TRUE., opt_rlend = (- 5))
        ELSE IF (iadv_rhotheta == 2) THEN
          CALL grad_green_gauss_cell_dycore_deconiface_135(z_rth_pr, p_patch, p_int, z_grad_rth, lacc = .TRUE., opt_rlstart = 3, opt_rlend = (- 5), opt_acc_async = .TRUE.)
        END IF
      END IF
      IF (istep == 1) THEN
        i_startblk = p_patch % edges % start_block((- 10))
        i_endblk = p_patch % edges % end_block((- 10))
        IF (i_endblk >= i_startblk) THEN
          CALL init_zero_contiguous_dp(z_rho_e(1, 1, i_startblk), nproma * nlev * (i_endblk - i_startblk + 1), opt_acc_async = .TRUE., lacc = .TRUE.)
          CALL init_zero_contiguous_dp(z_theta_v_e(1, 1, i_startblk), nproma * nlev * (i_endblk - i_startblk + 1), opt_acc_async = .TRUE., lacc = .TRUE.)
        END IF
        rl_start = 7
        rl_end = (- 9)
        i_startblk = p_patch % edges % start_block(7)
        i_endblk = p_patch % edges % end_block((- 9))
        IF (jg > 1 .OR. l_limited_area) THEN
          CALL init_zero_contiguous_dp(z_rho_e(1, 1, 1), nproma * nlev * i_startblk, opt_acc_async = .TRUE., lacc = .TRUE.)
          CALL init_zero_contiguous_dp(z_theta_v_e(1, 1, 1), nproma * nlev * i_startblk, opt_acc_async = .TRUE., lacc = .TRUE.)
        END IF
        DO jb = i_startblk, i_endblk
          CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, i_startidx, i_endidx, rl_start, rl_end)
          IF (iadv_rhotheta == 2) THEN
            DO je = i_startidx, i_endidx
              DO jk = 1, nlev
                lvn_pos = p_nh % prog(nnow) % vn(je, jk, jb) >= 0.0D0
                ilc0 = MERGE(p_patch % edges % cell_idx(je, jb, 1), p_patch % edges % cell_idx(je, jb, 2), lvn_pos)
                ibc0 = MERGE(p_patch % edges % cell_blk(je, jb, 1), p_patch % edges % cell_blk(je, jb, 2), lvn_pos)
                z_ntdistv_bary_1 = - (p_nh % prog(nnow) % vn(je, jk, jb) * dthalf + MERGE(p_int % pos_on_tplane_e(je, 1, 1, jb), p_int % pos_on_tplane_e(je, 2, 1, jb), lvn_pos)) * p_nh % metrics % deepatmo_gradh_mc(jk)
                z_ntdistv_bary_2 = - (p_nh % diag % vt(je, jk, jb) * dthalf + MERGE(p_int % pos_on_tplane_e(je, 1, 2, jb), p_int % pos_on_tplane_e(je, 2, 2, jb), lvn_pos)) * p_nh % metrics % deepatmo_gradh_mc(jk)
                distv_bary_1 = z_ntdistv_bary_1 * MERGE(p_patch % edges % primal_normal_cell(je, jb, 1) % v1, p_patch % edges % primal_normal_cell(je, jb, 2) % v1, lvn_pos) + z_ntdistv_bary_2 * MERGE(p_patch % edges % dual_normal_cell(je, jb, 1) % v1, p_patch % edges % dual_normal_cell(je, jb, 2) % v1, lvn_pos)
                distv_bary_2 = z_ntdistv_bary_1 * MERGE(p_patch % edges % primal_normal_cell(je, jb, 1) % v2, p_patch % edges % primal_normal_cell(je, jb, 2) % v2, lvn_pos) + z_ntdistv_bary_2 * MERGE(p_patch % edges % dual_normal_cell(je, jb, 1) % v2, p_patch % edges % dual_normal_cell(je, jb, 2) % v2, lvn_pos)
                z_rho_e(je, jk, jb) = REAL(p_nh % metrics % rho_ref_me(je, jk, jb), 8) + z_rth_pr(1, ilc0, jk, ibc0) + distv_bary_1 * z_grad_rth(1, ilc0, jk, ibc0) + distv_bary_2 * z_grad_rth(2, ilc0, jk, ibc0)
                z_theta_v_e(je, jk, jb) = REAL(p_nh % metrics % theta_ref_me(je, jk, jb), 8) + z_rth_pr(2, ilc0, jk, ibc0) + distv_bary_1 * z_grad_rth(3, ilc0, jk, ibc0) + distv_bary_2 * z_grad_rth(4, ilc0, jk, ibc0)
              END DO
            END DO
          ELSE
            DO je = i_startidx, i_endidx
              DO jk = 1, nlev
                z_rho_e(je, jk, jb) = p_int % c_lin_e(je, 1, jb) * p_nh % prog(nnow) % rho(p_patch % edges % cell_idx(je, jb, 1), jk, p_patch % edges % cell_blk(je, jb, 1)) + p_int % c_lin_e(je, 2, jb) * p_nh % prog(nnow) % rho(p_patch % edges % cell_idx(je, jb, 2), jk, p_patch % edges % cell_blk(je, jb, 2)) - dtime * (p_nh % prog(nnow) % vn(je, jk, jb) * p_patch % edges % inv_dual_edge_length(je, jb) * (p_nh % prog(nnow) % rho(p_patch % edges % cell_idx(je, jb, 2), jk, p_patch % edges % cell_blk(je, jb, 2)) - p_nh % prog(nnow) % rho(p_patch % edges % cell_idx(je, jb, 1), jk, p_patch % edges % cell_blk(je, jb, 1))) + p_nh % diag % vt(je, jk, jb) * p_patch % edges % inv_primal_edge_length(je, jb) * p_patch % edges % tangent_orientation(je, jb) * (z_rho_v(p_patch % edges % vertex_idx(je, jb, 2), jk, p_patch % edges % vertex_blk(je, jb, 2)) - z_rho_v(p_patch % edges % vertex_idx(je, jb, 1), jk, p_patch % edges % vertex_blk(je, jb, 1))))
                z_theta_v_e(je, jk, jb) = p_int % c_lin_e(je, 1, jb) * p_nh % prog(nnow) % theta_v(p_patch % edges % cell_idx(je, jb, 1), jk, p_patch % edges % cell_blk(je, jb, 1)) + p_int % c_lin_e(je, 2, jb) * p_nh % prog(nnow) % theta_v(p_patch % edges % cell_idx(je, jb, 2), jk, p_patch % edges % cell_blk(je, jb, 2)) - dtime * (p_nh % prog(nnow) % vn(je, jk, jb) * p_patch % edges % inv_dual_edge_length(je, jb) * (p_nh % prog(nnow) % theta_v(p_patch % edges % cell_idx(je, jb, 2), jk, p_patch % edges % cell_blk(je, jb, 2)) - p_nh % prog(nnow) % theta_v(p_patch % edges % cell_idx(je, jb, 1), jk, p_patch % edges % cell_blk(je, jb, 1))) + p_nh % diag % vt(je, jk, jb) * p_patch % edges % inv_primal_edge_length(je, jb) * p_patch % edges % tangent_orientation(je, jb) * (z_theta_v_v(p_patch % edges % vertex_idx(je, jb, 2), jk, p_patch % edges % vertex_blk(je, jb, 2)) - z_theta_v_v(p_patch % edges % vertex_idx(je, jb, 1), jk, p_patch % edges % vertex_blk(je, jb, 1))))
              END DO
            END DO
          END IF
        END DO
      ELSE IF (istep == 2 .AND. divdamp_type >= 3) THEN
        rl_start = 7
        rl_end = (- 10)
        i_startblk = p_patch % edges % start_block(7)
        i_endblk = p_patch % edges % end_block((- 10))
        DO jb = i_startblk, i_endblk
          CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, i_startidx, i_endidx, rl_start, rl_end)
          DO je = i_startidx, i_endidx
            DO jk = kstart_dd3d(jg), nlev
              z_graddiv_vn(jk, je, jb) = z_graddiv_vn(jk, je, jb) + p_nh % metrics % hmask_dd3d(je, jb) * p_nh % metrics % scalfac_dd3d(jk) * p_patch % edges % inv_dual_edge_length(je, jb) * (z_dwdz_dd(p_patch % edges % cell_idx(je, jb, 2), jk, p_patch % edges % cell_blk(je, jb, 2)) - z_dwdz_dd(p_patch % edges % cell_idx(je, jb, 1), jk, p_patch % edges % cell_blk(je, jb, 1)))
            END DO
          END DO
        END DO
      END IF
      rl_start = 10
      rl_end = (- 8)
      i_startblk = p_patch % edges % start_block(rl_start)
      i_endblk = p_patch % edges % end_block(rl_end)
      IF (istep == 1) THEN
        DO jb = i_startblk, i_endblk
          CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, i_startidx, i_endidx, rl_start, rl_end)
          IF (idyn_timestep == 1 .AND. l_child_vertnest) THEN
            DO je = i_startidx, i_endidx
              p_nh % diag % vn_ie_int(je, 1, jb) = p_nh % diag % vn_ie(je, nshift, jb)
            END DO
          END IF
          DO je = i_startidx, i_endidx
            DO jk = 1, nflatlev(jg) - 1
              z_gradh_exner(je, jk, jb) = p_patch % edges % inv_dual_edge_length(je, jb) * p_nh % metrics % deepatmo_gradh_mc(jk) * (z_exner_ex_pr(p_patch % edges % cell_idx(je, jb, 2), jk, p_patch % edges % cell_blk(je, jb, 2)) - z_exner_ex_pr(p_patch % edges % cell_idx(je, jb, 1), jk, p_patch % edges % cell_blk(je, jb, 1)))
            END DO
          END DO
          IF (igradp_method <= 3) THEN
            DO je = i_startidx, i_endidx
              DO jk = nflatlev(jg), nflat_gradp(jg)
                z_gradh_exner(je, jk, jb) = p_patch % edges % inv_dual_edge_length(je, jb) * p_nh % metrics % deepatmo_gradh_mc(jk) * (z_exner_ex_pr(p_patch % edges % cell_idx(je, jb, 2), jk, p_patch % edges % cell_blk(je, jb, 2)) - z_exner_ex_pr(p_patch % edges % cell_idx(je, jb, 1), jk, p_patch % edges % cell_blk(je, jb, 1))) - p_nh % metrics % ddxn_z_full(je, jk, jb) * (p_int % c_lin_e(je, 1, jb) * z_dexner_dz_c(1, p_patch % edges % cell_idx(je, jb, 1), jk, p_patch % edges % cell_blk(je, jb, 1)) + p_int % c_lin_e(je, 2, jb) * z_dexner_dz_c(1, p_patch % edges % cell_idx(je, jb, 2), jk, p_patch % edges % cell_blk(je, jb, 2)))
              END DO
            END DO
            DO je = i_startidx, i_endidx
              DO jk = nflat_gradp(jg) + 1, nlev
                z_gradh_exner(je, jk, jb) = p_patch % edges % inv_dual_edge_length(je, jb) * p_nh % metrics % deepatmo_gradh_mc(jk) * (z_exner_ex_pr(p_patch % edges % cell_idx(je, jb, 2), p_nh % metrics % vertidx_gradp(je, 2, jk, jb), p_patch % edges % cell_blk(je, jb, 2)) + p_nh % metrics % zdiff_gradp(je, 2, jk, jb) * (z_dexner_dz_c(1, p_patch % edges % cell_idx(je, jb, 2), p_nh % metrics % vertidx_gradp(je, 2, jk, jb), p_patch % edges % cell_blk(je, jb, 2)) + p_nh % metrics % zdiff_gradp(je, 2, jk, jb) * z_dexner_dz_c(2, p_patch % edges % cell_idx(je, jb, 2), p_nh % metrics % vertidx_gradp(je, 2, jk, jb), p_patch % edges % cell_blk(je, jb, 2))) - (z_exner_ex_pr(p_patch % edges % cell_idx(je, jb, 1), p_nh % metrics % vertidx_gradp(je, 1, jk, jb), p_patch % edges % cell_blk(je, jb, 1)) + p_nh % metrics % zdiff_gradp(je, 1, jk, jb) * (z_dexner_dz_c(1, p_patch % edges % cell_idx(je, jb, 1), p_nh % metrics % vertidx_gradp(je, 1, jk, jb), p_patch % edges % cell_blk(je, jb, 1)) + p_nh % metrics % zdiff_gradp(je, 1, jk, jb) * z_dexner_dz_c(2, p_patch % edges % cell_idx(je, jb, 1), p_nh % metrics % vertidx_gradp(je, 1, jk, jb), p_patch % edges % cell_blk(je, jb, 1)))))
              END DO
            END DO
          ELSE IF (igradp_method == 4 .OR. igradp_method == 5) THEN
            DO je = i_startidx, i_endidx
              DO jk = nflatlev(jg), nlev
                z_gradh_exner(je, jk, jb) = p_patch % edges % inv_dual_edge_length(je, jb) * p_nh % metrics % deepatmo_gradh_mc(jk) * (z_exner_ex_pr(p_patch % edges % cell_idx(je, jb, 2), p_nh % metrics % vertidx_gradp(je, 2, jk, jb) - 1, p_patch % edges % cell_blk(je, jb, 2)) * p_nh % metrics % coeff_gradp(je, 5, jk, jb) + z_exner_ex_pr(p_patch % edges % cell_idx(je, jb, 2), p_nh % metrics % vertidx_gradp(je, 2, jk, jb), p_patch % edges % cell_blk(je, jb, 2)) * p_nh % metrics % coeff_gradp(je, 6, jk, jb) + z_exner_ex_pr(p_patch % edges % cell_idx(je, jb, 2), p_nh % metrics % vertidx_gradp(je, 2, jk, jb) + 1, p_patch % edges % cell_blk(je, jb, 2)) * p_nh % metrics % coeff_gradp(je, 7, jk, jb) + z_exner_ex_pr(p_patch % edges % cell_idx(je, jb, 2), p_nh % metrics % vertidx_gradp(je, 2, jk, jb) + 2, p_patch % edges % cell_blk(je, jb, 2)) * p_nh % metrics % coeff_gradp(je, 8, jk, jb) - (z_exner_ex_pr(p_patch % edges % cell_idx(je, jb, 1), p_nh % metrics % vertidx_gradp(je, 1, jk, jb) - 1, p_patch % edges % cell_blk(je, jb, 1)) * p_nh % metrics % coeff_gradp(je, 1, jk, jb) + z_exner_ex_pr(p_patch % edges % cell_idx(je, jb, 1), p_nh % metrics % vertidx_gradp(je, 1, jk, jb), p_patch % edges % cell_blk(je, jb, 1)) * p_nh % metrics % coeff_gradp(je, 2, jk, jb) + z_exner_ex_pr(p_patch % edges % cell_idx(je, jb, 1), p_nh % metrics % vertidx_gradp(je, 1, jk, jb) + 1, p_patch % edges % cell_blk(je, jb, 1)) * p_nh % metrics % coeff_gradp(je, 3, jk, jb) + z_exner_ex_pr(p_patch % edges % cell_idx(je, jb, 1), p_nh % metrics % vertidx_gradp(je, 1, jk, jb) + 2, p_patch % edges % cell_blk(je, jb, 1)) * p_nh % metrics % coeff_gradp(je, 4, jk, jb)))
              END DO
            END DO
          END IF
          IF (igradp_method == 3) THEN
            DO je = i_startidx, i_endidx
              z_theta1 = p_nh % prog(nnow) % theta_v(p_patch % edges % cell_idx(je, jb, 1), p_nh % metrics % vertidx_gradp(je, 1, nlev, jb), p_patch % edges % cell_blk(je, jb, 1)) + p_nh % metrics % zdiff_gradp(je, 1, nlev, jb) * (p_nh % diag % theta_v_ic(p_patch % edges % cell_idx(je, jb, 1), p_nh % metrics % vertidx_gradp(je, 1, nlev, jb), p_patch % edges % cell_blk(je, jb, 1)) - p_nh % diag % theta_v_ic(p_patch % edges % cell_idx(je, jb, 1), p_nh % metrics % vertidx_gradp(je, 1, nlev, jb) + 1, p_patch % edges % cell_blk(je, jb, 1))) * p_nh % metrics % inv_ddqz_z_full(p_patch % edges % cell_idx(je, jb, 1), p_nh % metrics % vertidx_gradp(je, 1, nlev, jb), p_patch % edges % cell_blk(je, jb, 1))
              z_theta2 = p_nh % prog(nnow) % theta_v(p_patch % edges % cell_idx(je, jb, 2), p_nh % metrics % vertidx_gradp(je, 2, nlev, jb), p_patch % edges % cell_blk(je, jb, 2)) + p_nh % metrics % zdiff_gradp(je, 2, nlev, jb) * (p_nh % diag % theta_v_ic(p_patch % edges % cell_idx(je, jb, 2), p_nh % metrics % vertidx_gradp(je, 2, nlev, jb), p_patch % edges % cell_blk(je, jb, 2)) - p_nh % diag % theta_v_ic(p_patch % edges % cell_idx(je, jb, 2), p_nh % metrics % vertidx_gradp(je, 2, nlev, jb) + 1, p_patch % edges % cell_blk(je, jb, 2))) * p_nh % metrics % inv_ddqz_z_full(p_patch % edges % cell_idx(je, jb, 2), p_nh % metrics % vertidx_gradp(je, 2, nlev, jb), p_patch % edges % cell_blk(je, jb, 2))
              z_hydro_corr(je, jb) = 0.00976135730211817D0 * p_patch % edges % inv_dual_edge_length(je, jb) * (z_theta2 - z_theta1) * 4.0D0 / (z_theta1 + z_theta2) ** 2
            END DO
          ELSE IF (igradp_method == 5) THEN
            DO je = i_startidx, i_endidx
              ikp1 = MIN(nlev, p_nh % metrics % vertidx_gradp(je, 1, nlev, jb) + 2)
              ikp2 = MIN(nlev, p_nh % metrics % vertidx_gradp(je, 2, nlev, jb) + 2)
              z_theta1 = p_nh % prog(nnow) % theta_v(p_patch % edges % cell_idx(je, jb, 1), p_nh % metrics % vertidx_gradp(je, 1, nlev, jb) - 1, p_patch % edges % cell_blk(je, jb, 1)) * p_nh % metrics % coeff_gradp(je, 1, nlev, jb) + p_nh % prog(nnow) % theta_v(p_patch % edges % cell_idx(je, jb, 1), p_nh % metrics % vertidx_gradp(je, 1, nlev, jb), p_patch % edges % cell_blk(je, jb, 1)) * p_nh % metrics % coeff_gradp(je, 2, nlev, jb) + p_nh % prog(nnow) % theta_v(p_patch % edges % cell_idx(je, jb, 1), p_nh % metrics % vertidx_gradp(je, 1, nlev, jb) + 1, p_patch % edges % cell_blk(je, jb, 1)) * p_nh % metrics % coeff_gradp(je, 3, nlev, jb) + p_nh % prog(nnow) % theta_v(p_patch % edges % cell_idx(je, jb, 1), ikp1, p_patch % edges % cell_blk(je, jb, 1)) * p_nh % metrics % coeff_gradp(je, 4, nlev, jb)
              z_theta2 = p_nh % prog(nnow) % theta_v(p_patch % edges % cell_idx(je, jb, 2), p_nh % metrics % vertidx_gradp(je, 2, nlev, jb) - 1, p_patch % edges % cell_blk(je, jb, 2)) * p_nh % metrics % coeff_gradp(je, 5, nlev, jb) + p_nh % prog(nnow) % theta_v(p_patch % edges % cell_idx(je, jb, 2), p_nh % metrics % vertidx_gradp(je, 2, nlev, jb), p_patch % edges % cell_blk(je, jb, 2)) * p_nh % metrics % coeff_gradp(je, 6, nlev, jb) + p_nh % prog(nnow) % theta_v(p_patch % edges % cell_idx(je, jb, 2), p_nh % metrics % vertidx_gradp(je, 2, nlev, jb) + 1, p_patch % edges % cell_blk(je, jb, 2)) * p_nh % metrics % coeff_gradp(je, 7, nlev, jb) + p_nh % prog(nnow) % theta_v(p_patch % edges % cell_idx(je, jb, 2), ikp2, p_patch % edges % cell_blk(je, jb, 2)) * p_nh % metrics % coeff_gradp(je, 8, nlev, jb)
              z_hydro_corr(je, jb) = 0.00976135730211817D0 * p_patch % edges % inv_dual_edge_length(je, jb) * (z_theta2 - z_theta1) * 4.0D0 / (z_theta1 + z_theta2) ** 2
            END DO
          END IF
        END DO
      END IF
      IF (istep == 1 .AND. (igradp_method == 3 .OR. igradp_method == 5)) THEN
        DO jb = 1, nblks_gradp
          IF (jb == nblks_gradp) THEN
            nlen_gradp = npromz_gradp
          ELSE
            nlen_gradp = nproma_gradp
          END IF
          ishift = (jb - 1) * nproma_gradp
          DO je = 1, nlen_gradp
            ie = ishift + je
            z_gradh_exner(p_nh % metrics % pg_edgeidx(ie), p_nh % metrics % pg_vertidx(ie), p_nh % metrics % pg_edgeblk(ie)) = z_gradh_exner(p_nh % metrics % pg_edgeidx(ie), p_nh % metrics % pg_vertidx(ie), p_nh % metrics % pg_edgeblk(ie)) + p_nh % metrics % pg_exdist(ie) * z_hydro_corr(p_nh % metrics % pg_edgeidx(ie), p_nh % metrics % pg_edgeblk(ie))
          END DO
        END DO
      END IF
      DO jb = i_startblk, i_endblk
        CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, i_startidx, i_endidx, rl_start, rl_end)
        IF (istep == 2) THEN
          DO jk = 1, nlev
            DO je = i_startidx, i_endidx
              z_ddt_vn_apc = p_nh % diag % ddt_vn_apc_pc(je, jk, jb, nnow) * wgt_nnow_vel + p_nh % diag % ddt_vn_apc_pc(je, jk, jb, nnew) * wgt_nnew_vel
              z_ddt_vn_pgr = - 1004.64D0 * z_theta_v_e(je, jk, jb) * z_gradh_exner(je, jk, jb)
              z_ddt_vn_dyn = z_ddt_vn_apc + z_ddt_vn_pgr + p_nh % diag % ddt_vn_phy(je, jk, jb)
              p_nh % prog(nnew) % vn(je, jk, jb) = p_nh % prog(nnow) % vn(je, jk, jb) + dtime * z_ddt_vn_dyn
              IF (p_nh % diag % ddt_vn_adv_is_associated .OR. p_nh % diag % ddt_vn_cor_is_associated) THEN
                z_ddt_vn_cor = p_nh % diag % ddt_vn_cor_pc(je, jk, jb, nnow) * wgt_nnow_vel + p_nh % diag % ddt_vn_cor_pc(je, jk, jb, nnew) * wgt_nnew_vel
                IF (p_nh % diag % ddt_vn_adv_is_associated) THEN
                  p_nh % diag % ddt_vn_adv(je, jk, jb) = p_nh % diag % ddt_vn_adv(je, jk, jb) + r_nsubsteps * (z_ddt_vn_apc - z_ddt_vn_cor)
                END IF
                IF (p_nh % diag % ddt_vn_cor_is_associated) THEN
                  p_nh % diag % ddt_vn_cor(je, jk, jb) = p_nh % diag % ddt_vn_cor(je, jk, jb) + r_nsubsteps * z_ddt_vn_cor
                END IF
              END IF
              IF (p_nh % diag % ddt_vn_pgr_is_associated) THEN
                p_nh % diag % ddt_vn_pgr(je, jk, jb) = p_nh % diag % ddt_vn_pgr(je, jk, jb) + r_nsubsteps * z_ddt_vn_pgr
              END IF
              IF (p_nh % diag % ddt_vn_phd_is_associated) THEN
                p_nh % diag % ddt_vn_phd(je, jk, jb) = p_nh % diag % ddt_vn_phd(je, jk, jb) + r_nsubsteps * p_nh % diag % ddt_vn_phy(je, jk, jb)
              END IF
              IF (p_nh % diag % ddt_vn_dyn_is_associated) THEN
                p_nh % diag % ddt_vn_dyn(je, jk, jb) = p_nh % diag % ddt_vn_dyn(je, jk, jb) + r_nsubsteps * z_ddt_vn_dyn
              END IF
            END DO
          END DO
        ELSE
          DO jk = 1, nlev
            DO je = i_startidx, i_endidx
              p_nh % prog(nnew) % vn(je, jk, jb) = p_nh % prog(nnow) % vn(je, jk, jb) + dtime * (p_nh % diag % ddt_vn_apc_pc(je, jk, jb, nnow) - 1004.64D0 * z_theta_v_e(je, jk, jb) * z_gradh_exner(je, jk, jb) + p_nh % diag % ddt_vn_phy(je, jk, jb))
            END DO
          END DO
        END IF
        IF (istep == 2) THEN
          IF (divdamp_order == 4 .OR. divdamp_order == 24) THEN
            DO je = i_startidx, i_endidx
              DO jk = 1, nlev
                z_graddiv2_vn(je, jk) = p_int % geofac_grdiv(je, 1, jb) * z_graddiv_vn(jk, je, jb) + p_int % geofac_grdiv(je, 2, jb) * z_graddiv_vn(jk, p_patch % edges % quad_idx(je, jb, 1), p_patch % edges % quad_blk(je, jb, 1)) + p_int % geofac_grdiv(je, 3, jb) * z_graddiv_vn(jk, p_patch % edges % quad_idx(je, jb, 2), p_patch % edges % quad_blk(je, jb, 2)) + p_int % geofac_grdiv(je, 4, jb) * z_graddiv_vn(jk, p_patch % edges % quad_idx(je, jb, 3), p_patch % edges % quad_blk(je, jb, 3)) + p_int % geofac_grdiv(je, 5, jb) * z_graddiv_vn(jk, p_patch % edges % quad_idx(je, jb, 4), p_patch % edges % quad_blk(je, jb, 4))
              END DO
            END DO
          END IF
          IF (divdamp_order == 2 .OR. (divdamp_order == 24 .AND. scal_divdamp_o2 > 1D-06)) THEN
            DO jk = 1, nlev
              DO je = i_startidx, i_endidx
                z_d_vn_dmp = scal_divdamp_o2 * z_graddiv_vn(jk, je, jb)
                p_nh % prog(nnew) % vn(je, jk, jb) = p_nh % prog(nnew) % vn(je, jk, jb) + z_d_vn_dmp
                IF (p_nh % diag % ddt_vn_dmp_is_associated) THEN
                  p_nh % diag % ddt_vn_dmp(je, jk, jb) = p_nh % diag % ddt_vn_dmp(je, jk, jb) + z_d_vn_dmp * r_dtimensubsteps
                END IF
                IF (p_nh % diag % ddt_vn_dyn_is_associated) THEN
                  p_nh % diag % ddt_vn_dyn(je, jk, jb) = p_nh % diag % ddt_vn_dyn(je, jk, jb) + z_d_vn_dmp * r_dtimensubsteps
                END IF
              END DO
            END DO
          END IF
          IF (divdamp_order == 4 .OR. (divdamp_order == 24 .AND. divdamp_fac_o2 <= 4.0D0 * divdamp_fac)) THEN
            IF (l_limited_area .OR. jg > 1) THEN
              DO jk = 1, nlev
                DO je = i_startidx, i_endidx
                  z_d_vn_dmp = (scal_divdamp(jk) + bdy_divdamp(jk) * p_int % nudgecoeff_e(je, jb)) * z_graddiv2_vn(je, jk)
                  p_nh % prog(nnew) % vn(je, jk, jb) = p_nh % prog(nnew) % vn(je, jk, jb) + z_d_vn_dmp
                  IF (p_nh % diag % ddt_vn_dmp_is_associated) THEN
                    p_nh % diag % ddt_vn_dmp(je, jk, jb) = p_nh % diag % ddt_vn_dmp(je, jk, jb) + z_d_vn_dmp * r_dtimensubsteps
                  END IF
                  IF (p_nh % diag % ddt_vn_dyn_is_associated) THEN
                    p_nh % diag % ddt_vn_dyn(je, jk, jb) = p_nh % diag % ddt_vn_dyn(je, jk, jb) + z_d_vn_dmp * r_dtimensubsteps
                  END IF
                END DO
              END DO
            ELSE
              DO jk = 1, nlev
                DO je = i_startidx, i_endidx
                  z_d_vn_dmp = scal_divdamp(jk) * z_graddiv2_vn(je, jk)
                  p_nh % prog(nnew) % vn(je, jk, jb) = p_nh % prog(nnew) % vn(je, jk, jb) + z_d_vn_dmp
                  IF (p_nh % diag % ddt_vn_dmp_is_associated) THEN
                    p_nh % diag % ddt_vn_dmp(je, jk, jb) = p_nh % diag % ddt_vn_dmp(je, jk, jb) + z_d_vn_dmp * r_dtimensubsteps
                  END IF
                  IF (p_nh % diag % ddt_vn_dyn_is_associated) THEN
                    p_nh % diag % ddt_vn_dyn(je, jk, jb) = p_nh % diag % ddt_vn_dyn(je, jk, jb) + z_d_vn_dmp * r_dtimensubsteps
                  END IF
                END DO
              END DO
            END IF
          END IF
        END IF
        IF (is_iau_active) THEN
          DO jk = 1, nlev
            DO je = i_startidx, i_endidx
              z_d_vn_iau = iau_wgt_dyn * p_nh % diag % vn_incr(je, jk, jb)
              p_nh % prog(nnew) % vn(je, jk, jb) = p_nh % prog(nnew) % vn(je, jk, jb) + z_d_vn_iau
              IF (istep == 2) THEN
                IF (p_nh % diag % ddt_vn_iau_is_associated) THEN
                  p_nh % diag % ddt_vn_iau(je, jk, jb) = p_nh % diag % ddt_vn_iau(je, jk, jb) + z_d_vn_iau * r_dtimensubsteps
                END IF
                IF (p_nh % diag % ddt_vn_dyn_is_associated) THEN
                  p_nh % diag % ddt_vn_dyn(je, jk, jb) = p_nh % diag % ddt_vn_dyn(je, jk, jb) + z_d_vn_iau * r_dtimensubsteps
                END IF
              END IF
            END DO
          END DO
        END IF
        IF (rayleigh_type == 1) THEN
          DO jk = 1, nrdmax(jg)
            DO je = i_startidx, i_endidx
              z_ddt_vn_ray = - p_nh % metrics % rayleigh_vn(jk) * (p_nh % prog(nnew) % vn(je, jk, jb) - p_nh % ref % vn_ref(je, jk, jb))
              p_nh % prog(nnew) % vn(je, jk, jb) = p_nh % prog(nnew) % vn(je, jk, jb) + z_ddt_vn_ray * dtime
              IF (istep == 2) THEN
                IF (p_nh % diag % ddt_vn_ray_is_associated) THEN
                  p_nh % diag % ddt_vn_ray(je, jk, jb) = p_nh % diag % ddt_vn_ray(je, jk, jb) + z_ddt_vn_ray * r_nsubsteps
                END IF
                IF (p_nh % diag % ddt_vn_dyn_is_associated) THEN
                  p_nh % diag % ddt_vn_dyn(je, jk, jb) = p_nh % diag % ddt_vn_dyn(je, jk, jb) + z_ddt_vn_ray * r_nsubsteps
                END IF
              END IF
            END DO
          END DO
        END IF
      END DO
      IF (istep == 1 .AND. (l_limited_area .OR. jg > 1)) THEN
        rl_start = 1
        rl_end = 9
        i_startblk = p_patch % edges % start_block(1)
        i_endblk = p_patch % edges % end_block(9)
        DO jb = i_startblk, i_endblk
          CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, i_startidx, i_endidx, rl_start, rl_end)
          DO jk = 1, nlev
            DO je = i_startidx, i_endidx
              p_nh % prog(nnew) % vn(je, jk, jb) = p_nh % prog(nnow) % vn(je, jk, jb) + p_nh % diag % grf_tend_vn(je, jk, jb) * dtime
              IF (p_nh % diag % ddt_vn_grf_is_associated) THEN
                p_nh % diag % ddt_vn_grf(je, jk, jb) = p_nh % diag % ddt_vn_grf(je, jk, jb) + p_nh % diag % grf_tend_vn(je, jk, jb) * r_nsubsteps
              END IF
              IF (p_nh % diag % ddt_vn_dyn_is_associated) THEN
                p_nh % diag % ddt_vn_dyn(je, jk, jb) = p_nh % diag % ddt_vn_dyn(je, jk, jb) + p_nh % diag % grf_tend_vn(je, jk, jb) * r_nsubsteps
              END IF
            END DO
          END DO
        END DO
      END IF
      IF (jg > 1 .AND. grf_intmethod_e == 6 .AND. jstep == 0 .AND. istep == 1) THEN
        DO ic = 1, p_nh % metrics % bdy_mflx_e_dim
          je = p_nh % metrics % bdy_mflx_e_idx(ic)
          jb = p_nh % metrics % bdy_mflx_e_blk(ic)
          DO jk = 1, nlev
            p_nh % diag % grf_bdy_mflx(jk, ic, 2) = p_nh % diag % grf_tend_mflx(je, jk, jb)
            p_nh % diag % grf_bdy_mflx(jk, ic, 1) = prep_adv % mass_flx_me(je, jk, jb) - dt_shift * p_nh % diag % grf_bdy_mflx(jk, ic, 2)
          END DO
        END DO
      END IF
      IF (timers_level > 5) THEN
        CALL timer_stop(timer_solve_nh_vnupd)
        CALL timer_start(timer_solve_nh_exch)
      END IF
      IF (istep == 1) THEN
        CALL sync_patch_array_mult_f3din_dp_deconiface_136(2, p_patch, 2, lacc = .TRUE., f3din1 = p_nh % prog(nnew) % vn, f3din2 = z_rho_e, opt_varname = "vn_nnew and z_rho_e")
      ELSE
        CALL sync_patch_array_3d_dp_deconiface_137(2, p_patch, p_nh % prog(nnew) % vn, lacc = .TRUE., opt_varname = "vn_nnew")
      END IF
      IF (timers_level > 5) THEN
        CALL timer_stop(timer_solve_nh_exch)
        CALL timer_start(timer_solve_nh_edgecomp)
      END IF
      rl_start = 5
      rl_end = (- 10)
      i_startblk = p_patch % edges % start_block(rl_start)
      i_endblk = p_patch % edges % end_block(rl_end)
      DO jb = i_startblk, i_endblk
        CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, i_startidx, i_endidx, rl_start, rl_end)
        IF (istep == 1) THEN
          DO je = i_startidx, i_endidx
            DO jk = 1, nlev
              z_vn_avg(je, jk) = p_int % e_flx_avg(je, 1, jb) * p_nh % prog(nnew) % vn(je, jk, jb) + p_int % e_flx_avg(je, 2, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 1), jk, p_patch % edges % quad_blk(je, jb, 1)) + p_int % e_flx_avg(je, 3, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 2), jk, p_patch % edges % quad_blk(je, jb, 2)) + p_int % e_flx_avg(je, 4, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 3), jk, p_patch % edges % quad_blk(je, jb, 3)) + p_int % e_flx_avg(je, 5, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 4), jk, p_patch % edges % quad_blk(je, jb, 4))
              z_graddiv_vn(jk, je, jb) = p_int % geofac_grdiv(je, 1, jb) * p_nh % prog(nnew) % vn(je, jk, jb) + p_int % geofac_grdiv(je, 2, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 1), jk, p_patch % edges % quad_blk(je, jb, 1)) + p_int % geofac_grdiv(je, 3, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 2), jk, p_patch % edges % quad_blk(je, jb, 2)) + p_int % geofac_grdiv(je, 4, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 3), jk, p_patch % edges % quad_blk(je, jb, 3)) + p_int % geofac_grdiv(je, 5, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 4), jk, p_patch % edges % quad_blk(je, jb, 4))
              p_nh % diag % vt(je, jk, jb) = p_int % rbf_vec_coeff_e(1, je, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 1), jk, p_patch % edges % quad_blk(je, jb, 1)) + p_int % rbf_vec_coeff_e(2, je, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 2), jk, p_patch % edges % quad_blk(je, jb, 2)) + p_int % rbf_vec_coeff_e(3, je, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 3), jk, p_patch % edges % quad_blk(je, jb, 3)) + p_int % rbf_vec_coeff_e(4, je, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 4), jk, p_patch % edges % quad_blk(je, jb, 4))
            END DO
          END DO
        ELSE IF (itime_scheme >= 5) THEN
          DO je = i_startidx, i_endidx
            DO jk = 1, nlev
              z_vn_avg(je, jk) = p_int % e_flx_avg(je, 1, jb) * p_nh % prog(nnew) % vn(je, jk, jb) + p_int % e_flx_avg(je, 2, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 1), jk, p_patch % edges % quad_blk(je, jb, 1)) + p_int % e_flx_avg(je, 3, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 2), jk, p_patch % edges % quad_blk(je, jb, 2)) + p_int % e_flx_avg(je, 4, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 3), jk, p_patch % edges % quad_blk(je, jb, 3)) + p_int % e_flx_avg(je, 5, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 4), jk, p_patch % edges % quad_blk(je, jb, 4))
              p_nh % diag % vt(je, jk, jb) = p_int % rbf_vec_coeff_e(1, je, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 1), jk, p_patch % edges % quad_blk(je, jb, 1)) + p_int % rbf_vec_coeff_e(2, je, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 2), jk, p_patch % edges % quad_blk(je, jb, 2)) + p_int % rbf_vec_coeff_e(3, je, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 3), jk, p_patch % edges % quad_blk(je, jb, 3)) + p_int % rbf_vec_coeff_e(4, je, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 4), jk, p_patch % edges % quad_blk(je, jb, 4))
            END DO
          END DO
        ELSE
          DO je = i_startidx, i_endidx
            DO jk = 1, nlev
              z_vn_avg(je, jk) = p_int % e_flx_avg(je, 1, jb) * p_nh % prog(nnew) % vn(je, jk, jb) + p_int % e_flx_avg(je, 2, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 1), jk, p_patch % edges % quad_blk(je, jb, 1)) + p_int % e_flx_avg(je, 3, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 2), jk, p_patch % edges % quad_blk(je, jb, 2)) + p_int % e_flx_avg(je, 4, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 3), jk, p_patch % edges % quad_blk(je, jb, 3)) + p_int % e_flx_avg(je, 5, jb) * p_nh % prog(nnew) % vn(p_patch % edges % quad_idx(je, jb, 4), jk, p_patch % edges % quad_blk(je, jb, 4))
            END DO
          END DO
        END IF
        DO jk = 1, nlev
          DO je = i_startidx, i_endidx
            p_nh % diag % mass_fl_e(je, jk, jb) = z_rho_e(je, jk, jb) * z_vn_avg(je, jk) * p_nh % metrics % ddqz_z_full_e(je, jk, jb)
            z_theta_v_fl_e(je, jk, jb) = p_nh % diag % mass_fl_e(je, jk, jb) * z_theta_v_e(je, jk, jb)
          END DO
        END DO
        IF (lsave_mflx .AND. istep == 2) THEN
          DO je = i_startidx, i_endidx
            IF (p_patch % edges % refin_ctrl(je, jb) <= - 4 .AND. p_patch % edges % refin_ctrl(je, jb) >= - 6) THEN
              DO jk = 1, nlev
                p_nh % diag % mass_fl_e_sv(je, jk, jb) = p_nh % diag % mass_fl_e(je, jk, jb)
              END DO
            END IF
          END DO
        END IF
        IF (lprep_adv .AND. istep == 2) THEN
          IF (lclean_mflx) THEN
            DO jk = 1, nlev
              DO je = i_startidx, i_endidx
                prep_adv % vn_traj(je, jk, jb) = 0.0D0
                prep_adv % mass_flx_me(je, jk, jb) = 0.0D0
              END DO
            END DO
          END IF
          DO jk = 1, nlev
            DO je = i_startidx, i_endidx
              prep_adv % vn_traj(je, jk, jb) = prep_adv % vn_traj(je, jk, jb) + r_nsubsteps * z_vn_avg(je, jk)
              prep_adv % mass_flx_me(je, jk, jb) = prep_adv % mass_flx_me(je, jk, jb) + r_nsubsteps * p_nh % diag % mass_fl_e(je, jk, jb)
            END DO
          END DO
        END IF
        IF (istep == 1 .OR. itime_scheme >= 5) THEN
          DO jk = nflatlev(jg), nlev
            DO je = i_startidx, i_endidx
              z_w_concorr_me(je, jk, jb) = p_nh % prog(nnew) % vn(je, jk, jb) * p_nh % metrics % ddxn_z_full(je, jk, jb) + p_nh % diag % vt(je, jk, jb) * p_nh % metrics % ddxt_z_full(je, jk, jb)
            END DO
          END DO
        END IF
        IF (istep == 1) THEN
          DO jk = 2, nlev
            DO je = i_startidx, i_endidx
              p_nh % diag % vn_ie(je, jk, jb) = p_nh % metrics % wgtfac_e(je, jk, jb) * p_nh % prog(nnew) % vn(je, jk, jb) + (1.0D0 - p_nh % metrics % wgtfac_e(je, jk, jb)) * p_nh % prog(nnew) % vn(je, jk - 1, jb)
              z_vt_ie(je, jk, jb) = p_nh % metrics % wgtfac_e(je, jk, jb) * p_nh % diag % vt(je, jk, jb) + (1.0D0 - p_nh % metrics % wgtfac_e(je, jk, jb)) * p_nh % diag % vt(je, jk - 1, jb)
              z_kin_hor_e(je, jk, jb) = 0.5D0 * (p_nh % prog(nnew) % vn(je, jk, jb) ** 2 + p_nh % diag % vt(je, jk, jb) ** 2)
            END DO
          END DO
          IF (.NOT. l_vert_nested) THEN
            DO je = i_startidx, i_endidx
              p_nh % diag % vn_ie(je, 1, jb) = p_nh % prog(nnew) % vn(je, 1, jb)
              z_vt_ie(je, 1, jb) = p_nh % diag % vt(je, 1, jb)
              z_kin_hor_e(je, 1, jb) = 0.5D0 * (p_nh % prog(nnew) % vn(je, 1, jb) ** 2 + p_nh % diag % vt(je, 1, jb) ** 2)
              p_nh % diag % vn_ie(je, nlevp1, jb) = p_nh % metrics % wgtfacq_e(je, 1, jb) * p_nh % prog(nnew) % vn(je, nlev, jb) + p_nh % metrics % wgtfacq_e(je, 2, jb) * p_nh % prog(nnew) % vn(je, nlev - 1, jb) + p_nh % metrics % wgtfacq_e(je, 3, jb) * p_nh % prog(nnew) % vn(je, nlev - 2, jb)
            END DO
          ELSE
            DO je = i_startidx, i_endidx
              p_nh % diag % vn_ie(je, 1, jb) = p_nh % diag % vn_ie_ubc(je, 1, jb) + dt_linintp_ubc_nnew * p_nh % diag % vn_ie_ubc(je, 2, jb)
              z_vt_ie(je, 1, jb) = p_nh % diag % vt(je, 1, jb)
              z_kin_hor_e(je, 1, jb) = 0.5D0 * (p_nh % prog(nnew) % vn(je, 1, jb) ** 2 + p_nh % diag % vt(je, 1, jb) ** 2)
              p_nh % diag % vn_ie(je, nlevp1, jb) = p_nh % metrics % wgtfacq_e(je, 1, jb) * p_nh % prog(nnew) % vn(je, nlev, jb) + p_nh % metrics % wgtfacq_e(je, 2, jb) * p_nh % prog(nnew) % vn(je, nlev - 1, jb) + p_nh % metrics % wgtfacq_e(je, 3, jb) * p_nh % prog(nnew) % vn(je, nlev - 2, jb)
            END DO
          END IF
        END IF
      END DO
      IF (jg > 1 .AND. grf_intmethod_e == 6) THEN
        DO ic = 1, p_nh % metrics % bdy_mflx_e_dim
          je = p_nh % metrics % bdy_mflx_e_idx(ic)
          jb = p_nh % metrics % bdy_mflx_e_blk(ic)
          IF (lprep_adv .AND. istep == 2) THEN
            DO jk = 1, nlev
              prep_adv % mass_flx_me(je, jk, jb) = prep_adv % mass_flx_me(je, jk, jb) - r_nsubsteps * p_nh % diag % mass_fl_e(je, jk, jb)
              prep_adv % vn_traj(je, jk, jb) = prep_adv % vn_traj(je, jk, jb) - r_nsubsteps * p_nh % diag % mass_fl_e(je, jk, jb) / (z_rho_e(je, jk, jb) * p_nh % metrics % ddqz_z_full_e(je, jk, jb))
            END DO
          END IF
          DO jk = 1, nlev
            p_nh % diag % mass_fl_e(je, jk, jb) = p_nh % diag % grf_bdy_mflx(jk, ic, 1) + REAL(jstep, 8) * dtime * p_nh % diag % grf_bdy_mflx(jk, ic, 2)
            z_theta_v_fl_e(je, jk, jb) = p_nh % diag % mass_fl_e(je, jk, jb) * z_theta_v_e(je, jk, jb)
          END DO
          IF (lprep_adv .AND. istep == 2) THEN
            DO jk = 1, nlev
              prep_adv % mass_flx_me(je, jk, jb) = prep_adv % mass_flx_me(je, jk, jb) + r_nsubsteps * p_nh % diag % mass_fl_e(je, jk, jb)
              prep_adv % vn_traj(je, jk, jb) = prep_adv % vn_traj(je, jk, jb) + r_nsubsteps * p_nh % diag % mass_fl_e(je, jk, jb) / (z_rho_e(je, jk, jb) * p_nh % metrics % ddqz_z_full_e(je, jk, jb))
            END DO
          END IF
        END DO
      END IF
      IF (istep == 1 .OR. itime_scheme >= 5) THEN
        rl_start = 3
        rl_end = (- 5)
        i_startblk = p_patch % cells % start_block(3)
        i_endblk = p_patch % cells % end_block((- 5))
        DO jb = i_startblk, i_endblk
          CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, i_startidx, i_endidx, rl_start, rl_end)
          DO jc = i_startidx, i_endidx
            DO jk = nflatlev(jg), nlev
              z_w_concorr_mc(jc, jk) = p_int % e_bln_c_s(jc, 1, jb) * z_w_concorr_me(p_patch % cells % edge_idx(jc, jb, 1), jk, p_patch % cells % edge_blk(jc, jb, 1)) + p_int % e_bln_c_s(jc, 2, jb) * z_w_concorr_me(p_patch % cells % edge_idx(jc, jb, 2), jk, p_patch % cells % edge_blk(jc, jb, 2)) + p_int % e_bln_c_s(jc, 3, jb) * z_w_concorr_me(p_patch % cells % edge_idx(jc, jb, 3), jk, p_patch % cells % edge_blk(jc, jb, 3))
            END DO
          END DO
          DO jk = nflatlev(jg) + 1, nlev
            DO jc = i_startidx, i_endidx
              p_nh % diag % w_concorr_c(jc, jk, jb) = p_nh % metrics % wgtfac_c(jc, jk, jb) * z_w_concorr_mc(jc, jk) + (1.0D0 - p_nh % metrics % wgtfac_c(jc, jk, jb)) * z_w_concorr_mc(jc, jk - 1)
            END DO
          END DO
          DO jc = i_startidx, i_endidx
            p_nh % diag % w_concorr_c(jc, nlevp1, jb) = p_nh % metrics % wgtfacq_c(jc, 1, jb) * z_w_concorr_mc(jc, nlev) + p_nh % metrics % wgtfacq_c(jc, 2, jb) * z_w_concorr_mc(jc, nlev - 1) + p_nh % metrics % wgtfacq_c(jc, 3, jb) * z_w_concorr_mc(jc, nlev - 2)
          END DO
        END DO
      END IF
      IF (timers_level > 5) THEN
        CALL timer_stop(timer_solve_nh_edgecomp)
        CALL timer_start(timer_solve_nh_vimpl)
      END IF
      rl_start = 5
      rl_end = (- 4)
      i_startblk = p_patch % cells % start_block(rl_start)
      i_endblk = p_patch % cells % end_block(rl_end)
      IF (l_vert_nested) THEN
        jk_start = 2
      ELSE
        jk_start = 1
      END IF
      DO jb = i_startblk, i_endblk
        CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, i_startidx, i_endidx, rl_start, rl_end)
        DO jc = i_startidx, i_endidx
          DO jk = 1, nlev
            z_flxdiv_mass(jc, jk) = p_nh % metrics % deepatmo_divh_mc(jk) * (p_nh % diag % mass_fl_e(p_patch % cells % edge_idx(jc, jb, 1), jk, p_patch % cells % edge_blk(jc, jb, 1)) * p_int % geofac_div(jc, 1, jb) + p_nh % diag % mass_fl_e(p_patch % cells % edge_idx(jc, jb, 2), jk, p_patch % cells % edge_blk(jc, jb, 2)) * p_int % geofac_div(jc, 2, jb) + p_nh % diag % mass_fl_e(p_patch % cells % edge_idx(jc, jb, 3), jk, p_patch % cells % edge_blk(jc, jb, 3)) * p_int % geofac_div(jc, 3, jb))
            z_flxdiv_theta(jc, jk) = p_nh % metrics % deepatmo_divh_mc(jk) * (z_theta_v_fl_e(p_patch % cells % edge_idx(jc, jb, 1), jk, p_patch % cells % edge_blk(jc, jb, 1)) * p_int % geofac_div(jc, 1, jb) + z_theta_v_fl_e(p_patch % cells % edge_idx(jc, jb, 2), jk, p_patch % cells % edge_blk(jc, jb, 2)) * p_int % geofac_div(jc, 2, jb) + z_theta_v_fl_e(p_patch % cells % edge_idx(jc, jb, 3), jk, p_patch % cells % edge_blk(jc, jb, 3)) * p_int % geofac_div(jc, 3, jb))
          END DO
        END DO
        IF (l_vert_nested .AND. istep == 1) THEN
          DO jc = i_startidx, i_endidx
            p_nh % diag % theta_v_ic(jc, 1, jb) = p_nh % diag % theta_v_ic_ubc(jc, jb, 1) + dt_linintp_ubc * p_nh % diag % theta_v_ic_ubc(jc, jb, 2)
            p_nh % diag % rho_ic(jc, 1, jb) = p_nh % diag % rho_ic_ubc(jc, jb, 1) + dt_linintp_ubc * p_nh % diag % rho_ic_ubc(jc, jb, 2)
            z_mflx_top(jc, jb) = p_nh % diag % mflx_ic_ubc(jc, jb, 1) + dt_linintp_ubc * p_nh % diag % mflx_ic_ubc(jc, jb, 2)
          END DO
        END IF
        IF (istep == 2) THEN
          DO jk = 2, nlev
            DO jc = i_startidx, i_endidx
              z_w_expl(jc, jk) = p_nh % prog(nnow) % w(jc, jk, jb) + dtime * (wgt_nnow_vel * p_nh % diag % ddt_w_adv_pc(jc, jk, jb, nnow) + wgt_nnew_vel * p_nh % diag % ddt_w_adv_pc(jc, jk, jb, nnew) - 1004.64D0 * z_th_ddz_exner_c(jc, jk, jb))
              z_contr_w_fl_l(jc, jk) = p_nh % diag % rho_ic(jc, jk, jb) * (p_nh % metrics % vwind_expl_wgt(jc, jb) * p_nh % prog(nnow) % w(jc, jk, jb) - p_nh % diag % w_concorr_c(jc, jk, jb))
            END DO
          END DO
        ELSE
          DO jk = 2, nlev
            DO jc = i_startidx, i_endidx
              z_w_expl(jc, jk) = p_nh % prog(nnow) % w(jc, jk, jb) + dtime * (p_nh % diag % ddt_w_adv_pc(jc, jk, jb, nnow) - 1004.64D0 * z_th_ddz_exner_c(jc, jk, jb))
              z_contr_w_fl_l(jc, jk) = p_nh % diag % rho_ic(jc, jk, jb) * (p_nh % metrics % vwind_expl_wgt(jc, jb) * p_nh % prog(nnow) % w(jc, jk, jb) - p_nh % diag % w_concorr_c(jc, jk, jb))
            END DO
          END DO
        END IF
        DO jk = 1, nlev
          DO jc = i_startidx, i_endidx
            z_beta(jc, jk) = dtime * 287.04D0 * p_nh % prog(nnow) % exner(jc, jk, jb) / (717.5999999999999D0 * p_nh % prog(nnow) % rho(jc, jk, jb) * p_nh % prog(nnow) % theta_v(jc, jk, jb)) * p_nh % metrics % inv_ddqz_z_full(jc, jk, jb)
            z_alpha(jc, jk) = p_nh % metrics % vwind_impl_wgt(jc, jb) * p_nh % diag % theta_v_ic(jc, jk, jb) * p_nh % diag % rho_ic(jc, jk, jb)
          END DO
        END DO
        DO jc = i_startidx, i_endidx
          z_alpha(jc, nlevp1) = 0.0D0
          z_q(jc, 1) = 0.0D0
        END DO
        IF (.NOT. l_vert_nested) THEN
          DO jc = i_startidx, i_endidx
            p_nh % prog(nnew) % w(jc, 1, jb) = 0.0D0
            z_contr_w_fl_l(jc, 1) = 0.0D0
          END DO
        ELSE
          DO jc = i_startidx, i_endidx
            p_nh % prog(nnew) % w(jc, 1, jb) = p_nh % diag % w_ubc(jc, jb, 1) + dt_linintp_ubc_nnew * p_nh % diag % w_ubc(jc, jb, 2)
            z_contr_w_fl_l(jc, 1) = z_mflx_top(jc, jb) * p_nh % metrics % vwind_expl_wgt(jc, jb)
          END DO
        END IF
        DO jc = i_startidx, i_endidx
          p_nh % prog(nnew) % w(jc, nlevp1, jb) = p_nh % diag % w_concorr_c(jc, nlevp1, jb)
          z_contr_w_fl_l(jc, nlevp1) = 0.0D0
        END DO
        DO jc = i_startidx, i_endidx
          z_rho_expl(jc, 1) = p_nh % prog(nnow) % rho(jc, 1, jb) - dtime * p_nh % metrics % inv_ddqz_z_full(jc, 1, jb) * (z_flxdiv_mass(jc, 1) + z_contr_w_fl_l(jc, 1) * p_nh % metrics % deepatmo_divzu_mc(1) - z_contr_w_fl_l(jc, 2) * p_nh % metrics % deepatmo_divzl_mc(1))
          z_exner_expl(jc, 1) = p_nh % diag % exner_pr(jc, 1, jb) - z_beta(jc, 1) * (z_flxdiv_theta(jc, 1) + p_nh % diag % theta_v_ic(jc, 1, jb) * z_contr_w_fl_l(jc, 1) * p_nh % metrics % deepatmo_divzu_mc(1) - p_nh % diag % theta_v_ic(jc, 2, jb) * z_contr_w_fl_l(jc, 2) * p_nh % metrics % deepatmo_divzl_mc(1)) + dtime * p_nh % diag % ddt_exner_phy(jc, 1, jb)
        END DO
        DO jk = 2, nlev
          DO jc = i_startidx, i_endidx
            z_rho_expl(jc, jk) = p_nh % prog(nnow) % rho(jc, jk, jb) - dtime * p_nh % metrics % inv_ddqz_z_full(jc, jk, jb) * (z_flxdiv_mass(jc, jk) + z_contr_w_fl_l(jc, jk) * p_nh % metrics % deepatmo_divzu_mc(jk) - z_contr_w_fl_l(jc, jk + 1) * p_nh % metrics % deepatmo_divzl_mc(jk))
            z_exner_expl(jc, jk) = p_nh % diag % exner_pr(jc, jk, jb) - z_beta(jc, jk) * (z_flxdiv_theta(jc, jk) + p_nh % diag % theta_v_ic(jc, jk, jb) * z_contr_w_fl_l(jc, jk) * p_nh % metrics % deepatmo_divzu_mc(jk) - p_nh % diag % theta_v_ic(jc, jk + 1, jb) * z_contr_w_fl_l(jc, jk + 1) * p_nh % metrics % deepatmo_divzl_mc(jk)) + dtime * p_nh % diag % ddt_exner_phy(jc, jk, jb)
          END DO
        END DO
        IF (is_iau_active) THEN
          DO jk = 1, nlev
            DO jc = i_startidx, i_endidx
              z_rho_expl(jc, jk) = z_rho_expl(jc, jk) + iau_wgt_dyn * p_nh % diag % rho_incr(jc, jk, jb)
              z_exner_expl(jc, jk) = z_exner_expl(jc, jk) + iau_wgt_dyn * p_nh % diag % exner_incr(jc, jk, jb)
            END DO
          END DO
        END IF
        DO jk = 2, nlev
          DO jc = i_startidx, i_endidx
            z_gamma = dtime * 1004.64D0 * p_nh % metrics % vwind_impl_wgt(jc, jb) * p_nh % diag % theta_v_ic(jc, jk, jb) / p_nh % metrics % ddqz_z_half(jc, jk, jb)
            z_a = - z_gamma * z_beta(jc, jk - 1) * z_alpha(jc, jk - 1) * p_nh % metrics % deepatmo_divzu_mc(jk - 1)
            z_c = - z_gamma * z_beta(jc, jk) * z_alpha(jc, jk + 1) * p_nh % metrics % deepatmo_divzl_mc(jk)
            z_b = 1.0D0 + z_gamma * z_alpha(jc, jk) * (z_beta(jc, jk - 1) * p_nh % metrics % deepatmo_divzl_mc(jk - 1) + z_beta(jc, jk) * p_nh % metrics % deepatmo_divzu_mc(jk))
            z_g = 1.0D0 / (z_b + z_a * z_q(jc, jk - 1))
            z_q(jc, jk) = - z_c * z_g
            p_nh % prog(nnew) % w(jc, jk, jb) = z_w_expl(jc, jk) - z_gamma * (z_exner_expl(jc, jk - 1) - z_exner_expl(jc, jk))
            p_nh % prog(nnew) % w(jc, jk, jb) = (p_nh % prog(nnew) % w(jc, jk, jb) - z_a * p_nh % prog(nnew) % w(jc, jk - 1, jb)) * z_g
          END DO
        END DO
        DO jk = nlev - 1, 2, - 1
          DO jc = i_startidx, i_endidx
            p_nh % prog(nnew) % w(jc, jk, jb) = p_nh % prog(nnew) % w(jc, jk, jb) + p_nh % prog(nnew) % w(jc, jk + 1, jb) * z_q(jc, jk)
          END DO
        END DO
        IF (rayleigh_type == 2) THEN
          DO jk = 2, nrdmax(jg)
            DO jc = i_startidx, i_endidx
              p_nh % prog(nnew) % w(jc, jk, jb) = z_raylfac(jk) * p_nh % prog(nnew) % w(jc, jk, jb) + (1.0D0 - z_raylfac(jk)) * p_nh % prog(nnew) % w(jc, 1, jb)
            END DO
          END DO
        ELSE IF (rayleigh_type == 1) THEN
          DO jk = 2, nrdmax(jg)
            DO jc = i_startidx, i_endidx
              p_nh % prog(nnew) % w(jc, jk, jb) = p_nh % prog(nnew) % w(jc, jk, jb) - dtime * p_nh % metrics % rayleigh_w(jk) * (p_nh % prog(nnew) % w(jc, jk, jb) - p_nh % ref % w_ref(jc, jk, jb))
            END DO
          END DO
        END IF
        DO jk = jk_start, nlev
          DO jc = i_startidx, i_endidx
            p_nh % prog(nnew) % rho(jc, jk, jb) = z_rho_expl(jc, jk) - p_nh % metrics % vwind_impl_wgt(jc, jb) * dtime * p_nh % metrics % inv_ddqz_z_full(jc, jk, jb) * (p_nh % diag % rho_ic(jc, jk, jb) * p_nh % prog(nnew) % w(jc, jk, jb) * p_nh % metrics % deepatmo_divzu_mc(jk) - p_nh % diag % rho_ic(jc, jk + 1, jb) * p_nh % prog(nnew) % w(jc, jk + 1, jb) * p_nh % metrics % deepatmo_divzl_mc(jk))
            p_nh % prog(nnew) % exner(jc, jk, jb) = z_exner_expl(jc, jk) + p_nh % metrics % exner_ref_mc(jc, jk, jb) - z_beta(jc, jk) * (z_alpha(jc, jk) * p_nh % prog(nnew) % w(jc, jk, jb) * p_nh % metrics % deepatmo_divzu_mc(jk) - z_alpha(jc, jk + 1) * p_nh % prog(nnew) % w(jc, jk + 1, jb) * p_nh % metrics % deepatmo_divzl_mc(jk))
            p_nh % prog(nnew) % theta_v(jc, jk, jb) = p_nh % prog(nnow) % rho(jc, jk, jb) * p_nh % prog(nnow) % theta_v(jc, jk, jb) * ((p_nh % prog(nnew) % exner(jc, jk, jb) / p_nh % prog(nnow) % exner(jc, jk, jb) - 1.0D0) * 2.4999999999999996D0 + 1.0D0) / p_nh % prog(nnew) % rho(jc, jk, jb)
          END DO
        END DO
        IF (l_vert_nested) THEN
          DO jc = i_startidx, i_endidx
            p_nh % prog(nnew) % rho(jc, 1, jb) = z_rho_expl(jc, 1) - p_nh % metrics % vwind_impl_wgt(jc, jb) * dtime * p_nh % metrics % inv_ddqz_z_full(jc, 1, jb) * (z_mflx_top(jc, jb) * p_nh % metrics % deepatmo_divzu_mc(1) - p_nh % diag % rho_ic(jc, 2, jb) * p_nh % prog(nnew) % w(jc, 2, jb) * p_nh % metrics % deepatmo_divzl_mc(1))
            p_nh % prog(nnew) % exner(jc, 1, jb) = z_exner_expl(jc, 1) + p_nh % metrics % exner_ref_mc(jc, 1, jb) - z_beta(jc, 1) * (p_nh % metrics % vwind_impl_wgt(jc, jb) * p_nh % diag % theta_v_ic(jc, 1, jb) * z_mflx_top(jc, jb) * p_nh % metrics % deepatmo_divzu_mc(1) - z_alpha(jc, 2) * p_nh % prog(nnew) % w(jc, 2, jb) * p_nh % metrics % deepatmo_divzl_mc(1))
            p_nh % prog(nnew) % theta_v(jc, 1, jb) = p_nh % prog(nnow) % rho(jc, 1, jb) * p_nh % prog(nnow) % theta_v(jc, 1, jb) * ((p_nh % prog(nnew) % exner(jc, 1, jb) / p_nh % prog(nnow) % exner(jc, 1, jb) - 1.0D0) * 2.4999999999999996D0 + 1.0D0) / p_nh % prog(nnew) % rho(jc, 1, jb)
          END DO
        END IF
        IF (istep == 1 .AND. divdamp_type >= 3) THEN
          DO jk = kstart_dd3d(jg), nlev
            DO jc = i_startidx, i_endidx
              z_dwdz_dd(jc, jk, jb) = p_nh % metrics % inv_ddqz_z_full(jc, jk, jb) * ((p_nh % prog(nnew) % w(jc, jk, jb) - p_nh % prog(nnew) % w(jc, jk + 1, jb)) - (p_nh % diag % w_concorr_c(jc, jk, jb) - p_nh % diag % w_concorr_c(jc, jk + 1, jb)))
            END DO
          END DO
        END IF
        IF (lprep_adv .AND. istep == 2) THEN
          IF (lclean_mflx) THEN
            DO jk = 1, nlev
              DO jc = i_startidx, i_endidx
                prep_adv % mass_flx_ic(jc, jk, jb) = 0.0D0
                prep_adv % vol_flx_ic(jc, jk, jb) = 0.0D0
              END DO
            END DO
          END IF
          DO jk = 2, nlev
            DO jc = i_startidx, i_endidx
              z_a = r_nsubsteps * (z_contr_w_fl_l(jc, jk) + p_nh % diag % rho_ic(jc, jk, jb) * p_nh % metrics % vwind_impl_wgt(jc, jb) * p_nh % prog(nnew) % w(jc, jk, jb))
              prep_adv % mass_flx_ic(jc, jk, jb) = prep_adv % mass_flx_ic(jc, jk, jb) + z_a
              prep_adv % vol_flx_ic(jc, jk, jb) = prep_adv % vol_flx_ic(jc, jk, jb) + z_a / p_nh % diag % rho_ic(jc, jk, jb)
            END DO
          END DO
          IF (l_vert_nested) THEN
            DO jc = i_startidx, i_endidx
              prep_adv % mass_flx_ic(jc, 1, jb) = prep_adv % mass_flx_ic(jc, 1, jb) + r_nsubsteps * z_mflx_top(jc, jb)
              prep_adv % vol_flx_ic(jc, 1, jb) = prep_adv % vol_flx_ic(jc, 1, jb) + r_nsubsteps * z_mflx_top(jc, jb) / p_nh % diag % rho_ic(jc, 1, jb)
            END DO
          END IF
        END IF
        IF (istep == 1 .AND. idyn_timestep == 1) THEN
          DO jk = kstart_moist(jg), nlev
            DO jc = i_startidx, i_endidx
              p_nh % diag % exner_dyn_incr(jc, jk, jb) = p_nh % prog(nnow) % exner(jc, jk, jb)
            END DO
          END DO
        ELSE IF (istep == 2 .AND. idyn_timestep == ndyn_substeps_var(jg)) THEN
          DO jk = kstart_moist(jg), nlev
            DO jc = i_startidx, i_endidx
              p_nh % diag % exner_dyn_incr(jc, jk, jb) = p_nh % prog(nnew) % exner(jc, jk, jb) - (p_nh % diag % exner_dyn_incr(jc, jk, jb) + ndyn_substeps_var(jg) * dtime * p_nh % diag % ddt_exner_phy(jc, jk, jb))
            END DO
          END DO
        END IF
        IF (istep == 2 .AND. l_child_vertnest) THEN
          DO jc = i_startidx, i_endidx
            p_nh % diag % w_int(jc, jb, idyn_timestep) = 0.5D0 * (p_nh % prog(nnow) % w(jc, nshift, jb) + p_nh % prog(nnew) % w(jc, nshift, jb))
            p_nh % diag % theta_v_ic_int(jc, jb, idyn_timestep) = p_nh % diag % theta_v_ic(jc, nshift, jb)
            p_nh % diag % rho_ic_int(jc, jb, idyn_timestep) = p_nh % diag % rho_ic(jc, nshift, jb)
            p_nh % diag % mflx_ic_int(jc, jb, idyn_timestep) = p_nh % diag % rho_ic(jc, nshift, jb) * (p_nh % metrics % vwind_expl_wgt(jc, jb) * p_nh % prog(nnow) % w(jc, nshift, jb) + p_nh % metrics % vwind_impl_wgt(jc, jb) * p_nh % prog(nnew) % w(jc, nshift, jb))
          END DO
        END IF
      END DO
      IF (l_limited_area .OR. jg > 1) THEN
        rl_start = 1
        rl_end = 4
        i_startblk = p_patch % cells % start_block(1)
        i_endblk = p_patch % cells % end_block(4)
        DO jb = i_startblk, i_endblk
          CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, i_startidx, i_endidx, rl_start, rl_end)
          IF (istep == 1 .AND. my_process_is_mpi_all_seq()) THEN
            DO jk = 1, nlev
              DO jc = i_startidx, i_endidx
                p_nh % prog(nnew) % rho(jc, jk, jb) = p_nh % prog(nnow) % rho(jc, jk, jb) + dtime * p_nh % diag % grf_tend_rho(jc, jk, jb)
                p_nh % prog(nnew) % theta_v(jc, jk, jb) = p_nh % prog(nnow) % theta_v(jc, jk, jb) + dtime * p_nh % diag % grf_tend_thv(jc, jk, jb)
                p_nh % prog(nnew) % exner(jc, jk, jb) = EXP(0.4000000000000001D0 * LOG(0.0028704000000000004D0 * p_nh % prog(nnew) % rho(jc, jk, jb) * p_nh % prog(nnew) % theta_v(jc, jk, jb)))
                p_nh % prog(nnew) % w(jc, jk, jb) = p_nh % prog(nnow) % w(jc, jk, jb) + dtime * p_nh % diag % grf_tend_w(jc, jk, jb)
              END DO
            END DO
            DO jc = i_startidx, i_endidx
              p_nh % prog(nnew) % w(jc, nlevp1, jb) = p_nh % prog(nnow) % w(jc, nlevp1, jb) + dtime * p_nh % diag % grf_tend_w(jc, nlevp1, jb)
            END DO
          ELSE IF (istep == 1) THEN
            DO jk = 1, nlev
              DO jc = i_startidx, i_endidx
                p_nh % prog(nnew) % rho(jc, jk, jb) = p_nh % prog(nnow) % rho(jc, jk, jb) + dtime * p_nh % diag % grf_tend_rho(jc, jk, jb)
                p_nh % prog(nnew) % exner(jc, jk, jb) = p_nh % prog(nnow) % theta_v(jc, jk, jb) + dtime * p_nh % diag % grf_tend_thv(jc, jk, jb)
                p_nh % prog(nnew) % w(jc, jk, jb) = p_nh % prog(nnow) % w(jc, jk, jb) + dtime * p_nh % diag % grf_tend_w(jc, jk, jb)
              END DO
            END DO
            DO jc = i_startidx, i_endidx
              p_nh % prog(nnew) % w(jc, nlevp1, jb) = p_nh % prog(nnow) % w(jc, nlevp1, jb) + dtime * p_nh % diag % grf_tend_w(jc, nlevp1, jb)
            END DO
          END IF
          IF (istep == 1 .AND. divdamp_type >= 3) THEN
            DO jk = kstart_dd3d(jg), nlev
              DO jc = i_startidx, i_endidx
                z_dwdz_dd(jc, jk, jb) = p_nh % metrics % inv_ddqz_z_full(jc, jk, jb) * ((p_nh % prog(nnew) % w(jc, jk, jb) - p_nh % prog(nnew) % w(jc, jk + 1, jb)) - (p_nh % diag % w_concorr_c(jc, jk, jb) - p_nh % diag % w_concorr_c(jc, jk + 1, jb)))
              END DO
            END DO
          END IF
          IF (lprep_adv .AND. istep == 2) THEN
            IF (lclean_mflx) THEN
              prep_adv % mass_flx_ic(i_startidx : i_endidx, :, jb) = 0.0D0
            END IF
            DO jk = 2, nlev
              DO jc = i_startidx, i_endidx
                prep_adv % mass_flx_ic(jc, jk, jb) = prep_adv % mass_flx_ic(jc, jk, jb) + r_nsubsteps * p_nh % diag % rho_ic(jc, jk, jb) * (p_nh % metrics % vwind_expl_wgt(jc, jb) * p_nh % prog(nnow) % w(jc, jk, jb) + p_nh % metrics % vwind_impl_wgt(jc, jb) * p_nh % prog(nnew) % w(jc, jk, jb) - p_nh % diag % w_concorr_c(jc, jk, jb))
              END DO
            END DO
            IF (l_vert_nested) THEN
              DO jc = i_startidx, i_endidx
                prep_adv % mass_flx_ic(jc, 1, jb) = prep_adv % mass_flx_ic(jc, 1, jb) + r_nsubsteps * (p_nh % diag % mflx_ic_ubc(jc, jb, 1) + dt_linintp_ubc * p_nh % diag % mflx_ic_ubc(jc, jb, 2))
              END DO
            END IF
          END IF
        END DO
      END IF
      IF (timers_level > 5) THEN
        CALL timer_stop(timer_solve_nh_vimpl)
        CALL timer_start(timer_solve_nh_exch)
      END IF
      IF (istep == 1) THEN
        IF (divdamp_type >= 3) THEN
          CALL sync_patch_array_mult_f3din_dp_deconiface_138(1, p_patch, 2, lacc = .TRUE., f3din1 = p_nh % prog(nnew) % w, f3din2 = z_dwdz_dd, opt_varname = "w_nnew and z_dwdz_dd")
        ELSE
          CALL sync_patch_array_3d_dp_deconiface_139(1, p_patch, p_nh % prog(nnew) % w, lacc = .TRUE., opt_varname = "w_nnew")
        END IF
      ELSE
        CALL sync_patch_array_mult_f3din_dp_deconiface_140(1, p_patch, 3, lacc = .TRUE., f3din1 = p_nh % prog(nnew) % rho, f3din2 = p_nh % prog(nnew) % exner, f3din3 = p_nh % prog(nnew) % w, opt_varname = "rho, exner, w_nnew")
      END IF
      IF (timers_level > 5) CALL timer_stop(timer_solve_nh_exch)
    END DO
    IF (.NOT. my_process_is_mpi_all_seq()) THEN
      IF (l_limited_area .OR. jg > 1) THEN
        DO ic = 1, p_nh % metrics % bdy_halo_c_dim
          jb = p_nh % metrics % bdy_halo_c_blk(ic)
          jc = p_nh % metrics % bdy_halo_c_idx(ic)
          DO jk = 1, nlev
            p_nh % prog(nnew) % theta_v(jc, jk, jb) = p_nh % prog(nnew) % exner(jc, jk, jb)
            p_nh % prog(nnew) % exner(jc, jk, jb) = EXP(0.4000000000000001D0 * LOG(0.0028704000000000004D0 * p_nh % prog(nnew) % rho(jc, jk, jb) * p_nh % prog(nnew) % theta_v(jc, jk, jb)))
          END DO
        END DO
        rl_start = 1
        rl_end = 4
        i_startblk = p_patch % cells % start_block(1)
        i_endblk = p_patch % cells % end_block(4)
        DO jb = i_startblk, i_endblk
          CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, i_startidx, i_endidx, rl_start, rl_end)
          DO jk = 1, nlev
            DO jc = i_startidx, i_endidx
              p_nh % prog(nnew) % theta_v(jc, jk, jb) = p_nh % prog(nnew) % exner(jc, jk, jb)
              p_nh % prog(nnew) % exner(jc, jk, jb) = EXP(0.4000000000000001D0 * LOG(0.0028704000000000004D0 * p_nh % prog(nnew) % rho(jc, jk, jb) * p_nh % prog(nnew) % theta_v(jc, jk, jb)))
            END DO
          END DO
        END DO
      END IF
      rl_start = (- 5)
      rl_end = (- 8)
      i_startblk = p_patch % cells % start_block((- 5))
      i_endblk = p_patch % cells % end_block((- 8))
      DO jb = i_startblk, i_endblk
        CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, i_startidx, i_endidx, rl_start, rl_end)
        DO jc = i_startidx, i_endidx
          IF (p_nh % metrics % mask_prog_halo_c(jc, jb)) THEN
            DO jk = 1, nlev
              p_nh % prog(nnew) % theta_v(jc, jk, jb) = p_nh % prog(nnow) % rho(jc, jk, jb) * p_nh % prog(nnow) % theta_v(jc, jk, jb) * ((p_nh % prog(nnew) % exner(jc, jk, jb) / p_nh % prog(nnow) % exner(jc, jk, jb) - 1.0D0) * 2.4999999999999996D0 + 1.0D0) / p_nh % prog(nnew) % rho(jc, jk, jb)
            END DO
          END IF
        END DO
      END DO
    END IF
    IF (ltimer) CALL timer_stop(timer_solve_nh)
  END SUBROUTINE solve_nh
END MODULE mo_solve_nonhydro