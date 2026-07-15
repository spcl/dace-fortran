MODULE mo_dbg_nml
  IMPLICIT NONE
  INTEGER :: idbg_mxmn = 0
  NAMELIST /dbg_index_nml/ idbg_mxmn
  CONTAINS
END MODULE mo_dbg_nml
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
  INTEGER :: nold(10)
  INTEGER :: nnew(10)
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
  SUBROUTINE message(name, text, all_print)
    CHARACTER(LEN = *), INTENT(IN) :: name
    CHARACTER(LEN = *), INTENT(IN) :: text
    LOGICAL, INTENT(IN), OPTIONAL :: all_print
  END SUBROUTINE message
  SUBROUTINE warning(name, text)
    CHARACTER(LEN = *), INTENT(IN) :: name
    CHARACTER(LEN = *), INTENT(IN) :: text
  END SUBROUTINE warning
END MODULE mo_exception
MODULE mo_ext_data_types
  IMPLICIT NONE
  TYPE :: t_external_data
  END TYPE t_external_data
END MODULE mo_ext_data_types
MODULE mo_grid_config
  IMPLICIT NONE
  INTEGER :: n_dom
  CONTAINS
END MODULE mo_grid_config
MODULE mo_impl_constants
  IMPLICIT NONE
  INTEGER, PARAMETER :: max_char_length = 1024
END MODULE mo_impl_constants
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
MODULE mo_master_config
  IMPLICIT NONE
  CONTAINS
  LOGICAL FUNCTION isrestart()
    isrestart = .FALSE.
  END FUNCTION isrestart
  LOGICAL FUNCTION isinitfromrestart()
    isinitfromrestart = .FALSE.
  END FUNCTION isinitfromrestart
END MODULE mo_master_config
MODULE mo_math_types
  USE iso_c_binding, ONLY: c_int64_t
  IMPLICIT NONE
  TYPE :: t_cartesian_coordinates
    REAL(KIND = 8) :: x(3)
  END TYPE t_cartesian_coordinates
  TYPE :: t_geographical_coordinates
    REAL(KIND = 8) :: lon
    REAL(KIND = 8) :: lat
  END TYPE t_geographical_coordinates
  CONTAINS
END MODULE mo_math_types
MODULE mo_ocean_initialization
  IMPLICIT NONE
  CONTAINS
  FUNCTION is_initial_timestep(timestep)
    USE mo_master_config, ONLY: isinitfromrestart, isrestart
    INTEGER :: timestep
    LOGICAL :: is_initial_timestep
    IF (timestep == 1 .AND. .NOT. (isrestart() .OR. isinitfromrestart())) THEN
      is_initial_timestep = .TRUE.
    ELSE
      is_initial_timestep = .FALSE.
    END IF
  END FUNCTION is_initial_timestep
END MODULE mo_ocean_initialization
MODULE mo_ocean_nml
  IMPLICIT NONE
  INTEGER :: n_zlev
  INTEGER :: i_bc_veloc_lateral = 0
  INTEGER :: i_bc_veloc_top = 1
  INTEGER :: i_bc_veloc_bot = 1
  LOGICAL :: use_ssh_in_momentum_eq = .TRUE.
  INTEGER :: nonlinearcoriolis_type = 200
  LOGICAL :: l_anticipated_vorticity = .FALSE.
  INTEGER :: iswm_oce = 0
  REAL(KIND = 8) :: ab_const = 0.1D0
  REAL(KIND = 8) :: ab_beta = 0.6D0
  REAL(KIND = 8) :: ab_gam = 0.6D0
  REAL(KIND = 8) :: solver_tolerance = 1D-14
  REAL(KIND = 8) :: massmatrix_solver_tolerance = 1D-11
  INTEGER :: solver_max_restart_iterations = 100
  INTEGER :: solver_max_iter_per_restart = 200
  INTEGER :: solver_max_iter_per_restart_sp = 200
  REAL(KIND = 4) :: solver_tolerance_sp = 1E-11
  LOGICAL :: use_absolute_solver_tolerance = .TRUE.
  INTEGER :: select_transfer = 0
  INTEGER :: select_solver = 4
  INTEGER :: solver_firstguess = 2
  LOGICAL :: l_solver_compare = .FALSE.
  INTEGER :: solver_comp_nsteps = 100
  REAL(KIND = 8) :: solver_tolerance_comp = 1D-30
  LOGICAL :: l_lhs_direct = .FALSE.
  INTEGER :: select_lhs = 1
  INTEGER :: fast_performance_level = 50
  INTEGER :: mass_matrix_inversion_type = 0
  INTEGER :: velocitydiffusion_order = 1
  INTEGER :: laplacian_form = 1
  LOGICAL :: l_rigid_lid = .FALSE.
  LOGICAL :: l_edge_based = .TRUE.
  INTEGER :: horizonatlvelocity_verticaladvection_form = 1
  LOGICAL :: createsolvermatrix = .FALSE.
  NAMELIST /ocean_dynamics_nml/ ab_beta, ab_const, ab_gam, i_bc_veloc_bot, i_bc_veloc_lateral, i_bc_veloc_top, use_ssh_in_momentum_eq, iswm_oce, l_rigid_lid, l_edge_based, n_zlev, select_solver, use_absolute_solver_tolerance, solver_max_iter_per_restart, solver_max_restart_iterations, solver_tolerance, solver_max_iter_per_restart_sp, solver_tolerance_sp, select_lhs, select_transfer, l_lhs_direct, l_solver_compare, solver_tolerance_comp, solver_comp_nsteps, massmatrix_solver_tolerance, fast_performance_level, mass_matrix_inversion_type, nonlinearcoriolis_type, horizonatlvelocity_verticaladvection_form, solver_firstguess, createsolvermatrix
  INTEGER :: ppscheme_type = 4
  INTEGER :: vert_mix_type = 1
  REAL(KIND = 8) :: verticalviscosity_timeweight = 0.0D0
  REAL(KIND = 8) :: velocity_richardsoncoeff = 0.005D0
  REAL(KIND = 8) :: biharmonicvort_weight = 1.0D0
  REAL(KIND = 8) :: biharmonicdiv_weight = 1.0D0
  REAL(KIND = 8) :: harmonicvort_weight = 1.0D0
  REAL(KIND = 8) :: harmonicdiv_weight = 1.0D0
  NAMELIST /ocean_horizontal_diffusion_nml/ laplacian_form, velocitydiffusion_order, harmonicvort_weight, harmonicdiv_weight, biharmonicvort_weight, biharmonicdiv_weight
  NAMELIST /ocean_vertical_diffusion_nml/ ppscheme_type, vert_mix_type, verticalviscosity_timeweight, velocity_richardsoncoeff
  REAL(KIND = 8) :: oceanreferencedensity = 1025.022D0
  REAL(KIND = 8) :: oceanreferencedensity_inv
  NAMELIST /ocean_physics_nml/ oceanreferencedensity
  INTEGER :: iforc_oce = 0
  INTEGER :: forcing_windstress_u_type = 0
  INTEGER :: forcing_smooth_steps = 1
  REAL(KIND = 8) :: forcing_windstress_weight = 1.0D0
  NAMELIST /ocean_forcing_nml/ forcing_windstress_u_type, iforc_oce, forcing_smooth_steps, forcing_windstress_weight
  CONTAINS
END MODULE mo_ocean_nml
MODULE mo_ocean_physics_types
  IMPLICIT NONE
  TYPE :: t_ho_params
    REAL(KIND = 8), POINTER :: harmonicviscosity_coeff(:, :, :), biharmonicviscosity_coeff(:, :, :)
    REAL(KIND = 8), POINTER :: a_veloc_v(:, :, :)
    REAL(KIND = 8) :: a_veloc_v_back
    REAL(KIND = 8), POINTER, DIMENSION(:, :, :) :: velocity_windmixing
    REAL(KIND = 8) :: bottom_drag_coeff
  END TYPE t_ho_params
  TYPE(t_ho_params), PUBLIC, TARGET :: v_params
  CONTAINS
END MODULE mo_ocean_physics_types
MODULE mo_ocean_solve_aux
  IMPLICIT NONE
  TYPE :: t_ocean_solve_parm
    REAL(KIND = 8) :: tol
    INTEGER :: pt, nr, m, nblk, nblk_a, nidx, nidx_e
    LOGICAL :: use_atol
  END TYPE t_ocean_solve_parm
  CONTAINS
  SUBROUTINE ocean_solve_parm_init(this, pt, nr, m, nblk, nblk_a, nidx, nidx_e, tol, use_atol)
    CLASS(t_ocean_solve_parm), INTENT(INOUT) :: this
    INTEGER, INTENT(IN) :: pt, nr, m, nblk, nblk_a, nidx, nidx_e
    REAL(KIND = 8), INTENT(IN) :: tol
    LOGICAL :: use_atol
    this % pt = 60
    this % nr = nr
    this % m = m
    this % nblk = nblk
    this % nblk_a = nblk_a
    this % nidx = nidx
    this % nidx_e = nidx_e
    this % tol = tol
    this % use_atol = use_atol
  END SUBROUTINE ocean_solve_parm_init
END MODULE mo_ocean_solve_aux
MODULE mo_ocean_surface_types
  USE mo_math_types, ONLY: t_cartesian_coordinates
  IMPLICIT NONE
  TYPE :: t_ocean_surface
    REAL(KIND = 8), POINTER :: topbc_windstress_u(:, :), topbc_windstress_v(:, :)
    TYPE(t_cartesian_coordinates), ALLOCATABLE :: topbc_windstress_cc(:, :)
  END TYPE t_ocean_surface
  TYPE :: t_atmos_for_ocean
  END TYPE t_atmos_for_ocean
END MODULE mo_ocean_surface_types
MODULE mo_ocean_types
  USE mo_math_types, ONLY: t_cartesian_coordinates
  TYPE :: t_hydro_ocean_prog
    REAL(KIND = 8), POINTER, DIMENSION(:, :) :: h
    REAL(KIND = 8), POINTER, DIMENSION(:, :, :) :: vn
  END TYPE t_hydro_ocean_prog
  TYPE :: t_hydro_ocean_diag
    REAL(KIND = 8), POINTER, DIMENSION(:, :, :) :: rho, kin, press_hyd
    REAL(KIND = 8), POINTER, DIMENSION(:, :) :: thick_c
    TYPE(t_cartesian_coordinates), POINTER, DIMENSION(:, :, :) :: p_vn
    REAL(KIND = 8), POINTER, DIMENSION(:, :, :) :: zgrad_rho, w
    REAL(KIND = 8), POINTER, DIMENSION(:, :, :) :: vn_pred, vn_pred_ptp, veloc_adv_horz, veloc_adv_vert, laplacian_horz, laplacian_vert, grad, press_grad
    REAL(KIND = 8), POINTER, DIMENSION(:, :, :) :: vort
    TYPE(t_cartesian_coordinates), POINTER, DIMENSION(:, :, :) :: p_vn_dual
    REAL(KIND = 8), POINTER, DIMENSION(:, :) :: thick_e
  END TYPE t_hydro_ocean_diag
  TYPE :: t_hydro_ocean_aux
    REAL(KIND = 8), POINTER, DIMENSION(:, :, :) :: g_n, g_nm1, g_nimd
    REAL(KIND = 8), POINTER, DIMENSION(:, :) :: bc_bot_vn, bc_top_vn, bc_top_windstress
    REAL(KIND = 8), POINTER, DIMENSION(:, :) :: bc_top_u, bc_top_v, bc_total_top_potential
    REAL(KIND = 8), POINTER, DIMENSION(:, :) :: p_rhs_sfc_eq
    TYPE(t_cartesian_coordinates), POINTER, DIMENSION(:, :) :: bc_top_veloc_cc
  END TYPE t_hydro_ocean_aux
  TYPE :: t_operator_coeff
    REAL(KIND = 8), POINTER, DIMENSION(:, :, :, :) :: div_coeff
    REAL(KIND = 8), POINTER, DIMENSION(:, :, :, :) :: rot_coeff
    REAL(KIND = 8), POINTER, DIMENSION(:, :, :) :: grad_coeff
    REAL(KIND = 8), POINTER, DIMENSION(:, :, :) :: averagecellstoedges
    INTEGER, POINTER, DIMENSION(:, :, :) :: bnd_edges_per_vertex
    INTEGER, POINTER, DIMENSION(:, :, :, :) :: vertex_bnd_edge_idx
    INTEGER, POINTER, DIMENSION(:, :, :, :) :: vertex_bnd_edge_blk
    INTEGER, POINTER, DIMENSION(:, :, :, :) :: boundaryedge_coefficient_index
    REAL(KIND = 8), POINTER, DIMENSION(:, :, :, :) :: edge2edge_viacell_coeff
    REAL(KIND = 8), POINTER, DIMENSION(:, :, :) :: edge2edge_viacell_coeff_all
    REAL(KIND = 8), POINTER, DIMENSION(:, :, :, :) :: edge2edge_viavert_coeff
    TYPE(t_cartesian_coordinates), POINTER, DIMENSION(:, :, :, :) :: edge2cell_coeff_cc_t
    TYPE(t_cartesian_coordinates), POINTER, DIMENSION(:, :, :, :) :: edge2vert_coeff_cc
    TYPE(t_cartesian_coordinates), POINTER, DIMENSION(:, :, :, :) :: edge2vert_coeff_cc_t
  END TYPE t_operator_coeff
  TYPE :: t_solvercoeff_singleprecision
  END TYPE t_solvercoeff_singleprecision
  TYPE :: t_hydro_ocean_state
    TYPE(t_hydro_ocean_prog), POINTER :: p_prog(:)
    TYPE(t_hydro_ocean_diag) :: p_diag
    TYPE(t_hydro_ocean_aux) :: p_aux
  END TYPE t_hydro_ocean_state
END MODULE mo_ocean_types
MODULE mo_operator_ocean_coeff_3d
  IMPLICIT NONE
  INTEGER, PUBLIC :: no_dual_edges
  INTEGER, PUBLIC :: no_primal_edges
  CONTAINS
END MODULE mo_operator_ocean_coeff_3d
MODULE mo_parallel_config
  IMPLICIT NONE
  INTEGER :: nproma = 0
  INTEGER :: n_ghost_rows = 1
  LOGICAL :: l_log_checks = .FALSE.
  LOGICAL :: p_test_run = .FALSE.
  INTEGER :: itype_exch_barrier = 0
  INTEGER :: iorder_sendrecv = 1
  CONTAINS
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
  REAL(KIND = 8) :: dtime
  INTEGER :: timers_level
  LOGICAL :: activate_sync_timers
  CONTAINS
END MODULE mo_run_config
MODULE mo_timer
  IMPLICIT NONE
  INTEGER :: timer_exch_data, timer_exch_data_wait
  INTEGER :: timer_barrier
  INTEGER :: timer_ab_expl, timer_ab_rhs4sfc
  INTEGER :: timer_extra1, timer_extra2, timer_extra3, timer_extra4
  CONTAINS
END MODULE mo_timer
MODULE mo_util_stride
  USE iso_c_binding, ONLY: c_size_t
  USE iso_c_binding, ONLY: c_int, c_ptr
  USE iso_c_binding, ONLY: c_int, c_ptr
  IMPLICIT NONE
  INTERFACE
    SUBROUTINE util_stride_1d(f_out, elemsize, p1, p2) BIND(C)
      USE iso_c_binding, ONLY: c_int, c_ptr
      IMPLICIT NONE
      INTEGER(KIND = c_int), INTENT(OUT) :: f_out
      INTEGER(KIND = c_int), VALUE, INTENT(IN) :: elemsize
      TYPE(c_ptr), VALUE, INTENT(IN) :: p1, p2
    END SUBROUTINE util_stride_1d
    SUBROUTINE util_stride_2d(f_out, elemsize, p1, p2, p3) BIND(C)
      USE iso_c_binding, ONLY: c_int, c_ptr
      IMPLICIT NONE
      INTEGER(KIND = c_int), INTENT(OUT) :: f_out(2)
      INTEGER(KIND = c_int), VALUE, INTENT(IN) :: elemsize
      TYPE(c_ptr), VALUE, INTENT(IN) :: p1, p2, p3
    END SUBROUTINE util_stride_2d
    FUNCTION util_get_ptrdiff(a, b) RESULT(s) BIND(C, NAME = 'util_get_ptrdiff')
      USE iso_c_binding, ONLY: c_size_t
      IMPLICIT NONE
      INTEGER(KIND = c_size_t), INTENT(IN) :: a, b
      INTEGER(KIND = c_size_t) :: s
    END FUNCTION util_get_ptrdiff
  END INTERFACE
END MODULE mo_util_stride
MODULE mo_fortran_tools
  USE iso_c_binding, ONLY: c_ptr, c_f_pointer, c_loc, c_null_ptr
  IMPLICIT NONE
  TYPE :: t_ptr_3d_dp
    REAL(KIND = 8), POINTER :: p(:, :, :) => NULL()
  END TYPE t_ptr_3d_dp
  TYPE :: t_ptr_3d_sp
    REAL(KIND = 4), POINTER :: p(:, :, :) => NULL()
  END TYPE t_ptr_3d_sp
  INTERFACE init
    MODULE PROCEDURE init_zero_1d_dp
    MODULE PROCEDURE init_zero_1d_sp
    MODULE PROCEDURE init_zero_2d_dp
    MODULE PROCEDURE init_zero_2d_sp
    MODULE PROCEDURE init_zero_2d_i4
    MODULE PROCEDURE init_zero_3d_dp
    MODULE PROCEDURE init_zero_3d_sp
    MODULE PROCEDURE init_zero_3d_i4
    MODULE PROCEDURE init_zero_4d_dp
    MODULE PROCEDURE init_zero_4d_sp
    MODULE PROCEDURE init_zero_4d_i4
    MODULE PROCEDURE init_1d_dp
    MODULE PROCEDURE init_1d_sp
    MODULE PROCEDURE init_2d_dp
    MODULE PROCEDURE init_2d_sp
    MODULE PROCEDURE init_3d_dp
    MODULE PROCEDURE init_3d_sp
    MODULE PROCEDURE init_3d_spdp
    MODULE PROCEDURE init_5d_dp
    MODULE PROCEDURE init_5d_sp
    MODULE PROCEDURE init_5d_i4
    MODULE PROCEDURE init_5d_l
  END INTERFACE init
  CONTAINS
  SUBROUTINE init_zero_1d_dp(init_var, lacc, opt_acc_async_queue)
    REAL(KIND = 8), INTENT(OUT) :: init_var(:)
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_acc_async_queue
    INTEGER :: i1, m1
    LOGICAL :: lzacc
    INTEGER :: acc_async_queue
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL set_acc_async_queue(acc_async_queue, opt_acc_async_queue)
    m1 = SIZE(init_var, 1)
    DO i1 = 1, m1
      init_var(i1) = 0.0D0
    END DO
  END SUBROUTINE init_zero_1d_dp
  SUBROUTINE init_zero_1d_sp(init_var, lacc, opt_acc_async_queue)
    REAL(KIND = 4), INTENT(OUT) :: init_var(:)
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_acc_async_queue
    INTEGER :: i1, m1
    LOGICAL :: lzacc
    INTEGER :: acc_async_queue
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL set_acc_async_queue(acc_async_queue, opt_acc_async_queue)
    m1 = SIZE(init_var, 1)
    DO i1 = 1, m1
      init_var(i1) = 0.0D0
    END DO
  END SUBROUTINE init_zero_1d_sp
  SUBROUTINE init_zero_2d_dp(init_var, lacc, opt_acc_async_queue)
    REAL(KIND = 8), INTENT(OUT) :: init_var(:, :)
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_acc_async_queue
    INTEGER :: i1, i2, m1, m2
    LOGICAL :: lzacc
    INTEGER :: acc_async_queue
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL set_acc_async_queue(acc_async_queue, opt_acc_async_queue)
    m1 = SIZE(init_var, 1)
    m2 = SIZE(init_var, 2)
    DO i2 = 1, m2
      DO i1 = 1, m1
        init_var(i1, i2) = 0.0D0
      END DO
    END DO
  END SUBROUTINE init_zero_2d_dp
  SUBROUTINE init_zero_2d_sp(init_var, lacc, opt_acc_async_queue)
    REAL(KIND = 4), INTENT(OUT) :: init_var(:, :)
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_acc_async_queue
    INTEGER :: i1, i2, m1, m2
    LOGICAL :: lzacc
    INTEGER :: acc_async_queue
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL set_acc_async_queue(acc_async_queue, opt_acc_async_queue)
    m1 = SIZE(init_var, 1)
    m2 = SIZE(init_var, 2)
    DO i2 = 1, m2
      DO i1 = 1, m1
        init_var(i1, i2) = 0.0
      END DO
    END DO
  END SUBROUTINE init_zero_2d_sp
  SUBROUTINE init_zero_2d_i4(init_var, lacc, opt_acc_async_queue)
    INTEGER(KIND = 4), INTENT(OUT) :: init_var(:, :)
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_acc_async_queue
    INTEGER :: i1, i2, m1, m2
    LOGICAL :: lzacc
    INTEGER :: acc_async_queue
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL set_acc_async_queue(acc_async_queue, opt_acc_async_queue)
    m1 = SIZE(init_var, 1)
    m2 = SIZE(init_var, 2)
    DO i2 = 1, m2
      DO i1 = 1, m1
        init_var(i1, i2) = 0
      END DO
    END DO
  END SUBROUTINE init_zero_2d_i4
  SUBROUTINE init_zero_3d_dp(init_var, lacc, opt_acc_async_queue)
    REAL(KIND = 8), INTENT(OUT) :: init_var(:, :, :)
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_acc_async_queue
    INTEGER :: i1, i2, i3, m1, m2, m3
    LOGICAL :: lzacc
    INTEGER :: acc_async_queue
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL set_acc_async_queue(acc_async_queue, opt_acc_async_queue)
    m1 = SIZE(init_var, 1)
    m2 = SIZE(init_var, 2)
    m3 = SIZE(init_var, 3)
    DO i3 = 1, m3
      DO i2 = 1, m2
        DO i1 = 1, m1
          init_var(i1, i2, i3) = 0.0D0
        END DO
      END DO
    END DO
  END SUBROUTINE init_zero_3d_dp
  SUBROUTINE init_zero_3d_sp(init_var, lacc, opt_acc_async_queue)
    REAL(KIND = 4), INTENT(OUT) :: init_var(:, :, :)
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_acc_async_queue
    INTEGER :: i1, i2, i3, m1, m2, m3
    LOGICAL :: lzacc
    INTEGER :: acc_async_queue
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL set_acc_async_queue(acc_async_queue, opt_acc_async_queue)
    m1 = SIZE(init_var, 1)
    m2 = SIZE(init_var, 2)
    m3 = SIZE(init_var, 3)
    DO i3 = 1, m3
      DO i2 = 1, m2
        DO i1 = 1, m1
          init_var(i1, i2, i3) = 0.0
        END DO
      END DO
    END DO
  END SUBROUTINE init_zero_3d_sp
  SUBROUTINE init_zero_3d_i4(init_var, lacc, opt_acc_async_queue)
    INTEGER(KIND = 4), INTENT(OUT) :: init_var(:, :, :)
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_acc_async_queue
    INTEGER :: i1, i2, i3, m1, m2, m3
    LOGICAL :: lzacc
    INTEGER :: acc_async_queue
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL set_acc_async_queue(acc_async_queue, opt_acc_async_queue)
    m1 = SIZE(init_var, 1)
    m2 = SIZE(init_var, 2)
    m3 = SIZE(init_var, 3)
    DO i3 = 1, m3
      DO i2 = 1, m2
        DO i1 = 1, m1
          init_var(i1, i2, i3) = 0
        END DO
      END DO
    END DO
  END SUBROUTINE init_zero_3d_i4
  SUBROUTINE init_zero_4d_dp(init_var, lacc, opt_acc_async_queue)
    REAL(KIND = 8), INTENT(OUT) :: init_var(:, :, :, :)
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_acc_async_queue
    INTEGER :: i1, i2, i3, i4, m1, m2, m3, m4
    LOGICAL :: lzacc
    INTEGER :: acc_async_queue
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL set_acc_async_queue(acc_async_queue, opt_acc_async_queue)
    m1 = SIZE(init_var, 1)
    m2 = SIZE(init_var, 2)
    m3 = SIZE(init_var, 3)
    m4 = SIZE(init_var, 4)
    DO i4 = 1, m4
      DO i3 = 1, m3
        DO i2 = 1, m2
          DO i1 = 1, m1
            init_var(i1, i2, i3, i4) = 0.0D0
          END DO
        END DO
      END DO
    END DO
  END SUBROUTINE init_zero_4d_dp
  SUBROUTINE init_zero_4d_sp(init_var, lacc, opt_acc_async_queue)
    REAL(KIND = 4), INTENT(OUT) :: init_var(:, :, :, :)
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_acc_async_queue
    INTEGER :: i1, i2, i3, i4, m1, m2, m3, m4
    LOGICAL :: lzacc
    INTEGER :: acc_async_queue
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL set_acc_async_queue(acc_async_queue, opt_acc_async_queue)
    m1 = SIZE(init_var, 1)
    m2 = SIZE(init_var, 2)
    m3 = SIZE(init_var, 3)
    m4 = SIZE(init_var, 4)
    DO i4 = 1, m4
      DO i3 = 1, m3
        DO i2 = 1, m2
          DO i1 = 1, m1
            init_var(i1, i2, i3, i4) = 0.0
          END DO
        END DO
      END DO
    END DO
  END SUBROUTINE init_zero_4d_sp
  SUBROUTINE init_zero_4d_i4(init_var, lacc, opt_acc_async_queue)
    INTEGER(KIND = 4), INTENT(OUT) :: init_var(:, :, :, :)
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_acc_async_queue
    INTEGER :: i1, i2, i3, i4, m1, m2, m3, m4
    LOGICAL :: lzacc
    INTEGER :: acc_async_queue
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL set_acc_async_queue(acc_async_queue, opt_acc_async_queue)
    m1 = SIZE(init_var, 1)
    m2 = SIZE(init_var, 2)
    m3 = SIZE(init_var, 3)
    m4 = SIZE(init_var, 4)
    DO i4 = 1, m4
      DO i3 = 1, m3
        DO i2 = 1, m2
          DO i1 = 1, m1
            init_var(i1, i2, i3, i4) = 0
          END DO
        END DO
      END DO
    END DO
  END SUBROUTINE init_zero_4d_i4
  SUBROUTINE init_1d_dp(init_var, init_val, lacc, opt_acc_async_queue)
    REAL(KIND = 8), INTENT(OUT) :: init_var(:)
    REAL(KIND = 8), INTENT(IN) :: init_val
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_acc_async_queue
    INTEGER :: i1, m1
    LOGICAL :: lzacc
    INTEGER :: acc_async_queue
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL set_acc_async_queue(acc_async_queue, opt_acc_async_queue)
    m1 = SIZE(init_var, 1)
    DO i1 = 1, m1
      init_var(i1) = init_val
    END DO
  END SUBROUTINE init_1d_dp
  SUBROUTINE init_1d_sp(init_var, init_val, lacc, opt_acc_async_queue)
    REAL(KIND = 4), INTENT(OUT) :: init_var(:)
    REAL(KIND = 4), INTENT(IN) :: init_val
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_acc_async_queue
    INTEGER :: i1, m1
    LOGICAL :: lzacc
    INTEGER :: acc_async_queue
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL set_acc_async_queue(acc_async_queue, opt_acc_async_queue)
    m1 = SIZE(init_var, 1)
    DO i1 = 1, m1
      init_var(i1) = init_val
    END DO
  END SUBROUTINE init_1d_sp
  SUBROUTINE init_2d_dp(init_var, init_val, lacc, opt_acc_async_queue)
    REAL(KIND = 8), INTENT(OUT) :: init_var(:, :)
    REAL(KIND = 8), INTENT(IN) :: init_val
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_acc_async_queue
    INTEGER :: i1, i2, m1, m2
    LOGICAL :: lzacc
    INTEGER :: acc_async_queue
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL set_acc_async_queue(acc_async_queue, opt_acc_async_queue)
    m1 = SIZE(init_var, 1)
    m2 = SIZE(init_var, 2)
    DO i2 = 1, m2
      DO i1 = 1, m1
        init_var(i1, i2) = init_val
      END DO
    END DO
  END SUBROUTINE init_2d_dp
  SUBROUTINE init_2d_sp(init_var, init_val, lacc, opt_acc_async_queue)
    REAL(KIND = 4), INTENT(OUT) :: init_var(:, :)
    REAL(KIND = 4), INTENT(IN) :: init_val
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_acc_async_queue
    INTEGER :: i1, i2, m1, m2
    LOGICAL :: lzacc
    INTEGER :: acc_async_queue
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL set_acc_async_queue(acc_async_queue, opt_acc_async_queue)
    m1 = SIZE(init_var, 1)
    m2 = SIZE(init_var, 2)
    DO i2 = 1, m2
      DO i1 = 1, m1
        init_var(i1, i2) = init_val
      END DO
    END DO
  END SUBROUTINE init_2d_sp
  SUBROUTINE init_3d_dp(init_var, init_val, lacc, opt_acc_async_queue)
    REAL(KIND = 8), INTENT(OUT) :: init_var(:, :, :)
    REAL(KIND = 8), INTENT(IN) :: init_val
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_acc_async_queue
    INTEGER :: i1, i2, i3, m1, m2, m3
    LOGICAL :: lzacc
    INTEGER :: acc_async_queue
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL set_acc_async_queue(acc_async_queue, opt_acc_async_queue)
    m1 = SIZE(init_var, 1)
    m2 = SIZE(init_var, 2)
    m3 = SIZE(init_var, 3)
    DO i3 = 1, m3
      DO i2 = 1, m2
        DO i1 = 1, m1
          init_var(i1, i2, i3) = init_val
        END DO
      END DO
    END DO
  END SUBROUTINE init_3d_dp
  SUBROUTINE init_3d_sp(init_var, init_val, lacc, opt_acc_async_queue)
    REAL(KIND = 4), INTENT(OUT) :: init_var(:, :, :)
    REAL(KIND = 4), INTENT(IN) :: init_val
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_acc_async_queue
    INTEGER :: i1, i2, i3, m1, m2, m3
    LOGICAL :: lzacc
    INTEGER :: acc_async_queue
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL set_acc_async_queue(acc_async_queue, opt_acc_async_queue)
    m1 = SIZE(init_var, 1)
    m2 = SIZE(init_var, 2)
    m3 = SIZE(init_var, 3)
    DO i3 = 1, m3
      DO i2 = 1, m2
        DO i1 = 1, m1
          init_var(i1, i2, i3) = init_val
        END DO
      END DO
    END DO
  END SUBROUTINE init_3d_sp
  SUBROUTINE init_3d_spdp(init_var, init_val, lacc, opt_acc_async_queue)
    REAL(KIND = 4), INTENT(OUT) :: init_var(:, :, :)
    REAL(KIND = 8), INTENT(IN) :: init_val
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_acc_async_queue
    INTEGER :: i1, i2, i3, m1, m2, m3
    LOGICAL :: lzacc
    INTEGER :: acc_async_queue
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL set_acc_async_queue(acc_async_queue, opt_acc_async_queue)
    m1 = SIZE(init_var, 1)
    m2 = SIZE(init_var, 2)
    m3 = SIZE(init_var, 3)
    DO i3 = 1, m3
      DO i2 = 1, m2
        DO i1 = 1, m1
          init_var(i1, i2, i3) = REAL(init_val, kind = 4)
        END DO
      END DO
    END DO
  END SUBROUTINE init_3d_spdp
  SUBROUTINE init_5d_dp(init_var, init_val, lacc, opt_acc_async_queue)
    REAL(KIND = 8), INTENT(OUT) :: init_var(:, :, :, :, :)
    REAL(KIND = 8), INTENT(IN) :: init_val
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_acc_async_queue
    INTEGER :: i1, i2, i3, i4, i5, m1, m2, m3, m4, m5
    LOGICAL :: lzacc
    INTEGER :: acc_async_queue
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL set_acc_async_queue(acc_async_queue, opt_acc_async_queue)
    m1 = SIZE(init_var, 1)
    m2 = SIZE(init_var, 2)
    m3 = SIZE(init_var, 3)
    m4 = SIZE(init_var, 4)
    m5 = SIZE(init_var, 5)
    DO i5 = 1, m5
      DO i4 = 1, m4
        DO i3 = 1, m3
          DO i2 = 1, m2
            DO i1 = 1, m1
              init_var(i1, i2, i3, i4, i5) = init_val
            END DO
          END DO
        END DO
      END DO
    END DO
  END SUBROUTINE init_5d_dp
  SUBROUTINE init_5d_sp(init_var, init_val, lacc, opt_acc_async_queue)
    REAL(KIND = 4), INTENT(OUT) :: init_var(:, :, :, :, :)
    REAL(KIND = 4), INTENT(IN) :: init_val
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_acc_async_queue
    INTEGER :: i1, i2, i3, i4, i5, m1, m2, m3, m4, m5
    LOGICAL :: lzacc
    INTEGER :: acc_async_queue
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL set_acc_async_queue(acc_async_queue, opt_acc_async_queue)
    m1 = SIZE(init_var, 1)
    m2 = SIZE(init_var, 2)
    m3 = SIZE(init_var, 3)
    m4 = SIZE(init_var, 4)
    m5 = SIZE(init_var, 5)
    DO i5 = 1, m5
      DO i4 = 1, m4
        DO i3 = 1, m3
          DO i2 = 1, m2
            DO i1 = 1, m1
              init_var(i1, i2, i3, i4, i5) = init_val
            END DO
          END DO
        END DO
      END DO
    END DO
  END SUBROUTINE init_5d_sp
  SUBROUTINE init_5d_i4(init_var, init_val, lacc, opt_acc_async_queue)
    INTEGER(KIND = 4), INTENT(OUT) :: init_var(:, :, :, :, :)
    INTEGER(KIND = 4), INTENT(IN) :: init_val
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_acc_async_queue
    INTEGER :: i1, i2, i3, i4, i5, m1, m2, m3, m4, m5
    LOGICAL :: lzacc
    INTEGER :: acc_async_queue
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL set_acc_async_queue(acc_async_queue, opt_acc_async_queue)
    m1 = SIZE(init_var, 1)
    m2 = SIZE(init_var, 2)
    m3 = SIZE(init_var, 3)
    m4 = SIZE(init_var, 4)
    m5 = SIZE(init_var, 5)
    DO i5 = 1, m5
      DO i4 = 1, m4
        DO i3 = 1, m3
          DO i2 = 1, m2
            DO i1 = 1, m1
              init_var(i1, i2, i3, i4, i5) = init_val
            END DO
          END DO
        END DO
      END DO
    END DO
  END SUBROUTINE init_5d_i4
  SUBROUTINE init_5d_l(init_var, init_val, lacc, opt_acc_async_queue)
    LOGICAL, INTENT(OUT) :: init_var(:, :, :, :, :)
    LOGICAL, INTENT(IN) :: init_val
    LOGICAL, INTENT(IN) :: lacc
    INTEGER, INTENT(IN), OPTIONAL :: opt_acc_async_queue
    INTEGER :: i1, i2, i3, i4, i5, m1, m2, m3, m4, m5
    LOGICAL :: lzacc
    INTEGER :: acc_async_queue
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL set_acc_async_queue(acc_async_queue, opt_acc_async_queue)
    m1 = SIZE(init_var, 1)
    m2 = SIZE(init_var, 2)
    m3 = SIZE(init_var, 3)
    m4 = SIZE(init_var, 4)
    m5 = SIZE(init_var, 5)
    DO i5 = 1, m5
      DO i4 = 1, m4
        DO i3 = 1, m3
          DO i2 = 1, m2
            DO i1 = 1, m1
              init_var(i1, i2, i3, i4, i5) = init_val
            END DO
          END DO
        END DO
      END DO
    END DO
  END SUBROUTINE init_5d_l
  SUBROUTINE insert_dimension_r_dp_3_2_s(ptr_out, ptr_in, in_shape, new_dim_rank)
    INTEGER, INTENT(IN) :: in_shape(2), new_dim_rank
    REAL(KIND = 8), POINTER, INTENT(OUT) :: ptr_out(:, :, :)
    REAL(KIND = 8), TARGET, INTENT(IN) :: ptr_in
    INTEGER :: out_shape(3), i
    TYPE(c_ptr) :: cptr
    out_shape(1 : 2) = in_shape
    cptr = c_loc(ptr_in)
    DO i = 3, 3, - 1
      out_shape(i) = out_shape(i - 1)
    END DO
    out_shape(2) = 1
    CALL c_f_pointer(cptr, ptr_out, out_shape)
  END SUBROUTINE insert_dimension_r_dp_3_2_s
  SUBROUTINE insert_dimension_r_dp_3_2(ptr_out, ptr_in, new_dim_rank)
    USE mo_util_stride, ONLY: util_stride_1d, util_stride_2d
    REAL(KIND = 8), POINTER, INTENT(OUT) :: ptr_out(:, :, :)
    REAL(KIND = 8), TARGET, INTENT(IN) :: ptr_in(:, :)
    INTEGER, INTENT(IN) :: new_dim_rank
    INTEGER :: base_shape(2), in_shape(2), in_stride(2), out_shape(3), out_stride(3), i
    IF (SIZE(ptr_in) > 0) THEN
      in_shape = SHAPE(ptr_in)
      in_stride(1) = 1
      in_stride(2) = in_shape(1)
      IF (in_shape(1) > 1 .AND. in_shape(2) > 1) THEN
        CALL util_stride_2d(in_stride, 8, c_loc(ptr_in(1, 1)), c_loc(ptr_in(2, 1)), c_loc(ptr_in(1, 2)))
        base_shape(1) = in_stride(2)
      ELSE IF (in_shape(1) > 1) THEN
        CALL util_stride_1d(in_stride(1), 8, c_loc(ptr_in(1, 1)), c_loc(ptr_in(2, 1)))
        base_shape(1) = in_stride(1) * in_shape(1)
      ELSE IF (in_shape(2) > 1) THEN
        CALL util_stride_1d(in_stride(2), 8, c_loc(ptr_in(1, 1)), c_loc(ptr_in(1, 2)))
        base_shape(1) = in_stride(2)
      ELSE
        base_shape(1) = in_shape(1)
      END IF
      base_shape(2) = in_shape(2)
      CALL insert_dimension_r_dp_3_2_s(ptr_out, ptr_in(1, 1), base_shape, 2)
      IF (in_stride(1) > 1 .OR. in_stride(2) > in_shape(1) .OR. base_shape(1) /= in_shape(1)) THEN
        out_stride(1) = in_stride(1)
        out_stride(2) = 1
        out_shape(1 : 2) = in_shape
        DO i = 3, 3, - 1
          out_shape(i) = out_shape(i - 1)
          out_stride(i) = out_stride(i - 1)
        END DO
        out_stride(2) = 1
        out_shape(2) = 1
        out_shape = (out_shape - 1) * out_stride + 1
        ptr_out => ptr_out(: out_shape(1) : out_stride(1), : out_shape(2) : out_stride(2), : out_shape(3) : out_stride(3))
      END IF
    ELSE
      out_shape(1 : 2) = SHAPE(ptr_in)
      DO i = 3, 3, - 1
        out_shape(i) = out_shape(i - 1)
      END DO
      out_shape(2) = 1
      CALL c_f_pointer(c_null_ptr, ptr_out, out_shape)
    END IF
  END SUBROUTINE insert_dimension_r_dp_3_2
  PURE SUBROUTINE set_acc_host_or_device(lzacc, lacc)
    LOGICAL, INTENT(OUT) :: lzacc
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    lzacc = .FALSE.
  END SUBROUTINE set_acc_host_or_device
  SUBROUTINE set_acc_async_queue(acc_async_queue, opt_acc_async_queue)
    INTEGER, INTENT(OUT) :: acc_async_queue
    INTEGER, INTENT(IN), OPTIONAL :: opt_acc_async_queue
    acc_async_queue = 1
    acc_async_queue = opt_acc_async_queue
  END SUBROUTINE set_acc_async_queue
END MODULE mo_fortran_tools
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
  LOGICAL :: process_is_stdio
  INTEGER :: my_mpi_function
  INTEGER :: num_test_procs
  INTEGER :: p_work_pe0
  INTEGER :: p_pe_work
  INTEGER :: p_comm_work
  INTEGER :: p_comm_work_test
  INTEGER :: p_pe = 0
  INTEGER :: p_real_dp = 0
  INTEGER :: p_real_sp = 0
  INTEGER, PUBLIC :: comm_lev = 0, glob_comm(0 : 10), comm_proc0(0 : 10)
  CHARACTER(LEN = *), PARAMETER :: modname = 'mo_mpi'
  CONTAINS
  FUNCTION get_comm_acc_queue()
    INTEGER :: get_comm_acc_queue
    get_comm_acc_queue = 1
  END FUNCTION
  SUBROUTINE acc_wait_comms(queue)
    INTEGER, INTENT(IN) :: queue
  END SUBROUTINE
  INTEGER FUNCTION get_my_mpi_work_communicator()
    get_my_mpi_work_communicator = p_comm_work
  END FUNCTION get_my_mpi_work_communicator
  LOGICAL FUNCTION my_process_is_stdio()
    my_process_is_stdio = process_is_stdio
  END FUNCTION my_process_is_stdio
  LOGICAL FUNCTION my_process_is_mpi_test()
    my_process_is_mpi_test = (my_mpi_function == 1)
  END FUNCTION my_process_is_mpi_test
  LOGICAL FUNCTION my_process_is_mpi_parallel()
    my_process_is_mpi_parallel = process_is_mpi_parallel
  END FUNCTION my_process_is_mpi_parallel
  LOGICAL FUNCTION my_process_is_mpi_seq()
    my_process_is_mpi_seq = .FALSE.
  END FUNCTION my_process_is_mpi_seq
  SUBROUTINE finish(name, text)
    CHARACTER(LEN = *), INTENT(IN) :: name
    CHARACTER(LEN = *), INTENT(IN) :: text
  END SUBROUTINE finish
  SUBROUTINE abort_mpi
    CALL mpi_abort(0, 1, p_error)
    IF (p_error /= 0) THEN
      WRITE(0, '(a)') ' MPI_ABORT failed.'
      WRITE(0, '(a,i4)') ' Error =  ', p_error
    END IF
    CALL util_exit(1)
  END SUBROUTINE abort_mpi
  FUNCTION p_comm_size(communicator)
    INTEGER :: p_comm_size
    INTEGER, INTENT(IN) :: communicator
    CHARACTER(LEN = *), PARAMETER :: routine = modname // '::p_comm_size'
    INTEGER :: ierr
    CALL mpi_comm_size(communicator, p_comm_size, ierr)
    IF (ierr /= 0) CALL finish(routine, 'Error in MPI_COMM_SIZE operation!')
  END FUNCTION p_comm_size
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
  SUBROUTINE p_send_sp_3d(t_buffer, p_destination, p_tag, p_count, comm)
    REAL(KIND = 4), INTENT(IN) :: t_buffer(:, :, :)
    INTEGER, INTENT(IN) :: p_destination, p_tag
    INTEGER, OPTIONAL, INTENT(IN) :: p_count, comm
    INTEGER :: p_comm, icount
    p_comm = process_mpi_all_comm
    icount = SIZE(t_buffer)
    CALL mpi_send(t_buffer, icount, p_real_sp, p_destination, 1, p_comm, p_error)
  END SUBROUTINE p_send_sp_3d
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
  SUBROUTINE p_recv_sp_3d(t_buffer, p_source, p_tag, p_count, comm)
    REAL(KIND = 4), INTENT(OUT) :: t_buffer(:, :, :)
    INTEGER, INTENT(IN) :: p_source, p_tag
    INTEGER, OPTIONAL, INTENT(IN) :: p_count, comm
    INTEGER :: p_comm, icount
    p_comm = process_mpi_all_comm
    icount = SIZE(t_buffer)
    CALL mpi_recv(t_buffer, icount, p_real_sp, p_source, 1, p_comm, p_status, p_error)
  END SUBROUTINE p_recv_sp_3d
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
  SUBROUTINE p_bcast_sp_3d(t_buffer, p_source, comm)
    REAL(KIND = 4) :: t_buffer(:, :, :)
    INTEGER, INTENT(IN) :: p_source
    INTEGER, OPTIONAL, INTENT(IN) :: comm
    INTEGER :: p_comm
    p_comm = comm
    IF (process_mpi_all_size == 1) THEN
      RETURN
    ELSE
      CALL mpi_bcast(t_buffer, SIZE(t_buffer), p_real_sp, p_source, p_comm, p_error)
    END IF
  END SUBROUTINE p_bcast_sp_3d
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
  FUNCTION p_sum_dp_0d(zfield, comm, root) RESULT(p_sum)
    REAL(KIND = 8) :: p_sum
    REAL(KIND = 8), INTENT(IN) :: zfield
    INTEGER, INTENT(IN) :: comm
    INTEGER, INTENT(IN), OPTIONAL :: root
    INTEGER :: p_comm
    p_comm = comm
    IF (my_process_is_mpi_parallel()) THEN
      CALL mpi_allreduce(zfield, p_sum, 1, p_real_dp, 3, p_comm, p_error)
    ELSE
      p_sum = zfield
    END IF
  END FUNCTION p_sum_dp_0d
  SUBROUTINE p_minmax_common_dp(in_field, out_field, n, op, loc_op, proc_id, keyval, comm, root)
    INTEGER, INTENT(IN) :: n, op, loc_op
    REAL(KIND = 8), INTENT(IN) :: in_field(n)
    REAL(KIND = 8), INTENT(OUT) :: out_field(n)
    INTEGER, OPTIONAL, INTENT(INOUT) :: proc_id(n)
    INTEGER, OPTIONAL, INTENT(INOUT) :: keyval(n)
    INTEGER, OPTIONAL, INTENT(IN) :: root
    INTEGER, OPTIONAL, INTENT(IN) :: comm
    INTEGER :: p_comm, comm_size
    p_comm = comm
    comm_size = p_comm_size(comm)
    IF (comm_size > 1) THEN
      CALL mpi_reduce(in_field, out_field, 1, p_real_dp, op, root, p_comm, p_error)
    ELSE
      out_field = in_field
    END IF
  END SUBROUTINE p_minmax_common_dp
  FUNCTION p_max_dp_0d(zfield, proc_id, keyval, comm, root) RESULT(p_max)
    REAL(KIND = 8) :: p_max
    REAL(KIND = 8), INTENT(IN) :: zfield
    INTEGER, OPTIONAL, INTENT(INOUT) :: proc_id
    INTEGER, OPTIONAL, INTENT(INOUT) :: keyval
    INTEGER, OPTIONAL, INTENT(IN) :: root
    INTEGER, OPTIONAL, INTENT(IN) :: comm
    REAL(KIND = 8) :: temp_in(1), temp_out(1)
    temp_in(1) = zfield
    CALL p_minmax_common_dp(temp_in, temp_out, 1, 1, 11, comm = comm, root = root)
    p_max = temp_out(1)
  END FUNCTION p_max_dp_0d
  FUNCTION p_min_dp_0d(zfield, proc_id, keyval, comm, root) RESULT(p_min)
    REAL(KIND = 8) :: p_min
    REAL(KIND = 8), INTENT(IN) :: zfield
    INTEGER, OPTIONAL, INTENT(INOUT) :: proc_id
    INTEGER, OPTIONAL, INTENT(INOUT) :: keyval
    INTEGER, OPTIONAL, INTENT(IN) :: root
    INTEGER, OPTIONAL, INTENT(IN) :: comm
    REAL(KIND = 8) :: temp_in(1), temp_out(1)
    temp_in(1) = zfield
    CALL p_minmax_common_dp(temp_in, temp_out, 1, 2, 12, comm = comm, root = root)
    p_min = temp_out(1)
  END FUNCTION p_min_dp_0d
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
    USE mo_mpi, ONLY: acc_wait_comms, get_comm_acc_queue, my_process_is_mpi_seq, p_barrier, p_irecv_dp_deconiface_9 => p_irecv_dp, p_isend_dp_deconiface_11 => p_isend_dp, p_isend_dp_deconiface_13 => p_isend_dp, p_recv_dp_deconiface_12 => p_recv_dp, p_send_dp_deconiface_10 => p_send_dp, p_wait_noarg_deconiface_14 => p_wait_noarg
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
        CALL p_irecv_dp_deconiface_9(recv_buf(1, irs), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
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
        CALL p_send_dp_deconiface_10(send_buf(1, iss), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
      END DO
    ELSE IF (iorder_sendrecv == 2) THEN
      DO np = 1, p_pat % np_send
        pid = p_pat % pelist_send(np)
        iss = p_pat % send_startidx(np)
        icount = p_pat % send_count(np) * ndim2
        CALL p_isend_dp_deconiface_11(send_buf(1, iss), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
      END DO
      DO np = 1, p_pat % np_recv
        pid = p_pat % pelist_recv(np)
        irs = p_pat % recv_startidx(np)
        icount = p_pat % recv_count(np) * ndim2
        CALL p_recv_dp_deconiface_12(recv_buf(1, irs), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
      END DO
    ELSE IF (iorder_sendrecv == 3) THEN
      DO np = 1, p_pat % np_send
        pid = p_pat % pelist_send(np)
        iss = p_pat % send_startidx(np)
        icount = p_pat % send_count(np) * ndim2
        CALL p_isend_dp_deconiface_13(send_buf(1, iss), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
      END DO
    END IF
    IF (activate_sync_timers) CALL timer_start(timer_exch_data_wait)
    CALL p_wait_noarg_deconiface_14
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
  SUBROUTINE exchange_data_s3d(p_pat, lacc, recv, send, add)
    USE mo_mpi, ONLY: acc_wait_comms, get_comm_acc_queue, my_process_is_mpi_seq, p_barrier, p_irecv_sp_deconiface_15 => p_irecv_sp, p_isend_sp_deconiface_17 => p_isend_sp, p_isend_sp_deconiface_19 => p_isend_sp, p_recv_sp_deconiface_18 => p_recv_sp, p_send_sp_deconiface_16 => p_send_sp, p_wait_noarg_deconiface_20 => p_wait_noarg
    USE mo_parallel_config, ONLY: iorder_sendrecv, itype_exch_barrier, nproma
    USE mo_run_config, ONLY: activate_sync_timers
    USE mo_real_timer, ONLY: timer_start, timer_stop
    USE mo_timer, ONLY: timer_barrier, timer_exch_data, timer_exch_data_wait
    USE mo_exception, ONLY: finish
    CLASS(t_comm_pattern_orig), TARGET, INTENT(INOUT) :: p_pat
    LOGICAL, INTENT(IN) :: lacc
    REAL(KIND = 4), INTENT(INOUT), TARGET :: recv(:, :, :)
    REAL(KIND = 4), INTENT(IN), OPTIONAL, TARGET :: send(:, :, :)
    REAL(KIND = 4), INTENT(IN), OPTIONAL, TARGET :: add(:, :, :)
    CHARACTER(LEN = *), PARAMETER :: routine = modname // "::exchange_data_s3d"
    REAL(KIND = 4) :: send_buf(SIZE(recv, 2), p_pat % n_send), recv_buf(SIZE(recv, 2), p_pat % n_recv)
    INTEGER :: i, k, np, irs, iss, pid, icount, ndim2
    IF (my_process_is_mpi_seq()) THEN
      CALL exchange_data_s3d_seq(p_pat, .FALSE., recv, send, add)
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
        CALL p_irecv_sp_deconiface_15(recv_buf(1, irs), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
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
        CALL p_send_sp_deconiface_16(send_buf(1, iss), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
      END DO
    ELSE IF (iorder_sendrecv == 2) THEN
      DO np = 1, p_pat % np_send
        pid = p_pat % pelist_send(np)
        iss = p_pat % send_startidx(np)
        icount = p_pat % send_count(np) * ndim2
        CALL p_isend_sp_deconiface_17(send_buf(1, iss), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
      END DO
      DO np = 1, p_pat % np_recv
        pid = p_pat % pelist_recv(np)
        irs = p_pat % recv_startidx(np)
        icount = p_pat % recv_count(np) * ndim2
        CALL p_recv_sp_deconiface_18(recv_buf(1, irs), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
      END DO
    ELSE IF (iorder_sendrecv == 3) THEN
      DO np = 1, p_pat % np_send
        pid = p_pat % pelist_send(np)
        iss = p_pat % send_startidx(np)
        icount = p_pat % send_count(np) * ndim2
        CALL p_isend_sp_deconiface_19(send_buf(1, iss), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
      END DO
    END IF
    IF (activate_sync_timers) CALL timer_start(timer_exch_data_wait)
    CALL p_wait_noarg_deconiface_20
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
  END SUBROUTINE exchange_data_s3d
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
  END SUBROUTINE exchange_data_s3d_seq
  SUBROUTINE exchange_data_mult_mixprec(p_pat, lacc, nfields_dp, ndim2tot_dp, nfields_sp, ndim2tot_sp, recv_dp, send_dp, recv_sp, send_sp, nshift)
    USE mo_fortran_tools, ONLY: t_ptr_3d_dp, t_ptr_3d_sp
    USE mo_parallel_config, ONLY: iorder_sendrecv, itype_exch_barrier
    USE mo_run_config, ONLY: activate_sync_timers
    USE mo_real_timer, ONLY: timer_start, timer_stop
    USE mo_timer, ONLY: timer_barrier, timer_exch_data, timer_exch_data_wait
    USE mo_mpi, ONLY: acc_wait_comms, get_comm_acc_queue, my_process_is_mpi_seq, p_barrier, p_irecv_dp_deconiface_33 => p_irecv_dp, p_irecv_sp_deconiface_34 => p_irecv_sp, p_isend_dp_deconiface_37 => p_isend_dp, p_isend_dp_deconiface_41 => p_isend_dp, p_isend_sp_deconiface_38 => p_isend_sp, p_isend_sp_deconiface_42 => p_isend_sp, p_recv_dp_deconiface_39 => p_recv_dp, p_recv_sp_deconiface_40 => p_recv_sp, p_send_dp_deconiface_35 => p_send_dp, p_send_sp_deconiface_36 => p_send_sp, p_wait_noarg_deconiface_43 => p_wait_noarg
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
        CALL exchange_data_r3d_seq(p_pat, lacc, recv_dp(n) % p)
      END DO
      DO n = 1, nfields_sp
        CALL exchange_data_s3d_seq(p_pat, lacc, recv_sp(n) % p)
      END DO
      IF (activate_sync_timers) CALL timer_stop(timer_exch_data)
      RETURN
    END IF
    IF ((iorder_sendrecv == 1 .OR. iorder_sendrecv == 3) .AND. .NOT. my_process_is_mpi_seq()) THEN
      DO np = 1, p_pat % np_recv
        pid = p_pat % pelist_recv(np)
        irs = p_pat % recv_startidx(np)
        icount = p_pat % recv_count(np) * ndim2tot_dp
        IF (icount > 0) CALL p_irecv_dp_deconiface_33(recv_buf_dp(1, irs), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
        icount = p_pat % recv_count(np) * ndim2tot_sp
        IF (icount > 0) CALL p_irecv_sp_deconiface_34(recv_buf_sp(1, irs), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
      END DO
    END IF
    DO n = 1, nfields_dp
      IF (SIZE(recv_dp(n) % p, 2) == 1) kshift_dp(n) = 0
    END DO
    DO n = 1, nfields_sp
      IF (SIZE(recv_sp(n) % p, 2) == 1) kshift_sp(n) = 0
    END DO
    accum = 0
    DO n = 1, nfields_dp
      noffset_dp(n) = accum
      ndim2_dp(n) = SIZE(recv_dp(n) % p, 2) - kshift_dp(n)
      accum = accum + ndim2_dp(n)
    END DO
    accum = 0
    DO n = 1, nfields_sp
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
      DO n = 1, nfields_sp
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
        IF (icount > 0) CALL p_send_dp_deconiface_35(send_buf_dp(1, iss), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
        icount = p_pat % send_count(np) * ndim2tot_sp
        IF (icount > 0) CALL p_send_sp_deconiface_36(send_buf_sp(1, iss), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
      END DO
    ELSE IF (iorder_sendrecv == 2) THEN
      DO np = 1, p_pat % np_send
        pid = p_pat % pelist_send(np)
        iss = p_pat % send_startidx(np)
        icount = p_pat % send_count(np) * ndim2tot_dp
        IF (icount > 0) CALL p_isend_dp_deconiface_37(send_buf_dp(1, iss), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
        icount = p_pat % send_count(np) * ndim2tot_sp
        IF (icount > 0) CALL p_isend_sp_deconiface_38(send_buf_sp(1, iss), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
      END DO
      DO np = 1, p_pat % np_recv
        pid = p_pat % pelist_recv(np)
        irs = p_pat % recv_startidx(np)
        icount = p_pat % recv_count(np) * ndim2tot_dp
        IF (icount > 0) CALL p_recv_dp_deconiface_39(recv_buf_dp(1, irs), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
        icount = p_pat % recv_count(np) * ndim2tot_sp
        IF (icount > 0) CALL p_recv_sp_deconiface_40(recv_buf_sp(1, irs), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
      END DO
    ELSE IF (iorder_sendrecv == 3) THEN
      DO np = 1, p_pat % np_send
        pid = p_pat % pelist_send(np)
        iss = p_pat % send_startidx(np)
        icount = p_pat % send_count(np) * ndim2tot_dp
        IF (icount > 0) CALL p_isend_dp_deconiface_41(send_buf_dp(1, iss), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
        icount = p_pat % send_count(np) * ndim2tot_sp
        IF (icount > 0) CALL p_isend_sp_deconiface_42(send_buf_sp(1, iss), pid, 1, p_count = icount, comm = p_pat % comm, use_g2g = .FALSE.)
      END DO
    END IF
    IF (activate_sync_timers) CALL timer_start(timer_exch_data_wait)
    CALL p_wait_noarg_deconiface_43
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
      DO n = 1, nfields_sp
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
    USE mo_communication_types, ONLY: exchange_data_r3d_deconproc_1 => exchange_data_r3d, t_comm_pattern_orig
    TYPE(t_comm_pattern_orig), POINTER :: p_pat
    LOGICAL, INTENT(IN) :: lacc
    REAL(KIND = 8), INTENT(INOUT), TARGET :: recv(:, :, :)
    REAL(KIND = 8), INTENT(IN), OPTIONAL, TARGET :: send(:, :, :)
    REAL(KIND = 8), INTENT(IN), OPTIONAL, TARGET :: add(:, :, :)
    CALL exchange_data_r3d_deconproc_1(p_pat, lacc, recv, send, add)
  END SUBROUTINE exchange_data_r3d
  SUBROUTINE exchange_data_s3d(p_pat, lacc, recv, send, add)
    USE mo_communication_types, ONLY: exchange_data_s3d_deconproc_2 => exchange_data_s3d, t_comm_pattern_orig
    TYPE(t_comm_pattern_orig), POINTER :: p_pat
    LOGICAL, INTENT(IN) :: lacc
    REAL(KIND = 4), INTENT(INOUT), TARGET :: recv(:, :, :)
    REAL(KIND = 4), INTENT(IN), OPTIONAL, TARGET :: send(:, :, :)
    REAL(KIND = 4), INTENT(IN), OPTIONAL, TARGET :: add(:, :, :)
    CALL exchange_data_s3d_deconproc_2(p_pat, .FALSE., recv, send, add)
  END SUBROUTINE exchange_data_s3d
  SUBROUTINE exchange_data_mult_mixprec(p_pat, lacc, nfields_dp, ndim2tot_dp, nfields_sp, ndim2tot_sp, recv1_dp, send1_dp, recv2_dp, send2_dp, recv3_dp, send3_dp, recv4_dp, send4_dp, recv5_dp, send5_dp, recv1_sp, send1_sp, recv2_sp, send2_sp, recv3_sp, send3_sp, recv4_sp, send4_sp, recv5_sp, send5_sp, recv4d_dp, send4d_dp, recv4d_sp, send4d_sp, recv3d_arr_dp, recv3d_arr_sp, nshift)
    USE mo_communication_types, ONLY: exchange_data_mult_mixprec_deconproc_16 => exchange_data_mult_mixprec, t_comm_pattern_orig
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
    IF (i_sp /= nfields_sp) CALL finish(routine, "internal error nfields_sp")
    CALL exchange_data_mult_mixprec_deconproc_16(p_pat, lacc, nfields_dp, ndim2tot_dp, nfields_sp, ndim2tot_sp, recv_dp = recv_dp, recv_sp = recv_sp, nshift = nshift)
  END SUBROUTINE exchange_data_mult_mixprec
END MODULE mo_communication
MODULE mo_model_domain
  USE mo_math_types, ONLY: t_cartesian_coordinates, t_geographical_coordinates
  USE mo_decomposition_tools, ONLY: t_grid_domain_decomp_info
  USE mo_communication_types, ONLY: t_comm_pattern_orig, t_p_comm_pattern
  IMPLICIT NONE
  TYPE :: t_subset_range
    INTEGER :: start_block
    INTEGER :: start_index
    INTEGER :: end_block
    INTEGER :: end_index
    INTEGER :: block_size
    INTEGER :: size
    TYPE(t_patch), POINTER :: patch => NULL()
    INTEGER :: entity_location
    INTEGER, POINTER :: vertical_levels(:, :) => NULL()
    INTEGER :: no_of_holes
  END TYPE t_subset_range
  TYPE :: t_grid_cells
    INTEGER :: max_connectivity
    INTEGER, ALLOCATABLE :: neighbor_idx(:, :, :)
    INTEGER, ALLOCATABLE :: neighbor_blk(:, :, :)
    INTEGER, ALLOCATABLE :: edge_idx(:, :, :)
    INTEGER, ALLOCATABLE :: edge_blk(:, :, :)
    TYPE(t_geographical_coordinates), ALLOCATABLE :: center(:, :)
    TYPE(t_grid_domain_decomp_info) :: decomp_info
    TYPE(t_subset_range) :: all
    TYPE(t_subset_range) :: owned
    TYPE(t_subset_range) :: in_domain
  END TYPE t_grid_cells
  TYPE :: t_grid_edges
    INTEGER, ALLOCATABLE :: cell_idx(:, :, :)
    INTEGER, ALLOCATABLE :: cell_blk(:, :, :)
    INTEGER, ALLOCATABLE :: vertex_idx(:, :, :)
    INTEGER, ALLOCATABLE :: vertex_blk(:, :, :)
    REAL(KIND = 8), ALLOCATABLE :: tangent_orientation(:, :)
    TYPE(t_geographical_coordinates), ALLOCATABLE :: center(:, :)
    TYPE(t_cartesian_coordinates), POINTER :: primal_cart_normal(:, :) => NULL()
    TYPE(t_cartesian_coordinates), POINTER :: dual_cart_normal(:, :) => NULL()
    REAL(KIND = 8), POINTER :: primal_edge_length(:, :) => NULL()
    REAL(KIND = 8), ALLOCATABLE :: inv_primal_edge_length(:, :)
    REAL(KIND = 8), ALLOCATABLE :: inv_dual_edge_length(:, :)
    REAL(KIND = 8), ALLOCATABLE :: f_e(:, :)
    TYPE(t_grid_domain_decomp_info) :: decomp_info
    TYPE(t_subset_range) :: all
    TYPE(t_subset_range) :: owned
    TYPE(t_subset_range) :: in_domain
  END TYPE t_grid_edges
  TYPE :: t_grid_vertices
    INTEGER, ALLOCATABLE :: edge_idx(:, :, :)
    INTEGER, ALLOCATABLE :: edge_blk(:, :, :)
    INTEGER, ALLOCATABLE :: num_edges(:, :)
    TYPE(t_geographical_coordinates), ALLOCATABLE :: vertex(:, :)
    REAL(KIND = 8), ALLOCATABLE :: f_v(:, :)
    TYPE(t_grid_domain_decomp_info) :: decomp_info
    TYPE(t_subset_range) :: owned
    TYPE(t_subset_range) :: in_domain
  END TYPE t_grid_vertices
  TYPE :: t_patch
    INTEGER :: n_patch_cells
    INTEGER :: n_patch_edges
    INTEGER :: n_patch_verts
    INTEGER :: n_patch_cells_g
    INTEGER :: n_patch_edges_g
    INTEGER :: n_patch_verts_g
    INTEGER :: alloc_cell_blocks
    INTEGER :: nblks_c
    INTEGER :: nblks_e
    INTEGER :: nblks_v
    TYPE(t_grid_cells) :: cells
    TYPE(t_grid_edges) :: edges
    TYPE(t_grid_vertices) :: verts
    TYPE(t_comm_pattern_orig), POINTER :: comm_pat_c
    TYPE(t_comm_pattern_orig), POINTER :: comm_pat_c1
    TYPE(t_comm_pattern_orig), POINTER :: comm_pat_e
    TYPE(t_comm_pattern_orig), POINTER :: comm_pat_v
    TYPE(t_p_comm_pattern) :: comm_pat_work2test(3)
  END TYPE t_patch
  TYPE :: t_patch_vert
    REAL(KIND = 8), ALLOCATABLE :: del_zlev_m(:)
    INTEGER, POINTER :: dolic_c(:, :) => NULL()
    INTEGER, POINTER :: dolic_e(:, :) => NULL()
    INTEGER, POINTER :: vertex_bottomlevel(:, :) => NULL()
    REAL(KIND = 8), POINTER :: prism_thick_c(:, :, :), prism_thick_e(:, :, :), prism_thick_flat_sfc_c(:, :, :), prism_thick_flat_sfc_e(:, :, :), inv_prism_thick_c(:, :, :), inv_prism_thick_e(:, :, :), inv_prism_center_dist_e(:, :, :)
    REAL(KIND = 8), POINTER :: invconstantprismthickness(:, :, :) => NULL()
    REAL(KIND = 8), POINTER :: constantprismcenters_zdistance(:, :, :) => NULL()
    REAL(KIND = 8), POINTER :: constantprismcenters_invzdistance(:, :, :) => NULL()
  END TYPE t_patch_vert
  TYPE :: t_patch_3d
    TYPE(t_patch), POINTER :: p_patch_2d(:) => NULL()
    TYPE(t_patch_vert), POINTER :: p_patch_1d(:) => NULL()
    INTEGER, POINTER :: lsm_c(:, :, :) => NULL()
    INTEGER, POINTER :: lsm_e(:, :, :) => NULL()
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
MODULE mo_ocean_pp_scheme
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE icon_pp_edge_vnpredict_scheme(patch_3d, blockno, start_index, end_index, ocean_state, vn_predict, lacc)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d
    USE mo_ocean_types, ONLY: t_hydro_ocean_state
    USE mo_ocean_nml, ONLY: n_zlev, oceanreferencedensity, velocity_richardsoncoeff, verticalviscosity_timeweight
    USE mo_ocean_physics_types, ONLY: t_ho_params, v_params
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    INTEGER, INTENT(IN) :: blockno, start_index, end_index
    TYPE(t_hydro_ocean_state), TARGET :: ocean_state
    REAL(KIND = 8) :: vn_predict(:, :)
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: je, jk
    INTEGER :: cell_1_idx, cell_1_block, cell_2_idx, cell_2_block
    INTEGER :: levels
    LOGICAL :: lzacc
    REAL(KIND = 8) :: z_grav_rho, z_inv_oceanreferencedensity
    REAL(KIND = 8) :: density_differ_edge, dz, richardson_edge, z_shear_edge, vn_diff, new_velocity_friction
    REAL(KIND = 8), POINTER :: wind_mixing(:, :)
    REAL(KIND = 8) :: z_vert_density_grad_e(1 : n_zlev + 1)
    TYPE(t_patch), POINTER :: patch_2d
    TYPE(t_ho_params), POINTER :: params_oce
    CALL set_acc_host_or_device(lzacc, lacc)
    params_oce => v_params
    patch_2d => patch_3d % p_patch_2d(1)
    wind_mixing => params_oce % velocity_windmixing(:, :, blockno)
    levels = n_zlev
    z_grav_rho = 9.80665D0 / oceanreferencedensity
    z_inv_oceanreferencedensity = 1.0D0 / oceanreferencedensity
    DO je = start_index, end_index
      cell_1_idx = patch_2d % edges % cell_idx(je, blockno, 1)
      cell_1_block = patch_2d % edges % cell_blk(je, blockno, 1)
      cell_2_idx = patch_2d % edges % cell_idx(je, blockno, 2)
      cell_2_block = patch_2d % edges % cell_blk(je, blockno, 2)
      DO jk = 2, patch_3d % p_patch_1d(1) % dolic_e(je, blockno)
        z_vert_density_grad_e(jk) = 0.5D0 * (ocean_state % p_diag % zgrad_rho(cell_1_idx, jk, cell_1_block) + ocean_state % p_diag % zgrad_rho(cell_2_idx, jk, cell_2_block))
      END DO
      DO jk = 2, patch_3d % p_patch_1d(1) % dolic_e(je, blockno)
        dz = 0.5D0 * (patch_3d % p_patch_1d(1) % prism_thick_e(je, jk - 1, blockno) + patch_3d % p_patch_1d(1) % prism_thick_e(je, jk, blockno))
        density_differ_edge = z_vert_density_grad_e(jk) * dz
        vn_diff = ABS(vn_predict(je, jk) - vn_predict(je, jk - 1))
        z_shear_edge = 2.220446049250313D-16 + vn_diff ** 2
        richardson_edge = MAX(dz * z_grav_rho * density_differ_edge / z_shear_edge, 0.0D0)
        new_velocity_friction = params_oce % a_veloc_v_back * dz + velocity_richardsoncoeff / ((1.0D0 + 5.0D0 * richardson_edge) ** 2) + wind_mixing(je, jk)
        params_oce % a_veloc_v(je, jk, blockno) = verticalviscosity_timeweight * params_oce % a_veloc_v(je, jk, blockno) + (1.0D0 - verticalviscosity_timeweight) * new_velocity_friction
      END DO
    END DO
  END SUBROUTINE icon_pp_edge_vnpredict_scheme
END MODULE mo_ocean_pp_scheme
MODULE mo_ocean_solve_lhs
  IMPLICIT NONE
  CHARACTER(LEN = *), PARAMETER :: module_name = "mo_ocean_solve_lhs"
  TYPE :: t_lhs
    INTEGER :: nblk_loc, nidx_loc
    REAL(KIND = 8), ALLOCATABLE, DIMENSION(:, :, :) :: coef_l_wp
    INTEGER, ALLOCATABLE, DIMENSION(:, :, :) :: blk_loc, idx_loc
  END TYPE t_lhs
  CONTAINS
  SUBROUTINE lhs_dump_matrix(this, id, prefix, lprecon, lacc)
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_exception, ONLY: finish
    USE mo_mpi, ONLY: p_pe_work
    CLASS(t_lhs), INTENT(IN), TARGET :: this
    INTEGER, INTENT(IN) :: id
    LOGICAL, INTENT(IN) :: lprecon
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    CHARACTER(LEN = *), INTENT(IN) :: prefix
    CHARACTER(LEN = 128) :: filename
    INTEGER :: inz, iblk, iidx
    INTEGER, PARAMETER :: fileno = 501
    CHARACTER(LEN = *), PARAMETER :: routine = module_name // "::lhs_dump_matrix()"
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    IF (lprecon) CALL finish(routine, "cannot dump preconditioner matrix if no precon present")
    WRITE(filename, "(A,I0.4,A,i0.4,a)") TRIM(prefix) // "_", id, "_", p_pe_work, ".txt"
    OPEN(UNIT = fileno, FILE = TRIM(filename), STATUS = 'new')
    DO inz = 1, SIZE(this % coef_l_wp, 3)
      DO iblk = 1, this % nblk_loc
        DO iidx = 1, this % nidx_loc
          WRITE(501, "(2(a,2(i8.8,a)),es12.5)") "(", iidx, ":", iblk, "), ", "(", this % idx_loc(iidx, iblk, inz), ":", this % blk_loc(iidx, iblk, inz), ")", this % coef_l_wp(iidx, iblk, inz)
        END DO
      END DO
    END DO
    CLOSE(UNIT = fileno)
  END SUBROUTINE lhs_dump_matrix
END MODULE mo_ocean_solve_lhs
MODULE mo_ocean_solve_lhs_type
  USE mo_model_domain, ONLY: t_patch, t_patch_3d
  USE mo_ocean_types, ONLY: t_operator_coeff, t_solvercoeff_singleprecision
  IMPLICIT NONE
  TYPE, ABSTRACT :: t_lhs_agen
    LOGICAL :: is_const, use_shortcut
    LOGICAL :: is_init = .FALSE.
  END TYPE t_lhs_agen
  TYPE, EXTENDS(t_lhs_agen) :: t_primal_flip_flop_lhs
  END TYPE t_primal_flip_flop_lhs
  TYPE, EXTENDS(t_lhs_agen) :: t_surface_height_lhs
    TYPE(t_patch_3d), POINTER :: patch_3d => NULL()
    TYPE(t_patch), POINTER :: patch_2d => NULL()
    REAL(KIND = 8), POINTER :: thickness_e_wp(:, :) => NULL()
    TYPE(t_operator_coeff), POINTER :: op_coeffs_wp => NULL()
    TYPE(t_solvercoeff_singleprecision), POINTER :: op_coeffs_sp => NULL()
  END TYPE t_surface_height_lhs
  CONTAINS
  SUBROUTINE lhs_primal_flip_flop_construct(this, patch_3d, op_coeffs, jk, lacc)
    USE mo_model_domain, ONLY: t_patch_3d
    USE mo_ocean_types, ONLY: t_operator_coeff
    CLASS(t_primal_flip_flop_lhs), INTENT(INOUT) :: this
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    TYPE(t_operator_coeff), TARGET, INTENT(IN) :: op_coeffs
    INTEGER, INTENT(IN) :: jk
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
  END SUBROUTINE lhs_primal_flip_flop_construct
  SUBROUTINE lhs_surface_height_construct(this, patch_3d, thick_e, op_coeffs_wp, op_coeffs_sp, lacc)
    USE mo_model_domain, ONLY: t_patch_3d
    USE mo_ocean_types, ONLY: t_operator_coeff, t_solvercoeff_singleprecision
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_ocean_nml, ONLY: l_lhs_direct, select_lhs
    USE mo_exception, ONLY: finish
    CLASS(t_surface_height_lhs), INTENT(INOUT) :: this
    TYPE(t_patch_3d), POINTER, INTENT(IN) :: patch_3d
    REAL(KIND = 8), POINTER, INTENT(IN) :: thick_e(:, :)
    TYPE(t_operator_coeff), TARGET, INTENT(IN) :: op_coeffs_wp
    TYPE(t_solvercoeff_singleprecision), TARGET, INTENT(IN) :: op_coeffs_sp
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    this % patch_3d => patch_3d
    this % patch_2d => patch_3d % p_patch_2d(1)
    this % thickness_e_wp => thick_e
    this % op_coeffs_wp => op_coeffs_wp
    this % op_coeffs_sp => op_coeffs_sp
    this % is_const = .FALSE.
    this % use_shortcut = (select_lhs .GT. 2 .AND. select_lhs .LE. 3)
    IF (this % patch_2d % cells % max_connectivity .NE. 3 .AND. .NOT. l_lhs_direct) CALL finish("t_surface_height_lhs::lhs_surface_height_construct", "internal matrix implementation only works with triangular grids!")
    this % is_init = .TRUE.
  END SUBROUTINE lhs_surface_height_construct
END MODULE mo_ocean_solve_lhs_type
MODULE mo_ocean_solve_transfer
  IMPLICIT NONE
  TYPE, ABSTRACT :: t_transfer
  END TYPE t_transfer
  TYPE, EXTENDS(t_transfer) :: t_subset_transfer
  END TYPE t_subset_transfer
  TYPE, EXTENDS(t_transfer) :: t_trivial_transfer
  END TYPE t_trivial_transfer
  CONTAINS
  SUBROUTINE subset_transfer_construct(this, sync_type, patch_2d, redfac, mode, lacc)
    USE mo_model_domain, ONLY: t_patch
    CLASS(t_subset_transfer), INTENT(INOUT) :: this
    INTEGER, INTENT(IN) :: sync_type, redfac, mode
    TYPE(t_patch), POINTER :: patch_2d
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
  END SUBROUTINE subset_transfer_construct
  SUBROUTINE trivial_transfer_construct(this, sync_type, patch_2d, lacc)
    USE mo_model_domain, ONLY: t_patch
    CLASS(t_trivial_transfer), INTENT(INOUT) :: this
    INTEGER, INTENT(IN) :: sync_type
    TYPE(t_patch), POINTER :: patch_2d
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
  END SUBROUTINE trivial_transfer_construct
END MODULE mo_ocean_solve_transfer
MODULE mo_ocean_solve_backend
  USE mo_ocean_solve_lhs, ONLY: t_lhs
  USE mo_ocean_solve_transfer, ONLY: t_transfer
  IMPLICIT NONE
  CHARACTER(LEN = *), PARAMETER :: this_mod_name = 'mo_ocean_solve_backend'
  TYPE, ABSTRACT :: t_ocean_solve_backend
    TYPE(t_lhs) :: lhs
    CLASS(t_transfer), POINTER :: trans => NULL()
  END TYPE t_ocean_solve_backend
  CONTAINS
  SUBROUTINE ocean_solve_backend_dump_matrix(this, id, lprecon, lacc)
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_exception, ONLY: finish
    USE mo_ocean_solve_lhs, ONLY: lhs_dump_matrix_deconproc_74 => lhs_dump_matrix, lhs_dump_matrix_deconproc_75 => lhs_dump_matrix
    CLASS(t_ocean_solve_backend), INTENT(INOUT) :: this
    INTEGER, INTENT(IN) :: id
    LOGICAL, INTENT(IN) :: lprecon
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    CHARACTER(LEN = *), PARAMETER :: routine = this_mod_name // "::ocean_solve_t::ocean_solve_dump_matrix()"
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    IF (.NOT. ASSOCIATED(this % trans)) CALL finish(routine, "ocean_solve_t was not initialized")
    IF (lprecon) THEN
      CALL lhs_dump_matrix_deconproc_74(this % lhs, id, "ocean_matrix_precon_", .TRUE., lacc = lzacc)
    ELSE
      CALL lhs_dump_matrix_deconproc_75(this % lhs, id, "ocean_matrix_lhs_", .FALSE., lacc = lzacc)
    END IF
  END SUBROUTINE ocean_solve_backend_dump_matrix
END MODULE mo_ocean_solve_backend
MODULE mo_ocean_solve
  USE mo_ocean_solve_backend, ONLY: t_ocean_solve_backend
  IMPLICIT NONE
  TYPE :: t_ocean_solve
    CLASS(t_ocean_solve_backend), ALLOCATABLE :: act
    REAL(KIND = 8), ALLOCATABLE, DIMENSION(:, :), PUBLIC :: x_loc_wp
    REAL(KIND = 8), POINTER, DIMENSION(:, :), PUBLIC :: b_loc_wp
    REAL(KIND = 8), ALLOCATABLE, DIMENSION(:), PUBLIC :: res_loc_wp
    CHARACTER(LEN = 64), PUBLIC :: sol_type_name
    LOGICAL, PUBLIC :: is_init = .FALSE.
  END TYPE t_ocean_solve
  CONTAINS
  SUBROUTINE ocean_solve_dump_matrix(this, id, lprecon_in, lacc)
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_ocean_solve_backend, ONLY: ocean_solve_backend_dump_matrix_deconproc_90 => ocean_solve_backend_dump_matrix
    CLASS(t_ocean_solve), INTENT(INOUT) :: this
    INTEGER, INTENT(IN) :: id
    LOGICAL, INTENT(IN), OPTIONAL :: lprecon_in
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    LOGICAL :: lprecon
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    lprecon = .FALSE.
    CALL ocean_solve_backend_dump_matrix_deconproc_90(this % act, id, lprecon, lacc = lzacc)
  END SUBROUTINE ocean_solve_dump_matrix
  SUBROUTINE ocean_solve_solve(this, niter, niter_sp, lacc)
    CLASS(t_ocean_solve), INTENT(INOUT) :: this
    INTEGER, INTENT(OUT) :: niter, niter_sp
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
  END SUBROUTINE ocean_solve_solve
  SUBROUTINE ocean_solve_construct__t_surface_height_lhs__t_trivial_transfer(this, st, par, par_sp, lhs_agen, trans, lacc)
    USE mo_ocean_solve_aux, ONLY: t_ocean_solve_parm
    USE mo_ocean_solve_lhs_type, ONLY: t_surface_height_lhs
    USE mo_ocean_solve_transfer, ONLY: t_trivial_transfer
    CLASS(t_ocean_solve), TARGET, INTENT(INOUT) :: this
    INTEGER, INTENT(IN) :: st
    TYPE(t_ocean_solve_parm), INTENT(IN) :: par, par_sp
    TYPE(t_surface_height_lhs), TARGET, INTENT(IN) :: lhs_agen
    TYPE(t_trivial_transfer), TARGET, INTENT(IN) :: trans
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
  END SUBROUTINE ocean_solve_construct__t_surface_height_lhs__t_trivial_transfer
  SUBROUTINE ocean_solve_construct__t_surface_height_lhs__t_subset_transfer(this, st, par, par_sp, lhs_agen, trans, lacc)
    USE mo_ocean_solve_aux, ONLY: t_ocean_solve_parm
    USE mo_ocean_solve_lhs_type, ONLY: t_surface_height_lhs
    USE mo_ocean_solve_transfer, ONLY: t_subset_transfer
    CLASS(t_ocean_solve), TARGET, INTENT(INOUT) :: this
    INTEGER, INTENT(IN) :: st
    TYPE(t_ocean_solve_parm), INTENT(IN) :: par, par_sp
    TYPE(t_surface_height_lhs), TARGET, INTENT(IN) :: lhs_agen
    TYPE(t_subset_transfer), TARGET, INTENT(IN) :: trans
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
  END SUBROUTINE ocean_solve_construct__t_surface_height_lhs__t_subset_transfer
  SUBROUTINE ocean_solve_construct__t_primal_flip_flop_lhs__t_triv__1f49ae23(this, st, par, par_sp, lhs_agen, trans, lacc)
    USE mo_ocean_solve_aux, ONLY: t_ocean_solve_parm
    USE mo_ocean_solve_lhs_type, ONLY: t_primal_flip_flop_lhs
    USE mo_ocean_solve_transfer, ONLY: t_trivial_transfer
    CLASS(t_ocean_solve), TARGET, INTENT(INOUT) :: this
    INTEGER, INTENT(IN) :: st
    TYPE(t_ocean_solve_parm), INTENT(IN) :: par, par_sp
    TYPE(t_primal_flip_flop_lhs), TARGET, INTENT(IN) :: lhs_agen
    TYPE(t_trivial_transfer), TARGET, INTENT(IN) :: trans
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
  END SUBROUTINE ocean_solve_construct__t_primal_flip_flop_lhs__t_triv__1f49ae23
END MODULE mo_ocean_solve
MODULE mo_ocean_thermodyn
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE calc_internal_press_grad(patch_3d, rho, pressure_hyd, bc_total_top_potential, grad_coeff, press_grad, lacc)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_nml, ONLY: n_zlev, oceanreferencedensity_inv
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_grid_subset, ONLY: get_index_range
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    REAL(KIND = 8), INTENT(IN) :: rho(nproma, n_zlev, patch_3d % p_patch_2d(1) % alloc_cell_blocks)
    REAL(KIND = 8), INTENT(INOUT) :: pressure_hyd(nproma, n_zlev, patch_3d % p_patch_2d(1) % alloc_cell_blocks)
    REAL(KIND = 8), INTENT(IN) :: bc_total_top_potential(nproma, patch_3d % p_patch_2d(1) % alloc_cell_blocks)
    REAL(KIND = 8), INTENT(IN) :: grad_coeff(:, :, :)
    REAL(KIND = 8), INTENT(INOUT) :: press_grad(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_e)
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: je, jk, jb, jc, ic1, ic2, ib1, ib2
    INTEGER :: start_index, end_index
    REAL(KIND = 8) :: z_grav_rho_inv
    TYPE(t_subset_range), POINTER :: edges_in_domain
    TYPE(t_patch), POINTER :: patch_2d
    TYPE(t_subset_range), POINTER :: all_cells
    REAL(KIND = 8) :: press_l, press_r
    REAL(KIND = 8) :: thick1, thick2
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    z_grav_rho_inv = oceanreferencedensity_inv * 9.80665D0
    patch_2d => patch_3d % p_patch_2d(1)
    edges_in_domain => patch_2d % edges % in_domain
    all_cells => patch_2d % cells % all
    pressure_hyd(1 : nproma, 1 : n_zlev, 1 : patch_3d % p_patch_2d(1) % alloc_cell_blocks) = 0.0D0
    DO jb = all_cells % start_block, all_cells % end_block
      CALL get_index_range(all_cells, jb, start_index, end_index)
      DO jc = start_index, end_index
        pressure_hyd(jc, 1, jb) = rho(jc, 1, jb) * z_grav_rho_inv * patch_3d % p_patch_1d(1) % constantprismcenters_zdistance(jc, 1, jb) + bc_total_top_potential(jc, jb)
        DO jk = 2, patch_3d % p_patch_1d(1) % dolic_c(jc, jb)
          pressure_hyd(jc, jk, jb) = pressure_hyd(jc, jk - 1, jb) + 0.5D0 * (rho(jc, jk, jb) + rho(jc, jk - 1, jb)) * z_grav_rho_inv * patch_3d % p_patch_1d(1) % constantprismcenters_zdistance(jc, jk, jb)
        END DO
      END DO
    END DO
    DO jb = edges_in_domain % start_block, edges_in_domain % end_block
      CALL get_index_range(edges_in_domain, jb, start_index, end_index)
      DO je = start_index, end_index
        ic1 = patch_2d % edges % cell_idx(je, jb, 1)
        ib1 = patch_2d % edges % cell_blk(je, jb, 1)
        ic2 = patch_2d % edges % cell_idx(je, jb, 2)
        ib2 = patch_2d % edges % cell_blk(je, jb, 2)
        DO jk = 1, patch_3d % p_patch_1d(1) % dolic_e(je, jb)
          thick1 = patch_3d % p_patch_1d(1) % constantprismcenters_zdistance(ic1, jk, ib1)
          thick2 = patch_3d % p_patch_1d(1) % constantprismcenters_zdistance(ic2, jk, ib2)
          IF ((jk .EQ. patch_3d % p_patch_1d(1) % dolic_e(je, jb)) .AND. (ABS(thick1 - thick2) > 1E-10)) THEN
            press_l = pressure_hyd(ic2, jk, ib2)
            press_r = pressure_hyd(ic1, jk, ib1)
            IF (thick1 > thick2) THEN
              press_r = pressure_hyd(ic1, jk - 1, ib1) + 0.5D0 * (rho(ic1, jk - 1, ib1) + rho(ic1, jk, ib1)) * z_grav_rho_inv * thick2
            ELSE
              press_l = pressure_hyd(ic2, jk - 1, ib2) + 0.5D0 * (rho(ic2, jk - 1, ib2) + rho(ic2, jk, ib2)) * z_grav_rho_inv * thick1
            END IF
            press_grad(je, jk, jb) = (press_l - press_r) * grad_coeff(je, jk, jb)
          ELSE
            press_grad(je, jk, jb) = (pressure_hyd(ic2, jk, ib2) - pressure_hyd(ic1, jk, ib1)) * grad_coeff(je, jk, jb)
          END IF
        END DO
      END DO
    END DO
  END SUBROUTINE calc_internal_press_grad
END MODULE mo_ocean_thermodyn
MODULE mo_statistics
  IMPLICIT NONE
  CHARACTER(LEN = *), PARAMETER :: module_name = "mo_statistics"
  CONTAINS
  SUBROUTINE print_2dvalue_location(values, seek_value, in_subset)
    USE mo_model_domain, ONLY: t_subset_range
    USE mo_math_types, ONLY: t_geographical_coordinates
    USE mo_exception, ONLY: finish
    USE mo_grid_subset, ONLY: get_index_range
    REAL(KIND = 8), INTENT(IN) :: values(:, :)
    REAL(KIND = 8), INTENT(IN) :: seek_value
    TYPE(t_subset_range), TARGET, INTENT(IN) :: in_subset
    INTEGER :: block, start_index, end_index, idx
    TYPE(t_geographical_coordinates), POINTER :: geocoordinates(:, :)
    CHARACTER(LEN = *), PARAMETER :: method_name = module_name // ':print_cell_value_location'
    SELECT CASE (in_subset % entity_location)
    CASE (1)
      geocoordinates => in_subset % patch % cells % center
    CASE (2)
      geocoordinates => in_subset % patch % edges % center
    CASE (3)
      geocoordinates => in_subset % patch % verts % vertex
    CASE DEFAULT
      CALL finish(method_name, "unknown subset%entity_location")
    END SELECT
    DO block = in_subset % start_block, in_subset % end_block
      CALL get_index_range(in_subset, block, start_index, end_index)
      DO idx = start_index, end_index
        IF (values(idx, block) == seek_value) THEN
          WRITE(0, *) "Value ", seek_value, " found at lon=", geocoordinates(idx, block) % lon * 57.29577951308232D0, " lat=", geocoordinates(idx, block) % lat * 57.29577951308232D0
        END IF
      END DO
    END DO
  END SUBROUTINE print_2dvalue_location
  FUNCTION minmaxmean_2d_inrange(values, in_subset, lacc) RESULT(minmaxmean)
    USE mo_model_domain, ONLY: t_subset_range
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_exception, ONLY: warning
    USE mo_grid_subset, ONLY: get_index_range
    REAL(KIND = 8), INTENT(IN) :: values(:, :)
    TYPE(t_subset_range), INTENT(IN) :: in_subset
    REAL(KIND = 8) :: minmaxmean(3)
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    LOGICAL :: lzacc
    REAL(KIND = 8) :: min_in_block, max_in_block, min_value, max_value, sum_value
    INTEGER :: block, start_index, end_index, number_of_values, idx
    CALL set_acc_host_or_device(lzacc, lacc)
    IF (in_subset % no_of_holes > 0) CALL warning(module_name, "there are holes in the subset")
    CALL init_min_max(min_value, max_value)
    sum_value = 0.0D0
    number_of_values = 0
    IF (ASSOCIATED(in_subset % vertical_levels)) THEN
      DO block = in_subset % start_block, in_subset % end_block
        CALL get_index_range(in_subset, block, start_index, end_index)
        DO idx = start_index, end_index
          IF (in_subset % vertical_levels(idx, block) > 0) THEN
            min_value = MIN(min_value, values(idx, block))
            max_value = MAX(max_value, values(idx, block))
            sum_value = sum_value + values(idx, block)
            number_of_values = number_of_values + 1
          END IF
        END DO
      END DO
    ELSE
      DO block = in_subset % start_block, in_subset % end_block
        CALL get_index_range(in_subset, block, start_index, end_index)
        min_in_block = MINVAL(values(start_index : end_index, block))
        max_in_block = MAXVAL(values(start_index : end_index, block))
        min_value = MIN(min_value, min_in_block)
        max_value = MAX(max_value, max_in_block)
        sum_value = sum_value + SUM(values(start_index : end_index, block))
      END DO
      number_of_values = in_subset % size
    END IF
    CALL gather_minmaxmean(min_value, max_value, sum_value, number_of_values, minmaxmean)
  END FUNCTION minmaxmean_2d_inrange
  SUBROUTINE init_min_max(min_value, max_value)
    REAL(KIND = 8), INTENT(OUT) :: min_value, max_value
    min_value = 1D+16
    max_value = (- 1D+16)
  END SUBROUTINE init_min_max
  SUBROUTINE gather_minmaxmean(min_value, max_value, sum_value, number_of_values, minmaxmean)
    USE mo_mpi, ONLY: get_my_mpi_work_communicator, my_process_is_mpi_parallel, p_max_dp_0d_deconiface_71 => p_max_dp_0d, p_min_dp_0d_deconiface_70 => p_min_dp_0d, p_sum_dp_0d_deconiface_72 => p_sum_dp_0d, p_sum_dp_0d_deconiface_73 => p_sum_dp_0d
    REAL(KIND = 8), INTENT(IN) :: min_value, max_value, sum_value
    INTEGER, INTENT(IN) :: number_of_values
    REAL(KIND = 8), INTENT(OUT) :: minmaxmean(3)
    REAL(KIND = 8) :: global_number_of_values
    INTEGER :: communicator
    IF (my_process_is_mpi_parallel()) THEN
      communicator = get_my_mpi_work_communicator()
      minmaxmean(1) = p_min_dp_0d_deconiface_70(min_value, comm = communicator)
      minmaxmean(2) = p_max_dp_0d_deconiface_71(max_value, comm = communicator)
      global_number_of_values = p_sum_dp_0d_deconiface_72(REAL(number_of_values, 8), comm = communicator)
      minmaxmean(3) = p_sum_dp_0d_deconiface_73(sum_value, comm = communicator) / global_number_of_values
    ELSE
      minmaxmean(1) = min_value
      minmaxmean(2) = max_value
      minmaxmean(3) = sum_value / REAL(number_of_values, 8)
    END IF
  END SUBROUTINE gather_minmaxmean
END MODULE mo_statistics
MODULE mo_sync
  IMPLICIT NONE
  INTEGER, SAVE :: log_unit = - 1
  LOGICAL, SAVE :: do_sync_checks = .TRUE.
  INTERFACE sync_patch_array_mult
    MODULE PROCEDURE sync_patch_array_mult_f3din_sp
    MODULE PROCEDURE sync_patch_array_mult_f3din_f4din_sp
    MODULE PROCEDURE sync_patch_array_mult_f3din_f3din_arr_sp
    MODULE PROCEDURE sync_patch_array_mult_f3din_dp
    MODULE PROCEDURE sync_patch_array_mult_f3din_f4din_dp
    MODULE PROCEDURE sync_patch_array_mult_f3din_f3din_arr_dp
  END INTERFACE
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
    USE mo_communication, ONLY: exchange_data_r3d_deconiface_76 => exchange_data_r3d
    INTEGER, INTENT(IN) :: typ
    TYPE(t_patch), TARGET, INTENT(IN) :: p_patch
    REAL(KIND = 8), INTENT(INOUT) :: arr(:, :, :)
    LOGICAL, INTENT(IN) :: lacc
    CHARACTER(LEN = *), TARGET, INTENT(IN), OPTIONAL :: opt_varname
    IF (p_test_run .AND. do_sync_checks) CALL check_patch_array_3d_dp(typ, p_patch, arr, lacc = lacc, opt_varname = opt_varname)
    IF (my_process_is_mpi_parallel()) THEN
      CALL exchange_data_r3d_deconiface_76(p_pat = comm_pat_of_type(p_patch, typ), lacc = lacc, recv = arr)
    END IF
  END SUBROUTINE sync_patch_array_3d_dp
  SUBROUTINE sync_patch_array_2d_dp(typ, p_patch, arr, lacc, opt_varname)
    USE mo_model_domain, ONLY: t_patch
    USE mo_fortran_tools, ONLY: insert_dimension_r_dp_3_2_deconiface_81 => insert_dimension_r_dp_3_2
    INTEGER, INTENT(IN) :: typ
    TYPE(t_patch), INTENT(IN) :: p_patch
    REAL(KIND = 8), TARGET, INTENT(INOUT) :: arr(:, :)
    LOGICAL, INTENT(IN) :: lacc
    CHARACTER*(*), INTENT(IN), OPTIONAL :: opt_varname
    REAL(KIND = 8), POINTER :: arr3(:, :, :)
    CALL insert_dimension_r_dp_3_2_deconiface_81(arr3, arr, 2)
    CALL sync_patch_array_3d_dp(1, p_patch, arr3, lacc = lacc, opt_varname = opt_varname)
  END SUBROUTINE sync_patch_array_2d_dp
  SUBROUTINE sync_patch_array_mult_f3din_dp(typ, p_patch, nfields, lacc, f3din1, f3din2, f3din3, f3din4, f3din5, opt_varname)
    USE mo_model_domain, ONLY: t_patch
    INTEGER, INTENT(IN) :: typ
    TYPE(t_patch), INTENT(IN), TARGET :: p_patch
    INTEGER, INTENT(IN) :: nfields
    LOGICAL, INTENT(IN) :: lacc
    REAL(KIND = 8), TARGET, INTENT(INOUT) :: f3din1(:, :, :)
    REAL(KIND = 8), TARGET, OPTIONAL, INTENT(INOUT) :: f3din2(:, :, :), f3din3(:, :, :), f3din4(:, :, :), f3din5(:, :, :)
    CHARACTER(LEN = *), TARGET, INTENT(IN), OPTIONAL :: opt_varname
    CALL sync_patch_array_mult_mixprec(typ = 3, p_patch = p_patch, nfields_sp = 0, nfields_dp = 3, lacc = lacc, f3din1_dp = f3din1, f3din2_dp = f3din2, f3din3_dp = f3din3, f3din4_dp = f3din4, f3din5_dp = f3din5, opt_varname = opt_varname)
  END SUBROUTINE sync_patch_array_mult_f3din_dp
  SUBROUTINE sync_patch_array_mult_f3din_f4din_dp(typ, p_patch, nfields, lacc, f3din1, f3din2, f3din3, f3din4, f3din5, f4din, opt_varname)
    USE mo_model_domain, ONLY: t_patch
    INTEGER, INTENT(IN) :: typ
    TYPE(t_patch), INTENT(IN), TARGET :: p_patch
    INTEGER, INTENT(IN) :: nfields
    LOGICAL, INTENT(IN) :: lacc
    REAL(KIND = 8), TARGET, INTENT(INOUT) :: f4din(:, :, :, :)
    REAL(KIND = 8), TARGET, OPTIONAL, INTENT(INOUT) :: f3din1(:, :, :), f3din2(:, :, :), f3din3(:, :, :), f3din4(:, :, :), f3din5(:, :, :)
    CHARACTER(LEN = *), TARGET, INTENT(IN), OPTIONAL :: opt_varname
    CALL sync_patch_array_mult_mixprec(typ = typ, p_patch = p_patch, nfields_sp = 0, nfields_dp = nfields, lacc = lacc, f3din1_dp = f3din1, f3din2_dp = f3din2, f3din3_dp = f3din3, f3din4_dp = f3din4, f3din5_dp = f3din5, f4din_dp = f4din, opt_varname = opt_varname)
  END SUBROUTINE sync_patch_array_mult_f3din_f4din_dp
  SUBROUTINE sync_patch_array_mult_f3din_f3din_arr_dp(typ, p_patch, nfields, lacc, f3din1, f3din2, f3din3, f3din4, f3din5, f3din_arr, opt_varname)
    USE mo_model_domain, ONLY: t_patch
    USE mo_fortran_tools, ONLY: t_ptr_3d_dp
    INTEGER, INTENT(IN) :: typ
    TYPE(t_patch), INTENT(IN), TARGET :: p_patch
    INTEGER, INTENT(IN) :: nfields
    LOGICAL, INTENT(IN) :: lacc
    REAL(KIND = 8), TARGET, OPTIONAL, INTENT(INOUT) :: f3din1(:, :, :), f3din2(:, :, :), f3din3(:, :, :), f3din4(:, :, :), f3din5(:, :, :)
    TYPE(t_ptr_3d_dp), INTENT(INOUT) :: f3din_arr(:)
    CHARACTER(LEN = *), TARGET, INTENT(IN), OPTIONAL :: opt_varname
    CALL sync_patch_array_mult_mixprec(typ = typ, p_patch = p_patch, nfields_sp = 0, nfields_dp = nfields, lacc = lacc, f3din1_dp = f3din1, f3din2_dp = f3din2, f3din3_dp = f3din3, f3din4_dp = f3din4, f3din5_dp = f3din5, f3din_arr_dp = f3din_arr, opt_varname = opt_varname)
  END SUBROUTINE sync_patch_array_mult_f3din_f3din_arr_dp
  SUBROUTINE sync_patch_array_mult_f3din_sp(typ, p_patch, nfields, lacc, f3din1, f3din2, f3din3, f3din4, f3din5, opt_varname)
    USE mo_model_domain, ONLY: t_patch
    INTEGER, INTENT(IN) :: typ
    TYPE(t_patch), INTENT(IN), TARGET :: p_patch
    INTEGER, INTENT(IN) :: nfields
    LOGICAL, INTENT(IN) :: lacc
    REAL(KIND = 4), TARGET, INTENT(INOUT) :: f3din1(:, :, :)
    REAL(KIND = 4), TARGET, OPTIONAL, INTENT(INOUT) :: f3din2(:, :, :), f3din3(:, :, :), f3din4(:, :, :), f3din5(:, :, :)
    CHARACTER(LEN = *), TARGET, INTENT(IN), OPTIONAL :: opt_varname
    CALL sync_patch_array_mult_mixprec(typ = typ, p_patch = p_patch, nfields_sp = nfields, nfields_dp = 0, lacc = lacc, f3din1_sp = f3din1, f3din2_sp = f3din2, f3din3_sp = f3din3, f3din4_sp = f3din4, f3din5_sp = f3din5, opt_varname = opt_varname)
  END SUBROUTINE sync_patch_array_mult_f3din_sp
  SUBROUTINE sync_patch_array_mult_f3din_f4din_sp(typ, p_patch, nfields, lacc, f3din1, f3din2, f3din3, f3din4, f3din5, f4din, opt_varname)
    USE mo_model_domain, ONLY: t_patch
    INTEGER, INTENT(IN) :: typ
    TYPE(t_patch), INTENT(IN), TARGET :: p_patch
    INTEGER, INTENT(IN) :: nfields
    LOGICAL, INTENT(IN) :: lacc
    REAL(KIND = 4), TARGET, INTENT(INOUT) :: f4din(:, :, :, :)
    REAL(KIND = 4), TARGET, OPTIONAL, INTENT(INOUT) :: f3din1(:, :, :), f3din2(:, :, :), f3din3(:, :, :), f3din4(:, :, :), f3din5(:, :, :)
    CHARACTER(LEN = *), TARGET, INTENT(IN), OPTIONAL :: opt_varname
    CALL sync_patch_array_mult_mixprec(typ = typ, p_patch = p_patch, nfields_sp = nfields, nfields_dp = 0, lacc = lacc, f3din1_sp = f3din1, f3din2_sp = f3din2, f3din3_sp = f3din3, f3din4_sp = f3din4, f3din5_sp = f3din5, f4din_sp = f4din, opt_varname = opt_varname)
  END SUBROUTINE sync_patch_array_mult_f3din_f4din_sp
  SUBROUTINE sync_patch_array_mult_f3din_f3din_arr_sp(typ, p_patch, nfields, lacc, f3din1, f3din2, f3din3, f3din4, f3din5, f3din_arr, opt_varname)
    USE mo_model_domain, ONLY: t_patch
    USE mo_fortran_tools, ONLY: t_ptr_3d_sp
    INTEGER, INTENT(IN) :: typ
    TYPE(t_patch), INTENT(IN), TARGET :: p_patch
    INTEGER, INTENT(IN) :: nfields
    LOGICAL, INTENT(IN) :: lacc
    REAL(KIND = 4), TARGET, OPTIONAL, INTENT(INOUT) :: f3din1(:, :, :), f3din2(:, :, :), f3din3(:, :, :), f3din4(:, :, :), f3din5(:, :, :)
    TYPE(t_ptr_3d_sp), INTENT(INOUT) :: f3din_arr(:)
    CHARACTER(LEN = *), TARGET, INTENT(IN), OPTIONAL :: opt_varname
    CALL sync_patch_array_mult_mixprec(typ = typ, p_patch = p_patch, nfields_sp = nfields, nfields_dp = 0, lacc = lacc, f3din1_sp = f3din1, f3din2_sp = f3din2, f3din3_sp = f3din3, f3din4_sp = f3din4, f3din5_sp = f3din5, f3din_arr_sp = f3din_arr, opt_varname = opt_varname)
  END SUBROUTINE sync_patch_array_mult_f3din_f3din_arr_sp
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
    INTEGER :: i
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
      IF (PRESENT(f4din_dp)) THEN
        DO i = 1, SIZE(f4din_dp, 4)
          CALL check_patch_array_3d_dp(typ, p_patch, f4din_dp(:, :, :, i), lacc = lacc, opt_varname = opt_varname)
        END DO
      END IF
      IF (PRESENT(f4din_sp)) THEN
        DO i = 1, SIZE(f4din_sp, 4)
          CALL check_patch_array_3d_sp(typ, p_patch, f4din_sp(:, :, :, i), lacc = lacc, opt_varname = opt_varname)
        END DO
      END IF
      IF (PRESENT(f3din1_dp)) CALL check_patch_array_3d_dp(typ, p_patch, f3din1_dp, lacc = lacc, opt_varname = opt_varname)
      IF (PRESENT(f3din2_dp)) CALL check_patch_array_3d_dp(typ, p_patch, f3din2_dp, lacc = lacc, opt_varname = opt_varname)
      IF (PRESENT(f3din3_dp)) CALL check_patch_array_3d_dp(typ, p_patch, f3din3_dp, lacc = lacc, opt_varname = opt_varname)
      IF (PRESENT(f3din4_dp)) CALL check_patch_array_3d_dp(typ, p_patch, f3din4_dp, lacc = lacc, opt_varname = opt_varname)
      IF (PRESENT(f3din5_dp)) CALL check_patch_array_3d_dp(typ, p_patch, f3din5_dp, lacc = lacc, opt_varname = opt_varname)
      IF (PRESENT(f3din1_sp)) CALL check_patch_array_3d_sp(typ, p_patch, f3din1_sp, lacc = lacc, opt_varname = opt_varname)
      IF (PRESENT(f3din2_sp)) CALL check_patch_array_3d_sp(typ, p_patch, f3din2_sp, lacc = lacc, opt_varname = opt_varname)
      IF (PRESENT(f3din3_sp)) CALL check_patch_array_3d_sp(typ, p_patch, f3din3_sp, lacc = lacc, opt_varname = opt_varname)
      IF (PRESENT(f3din4_sp)) CALL check_patch_array_3d_sp(typ, p_patch, f3din4_sp, lacc = lacc, opt_varname = opt_varname)
      IF (PRESENT(f3din5_sp)) CALL check_patch_array_3d_sp(typ, p_patch, f3din5_sp, lacc = lacc, opt_varname = opt_varname)
      IF (PRESENT(f3din_arr_dp)) THEN
        DO i = 1, SIZE(f3din_arr_dp)
          CALL check_patch_array_3d_dp(typ = typ, p_patch = p_patch, arr = f3din_arr_dp(i) % p, lacc = lacc, opt_varname = opt_varname)
        END DO
      END IF
      IF (PRESENT(f3din_arr_sp)) THEN
        DO i = 1, SIZE(f3din_arr_sp)
          CALL check_patch_array_3d_sp(typ, p_patch, f3din_arr_sp(i) % p, lacc = lacc, opt_varname = opt_varname)
        END DO
      END IF
    END IF
    IF (my_process_is_mpi_parallel()) THEN
      IF (PRESENT(f4din_dp)) THEN
        ndim2tot_dp = SIZE(f4din_dp, 4) * SIZE(f4din_dp, 2)
      ELSE
        ndim2tot_dp = 0
      END IF
      IF (PRESENT(f3din1_dp)) ndim2tot_dp = ndim2tot_dp + SIZE(f3din1_dp, 2)
      IF (PRESENT(f3din2_dp)) ndim2tot_dp = ndim2tot_dp + SIZE(f3din2_dp, 2)
      IF (PRESENT(f3din3_dp)) ndim2tot_dp = ndim2tot_dp + SIZE(f3din3_dp, 2)
      IF (PRESENT(f3din4_dp)) ndim2tot_dp = ndim2tot_dp + SIZE(f3din4_dp, 2)
      IF (PRESENT(f3din5_dp)) ndim2tot_dp = ndim2tot_dp + SIZE(f3din5_dp, 2)
      IF (PRESENT(f3din_arr_dp)) THEN
        DO i = 1, SIZE(f3din_arr_dp)
          ndim2tot_dp = ndim2tot_dp + SIZE(f3din_arr_dp(i) % p, 2)
        END DO
      END IF
      IF (PRESENT(f4din_sp)) THEN
        ndim2tot_sp = SIZE(f4din_sp, 4) * SIZE(f4din_sp, 2)
      ELSE
        ndim2tot_sp = 0
      END IF
      IF (PRESENT(f3din1_sp)) ndim2tot_sp = ndim2tot_sp + SIZE(f3din1_sp, 2)
      IF (PRESENT(f3din2_sp)) ndim2tot_sp = ndim2tot_sp + SIZE(f3din2_sp, 2)
      IF (PRESENT(f3din3_sp)) ndim2tot_sp = ndim2tot_sp + SIZE(f3din3_sp, 2)
      IF (PRESENT(f3din4_sp)) ndim2tot_sp = ndim2tot_sp + SIZE(f3din4_sp, 2)
      IF (PRESENT(f3din5_sp)) ndim2tot_sp = ndim2tot_sp + SIZE(f3din5_sp, 2)
      IF (PRESENT(f3din_arr_sp)) THEN
        DO i = 1, SIZE(f3din_arr_sp)
          ndim2tot_sp = ndim2tot_sp + SIZE(f3din_arr_sp(i) % p, 2)
        END DO
      END IF
      CALL exchange_data_mult_mixprec(p_pat = p_pat, lacc = lacc, nfields_dp = nfields_dp, ndim2tot_dp = ndim2tot_dp, nfields_sp = nfields_sp, ndim2tot_sp = ndim2tot_sp, recv1_dp = f3din1_dp, recv2_dp = f3din2_dp, recv3_dp = f3din3_dp, recv4_dp = f3din4_dp, recv5_dp = f3din5_dp, recv1_sp = f3din1_sp, recv2_sp = f3din2_sp, recv3_sp = f3din3_sp, recv4_sp = f3din4_sp, recv5_sp = f3din5_sp, recv4d_dp = f4din_dp, recv4d_sp = f4din_sp, recv3d_arr_dp = f3din_arr_dp, recv3d_arr_sp = f3din_arr_sp)
    END IF
  END SUBROUTINE sync_patch_array_mult_mixprec
  SUBROUTINE check_patch_array_3d_sp(typ, p_patch, arr, lacc, opt_varname)
    USE mo_model_domain, ONLY: t_patch
    USE mo_parallel_config, ONLY: blk_no, idx_no, l_log_checks, n_ghost_rows, nproma, p_test_run
    USE mo_communication_types, ONLY: t_comm_pattern_orig
    USE mo_io_units, ONLY: filename_max, find_next_free_unit
    USE mo_exception, ONLY: finish
    USE mo_mpi, ONLY: comm_lev, comm_proc0, glob_comm, my_process_is_mpi_test, num_test_procs, p_bcast_sp_3d_deconiface_85 => p_bcast_sp_3d, p_bcast_sp_3d_deconiface_87 => p_bcast_sp_3d, p_bcast_sp_3d_deconiface_89 => p_bcast_sp_3d, p_comm_work_test, p_pe, p_pe_work, p_recv_sp_3d_deconiface_88 => p_recv_sp_3d, p_send_sp_3d_deconiface_86 => p_send_sp_3d, p_work_pe0, process_mpi_all_test_id
    USE mo_communication, ONLY: exchange_data_s3d_deconiface_84 => exchange_data_s3d
    INTEGER, INTENT(IN) :: typ
    TYPE(t_patch), INTENT(IN), TARGET :: p_patch
    REAL(KIND = 4), INTENT(IN) :: arr(:, :, :)
    LOGICAL, INTENT(IN) :: lacc
    CHARACTER(LEN = *), INTENT(IN), OPTIONAL :: opt_varname
    REAL(KIND = 4), ALLOCATABLE :: arr_g(:, :, :)
    INTEGER :: j, jb, jl, jb_g, jl_g, n, ndim2, ndim3, nblks_g, flag, jk
    INTEGER :: ityp, ndim, ndim_g, jk_min_err
    INTEGER :: nerr(0 : n_ghost_rows), shape_recv(3)
    INTEGER, POINTER :: p_glb_index(:), p_decomp_domain(:, :)
    TYPE(t_comm_pattern_orig), POINTER :: p_pat_work2test
    LOGICAL :: l_my_process_is_mpi_test
    CHARACTER(LEN = 256) :: varname, cfmt
    INTEGER :: varname_tlen
    CHARACTER(LEN = filename_max) :: log_file
    REAL(KIND = 4) :: absmax
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
      CALL exchange_data_s3d_deconiface_84(p_pat = p_pat_work2test, lacc = .FALSE., recv = arr_g, send = arr)
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
        CALL p_bcast_sp_3d_deconiface_85(arr_g(:, :, 1 : nblks_g), process_mpi_all_test_id, comm = p_comm_work_test)
      ELSE
        CALL p_send_sp_3d_deconiface_86(arr_g(:, :, 1 : nblks_g), comm_proc0(comm_lev) + p_work_pe0, 1)
      END IF
      DEALLOCATE(arr_g)
    ELSE
      ALLOCATE(arr_g(nproma, ndim2, nblks_g))
      IF (comm_lev == 0) THEN
        CALL p_bcast_sp_3d_deconiface_87(arr_g(:, :, 1 : nblks_g), process_mpi_all_test_id, comm = p_comm_work_test)
      ELSE
        IF (p_pe_work == comm_proc0(comm_lev)) CALL p_recv_sp_3d_deconiface_88(arr_g(:, :, 1 : nblks_g), process_mpi_all_test_id, 1)
        CALL p_bcast_sp_3d_deconiface_89(arr_g(:, :, 1 : nblks_g), 0, comm = glob_comm(comm_lev))
      END IF
      nerr(:) = 0
      absmax = 0.0
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
        IF (ALL(arr == 0.0)) THEN
          WRITE(log_unit, cfmt) nerr(0 : n), varname(1 : varname_tlen), ': ALL 0 !!!'
        ELSE
          WRITE(log_unit, cfmt) nerr(0 : n), varname(1 : varname_tlen)
        END IF
        IF (absmax > 0.0) WRITE(log_unit, *) 'Max abs inner err:', absmax
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
  END SUBROUTINE check_patch_array_3d_sp
  SUBROUTINE check_patch_array_3d_dp(typ, p_patch, arr, lacc, opt_varname)
    USE mo_model_domain, ONLY: t_patch
    USE mo_parallel_config, ONLY: blk_no, idx_no, l_log_checks, n_ghost_rows, nproma, p_test_run
    USE mo_communication_types, ONLY: t_comm_pattern_orig
    USE mo_io_units, ONLY: filename_max, find_next_free_unit
    USE mo_exception, ONLY: finish
    USE mo_mpi, ONLY: comm_lev, comm_proc0, glob_comm, my_process_is_mpi_test, num_test_procs, p_bcast_dp_3d_deconiface_91 => p_bcast_dp_3d, p_bcast_dp_3d_deconiface_93 => p_bcast_dp_3d, p_bcast_dp_3d_deconiface_95 => p_bcast_dp_3d, p_comm_work_test, p_pe, p_pe_work, p_recv_dp_3d_deconiface_94 => p_recv_dp_3d, p_send_dp_3d_deconiface_92 => p_send_dp_3d, p_work_pe0, process_mpi_all_test_id
    USE mo_communication, ONLY: exchange_data_r3d_deconiface_90 => exchange_data_r3d
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
      CALL exchange_data_r3d_deconiface_90(p_pat = p_pat_work2test, lacc = .FALSE., recv = arr_g, send = arr)
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
        CALL p_bcast_dp_3d_deconiface_91(arr_g(:, :, 1 : nblks_g), process_mpi_all_test_id, comm = p_comm_work_test)
      ELSE
        CALL p_send_dp_3d_deconiface_92(arr_g(:, :, 1 : nblks_g), comm_proc0(comm_lev) + p_work_pe0, 1)
      END IF
      DEALLOCATE(arr_g)
    ELSE
      ALLOCATE(arr_g(nproma, ndim2, nblks_g))
      IF (comm_lev == 0) THEN
        CALL p_bcast_dp_3d_deconiface_93(arr_g(:, :, 1 : nblks_g), process_mpi_all_test_id, comm = p_comm_work_test)
      ELSE
        IF (p_pe_work == comm_proc0(comm_lev)) CALL p_recv_dp_3d_deconiface_94(arr_g(:, :, 1 : nblks_g), process_mpi_all_test_id, 1)
        CALL p_bcast_dp_3d_deconiface_95(arr_g(:, :, 1 : nblks_g), 0, comm = glob_comm(comm_lev))
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
MODULE mo_ocean_math_operators
  IMPLICIT NONE
  INTERFACE smooth_oncells
    MODULE PROCEDURE smooth_oncells_3d
    MODULE PROCEDURE smooth_oncells_2d
  END INTERFACE
  CONTAINS
  SUBROUTINE map_edges2vert_3d(patch_2d, vn, edge2vert_coeff_cc, vn_dual, lacc)
    USE mo_model_domain, ONLY: t_patch, t_subset_range
    USE mo_math_types, ONLY: t_cartesian_coordinates
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_nml, ONLY: n_zlev
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_grid_subset, ONLY: get_index_range
    TYPE(t_patch), TARGET, INTENT(IN) :: patch_2d
    REAL(KIND = 8), INTENT(IN) :: vn(:, :, :)
    TYPE(t_cartesian_coordinates), INTENT(IN) :: edge2vert_coeff_cc(:, :, :, :)
    TYPE(t_cartesian_coordinates) :: vn_dual(nproma, n_zlev, patch_2d % nblks_v)
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: start_level, end_level
    INTEGER :: vertexindex, level, blockno, vertexconnect
    INTEGER :: edgeofvertex_index, edgeofvertex_block
    INTEGER :: start_index_v, end_index_v
    TYPE(t_subset_range), POINTER :: verts_in_domain
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    verts_in_domain => patch_2d % verts % in_domain
    start_level = 1
    end_level = n_zlev
    DO blockno = verts_in_domain % start_block, verts_in_domain % end_block
      CALL get_index_range(verts_in_domain, blockno, start_index_v, end_index_v)
      DO level = 1, n_zlev
        DO vertexindex = 1, nproma
          vn_dual(vertexindex, level, blockno) % x(1) = 0.0D0
          vn_dual(vertexindex, level, blockno) % x(2) = 0.0D0
          vn_dual(vertexindex, level, blockno) % x(3) = 0.0D0
        END DO
      END DO
      DO vertexindex = start_index_v, end_index_v
        DO vertexconnect = 1, patch_2d % verts % num_edges(vertexindex, blockno)
          edgeofvertex_index = patch_2d % verts % edge_idx(vertexindex, blockno, vertexconnect)
          edgeofvertex_block = patch_2d % verts % edge_blk(vertexindex, blockno, vertexconnect)
          IF (edgeofvertex_index > 0) THEN
            DO level = start_level, end_level
              vn_dual(vertexindex, level, blockno) % x = vn_dual(vertexindex, level, blockno) % x + edge2vert_coeff_cc(vertexindex, level, blockno, vertexconnect) % x * vn(edgeofvertex_index, level, edgeofvertex_block)
            END DO
          END IF
        END DO
      END DO
    END DO
  END SUBROUTINE map_edges2vert_3d
  SUBROUTINE grad_fd_norm_oce_3d(psi_c, patch_3d, grad_coeff, grad_norm_psi_e)
    USE mo_model_domain, ONLY: t_patch_3d, t_subset_range
    USE mo_grid_subset, ONLY: get_index_range
    TYPE(t_patch_3d), TARGET :: patch_3d
    REAL(KIND = 8) :: grad_coeff(:, :, :)
    REAL(KIND = 8) :: psi_c(:, :, :)
    REAL(KIND = 8), INTENT(INOUT) :: grad_norm_psi_e(:, :, :)
    INTEGER :: start_edge_index, end_edge_index, blockno
    TYPE(t_subset_range), POINTER :: edges_in_domain
    edges_in_domain => patch_3d % p_patch_2d(1) % edges % in_domain
    DO blockno = edges_in_domain % start_block, edges_in_domain % end_block
      CALL get_index_range(edges_in_domain, blockno, start_edge_index, end_edge_index)
      CALL grad_fd_norm_oce_3d_onblock(psi_c, patch_3d, grad_coeff(:, :, blockno), grad_norm_psi_e(:, :, blockno), start_edge_index, end_edge_index, blockno)
    END DO
  END SUBROUTINE grad_fd_norm_oce_3d
  SUBROUTINE grad_fd_norm_oce_3d_onblock(psi_c, patch_3d, grad_coeff, grad_norm_psi_e, start_edge_index, end_edge_index, blockno, lacc)
    USE mo_model_domain, ONLY: t_patch_3d
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_nml, ONLY: n_zlev
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    REAL(KIND = 8), INTENT(IN) :: grad_coeff(:, :)
    REAL(KIND = 8), INTENT(IN) :: psi_c(nproma, n_zlev, patch_3d % p_patch_2d(1) % alloc_cell_blocks)
    REAL(KIND = 8), INTENT(INOUT) :: grad_norm_psi_e(nproma, n_zlev)
    INTEGER, INTENT(IN) :: start_edge_index, end_edge_index, blockno
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: je, level
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    DO je = start_edge_index, end_edge_index
      DO level = 1, patch_3d % p_patch_1d(1) % dolic_e(je, blockno)
        grad_norm_psi_e(je, level) = grad_coeff(je, level) * (psi_c(patch_3d % p_patch_2d(1) % edges % cell_idx(je, blockno, 2), level, patch_3d % p_patch_2d(1) % edges % cell_blk(je, blockno, 2)) - psi_c(patch_3d % p_patch_2d(1) % edges % cell_idx(je, blockno, 1), level, patch_3d % p_patch_2d(1) % edges % cell_blk(je, blockno, 1)))
      END DO
    END DO
  END SUBROUTINE grad_fd_norm_oce_3d_onblock
  SUBROUTINE grad_vector(cellvector, patch_3d, grad_coeff, gradvector)
    USE mo_model_domain, ONLY: t_patch_3d, t_subset_range
    USE mo_math_types, ONLY: t_cartesian_coordinates
    USE mo_grid_subset, ONLY: get_index_range
    USE mo_ocean_nml, ONLY: n_zlev
    TYPE(t_patch_3d), TARGET :: patch_3d
    REAL(KIND = 8) :: grad_coeff(:, :, :)
    TYPE(t_cartesian_coordinates) :: cellvector(:, :, :)
    TYPE(t_cartesian_coordinates) :: gradvector(:, :, :)
    INTEGER :: start_edge_index, end_edge_index, blockno, je, level
    INTEGER, DIMENSION(:, :, :), POINTER :: idx, blk
    TYPE(t_subset_range), POINTER :: edges_in_domain
    edges_in_domain => patch_3d % p_patch_2d(1) % edges % in_domain
    idx => patch_3d % p_patch_2d(1) % edges % cell_idx
    blk => patch_3d % p_patch_2d(1) % edges % cell_blk
    DO blockno = edges_in_domain % start_block, edges_in_domain % end_block
      CALL get_index_range(edges_in_domain, blockno, start_edge_index, end_edge_index)
      DO je = start_edge_index, end_edge_index
        DO level = 1, patch_3d % p_patch_1d(1) % dolic_e(je, blockno)
          gradvector(je, level, blockno) % x = grad_coeff(je, level, blockno) * (cellvector(idx(je, blockno, 2), level, blk(je, blockno, 2)) % x - cellvector(idx(je, blockno, 1), level, blk(je, blockno, 1)) % x)
        END DO
        DO level = patch_3d % p_patch_1d(1) % dolic_e(je, blockno) + 1, n_zlev
          gradvector(je, level, blockno) % x = 0.0D0
        END DO
      END DO
    END DO
  END SUBROUTINE grad_vector
  SUBROUTINE div_vector_ontriangle(patch_3d, edgevector, divvector, div_coeff)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_math_types, ONLY: t_cartesian_coordinates
    USE mo_exception, ONLY: finish
    USE mo_grid_subset, ONLY: get_index_range
    USE mo_ocean_nml, ONLY: n_zlev
    TYPE(t_patch_3d), TARGET :: patch_3d
    TYPE(t_cartesian_coordinates) :: edgevector(:, :, :)
    TYPE(t_cartesian_coordinates) :: divvector(:, :, :)
    REAL(KIND = 8) :: div_coeff(:, :, :, :)
    INTEGER :: start_index, end_index, cell_index, level, blockno
    TYPE(t_subset_range), POINTER :: cells_in_domain
    INTEGER, DIMENSION(:, :, :), POINTER :: idx, blk
    TYPE(t_patch), POINTER :: patch_2d
    patch_2d => patch_3d % p_patch_2d(1)
    IF (patch_2d % cells % max_connectivity /= 3) THEN
      CALL finish('div_vector_onTriangle', 'cells%max_connectivity /= 3')
    END IF
    cells_in_domain => patch_2d % cells % in_domain
    idx => patch_3d % p_patch_2d(1) % cells % edge_idx
    blk => patch_3d % p_patch_2d(1) % cells % edge_blk
    DO blockno = cells_in_domain % start_block, cells_in_domain % end_block
      CALL get_index_range(cells_in_domain, blockno, start_index, end_index)
      DO cell_index = start_index, end_index
        DO level = 1, patch_3d % p_patch_1d(1) % dolic_c(cell_index, blockno)
          divvector(cell_index, level, blockno) % x = edgevector(idx(cell_index, blockno, 1), level, blk(cell_index, blockno, 1)) % x * div_coeff(cell_index, level, blockno, 1) + edgevector(idx(cell_index, blockno, 2), level, blk(cell_index, blockno, 2)) % x * div_coeff(cell_index, level, blockno, 2) + edgevector(idx(cell_index, blockno, 3), level, blk(cell_index, blockno, 3)) % x * div_coeff(cell_index, level, blockno, 3)
        END DO
        DO level = patch_3d % p_patch_1d(1) % dolic_c(cell_index, blockno) + 1, n_zlev
          divvector(cell_index, level, blockno) % x = 0.0D0
        END DO
      END DO
    END DO
  END SUBROUTINE div_vector_ontriangle
  SUBROUTINE div_oce_3d_mlevels_ontriangles(vec_e, patch_3d, div_coeff, div_vec_c, opt_start_level, opt_end_level, subset_range, lacc)
    USE mo_model_domain, ONLY: t_patch_3d, t_subset_range
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_grid_subset, ONLY: get_index_range
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    REAL(KIND = 8), INTENT(IN) :: vec_e(:, :, :)
    REAL(KIND = 8), INTENT(IN) :: div_coeff(:, :, :, :)
    REAL(KIND = 8), INTENT(INOUT) :: div_vec_c(:, :, :)
    INTEGER, INTENT(IN), OPTIONAL :: opt_start_level
    INTEGER, INTENT(IN), OPTIONAL :: opt_end_level
    TYPE(t_subset_range), TARGET, INTENT(IN), OPTIONAL :: subset_range
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: start_level, end_level
    INTEGER :: blockno, start_block, end_block
    INTEGER :: start_index, end_index
    TYPE(t_subset_range), POINTER :: cells_subset
    LOGICAL :: lzacc
    cells_subset => subset_range
    start_block = cells_subset % start_block
    end_block = cells_subset % end_block
    start_level = opt_start_level
    end_level = opt_end_level
    CALL set_acc_host_or_device(lzacc, lacc)
    DO blockno = start_block, end_block
      CALL get_index_range(cells_subset, blockno, start_index, end_index)
      CALL div_oce_3d_ontriangles_onblock(vec_e, patch_3d, div_coeff, div_vec_c(:, :, blockno), blockno, start_index, end_index, start_level, end_level, lacc = lzacc)
    END DO
  END SUBROUTINE div_oce_3d_mlevels_ontriangles
  SUBROUTINE div_oce_3d_ontriangles_onblock(vec_e, patch_3d, div_coeff, div_vec_c, blockno, start_index, end_index, start_level, end_level, lacc)
    USE mo_model_domain, ONLY: t_patch_3d
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    REAL(KIND = 8), INTENT(IN) :: vec_e(:, :, :)
    REAL(KIND = 8), INTENT(IN) :: div_coeff(:, :, :, :)
    REAL(KIND = 8), INTENT(INOUT) :: div_vec_c(:, :)
    INTEGER, INTENT(IN) :: blockno, start_index, end_index
    INTEGER, INTENT(IN) :: start_level, end_level
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: jc, level
    INTEGER, DIMENSION(:, :), POINTER :: dolic_c
    LOGICAL :: lzacc
    dolic_c => patch_3d % p_patch_1d(1) % dolic_c
    CALL set_acc_host_or_device(lzacc, lacc)
    div_vec_c(:, :) = 0.0D0
    DO jc = start_index, end_index
      DO level = start_level, MIN(end_level, dolic_c(jc, blockno))
        div_vec_c(jc, level) = vec_e(patch_3d % p_patch_2d(1) % cells % edge_idx(jc, blockno, 1), level, patch_3d % p_patch_2d(1) % cells % edge_blk(jc, blockno, 1)) * div_coeff(jc, level, blockno, 1) + vec_e(patch_3d % p_patch_2d(1) % cells % edge_idx(jc, blockno, 2), level, patch_3d % p_patch_2d(1) % cells % edge_blk(jc, blockno, 2)) * div_coeff(jc, level, blockno, 2) + vec_e(patch_3d % p_patch_2d(1) % cells % edge_idx(jc, blockno, 3), level, patch_3d % p_patch_2d(1) % cells % edge_blk(jc, blockno, 3)) * div_coeff(jc, level, blockno, 3)
      END DO
    END DO
  END SUBROUTINE div_oce_3d_ontriangles_onblock
  SUBROUTINE div_oce_3d_general_onblock(vec_e, patch_3d, div_coeff, div_vec_c, blockno, start_index, end_index, start_level, end_level, lacc)
    USE mo_model_domain, ONLY: t_patch_3d
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    REAL(KIND = 8), INTENT(IN) :: vec_e(:, :, :)
    REAL(KIND = 8), INTENT(IN) :: div_coeff(:, :, :, :)
    REAL(KIND = 8), INTENT(INOUT) :: div_vec_c(:, :)
    INTEGER, INTENT(IN) :: blockno, start_index, end_index
    INTEGER, INTENT(IN) :: start_level, end_level
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: jc, level, max_connectivity, c
    INTEGER, DIMENSION(:, :), POINTER :: dolic_c
    REAL(KIND = 8) :: temp_div_vec
    LOGICAL :: lzacc
    dolic_c => patch_3d % p_patch_1d(1) % dolic_c
    max_connectivity = patch_3d % p_patch_2d(1) % cells % max_connectivity
    CALL set_acc_host_or_device(lzacc, lacc)
    div_vec_c(:, :) = 0.0D0
    DO jc = start_index, end_index
      DO level = 1, MIN(end_level, dolic_c(jc, blockno))
        temp_div_vec = 0.0D0
        DO c = 1, max_connectivity
          IF (patch_3d % p_patch_2d(1) % cells % edge_idx(jc, blockno, c) > 0) THEN
            temp_div_vec = temp_div_vec + vec_e(patch_3d % p_patch_2d(1) % cells % edge_idx(jc, blockno, c), level, patch_3d % p_patch_2d(1) % cells % edge_blk(jc, blockno, c)) * div_coeff(jc, level, blockno, c)
          END IF
        END DO
        div_vec_c(jc, level) = temp_div_vec
      END DO
    END DO
  END SUBROUTINE div_oce_3d_general_onblock
  SUBROUTINE div_oce_3d_mlevels(vec_e, patch_3d, div_coeff, div_vec_c, opt_start_level, opt_end_level, subset_range, lacc)
    USE mo_model_domain, ONLY: t_patch_3d, t_subset_range
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_ocean_nml, ONLY: n_zlev
    USE mo_grid_subset, ONLY: get_index_range
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    REAL(KIND = 8), INTENT(IN) :: vec_e(:, :, :)
    REAL(KIND = 8), INTENT(IN) :: div_coeff(:, :, :, :)
    REAL(KIND = 8), INTENT(INOUT) :: div_vec_c(:, :, :)
    INTEGER, INTENT(IN), OPTIONAL :: opt_start_level
    INTEGER, INTENT(IN), OPTIONAL :: opt_end_level
    TYPE(t_subset_range), TARGET, INTENT(IN), OPTIONAL :: subset_range
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: start_level, end_level
    INTEGER :: jc, level, blockno, max_connectivity, edgeofcell
    INTEGER :: start_index, end_index, start_block, end_block
    TYPE(t_subset_range), POINTER :: cells_subset
    INTEGER, DIMENSION(:, :), POINTER :: dolic_c
    REAL(KIND = 8) :: temp_div_vec
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    IF (patch_3d % p_patch_2d(1) % cells % max_connectivity == 3) THEN
      CALL div_oce_3d_mlevels_ontriangles(vec_e, patch_3d, div_coeff, div_vec_c, opt_start_level, opt_end_level, subset_range, lacc = lzacc)
      RETURN
    END IF
    IF (PRESENT(subset_range)) THEN
      cells_subset => subset_range
    ELSE
      cells_subset => patch_3d % p_patch_2d(1) % cells % in_domain
    END IF
    start_block = cells_subset % start_block
    end_block = cells_subset % end_block
    start_level = 1
    end_level = n_zlev
    max_connectivity = patch_3d % p_patch_2d(1) % cells % max_connectivity
    dolic_c => patch_3d % p_patch_1d(1) % dolic_c
    DO blockno = start_block, end_block
      CALL get_index_range(cells_subset, blockno, start_index, end_index)
      div_vec_c(:, :, blockno) = 0.0D0
      DO jc = start_index, end_index
        DO level = start_level, MIN(end_level, dolic_c(jc, blockno))
          temp_div_vec = 0.0D0
          DO edgeofcell = 1, max_connectivity
            IF (patch_3d % p_patch_2d(1) % cells % edge_idx(jc, blockno, edgeofcell) > 0) THEN
              temp_div_vec = temp_div_vec + vec_e(patch_3d % p_patch_2d(1) % cells % edge_idx(jc, blockno, edgeofcell), level, patch_3d % p_patch_2d(1) % cells % edge_blk(jc, blockno, edgeofcell)) * div_coeff(jc, level, blockno, edgeofcell)
            END IF
          END DO
          div_vec_c(jc, level, blockno) = temp_div_vec
        END DO
      END DO
    END DO
  END SUBROUTINE div_oce_3d_mlevels
  SUBROUTINE grad_fd_norm_oce_2d_onblock(psi_c, patch_2d, grad_coeff, grad_norm_psi_e, start_index, end_index, blockno, lacc)
    USE mo_model_domain, ONLY: t_patch
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    TYPE(t_patch), TARGET, INTENT(IN) :: patch_2d
    REAL(KIND = 8), INTENT(IN) :: psi_c(:, :)
    REAL(KIND = 8), INTENT(IN) :: grad_coeff(:)
    REAL(KIND = 8), INTENT(INOUT) :: grad_norm_psi_e(:)
    INTEGER, INTENT(IN) :: start_index, end_index, blockno
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: je
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    DO je = start_index, end_index
      grad_norm_psi_e(je) = (psi_c(patch_2d % edges % cell_idx(je, blockno, 2), patch_2d % edges % cell_blk(je, blockno, 2)) - psi_c(patch_2d % edges % cell_idx(je, blockno, 1), patch_2d % edges % cell_blk(je, blockno, 1))) * grad_coeff(je)
    END DO
  END SUBROUTINE grad_fd_norm_oce_2d_onblock
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
      CALL get_index_range(verts_in_domain, blockno, start_index_v, end_index_v)
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
  SUBROUTINE verticalderiv_vec_midlevel_on_block(patch_3d, vec_in, vertderiv_vec, start_level, blockno, start_index, end_index, lacc)
    USE mo_model_domain, ONLY: t_patch_3d
    USE mo_math_types, ONLY: t_cartesian_coordinates
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_nml, ONLY: n_zlev
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    TYPE(t_cartesian_coordinates), INTENT(IN) :: vec_in(nproma, n_zlev)
    INTEGER, INTENT(IN) :: start_level
    INTEGER, INTENT(IN) :: blockno, start_index, end_index
    TYPE(t_cartesian_coordinates), INTENT(INOUT) :: vertderiv_vec(:, :)
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: jk, jc
    LOGICAL :: lzacc
    REAL(KIND = 8), POINTER :: inv_prism_center_distance(:, :)
    CALL set_acc_host_or_device(lzacc, lacc)
    inv_prism_center_distance => patch_3d % p_patch_1d(1) % constantprismcenters_invzdistance(:, :, blockno)
    DO jc = start_index, end_index
      DO jk = 2, patch_3d % p_patch_1d(1) % dolic_c(jc, blockno)
        vertderiv_vec(jc, jk) % x = (vec_in(jc, jk - 1) % x - vec_in(jc, jk) % x) * inv_prism_center_distance(jc, jk)
      END DO
    END DO
  END SUBROUTINE verticalderiv_vec_midlevel_on_block
  SUBROUTINE verticaldiv_vector_onfulllevels_on_block(patch_3d, vector_in, vertdiv_vector, start_level, blockno, start_index, end_index)
    USE mo_model_domain, ONLY: t_patch_3d
    USE mo_math_types, ONLY: t_cartesian_coordinates
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    TYPE(t_cartesian_coordinates) :: vector_in(:, :)
    INTEGER, INTENT(IN) :: start_level
    INTEGER, INTENT(IN) :: blockno, start_index, end_index
    TYPE(t_cartesian_coordinates) :: vertdiv_vector(:, :)
    INTEGER :: jk, jc
    REAL(KIND = 8), POINTER :: inv_prism_thickness(:, :)
    inv_prism_thickness => patch_3d % p_patch_1d(1) % invconstantprismthickness(:, :, blockno)
    DO jc = start_index, end_index
      DO jk = 1, patch_3d % p_patch_1d(1) % dolic_c(jc, blockno)
        vertdiv_vector(jc, jk) % x = (vector_in(jc, jk) % x - vector_in(jc, jk + 1) % x) * inv_prism_thickness(jc, jk)
      END DO
    END DO
  END SUBROUTINE verticaldiv_vector_onfulllevels_on_block
  SUBROUTINE smooth_oncells_3d(patch_3d, in_value, out_value, smooth_weights, has_missvalue, missvalue)
    USE mo_model_domain, ONLY: t_patch_3d, t_subset_range
    USE mo_grid_subset, ONLY: get_index_range
    USE mo_sync, ONLY: sync_patch_array_3d_dp_deconiface_132 => sync_patch_array_3d_dp
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    REAL(KIND = 8), INTENT(IN) :: in_value(:, :, :)
    REAL(KIND = 8), INTENT(INOUT) :: out_value(:, :, :)
    REAL(KIND = 8), INTENT(IN) :: smooth_weights(1 : 2)
    LOGICAL, INTENT(IN) :: has_missvalue
    REAL(KIND = 8), INTENT(IN) :: missvalue
    INTEGER :: max_connectivity, blockno, start_index, end_index, jc, level, neigbor, neigbor_index, neigbor_block
    REAL(KIND = 8) :: numberofneigbors, neigbors_weight
    TYPE(t_subset_range), POINTER :: cells_indomain
    cells_indomain => patch_3d % p_patch_2d(1) % cells % owned
    max_connectivity = patch_3d % p_patch_2d(1) % cells % max_connectivity
    IF (has_missvalue) THEN
      DO blockno = cells_indomain % start_block, cells_indomain % end_block
        CALL get_index_range(cells_indomain, blockno, start_index, end_index)
        out_value(:, :, blockno) = 0.0D0
        DO jc = start_index, end_index
          DO level = 1, patch_3d % p_patch_1d(1) % dolic_c(jc, blockno)
            numberofneigbors = 0.0D0
            DO neigbor = 1, max_connectivity
              neigbor_index = patch_3d % p_patch_2d(1) % cells % neighbor_idx(jc, blockno, neigbor)
              neigbor_block = patch_3d % p_patch_2d(1) % cells % neighbor_blk(jc, blockno, neigbor)
              IF (neigbor_block > 0) THEN
                IF (patch_3d % p_patch_1d(1) % dolic_c(neigbor_index, neigbor_block) >= level .AND. in_value(neigbor_index, level, neigbor_block) /= missvalue) THEN
                  out_value(jc, level, blockno) = out_value(jc, level, blockno) + in_value(neigbor_index, level, neigbor_block)
                  numberofneigbors = numberofneigbors + 1.0D0
                END IF
              END IF
            END DO
            IF (numberofneigbors > 0.0D0) THEN
              IF (in_value(jc, level, blockno) /= missvalue) THEN
                out_value(jc, level, blockno) = out_value(jc, level, blockno) * smooth_weights(2) / numberofneigbors + in_value(jc, level, blockno) * smooth_weights(1)
              ELSE
                out_value(jc, level, blockno) = out_value(jc, level, blockno) / numberofneigbors
              END IF
            ELSE
              out_value(jc, level, blockno) = in_value(jc, level, blockno)
            END IF
          END DO
        END DO
      END DO
    ELSE
      DO blockno = cells_indomain % start_block, cells_indomain % end_block
        CALL get_index_range(cells_indomain, blockno, start_index, end_index)
        out_value(:, :, blockno) = 0.0D0
        DO jc = start_index, end_index
          DO level = 1, patch_3d % p_patch_1d(1) % dolic_c(jc, blockno)
            numberofneigbors = 0.0D0
            DO neigbor = 1, max_connectivity
              neigbor_index = patch_3d % p_patch_2d(1) % cells % neighbor_idx(jc, blockno, neigbor)
              neigbor_block = patch_3d % p_patch_2d(1) % cells % neighbor_blk(jc, blockno, neigbor)
              IF (neigbor_block > 0) THEN
                IF (patch_3d % p_patch_1d(1) % dolic_c(neigbor_index, neigbor_block) >= level) THEN
                  out_value(jc, level, blockno) = out_value(jc, level, blockno) + in_value(neigbor_index, level, neigbor_block)
                  numberofneigbors = numberofneigbors + 1.0D0
                END IF
              END IF
            END DO
            IF (numberofneigbors > 0.0D0) THEN
              neigbors_weight = smooth_weights(2) / numberofneigbors
              out_value(jc, level, blockno) = out_value(jc, level, blockno) * neigbors_weight + in_value(jc, level, blockno) * smooth_weights(1)
            ELSE
              out_value(jc, level, blockno) = in_value(jc, level, blockno)
            END IF
          END DO
        END DO
      END DO
    END IF
    CALL sync_patch_array_3d_dp_deconiface_132(1, patch_3d % p_patch_2d(1), out_value, lacc = .FALSE.)
  END SUBROUTINE smooth_oncells_3d
  SUBROUTINE smooth_oncells_2d(patch_3d, in_value, out_value, smooth_weights, has_missvalue, missvalue, lacc)
    USE mo_model_domain, ONLY: t_patch_3d, t_subset_range
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_grid_subset, ONLY: get_index_range
    USE mo_sync, ONLY: sync_patch_array_2d_dp_deconiface_133 => sync_patch_array_2d_dp
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    REAL(KIND = 8), INTENT(IN) :: in_value(:, :)
    REAL(KIND = 8), INTENT(INOUT) :: out_value(:, :)
    REAL(KIND = 8), INTENT(IN) :: smooth_weights(1 : 2)
    LOGICAL, INTENT(IN) :: has_missvalue
    REAL(KIND = 8), INTENT(IN) :: missvalue
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: max_connectivity, blockno, start_index, end_index, jc, level, neigbor, neigbor_index, neigbor_block
    REAL(KIND = 8) :: numberofneigbors, neigbors_weight
    TYPE(t_subset_range), POINTER :: cells_indomain
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    cells_indomain => patch_3d % p_patch_2d(1) % cells % owned
    max_connectivity = patch_3d % p_patch_2d(1) % cells % max_connectivity
    IF (has_missvalue) THEN
      DO blockno = cells_indomain % start_block, cells_indomain % end_block
        CALL get_index_range(cells_indomain, blockno, start_index, end_index)
        out_value(:, blockno) = 0.0D0
        DO jc = start_index, end_index
          DO level = 1, MIN(patch_3d % p_patch_1d(1) % dolic_c(jc, blockno), 1)
            numberofneigbors = 0.0D0
            DO neigbor = 1, max_connectivity
              neigbor_index = patch_3d % p_patch_2d(1) % cells % neighbor_idx(jc, blockno, neigbor)
              neigbor_block = patch_3d % p_patch_2d(1) % cells % neighbor_blk(jc, blockno, neigbor)
              IF (neigbor_block > 0) THEN
                IF (patch_3d % p_patch_1d(1) % dolic_c(neigbor_index, neigbor_block) >= level .AND. in_value(neigbor_index, neigbor_block) /= missvalue) THEN
                  out_value(jc, blockno) = out_value(jc, blockno) + in_value(neigbor_index, neigbor_block)
                  numberofneigbors = numberofneigbors + 1.0D0
                END IF
              END IF
            END DO
            IF (numberofneigbors > 0.0D0) THEN
              IF (in_value(jc, blockno) /= missvalue) THEN
                out_value(jc, blockno) = out_value(jc, blockno) * smooth_weights(2) / numberofneigbors + in_value(jc, blockno) * smooth_weights(1)
              ELSE
                out_value(jc, blockno) = out_value(jc, blockno) / numberofneigbors
              END IF
            ELSE
              out_value(jc, blockno) = in_value(jc, blockno)
            END IF
          END DO
        END DO
      END DO
    ELSE
      DO blockno = cells_indomain % start_block, cells_indomain % end_block
        CALL get_index_range(cells_indomain, blockno, start_index, end_index)
        out_value(:, blockno) = 0.0D0
        DO jc = start_index, end_index
          DO level = 1, MIN(patch_3d % p_patch_1d(1) % dolic_c(jc, blockno), 1)
            numberofneigbors = 0.0D0
            DO neigbor = 1, max_connectivity
              neigbor_index = patch_3d % p_patch_2d(1) % cells % neighbor_idx(jc, blockno, neigbor)
              neigbor_block = patch_3d % p_patch_2d(1) % cells % neighbor_blk(jc, blockno, neigbor)
              IF (neigbor_block > 0) THEN
                IF (patch_3d % p_patch_1d(1) % dolic_c(neigbor_index, neigbor_block) >= level) THEN
                  out_value(jc, blockno) = out_value(jc, blockno) + in_value(neigbor_index, neigbor_block)
                  numberofneigbors = numberofneigbors + 1.0D0
                END IF
              END IF
            END DO
            IF (numberofneigbors > 0.0D0) THEN
              neigbors_weight = smooth_weights(2) / numberofneigbors
              out_value(jc, blockno) = out_value(jc, blockno) * neigbors_weight + in_value(jc, blockno) * smooth_weights(1)
            ELSE
              out_value(jc, blockno) = in_value(jc, blockno)
            END IF
          END DO
        END DO
      END DO
    END IF
    CALL sync_patch_array_2d_dp_deconiface_133(1, patch_3d % p_patch_2d(1), out_value, lacc = lzacc)
  END SUBROUTINE smooth_oncells_2d
END MODULE mo_ocean_math_operators
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
    USE mo_sync, ONLY: sync_patch_array_3d_dp_deconiface_134 => sync_patch_array_3d_dp
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
    CALL sync_patch_array_3d_dp_deconiface_134(3, patch_2d, vort_v, lacc = lzacc)
    IF (.NOT. l_anticipated_vorticity) THEN
      DO blockno = edges_in_domain % start_block, edges_in_domain % end_block
        CALL get_index_range(edges_in_domain, blockno, start_edge_index, end_edge_index)
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
        CALL get_index_range(edges_in_domain, blockno, start_edge_index, end_edge_index)
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
  SUBROUTINE nonlinear_coriolis_3d(patch_3d, vn, p_vn_dual, vort_v, operators_coefficients, vort_flux, lacc)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_nml, ONLY: fast_performance_level, n_zlev
    USE mo_math_types, ONLY: t_cartesian_coordinates
    USE mo_ocean_types, ONLY: t_operator_coeff
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_ocean_math_operators, ONLY: rot_vertex_ocean_3d
    USE mo_sync, ONLY: sync_patch_array_3d_dp_deconiface_135 => sync_patch_array_3d_dp
    USE mo_grid_subset, ONLY: get_index_range
    USE mo_operator_ocean_coeff_3d, ONLY: no_dual_edges
    TYPE(t_patch_3d), POINTER, INTENT(IN) :: patch_3d
    REAL(KIND = 8), INTENT(INOUT) :: vn(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_e)
    TYPE(t_cartesian_coordinates), INTENT(INOUT) :: p_vn_dual(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_v)
    REAL(KIND = 8), INTENT(INOUT) :: vort_v(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_v)
    TYPE(t_operator_coeff), TARGET, INTENT(IN) :: operators_coefficients
    REAL(KIND = 8), INTENT(INOUT) :: vort_flux(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_e)
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: startlevel, endlevel
    INTEGER :: je, level, blockno
    INTEGER :: il_e, ib_e
    INTEGER :: start_edge_index, end_edge_index
    INTEGER :: ictr, neighbor, vertex_edge
    INTEGER :: il_v, ib_v
    LOGICAL :: lzacc
    REAL(KIND = 8) :: vort_global
    TYPE(t_subset_range), POINTER :: edges_in_domain
    TYPE(t_patch), POINTER :: patch_2d
    CALL set_acc_host_or_device(lzacc, lacc)
    IF (fast_performance_level > 10) THEN
      CALL nonlinear_coriolis_3d_fast_scalar(patch_3d, vn, p_vn_dual, vort_v, operators_coefficients, vort_flux, lacc = lzacc)
      RETURN
    END IF
    patch_2d => patch_3d % p_patch_2d(1)
    edges_in_domain => patch_2d % edges % in_domain
    startlevel = 1
    endlevel = n_zlev
    CALL rot_vertex_ocean_3d(patch_3d, vn, p_vn_dual, operators_coefficients, vort_v)
    CALL sync_patch_array_3d_dp_deconiface_135(3, patch_2d, vort_v, lacc = lzacc)
    DO blockno = edges_in_domain % start_block, edges_in_domain % end_block
      CALL get_index_range(edges_in_domain, blockno, start_edge_index, end_edge_index)
      level_loop:DO level = startlevel, endlevel
        edge_idx_loop:DO je = start_edge_index, end_edge_index
          IF (patch_3d % lsm_e(je, level, blockno) == (- 2)) THEN
            vort_flux(je, level, blockno) = 0.0D0
            DO neighbor = 1, 2
              IF (neighbor == 1) ictr = 0
              IF (neighbor == 2) ictr = no_dual_edges
              il_v = patch_2d % edges % vertex_idx(je, blockno, neighbor)
              ib_v = patch_2d % edges % vertex_blk(je, blockno, neighbor)
              vort_global = (vort_v(il_v, level, ib_v) + patch_2d % verts % f_v(il_v, ib_v))
              DO vertex_edge = 1, patch_2d % verts % num_edges(il_v, ib_v)
                ictr = ictr + 1
                il_e = patch_2d % verts % edge_idx(il_v, ib_v, vertex_edge)
                ib_e = patch_2d % verts % edge_blk(il_v, ib_v, vertex_edge)
                vort_flux(je, level, blockno) = vort_flux(je, level, blockno) + vn(il_e, level, ib_e) * vort_global * operators_coefficients % edge2edge_viavert_coeff(je, level, blockno, ictr)
              END DO
            END DO
          ELSE
            vort_flux(je, level, blockno) = 0.0D0
          END IF
        END DO edge_idx_loop
      END DO level_loop
    END DO
  END SUBROUTINE nonlinear_coriolis_3d
  SUBROUTINE map_edges2edges_viacell_3d_mlev_const_z(patch_3d, vn_e, operators_coefficients, out_vn_e, lacc)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_nml, ONLY: fast_performance_level, n_zlev
    USE mo_ocean_types, ONLY: t_operator_coeff
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_grid_subset, ONLY: get_index_range
    USE mo_operator_ocean_coeff_3d, ONLY: no_primal_edges
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    REAL(KIND = 8), INTENT(IN) :: vn_e(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_e)
    TYPE(t_operator_coeff), INTENT(IN) :: operators_coefficients
    REAL(KIND = 8), INTENT(INOUT) :: out_vn_e(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_e)
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: startlevel, endlevel, start_block, end_block
    INTEGER :: start_edge_index, end_edge_index
    INTEGER :: il_e, ib_e, il_c, ib_c, ictr
    INTEGER :: je, blockno, level, ie
    REAL(KIND = 8) :: thick_edge
    TYPE(t_subset_range), POINTER :: edges_indomain
    TYPE(t_patch), POINTER :: patch_2d
    LOGICAL :: lzacc
    patch_2d => patch_3d % p_patch_2d(1)
    CALL set_acc_host_or_device(lzacc, lacc)
    IF (patch_2d % cells % max_connectivity == 3 .AND. fast_performance_level > 10) THEN
      CALL map_edges2edges_viacell_3d_mlev_constz_ontriangles(patch_3d, vn_e, operators_coefficients, out_vn_e, lacc = lzacc)
      RETURN
    END IF
    edges_indomain => patch_2d % edges % in_domain
    startlevel = 1
    endlevel = n_zlev
    start_block = edges_indomain % start_block
    end_block = edges_indomain % end_block
    DO blockno = start_block, end_block
      CALL get_index_range(edges_indomain, blockno, start_edge_index, end_edge_index)
      out_vn_e(:, :, blockno) = 0.0D0
      level_loop_e:DO level = startlevel, endlevel
        edge_idx_loop:DO je = start_edge_index, end_edge_index
          IF (patch_3d % lsm_e(je, level, blockno) == (- 2)) THEN
            out_vn_e(je, level, blockno) = 0.0D0
            ictr = 0
            il_c = patch_2d % edges % cell_idx(je, blockno, 1)
            ib_c = patch_2d % edges % cell_blk(je, blockno, 1)
            DO ie = 1, no_primal_edges
              ictr = ictr + 1
              il_e = patch_2d % cells % edge_idx(il_c, ib_c, ie)
              ib_e = patch_2d % cells % edge_blk(il_c, ib_c, ie)
              IF (il_e > 0) THEN
                thick_edge = patch_3d % p_patch_1d(1) % prism_thick_e(il_e, level, ib_e)
                out_vn_e(je, level, blockno) = out_vn_e(je, level, blockno) + vn_e(il_e, level, ib_e) * operators_coefficients % edge2edge_viacell_coeff(je, level, blockno, ictr) * thick_edge
              END IF
            END DO
            ictr = no_primal_edges
            il_c = patch_2d % edges % cell_idx(je, blockno, 2)
            ib_c = patch_2d % edges % cell_blk(je, blockno, 2)
            DO ie = 1, no_primal_edges
              ictr = ictr + 1
              il_e = patch_2d % cells % edge_idx(il_c, ib_c, ie)
              ib_e = patch_2d % cells % edge_blk(il_c, ib_c, ie)
              IF (il_e > 0) THEN
                thick_edge = patch_3d % p_patch_1d(1) % prism_thick_e(il_e, level, ib_e)
                out_vn_e(je, level, blockno) = out_vn_e(je, level, blockno) + vn_e(il_e, level, ib_e) * operators_coefficients % edge2edge_viacell_coeff(je, level, blockno, ictr) * thick_edge
              END IF
            END DO
          END IF
        END DO edge_idx_loop
      END DO level_loop_e
    END DO
  END SUBROUTINE map_edges2edges_viacell_3d_mlev_const_z
  SUBROUTINE map_edges2edges_viacell_3d_mlev_constz_ontriangles(patch_3d, vn_e, operators_coefficients, out_vn_e, lacc)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_nml, ONLY: n_zlev
    USE mo_ocean_types, ONLY: t_operator_coeff
    USE mo_operator_ocean_coeff_3d, ONLY: no_primal_edges
    USE mo_exception, ONLY: finish
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_grid_subset, ONLY: get_index_range
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    REAL(KIND = 8), INTENT(IN) :: vn_e(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_e)
    TYPE(t_operator_coeff), INTENT(IN) :: operators_coefficients
    REAL(KIND = 8), INTENT(INOUT) :: out_vn_e(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_e)
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: startlevel, endlevel
    INTEGER :: start_edge_index, end_edge_index
    INTEGER :: je, blockno, level, start_block, end_block
    INTEGER :: cell_1_index, cell_2_index, cell_1_block, cell_2_block
    INTEGER :: edge_11_index, edge_12_index, edge_13_index
    INTEGER :: edge_11_block, edge_12_block, edge_13_block
    INTEGER :: edge_21_index, edge_22_index, edge_23_index
    INTEGER :: edge_21_block, edge_22_block, edge_23_block
    TYPE(t_subset_range), POINTER :: edges_in_domain
    TYPE(t_patch), POINTER :: patch_2d
    LOGICAL :: lzacc
    INTEGER, POINTER :: dolic_e(:, :)
    IF (no_primal_edges /= 3) CALL finish('map_edges2edges_viacell triangle version', 'no_primal_edges /= 3')
    CALL set_acc_host_or_device(lzacc, lacc)
    patch_2d => patch_3d % p_patch_2d(1)
    edges_in_domain => patch_2d % edges % in_domain
    startlevel = 1
    endlevel = n_zlev
    start_block = edges_in_domain % start_block
    end_block = edges_in_domain % end_block
    dolic_e => patch_3d % p_patch_1d(1) % dolic_e
    DO blockno = start_block, end_block
      CALL get_index_range(edges_in_domain, blockno, start_edge_index, end_edge_index)
      DO level = 1, n_zlev
        DO je = 1, nproma
          out_vn_e(je, level, blockno) = 0.0D0
        END DO
      END DO
      DO je = start_edge_index, end_edge_index
        IF (patch_3d % p_patch_1d(1) % dolic_e(je, blockno) < 1) CYCLE
        cell_1_index = patch_2d % edges % cell_idx(je, blockno, 1)
        cell_1_block = patch_2d % edges % cell_blk(je, blockno, 1)
        cell_2_index = patch_2d % edges % cell_idx(je, blockno, 2)
        cell_2_block = patch_2d % edges % cell_blk(je, blockno, 2)
        edge_11_index = patch_2d % cells % edge_idx(cell_1_index, cell_1_block, 1)
        edge_12_index = patch_2d % cells % edge_idx(cell_1_index, cell_1_block, 2)
        edge_13_index = patch_2d % cells % edge_idx(cell_1_index, cell_1_block, 3)
        edge_11_block = patch_2d % cells % edge_blk(cell_1_index, cell_1_block, 1)
        edge_12_block = patch_2d % cells % edge_blk(cell_1_index, cell_1_block, 2)
        edge_13_block = patch_2d % cells % edge_blk(cell_1_index, cell_1_block, 3)
        edge_21_index = patch_2d % cells % edge_idx(cell_2_index, cell_2_block, 1)
        edge_22_index = patch_2d % cells % edge_idx(cell_2_index, cell_2_block, 2)
        edge_23_index = patch_2d % cells % edge_idx(cell_2_index, cell_2_block, 3)
        edge_21_block = patch_2d % cells % edge_blk(cell_2_index, cell_2_block, 1)
        edge_22_block = patch_2d % cells % edge_blk(cell_2_index, cell_2_block, 2)
        edge_23_block = patch_2d % cells % edge_blk(cell_2_index, cell_2_block, 3)
        DO level = startlevel, dolic_e(je, blockno)
          out_vn_e(je, level, blockno) = (vn_e(edge_11_index, level, edge_11_block) * operators_coefficients % edge2edge_viacell_coeff(je, level, blockno, 1) * patch_3d % p_patch_1d(1) % prism_thick_e(edge_11_index, level, edge_11_block) + vn_e(edge_12_index, level, edge_12_block) * operators_coefficients % edge2edge_viacell_coeff(je, level, blockno, 2) * patch_3d % p_patch_1d(1) % prism_thick_e(edge_12_index, level, edge_12_block) + vn_e(edge_13_index, level, edge_13_block) * operators_coefficients % edge2edge_viacell_coeff(je, level, blockno, 3) * patch_3d % p_patch_1d(1) % prism_thick_e(edge_13_index, level, edge_13_block)) + (vn_e(edge_21_index, level, edge_21_block) * operators_coefficients % edge2edge_viacell_coeff(je, level, blockno, 4) * patch_3d % p_patch_1d(1) % prism_thick_e(edge_21_index, level, edge_21_block) + vn_e(edge_22_index, level, edge_22_block) * operators_coefficients % edge2edge_viacell_coeff(je, level, blockno, 5) * patch_3d % p_patch_1d(1) % prism_thick_e(edge_22_index, level, edge_22_block) + vn_e(edge_23_index, level, edge_23_block) * operators_coefficients % edge2edge_viacell_coeff(je, level, blockno, 6) * patch_3d % p_patch_1d(1) % prism_thick_e(edge_23_index, level, edge_23_block))
        END DO
      END DO
    END DO
  END SUBROUTINE map_edges2edges_viacell_3d_mlev_constz_ontriangles
  SUBROUTINE map_edges2edges_viacell_2d_constz(patch_3d, vn_e, operators_coefficients, out_vn_e, lacc)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_types, ONLY: t_operator_coeff
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_ocean_nml, ONLY: fast_performance_level
    USE mo_grid_subset, ONLY: get_index_range
    USE mo_operator_ocean_coeff_3d, ONLY: no_primal_edges
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    REAL(KIND = 8), INTENT(IN) :: vn_e(nproma, patch_3d % p_patch_2d(1) % nblks_e)
    TYPE(t_operator_coeff), INTENT(IN) :: operators_coefficients
    REAL(KIND = 8), INTENT(INOUT) :: out_vn_e(nproma, patch_3d % p_patch_2d(1) % nblks_e)
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: startlevel, endlevel, start_block, end_block
    INTEGER :: start_edge_index, end_edge_index
    INTEGER :: il_e, ib_e, il_c, ib_c, ictr
    INTEGER :: je, blockno, level, ie
    REAL(KIND = 8) :: thick_edge
    TYPE(t_subset_range), POINTER :: edges_indomain
    TYPE(t_patch), POINTER :: patch_2d
    LOGICAL :: lzacc
    patch_2d => patch_3d % p_patch_2d(1)
    CALL set_acc_host_or_device(lzacc, lacc)
    IF (patch_2d % cells % max_connectivity == 3 .AND. fast_performance_level > 10) THEN
      CALL map_edges2edges_viacell_2d_constz_ontriangles(patch_3d, vn_e, operators_coefficients, out_vn_e, lacc = lzacc)
      RETURN
    END IF
    edges_indomain => patch_2d % edges % in_domain
    startlevel = 1
    endlevel = 1
    start_block = edges_indomain % start_block
    end_block = edges_indomain % end_block
    DO blockno = start_block, end_block
      CALL get_index_range(edges_indomain, blockno, start_edge_index, end_edge_index)
      out_vn_e(:, blockno) = 0.0D0
      edge_idx_loop:DO je = start_edge_index, end_edge_index
        endlevel = patch_3d % p_patch_1d(1) % dolic_e(je, blockno)
        level_loop_e:DO level = startlevel, endlevel
          IF (patch_3d % lsm_e(je, level, blockno) == (- 2)) THEN
            ictr = 0
            il_c = patch_2d % edges % cell_idx(je, blockno, 1)
            ib_c = patch_2d % edges % cell_blk(je, blockno, 1)
            DO ie = 1, no_primal_edges
              ictr = ictr + 1
              il_e = patch_2d % cells % edge_idx(il_c, ib_c, ie)
              ib_e = patch_2d % cells % edge_blk(il_c, ib_c, ie)
              IF (il_e > 0) THEN
                thick_edge = patch_3d % p_patch_1d(1) % prism_thick_e(il_e, level, ib_e)
                out_vn_e(je, blockno) = out_vn_e(je, blockno) + vn_e(il_e, ib_e) * operators_coefficients % edge2edge_viacell_coeff(je, level, blockno, ictr) * thick_edge
              END IF
            END DO
            ictr = no_primal_edges
            il_c = patch_2d % edges % cell_idx(je, blockno, 2)
            ib_c = patch_2d % edges % cell_blk(je, blockno, 2)
            DO ie = 1, no_primal_edges
              ictr = ictr + 1
              il_e = patch_2d % cells % edge_idx(il_c, ib_c, ie)
              ib_e = patch_2d % cells % edge_blk(il_c, ib_c, ie)
              IF (il_e > 0) THEN
                thick_edge = patch_3d % p_patch_1d(1) % prism_thick_e(il_e, level, ib_e)
                out_vn_e(je, blockno) = out_vn_e(je, blockno) + vn_e(il_e, ib_e) * operators_coefficients % edge2edge_viacell_coeff(je, level, blockno, ictr) * thick_edge
              END IF
            END DO
          END IF
        END DO level_loop_e
      END DO edge_idx_loop
    END DO
  END SUBROUTINE map_edges2edges_viacell_2d_constz
  SUBROUTINE map_edges2edges_viacell_2d_constz_ontriangles(patch_3d, vn_e, operators_coefficients, out_vn_e, lacc)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_types, ONLY: t_operator_coeff
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_grid_subset, ONLY: get_index_range
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    REAL(KIND = 8), INTENT(IN) :: vn_e(nproma, patch_3d % p_patch_2d(1) % nblks_e)
    TYPE(t_operator_coeff), INTENT(IN) :: operators_coefficients
    REAL(KIND = 8), INTENT(INOUT) :: out_vn_e(nproma, patch_3d % p_patch_2d(1) % nblks_e)
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: cell_1_index, cell_2_index, cell_1_block, cell_2_block
    INTEGER :: edge_1_1_index, edge_1_2_index, edge_1_3_index
    INTEGER :: edge_2_1_index, edge_2_2_index, edge_2_3_index
    INTEGER :: edge_1_1_block, edge_1_2_block, edge_1_3_block
    INTEGER :: edge_2_1_block, edge_2_2_block, edge_2_3_block
    INTEGER :: je, blockno, start_edge_index, end_edge_index, level
    INTEGER :: start_block, end_block
    LOGICAL :: lzacc
    TYPE(t_subset_range), POINTER :: edges_indomain
    TYPE(t_patch), POINTER :: patch_2d
    INTEGER, POINTER :: dolic_e(:, :)
    patch_2d => patch_3d % p_patch_2d(1)
    edges_indomain => patch_2d % edges % in_domain
    dolic_e => patch_3d % p_patch_1d(1) % dolic_e
    start_block = edges_indomain % start_block
    end_block = edges_indomain % end_block
    CALL set_acc_host_or_device(lzacc, lacc)
    DO blockno = start_block, end_block
      CALL get_index_range(edges_indomain, blockno, start_edge_index, end_edge_index)
      DO je = start_edge_index, end_edge_index
        out_vn_e(je, blockno) = 0.0D0
        DO level = 1, MIN(1, dolic_e(je, blockno))
          cell_1_index = patch_2d % edges % cell_idx(je, blockno, 1)
          cell_1_block = patch_2d % edges % cell_blk(je, blockno, 1)
          cell_2_index = patch_2d % edges % cell_idx(je, blockno, 2)
          cell_2_block = patch_2d % edges % cell_blk(je, blockno, 2)
          edge_1_1_index = patch_2d % cells % edge_idx(cell_1_index, cell_1_block, 1)
          edge_1_2_index = patch_2d % cells % edge_idx(cell_1_index, cell_1_block, 2)
          edge_1_3_index = patch_2d % cells % edge_idx(cell_1_index, cell_1_block, 3)
          edge_2_1_index = patch_2d % cells % edge_idx(cell_2_index, cell_2_block, 1)
          edge_2_2_index = patch_2d % cells % edge_idx(cell_2_index, cell_2_block, 2)
          edge_2_3_index = patch_2d % cells % edge_idx(cell_2_index, cell_2_block, 3)
          edge_1_1_block = patch_2d % cells % edge_blk(cell_1_index, cell_1_block, 1)
          edge_1_2_block = patch_2d % cells % edge_blk(cell_1_index, cell_1_block, 2)
          edge_1_3_block = patch_2d % cells % edge_blk(cell_1_index, cell_1_block, 3)
          edge_2_1_block = patch_2d % cells % edge_blk(cell_2_index, cell_2_block, 1)
          edge_2_2_block = patch_2d % cells % edge_blk(cell_2_index, cell_2_block, 2)
          edge_2_3_block = patch_2d % cells % edge_blk(cell_2_index, cell_2_block, 3)
          out_vn_e(je, blockno) = vn_e(edge_1_1_index, edge_1_1_block) * operators_coefficients % edge2edge_viacell_coeff_all(1, je, blockno) + vn_e(edge_1_2_index, edge_1_2_block) * operators_coefficients % edge2edge_viacell_coeff_all(2, je, blockno) + vn_e(edge_1_3_index, edge_1_3_block) * operators_coefficients % edge2edge_viacell_coeff_all(3, je, blockno) + vn_e(edge_2_1_index, edge_2_1_block) * operators_coefficients % edge2edge_viacell_coeff_all(4, je, blockno) + vn_e(edge_2_2_index, edge_2_2_block) * operators_coefficients % edge2edge_viacell_coeff_all(5, je, blockno) + vn_e(edge_2_3_index, edge_2_3_block) * operators_coefficients % edge2edge_viacell_coeff_all(6, je, blockno)
        END DO
      END DO
    END DO
  END SUBROUTINE map_edges2edges_viacell_2d_constz_ontriangles
  SUBROUTINE map_edges2edges_viacell_2d_per_level(patch_3d, vn_e, operators_coefficients, out_vn_e, level)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_types, ONLY: t_operator_coeff
    USE mo_grid_subset, ONLY: get_index_range
    USE mo_operator_ocean_coeff_3d, ONLY: no_primal_edges
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    REAL(KIND = 8), INTENT(IN) :: vn_e(nproma, patch_3d % p_patch_2d(1) % nblks_e)
    TYPE(t_operator_coeff), INTENT(IN) :: operators_coefficients
    REAL(KIND = 8), INTENT(INOUT) :: out_vn_e(nproma, patch_3d % p_patch_2d(1) % nblks_e)
    INTEGER, INTENT(IN) :: level
    INTEGER :: start_edge_index, end_edge_index
    INTEGER :: il_e, ib_e, il_c, ib_c, ictr
    INTEGER :: je, blockno, ie
    REAL(KIND = 8) :: thick_edge, thick_cell
    TYPE(t_subset_range), POINTER :: edges_indomain
    TYPE(t_patch), POINTER :: patch_2d
    patch_2d => patch_3d % p_patch_2d(1)
    edges_indomain => patch_2d % edges % in_domain
    DO blockno = edges_indomain % start_block, edges_indomain % end_block
      CALL get_index_range(edges_indomain, blockno, start_edge_index, end_edge_index)
      out_vn_e(:, blockno) = 0.0D0
      edge_idx_loop:DO je = start_edge_index, end_edge_index
        IF (patch_3d % lsm_e(je, 1, blockno) == (- 2)) THEN
          ictr = 0
          il_c = patch_2d % edges % cell_idx(je, blockno, 1)
          ib_c = patch_2d % edges % cell_blk(je, blockno, 1)
          thick_cell = patch_3d % p_patch_1d(1) % prism_thick_c(il_c, 1, ib_c)
          DO ie = 1, no_primal_edges
            ictr = ictr + 1
            il_e = patch_2d % cells % edge_idx(il_c, ib_c, ie)
            ib_e = patch_2d % cells % edge_blk(il_c, ib_c, ie)
            thick_edge = patch_3d % p_patch_1d(1) % prism_thick_e(il_e, 1, ib_e)
            out_vn_e(je, blockno) = out_vn_e(je, blockno) + vn_e(il_e, ib_e) * operators_coefficients % edge2edge_viacell_coeff(je, 1, blockno, ictr) * (thick_edge / thick_cell)
          END DO
          ictr = no_primal_edges
          il_c = patch_2d % edges % cell_idx(je, blockno, 2)
          ib_c = patch_2d % edges % cell_blk(je, blockno, 2)
          thick_cell = patch_3d % p_patch_1d(1) % prism_thick_c(il_c, 1, ib_c)
          DO ie = 1, no_primal_edges
            ictr = ictr + 1
            il_e = patch_2d % cells % edge_idx(il_c, ib_c, ie)
            ib_e = patch_2d % cells % edge_blk(il_c, ib_c, ie)
            thick_edge = patch_3d % p_patch_1d(1) % prism_thick_e(il_e, 1, ib_e)
            out_vn_e(je, blockno) = out_vn_e(je, blockno) + vn_e(il_e, ib_e) * operators_coefficients % edge2edge_viacell_coeff(je, 1, blockno, ictr) * (thick_edge / thick_cell)
          END DO
        END IF
      END DO edge_idx_loop
    END DO
  END SUBROUTINE map_edges2edges_viacell_2d_per_level
  SUBROUTINE map_cell2edges_3d_mlevels(patch_3d, p_vn_c, ptp_vn, operators_coefficients, opt_startlevel, opt_endlevel, subset_range, lacc)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_math_types, ONLY: t_cartesian_coordinates
    USE mo_ocean_types, ONLY: t_operator_coeff
    USE mo_ocean_nml, ONLY: n_zlev
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_grid_subset, ONLY: get_index_range
    USE mo_parallel_config, ONLY: nproma
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    TYPE(t_cartesian_coordinates), INTENT(IN) :: p_vn_c(:, :, :)
    REAL(KIND = 8), INTENT(INOUT) :: ptp_vn(:, :, :)
    TYPE(t_operator_coeff), INTENT(IN) :: operators_coefficients
    INTEGER, INTENT(IN), OPTIONAL :: opt_startlevel
    INTEGER, INTENT(IN), OPTIONAL :: opt_endlevel
    TYPE(t_subset_range), TARGET, INTENT(IN), OPTIONAL :: subset_range
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: startlevel, endlevel
    INTEGER :: start_edge_index, end_edge_index
    INTEGER :: je, blockno, level
    INTEGER :: cell_1_index, cell_1_block, cell_2_index, cell_2_block
    LOGICAL :: lzacc
    TYPE(t_subset_range), POINTER :: edges_in_domain
    TYPE(t_patch), POINTER :: patch_2d
    patch_2d => patch_3d % p_patch_2d(1)
    edges_in_domain => patch_2d % edges % in_domain
    startlevel = 1
    endlevel = n_zlev
    CALL set_acc_host_or_device(lzacc, lacc)
    DO blockno = edges_in_domain % start_block, edges_in_domain % end_block
      CALL get_index_range(edges_in_domain, blockno, start_edge_index, end_edge_index)
      DO level = 1, n_zlev
        DO je = 1, nproma
          ptp_vn(je, level, blockno) = 0.0D0
        END DO
      END DO
      DO je = start_edge_index, end_edge_index
        cell_1_index = patch_2d % edges % cell_idx(je, blockno, 1)
        cell_1_block = patch_2d % edges % cell_blk(je, blockno, 1)
        cell_2_index = patch_2d % edges % cell_idx(je, blockno, 2)
        cell_2_block = patch_2d % edges % cell_blk(je, blockno, 2)
        DO level = 1, MIN(endlevel, patch_3d % p_patch_1d(1) % dolic_e(je, blockno))
          ptp_vn(je, level, blockno) = DOT_PRODUCT(p_vn_c(cell_1_index, level, cell_1_block) % x, operators_coefficients % edge2cell_coeff_cc_t(je, level, blockno, 1) % x) + DOT_PRODUCT(p_vn_c(cell_2_index, level, cell_2_block) % x, operators_coefficients % edge2cell_coeff_cc_t(je, level, blockno, 2) % x)
        END DO
      END DO
    END DO
  END SUBROUTINE map_cell2edges_3d_mlevels
  SUBROUTINE map_cell2edges_3d_1level(patch_3d, p_vn_c, ptp_vn, operators_coefficients, level, subset_range, lacc)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_math_types, ONLY: t_cartesian_coordinates
    USE mo_ocean_types, ONLY: t_operator_coeff
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_grid_subset, ONLY: get_index_range
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    TYPE(t_cartesian_coordinates), INTENT(IN) :: p_vn_c(:, :)
    REAL(KIND = 8), INTENT(INOUT) :: ptp_vn(:, :)
    TYPE(t_operator_coeff), INTENT(IN) :: operators_coefficients
    INTEGER, INTENT(IN) :: level
    TYPE(t_subset_range), TARGET, INTENT(IN), OPTIONAL :: subset_range
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: start_edge_index, end_edge_index
    INTEGER :: je, blockno
    INTEGER :: cell_1_index, cell_1_block, cell_2_index, cell_2_block
    TYPE(t_subset_range), POINTER :: edges_in_domain
    TYPE(t_patch), POINTER :: patch_2d
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    patch_2d => patch_3d % p_patch_2d(1)
    edges_in_domain => patch_2d % edges % in_domain
    DO blockno = edges_in_domain % start_block, edges_in_domain % end_block
      ptp_vn(:, blockno) = 0.0D0
      CALL get_index_range(edges_in_domain, blockno, start_edge_index, end_edge_index)
      DO je = start_edge_index, end_edge_index
        IF (patch_3d % p_patch_1d(1) % dolic_e(je, blockno) > 0) THEN
          cell_1_index = patch_2d % edges % cell_idx(je, blockno, 1)
          cell_1_block = patch_2d % edges % cell_blk(je, blockno, 1)
          cell_2_index = patch_2d % edges % cell_idx(je, blockno, 2)
          cell_2_block = patch_2d % edges % cell_blk(je, blockno, 2)
          ptp_vn(je, blockno) = DOT_PRODUCT(p_vn_c(cell_1_index, cell_1_block) % x, operators_coefficients % edge2cell_coeff_cc_t(je, 1, blockno, 1) % x) + DOT_PRODUCT(p_vn_c(cell_2_index, cell_2_block) % x, operators_coefficients % edge2cell_coeff_cc_t(je, 1, blockno, 2) % x)
        END IF
      END DO
    END DO
  END SUBROUTINE map_cell2edges_3d_1level
  SUBROUTINE map_vec_prismtop2center_on_block(patch_3d, vec_top, vec_center, blockno, start_cell_index, end_cell_index, lacc)
    USE mo_model_domain, ONLY: t_patch_3d
    USE mo_math_types, ONLY: t_cartesian_coordinates
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    TYPE(t_cartesian_coordinates), INTENT(IN) :: vec_top(:, :)
    TYPE(t_cartesian_coordinates), INTENT(INOUT) :: vec_center(:, :)
    INTEGER, INTENT(IN) :: blockno, start_cell_index, end_cell_index
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: level, jc
    INTEGER :: start_level, end_level
    LOGICAL :: lzacc
    REAL(KIND = 8), POINTER :: prism_center_distance(:, :), prism_thick(:, :)
    start_level = 1
    CALL set_acc_host_or_device(lzacc, lacc)
    prism_center_distance => patch_3d % p_patch_1d(1) % constantprismcenters_zdistance(:, :, blockno)
    prism_thick => patch_3d % p_patch_1d(1) % prism_thick_flat_sfc_c(:, :, blockno)
    DO jc = start_cell_index, end_cell_index
      end_level = patch_3d % p_patch_1d(1) % dolic_c(jc, blockno)
      DO level = start_level, end_level
        vec_center(jc, level) % x = (prism_center_distance(jc, level) * vec_top(jc, level) % x + prism_center_distance(jc, level + 1) * vec_top(jc, level + 1) % x) / (2.0D0 * prism_thick(jc, level))
      END DO
    END DO
  END SUBROUTINE map_vec_prismtop2center_on_block
  SUBROUTINE map_scalar_prismtop2center_onblock(patch_3d, scalar_top, scalar_center, blockno, start_cell_index, end_cell_index)
    USE mo_model_domain, ONLY: t_patch_3d
    TYPE(t_patch_3d), TARGET :: patch_3d
    REAL(KIND = 8) :: scalar_top(:, :)
    REAL(KIND = 8) :: scalar_center(:, :)
    INTEGER, INTENT(IN) :: blockno, start_cell_index, end_cell_index
    INTEGER :: level, jc
    REAL(KIND = 8), POINTER :: prism_center_distance(:, :), prism_thick(:, :)
    prism_center_distance => patch_3d % p_patch_1d(1) % constantprismcenters_zdistance(:, :, blockno)
    prism_thick => patch_3d % p_patch_1d(1) % prism_thick_flat_sfc_c(:, :, blockno)
    DO jc = start_cell_index, end_cell_index
      DO level = 1, patch_3d % p_patch_1d(1) % dolic_c(jc, blockno)
        scalar_center(jc, level) = (prism_center_distance(jc, level) * scalar_top(jc, level) + prism_center_distance(jc, level + 1) * scalar_top(jc, level + 1)) / (2.0D0 * prism_thick(jc, level))
      END DO
    END DO
  END SUBROUTINE map_scalar_prismtop2center_onblock
  SUBROUTINE map_vector_center2prismtop_onblock(patch_3d, vector_center, vector_top, blockno, start_cell_index, end_cell_index)
    USE mo_model_domain, ONLY: t_patch_3d
    USE mo_math_types, ONLY: t_cartesian_coordinates
    TYPE(t_patch_3d), TARGET :: patch_3d
    TYPE(t_cartesian_coordinates) :: vector_center(:, :)
    TYPE(t_cartesian_coordinates) :: vector_top(:, :)
    INTEGER, INTENT(IN) :: blockno, start_cell_index, end_cell_index
    INTEGER :: cell_index, level
    REAL(KIND = 8), POINTER :: inv_prism_center_distance(:, :), prism_thick(:, :)
    inv_prism_center_distance => patch_3d % p_patch_1d(1) % constantprismcenters_invzdistance(:, :, blockno)
    prism_thick => patch_3d % p_patch_1d(1) % prism_thick_flat_sfc_c(:, :, blockno)
    vector_top(:, :) % x(1) = 0.0D0
    vector_top(:, :) % x(2) = 0.0D0
    vector_top(:, :) % x(3) = 0.0D0
    DO cell_index = start_cell_index, end_cell_index
      vector_top(cell_index, 1) % x = 0.0D0
      DO level = 2, patch_3d % p_patch_1d(1) % dolic_c(cell_index, blockno)
        vector_top(cell_index, level) % x = (vector_center(cell_index, level - 1) % x * prism_thick(cell_index, level - 1) + vector_center(cell_index, level) % x * prism_thick(cell_index, level)) * 2.0D0 * inv_prism_center_distance(cell_index, level)
      END DO
    END DO
  END SUBROUTINE map_vector_center2prismtop_onblock
END MODULE mo_scalar_product
MODULE mo_util_dbg_prnt
  IMPLICIT NONE
  INTERFACE dbg_print
    MODULE PROCEDURE dbg_print_2d
    MODULE PROCEDURE dbg_print_3d
  END INTERFACE
  CONTAINS
  SUBROUTINE dbg_print_3d(description, p_array, place, indetail_level, in_subset)
    USE mo_model_domain, ONLY: t_subset_range
    CHARACTER(LEN = *), INTENT(IN) :: description
    REAL(KIND = 8), INTENT(IN) :: p_array(:, :, :)
    CHARACTER(LEN = *), INTENT(IN) :: place
    INTEGER, INTENT(IN) :: indetail_level
    TYPE(t_subset_range), TARGET, OPTIONAL :: in_subset
  END SUBROUTINE dbg_print_3d
  SUBROUTINE dbg_print_2d(description, p_array, place, indetail_level, in_subset)
    USE mo_model_domain, ONLY: t_subset_range
    CHARACTER(LEN = *), INTENT(IN) :: description
    REAL(KIND = 8), INTENT(IN) :: p_array(:, :)
    CHARACTER(LEN = *), INTENT(IN) :: place
    INTEGER, INTENT(IN) :: indetail_level
    TYPE(t_subset_range), TARGET, OPTIONAL :: in_subset
  END SUBROUTINE dbg_print_2d
  SUBROUTINE debug_print_maxminmean(description, minmaxmean, place, indetail_level)
    USE mo_dbg_nml, ONLY: idbg_mxmn
    USE mo_mpi, ONLY: my_process_is_stdio
    CHARACTER(LEN = *), INTENT(IN) :: description
    REAL(KIND = 8), INTENT(IN) :: minmaxmean(3)
    CHARACTER(LEN = *), INTENT(IN) :: place
    INTEGER, INTENT(IN) :: indetail_level
992 FORMAT(A, A12, ':', A27, '  :', 1P, G26.18, 1P, G26.18, 1P, G26.18)
    IF (idbg_mxmn >= 1) THEN
      IF (my_process_is_stdio()) WRITE(0, 992) ' MAX/MIN/MEAN ', TRIM(place), TRIM(description), minmaxmean(2), minmaxmean(1), minmaxmean(3)
    END IF
  END SUBROUTINE debug_print_maxminmean
END MODULE mo_util_dbg_prnt
MODULE mo_ocean_boundcond
  IMPLICIT NONE
  CHARACTER(LEN = 12) :: str_module = 'oceBoundCond'
  INTEGER :: idt_src = 1
  INTEGER :: current_step = 0
  CONTAINS
  SUBROUTINE top_bound_cond_horz_veloc(patch_3d, ocean_state, p_op_coeff, p_oce_sfc, lacc)
    USE mo_model_domain, ONLY: t_patch_3d
    USE mo_ocean_types, ONLY: t_hydro_ocean_state, t_operator_coeff
    USE mo_ocean_surface_types, ONLY: t_ocean_surface
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_ocean_nml, ONLY: forcing_windstress_u_type
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    TYPE(t_hydro_ocean_state), INTENT(INOUT) :: ocean_state
    TYPE(t_operator_coeff), INTENT(IN) :: p_op_coeff
    TYPE(t_ocean_surface) :: p_oce_sfc
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    IF (forcing_windstress_u_type > 100 .OR. forcing_windstress_u_type == 0) THEN
      CALL top_bound_cond_horz_veloc_onedges(patch_3d, ocean_state, p_op_coeff)
    ELSE
      CALL top_bound_cond_horz_veloc_fromcells(patch_3d, ocean_state, p_op_coeff, p_oce_sfc, lacc = lzacc)
    END IF
  END SUBROUTINE top_bound_cond_horz_veloc
  SUBROUTINE top_bound_cond_horz_veloc_onedges(patch_3d, ocean_state, p_op_coeff)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_ocean_types, ONLY: t_hydro_ocean_state, t_operator_coeff
    USE mo_parallel_config, ONLY: nproma
    USE mo_master_config, ONLY: isrestart
    USE mo_ocean_nml, ONLY: forcing_smooth_steps, forcing_windstress_weight, i_bc_veloc_top, iswm_oce, oceanreferencedensity
    USE mo_grid_subset, ONLY: get_index_range
    USE mo_exception, ONLY: finish
    USE mo_util_dbg_prnt, ONLY: dbg_print_2d_deconiface_138 => dbg_print_2d
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    TYPE(t_hydro_ocean_state), INTENT(INOUT) :: ocean_state
    TYPE(t_operator_coeff), INTENT(IN) :: p_op_coeff
    INTEGER :: je, jb
    INTEGER :: start_index, end_index
    REAL(KIND = 8) :: z_scale(nproma, patch_3d % p_patch_2d(1) % nblks_e)
    REAL(KIND = 8) :: smooth_coeff
    TYPE(t_subset_range), POINTER :: all_edges
    TYPE(t_patch), POINTER :: patch_2d
    patch_2d => patch_3d % p_patch_2d(1)
    all_edges => patch_2d % edges % all
    IF (isrestart() .OR. i_bc_veloc_top /= 4) THEN
      smooth_coeff = 1.0D0
    ELSE
      smooth_coeff = MIN(REAL(current_step, 8) / REAL(forcing_smooth_steps, 8), 1.0D0)
      current_step = current_step + 1
    END IF
    IF (iswm_oce == 1) THEN
      DO jb = all_edges % start_block, all_edges % end_block
        CALL get_index_range(all_edges, jb, start_index, end_index)
        DO je = start_index, end_index
          z_scale(je, jb) = 1.0D0 / (oceanreferencedensity * ocean_state % p_diag % thick_e(je, jb))
        END DO
      END DO
    ELSE IF (iswm_oce /= 1) THEN
      DO jb = all_edges % start_block, all_edges % end_block
        z_scale(:, jb) = 1.0D0 / oceanreferencedensity
      END DO
    END IF
    SELECT CASE (i_bc_veloc_top)
    CASE (0)
      DO jb = all_edges % start_block, all_edges % end_block
        CALL get_index_range(all_edges, jb, start_index, end_index)
        DO je = start_index, end_index
          ocean_state % p_aux % bc_top_vn(je, jb) = 0.0D0
        END DO
      END DO
    CASE (1, 4)
      DO jb = all_edges % start_block, all_edges % end_block
        CALL get_index_range(all_edges, jb, start_index, end_index)
        DO je = start_index, end_index
          ocean_state % p_aux % bc_top_vn(je, jb) = ocean_state % p_aux % bc_top_windstress(je, jb) * smooth_coeff * z_scale(je, jb) * forcing_windstress_weight
        END DO
      END DO
    CASE DEFAULT
      CALL finish("top_bound_cond_horz_veloc", "unknown i_bc_veloc_top")
    END SELECT
    idt_src = 3
    CALL dbg_print_2d_deconiface_138('top bound.cond. vn', ocean_state % p_aux % bc_top_vn, str_module, idt_src, in_subset = patch_2d % edges % owned)
  END SUBROUTINE top_bound_cond_horz_veloc_onedges
  SUBROUTINE top_bound_cond_horz_veloc_fromcells(patch_3d, ocean_state, p_op_coeff, p_oce_sfc, lacc)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_ocean_types, ONLY: t_hydro_ocean_state, t_operator_coeff
    USE mo_ocean_surface_types, ONLY: t_ocean_surface
    USE mo_parallel_config, ONLY: nproma
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_master_config, ONLY: isrestart
    USE mo_ocean_nml, ONLY: forcing_smooth_steps, forcing_windstress_weight, i_bc_veloc_top, iswm_oce, oceanreferencedensity
    USE mo_grid_subset, ONLY: get_index_range
    USE mo_exception, ONLY: finish
    USE mo_scalar_product, ONLY: map_cell2edges_3d_1level_deconiface_139 => map_cell2edges_3d_1level
    USE mo_util_dbg_prnt, ONLY: dbg_print_2d_deconiface_140 => dbg_print_2d, dbg_print_2d_deconiface_141 => dbg_print_2d, dbg_print_2d_deconiface_142 => dbg_print_2d
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    TYPE(t_hydro_ocean_state), INTENT(INOUT) :: ocean_state
    TYPE(t_operator_coeff), INTENT(IN) :: p_op_coeff
    TYPE(t_ocean_surface) :: p_oce_sfc
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: jc, jb
    INTEGER :: start_index, end_index
    REAL(KIND = 8) :: z_scale(nproma, patch_3d % p_patch_2d(1) % alloc_cell_blocks)
    REAL(KIND = 8) :: smooth_coeff, stress_coeff
    LOGICAL :: lzacc
    TYPE(t_subset_range), POINTER :: all_cells
    TYPE(t_patch), POINTER :: patch_2d
    patch_2d => patch_3d % p_patch_2d(1)
    all_cells => patch_2d % cells % all
    CALL set_acc_host_or_device(lzacc, lacc)
    IF (isrestart()) THEN
      smooth_coeff = 1.0D0
    ELSE
      smooth_coeff = MIN(REAL(current_step, 8) / REAL(forcing_smooth_steps, 8), 1.0D0)
      current_step = current_step + 1
    END IF
    IF (iswm_oce == 1) THEN
      DO jb = all_cells % start_block, all_cells % end_block
        z_scale(:, jb) = 1.0D0 / (oceanreferencedensity * ocean_state % p_diag % thick_c(:, jb))
      END DO
    ELSE IF (iswm_oce /= 1) THEN
      DO jb = all_cells % start_block, all_cells % end_block
        z_scale(:, jb) = 1.0D0 / oceanreferencedensity
      END DO
    END IF
    SELECT CASE (i_bc_veloc_top)
    CASE (0)
      DO jb = all_cells % start_block, all_cells % end_block
        ocean_state % p_aux % bc_top_u(:, jb) = 0.0D0
        ocean_state % p_aux % bc_top_v(:, jb) = 0.0D0
        CALL get_index_range(all_cells, jb, start_index, end_index)
        DO jc = start_index, end_index
          ocean_state % p_aux % bc_top_veloc_cc(jc, jb) % x = 0.0D0
        END DO
      END DO
    CASE (1)
      DO jb = all_cells % start_block, all_cells % end_block
        CALL get_index_range(all_cells, jb, start_index, end_index)
        DO jc = start_index, end_index
          IF (patch_3d % p_patch_1d(1) % dolic_c(jc, jb) > 0) THEN
            stress_coeff = z_scale(jc, jb)
            ocean_state % p_aux % bc_top_u(jc, jb) = p_oce_sfc % topbc_windstress_u(jc, jb) * stress_coeff
            ocean_state % p_aux % bc_top_v(jc, jb) = p_oce_sfc % topbc_windstress_v(jc, jb) * stress_coeff
            ocean_state % p_aux % bc_top_veloc_cc(jc, jb) % x = p_oce_sfc % topbc_windstress_cc(jc, jb) % x * stress_coeff
          END IF
        END DO
      END DO
    CASE (4)
      DO jb = all_cells % start_block, all_cells % end_block
        CALL get_index_range(all_cells, jb, start_index, end_index)
        DO jc = start_index, end_index
          IF (patch_3d % lsm_c(jc, 1, jb) <= (- 1)) THEN
            stress_coeff = smooth_coeff * z_scale(jc, jb) * forcing_windstress_weight
            ocean_state % p_aux % bc_top_u(jc, jb) = p_oce_sfc % topbc_windstress_u(jc, jb) * stress_coeff
            ocean_state % p_aux % bc_top_v(jc, jb) = p_oce_sfc % topbc_windstress_v(jc, jb) * stress_coeff
            ocean_state % p_aux % bc_top_veloc_cc(jc, jb) % x = p_oce_sfc % topbc_windstress_cc(jc, jb) % x * stress_coeff
          END IF
        END DO
      END DO
    CASE DEFAULT
      CALL finish("top_bound_cond_horz_veloc", "unknown i_bc_veloc_top")
    END SELECT
    CALL map_cell2edges_3d_1level_deconiface_139(patch_3d, ocean_state % p_aux % bc_top_veloc_cc, ocean_state % p_aux % bc_top_vn, p_op_coeff, level = 1, lacc = lzacc)
    idt_src = 2
    CALL dbg_print_2d_deconiface_140('top bound.cond. u', ocean_state % p_aux % bc_top_u, str_module, idt_src, in_subset = patch_2d % cells % owned)
    CALL dbg_print_2d_deconiface_141('top bound.cond. v', ocean_state % p_aux % bc_top_v, str_module, idt_src, in_subset = patch_2d % cells % owned)
    idt_src = 3
    CALL dbg_print_2d_deconiface_142('top bound.cond. vn', ocean_state % p_aux % bc_top_vn, str_module, idt_src, in_subset = patch_2d % edges % owned)
  END SUBROUTINE top_bound_cond_horz_veloc_fromcells
  SUBROUTINE velocitybottomboundarycondition_onblock(patch_3d, blockno, start_edge_index, end_edge_index, vn_old, vn_pred, bc_bot_vn, lacc)
    USE mo_model_domain, ONLY: t_patch_3d
    USE mo_impl_constants, ONLY: max_char_length
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_ocean_nml, ONLY: i_bc_veloc_bot
    USE mo_ocean_physics_types, ONLY: v_params
    USE mo_exception, ONLY: finish, message
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    INTEGER, INTENT(IN) :: blockno, start_edge_index, end_edge_index
    REAL(KIND = 8) :: vn_old(:, :), vn_pred(:, :)
    REAL(KIND = 8) :: bc_bot_vn(:)
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: bottom_level, je
    REAL(KIND = 8) :: norm, vn_max, vn
    LOGICAL :: lzacc
    CHARACTER(LEN = max_char_length), PARAMETER :: routine = ('mo_ocean_boundcond:VelocityBottomBoundaryCondition_onBlock')
    CALL set_acc_host_or_device(lzacc, lacc)
    SELECT CASE (i_bc_veloc_bot)
    CASE (0)
      DO je = start_edge_index, end_edge_index
        bc_bot_vn(je) = 0.0D0
      END DO
    CASE (1)
      DO je = start_edge_index, end_edge_index
        bottom_level = patch_3d % p_patch_1d(1) % dolic_e(je, blockno)
        IF (bottom_level > 0) THEN
          vn_max = MAX(ABS(vn_old(je, bottom_level)), ABS(vn_pred(je, bottom_level)), ABS(vn_old(je, bottom_level) - vn_pred(je, bottom_level)))
          norm = SQRT(vn_max * vn_max)
          bc_bot_vn(je) = v_params % bottom_drag_coeff * norm * vn_pred(je, bottom_level)
        END IF
      END DO
    CASE (2)
      DO je = start_edge_index, end_edge_index
        bottom_level = patch_3d % p_patch_1d(1) % dolic_e(je, blockno)
        IF (bottom_level > 0) THEN
          vn = vn_old(je, bottom_level)
          norm = SQRT(vn * vn)
          bc_bot_vn(je) = v_params % bottom_drag_coeff * norm * vn_old(je, bottom_level)
        END IF
      END DO
    CASE (3)
      CALL message(TRIM(routine), 'TOPOGRAPHY_SLOPE bottom velocity boundary conditions not implemented yet')
      CALL finish(TRIM(routine), 'TOPOGRAPHY_SLOPE bottom velocity boundary conditions not implemented yet')
    CASE DEFAULT
      CALL message(TRIM(routine), 'choosen wrong bottom velocity boundary conditions')
    END SELECT
  END SUBROUTINE velocitybottomboundarycondition_onblock
END MODULE mo_ocean_boundcond
MODULE mo_ocean_velocity_advection
  IMPLICIT NONE
  CHARACTER(LEN = 12) :: str_module = 'oceVelocAdv '
  INTEGER :: idt_src = 1
  CONTAINS
  SUBROUTINE veloc_adv_horz_mimetic(patch_3d, vn_old, vn_new, p_diag, veloc_adv_horz_e, ocean_coefficients, lacc)
    USE mo_model_domain, ONLY: t_patch_3d
    USE mo_ocean_types, ONLY: t_hydro_ocean_diag, t_operator_coeff
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_ocean_nml, ONLY: nonlinearcoriolis_type
    TYPE(t_patch_3d), POINTER, INTENT(IN) :: patch_3d
    REAL(KIND = 8), POINTER, INTENT(INOUT) :: vn_old(:, :, :)
    REAL(KIND = 8), POINTER, INTENT(INOUT) :: vn_new(:, :, :)
    TYPE(t_hydro_ocean_diag) :: p_diag
    REAL(KIND = 8), POINTER, INTENT(INOUT) :: veloc_adv_horz_e(:, :, :)
    TYPE(t_operator_coeff), TARGET, INTENT(IN) :: ocean_coefficients
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    IF (nonlinearcoriolis_type == 200) THEN
      CALL veloc_adv_horz_mimetic_rot(patch_3d, vn_old, p_diag, veloc_adv_horz_e, ocean_coefficients, lacc = lzacc)
    ELSE IF (nonlinearcoriolis_type == 201) THEN
      CALL veloc_adv_horz_mimetic_classiccgrid(patch_3d, vn_old, p_diag, veloc_adv_horz_e, ocean_coefficients)
    ELSE IF (nonlinearcoriolis_type == 0) THEN
      CALL calculate_only_kineticgrad(patch_3d, vn_old, p_diag, veloc_adv_horz_e, ocean_coefficients)
    END IF
  END SUBROUTINE veloc_adv_horz_mimetic
  SUBROUTINE veloc_adv_vert_mimetic(patch_3d, p_diag, ocean_coefficients, veloc_adv_vert_e, lacc)
    USE mo_model_domain, ONLY: t_patch_3d
    USE mo_ocean_types, ONLY: t_hydro_ocean_diag, t_operator_coeff
    USE mo_fortran_tools, ONLY: init_zero_3d_dp_deconiface_143 => init_zero_3d_dp, set_acc_host_or_device
    USE mo_ocean_nml, ONLY: horizonatlvelocity_verticaladvection_form
    USE mo_exception, ONLY: finish
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    TYPE(t_hydro_ocean_diag) :: p_diag
    TYPE(t_operator_coeff), INTENT(IN) :: ocean_coefficients
    REAL(KIND = 8), INTENT(INOUT) :: veloc_adv_vert_e(:, :, :)
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    SELECT CASE (horizonatlvelocity_verticaladvection_form)
    CASE (1)
      CALL veloc_adv_vert_mimetic_rot(patch_3d, p_diag, ocean_coefficients, veloc_adv_vert_e, lacc = lzacc)
    CASE (2)
      CALL veloc_adv_vert_mimetic_div(patch_3d, p_diag, ocean_coefficients, veloc_adv_vert_e)
    CASE (3)
      CALL veloc_adv_vert_rot(patch_3d, p_diag, ocean_coefficients, veloc_adv_vert_e)
    CASE (0)
      CALL init_zero_3d_dp_deconiface_143(veloc_adv_vert_e, lacc = lzacc)
    CASE DEFAULT
      CALL finish("veloc_adv_vert_mimetic", "unknown HorizonatlVelocity_VerticalAdvection_form")
    END SELECT
  END SUBROUTINE veloc_adv_vert_mimetic
  SUBROUTINE veloc_adv_horz_mimetic_rot(patch_3d, vn, p_diag, veloc_adv_horz_e, ocean_coefficients, lacc)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_ocean_types, ONLY: t_hydro_ocean_diag, t_operator_coeff
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_scalar_product, ONLY: nonlinear_coriolis_3d
    USE mo_grid_subset, ONLY: get_index_range
    USE mo_ocean_math_operators, ONLY: grad_fd_norm_oce_3d_onblock
    USE mo_util_dbg_prnt, ONLY: dbg_print_3d_deconiface_144 => dbg_print_3d, dbg_print_3d_deconiface_145 => dbg_print_3d, dbg_print_3d_deconiface_146 => dbg_print_3d
    TYPE(t_patch_3d), POINTER, INTENT(IN) :: patch_3d
    REAL(KIND = 8), POINTER, INTENT(INOUT) :: vn(:, :, :)
    TYPE(t_hydro_ocean_diag) :: p_diag
    REAL(KIND = 8), POINTER, INTENT(INOUT) :: veloc_adv_horz_e(:, :, :)
    TYPE(t_operator_coeff), INTENT(IN), TARGET :: ocean_coefficients
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: blockno, startlevel
    INTEGER :: start_edge_index, end_edge_index
    LOGICAL :: lzacc
    TYPE(t_subset_range), POINTER :: edges_in_domain
    TYPE(t_patch), POINTER :: patch_2d
    patch_2d => patch_3d % p_patch_2d(1)
    edges_in_domain => patch_2d % edges % in_domain
    startlevel = 1
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL nonlinear_coriolis_3d(patch_3d, vn, p_diag % p_vn_dual, p_diag % vort, ocean_coefficients, veloc_adv_horz_e, lacc = lzacc)
    DO blockno = edges_in_domain % start_block, edges_in_domain % end_block
      CALL get_index_range(edges_in_domain, blockno, start_edge_index, end_edge_index)
      CALL grad_fd_norm_oce_3d_onblock(p_diag % kin, patch_3d, ocean_coefficients % grad_coeff(:, :, blockno), p_diag % grad(:, :, blockno), start_edge_index, end_edge_index, blockno, lacc = lzacc)
    END DO
    idt_src = 3
    CALL dbg_print_3d_deconiface_144('HorzMimRot: kin energy', p_diag % kin, str_module, idt_src, patch_2d % cells % owned)
    CALL dbg_print_3d_deconiface_145('HorzMimRot: vorticity', p_diag % vort, str_module, idt_src, patch_2d % verts % owned)
    CALL dbg_print_3d_deconiface_146('HorzMimRot: grad kin en', p_diag % grad, str_module, idt_src, patch_2d % edges % owned)
  END SUBROUTINE veloc_adv_horz_mimetic_rot
  SUBROUTINE calculate_only_kineticgrad(patch_3d, vn, p_diag, veloc_adv_horz_e, ocean_coefficients)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_ocean_types, ONLY: t_hydro_ocean_diag, t_operator_coeff
    USE mo_fortran_tools, ONLY: init_zero_3d_dp_deconiface_147 => init_zero_3d_dp
    USE mo_grid_subset, ONLY: get_index_range
    USE mo_ocean_math_operators, ONLY: grad_fd_norm_oce_3d_onblock
    USE mo_util_dbg_prnt, ONLY: dbg_print_3d_deconiface_148 => dbg_print_3d, dbg_print_3d_deconiface_149 => dbg_print_3d
    TYPE(t_patch_3d), TARGET :: patch_3d
    REAL(KIND = 8), POINTER, INTENT(INOUT) :: vn(:, :, :)
    TYPE(t_hydro_ocean_diag) :: p_diag
    REAL(KIND = 8), POINTER, INTENT(INOUT) :: veloc_adv_horz_e(:, :, :)
    TYPE(t_operator_coeff), INTENT(IN) :: ocean_coefficients
    INTEGER :: blockno, start_edge_index, end_edge_index
    TYPE(t_subset_range), POINTER :: edges_in_domain
    TYPE(t_patch), POINTER :: patch_2d
    patch_2d => patch_3d % p_patch_2d(1)
    edges_in_domain => patch_2d % edges % in_domain
    CALL init_zero_3d_dp_deconiface_147(veloc_adv_horz_e, lacc = .FALSE.)
    DO blockno = edges_in_domain % start_block, edges_in_domain % end_block
      CALL get_index_range(edges_in_domain, blockno, start_edge_index, end_edge_index)
      CALL grad_fd_norm_oce_3d_onblock(p_diag % kin, patch_3d, ocean_coefficients % grad_coeff(:, :, blockno), p_diag % grad(:, :, blockno), start_edge_index, end_edge_index, blockno)
    END DO
    idt_src = 3
    CALL dbg_print_3d_deconiface_148('HorzMimRot: kin energy', p_diag % kin, str_module, idt_src, patch_2d % cells % owned)
    CALL dbg_print_3d_deconiface_149('HorzMimRot: grad kin en', p_diag % grad, str_module, idt_src, patch_2d % edges % owned)
  END SUBROUTINE calculate_only_kineticgrad
  SUBROUTINE veloc_adv_horz_mimetic_classiccgrid(patch_3d, vn, p_diag, veloc_adv_horz_e, ocean_coefficients)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_nml, ONLY: n_zlev
    USE mo_ocean_types, ONLY: t_hydro_ocean_diag, t_operator_coeff
    USE mo_ocean_math_operators, ONLY: grad_fd_norm_oce_3d_onblock, rot_vertex_ocean_3d
    USE mo_sync, ONLY: sync_patch_array_3d_dp_deconiface_150 => sync_patch_array_3d_dp
    USE mo_grid_subset, ONLY: get_index_range
    USE mo_util_dbg_prnt, ONLY: dbg_print_2d_deconiface_153 => dbg_print_2d, dbg_print_3d_deconiface_151 => dbg_print_3d, dbg_print_3d_deconiface_152 => dbg_print_3d, dbg_print_3d_deconiface_154 => dbg_print_3d, dbg_print_3d_deconiface_155 => dbg_print_3d
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    REAL(KIND = 8), INTENT(INOUT) :: vn(1 : nproma, 1 : n_zlev, 1 : patch_3d % p_patch_2d(1) % nblks_e)
    TYPE(t_hydro_ocean_diag) :: p_diag
    REAL(KIND = 8), INTENT(INOUT) :: veloc_adv_horz_e(1 : nproma, 1 : n_zlev, 1 : patch_3d % p_patch_2d(1) % nblks_e)
    TYPE(t_operator_coeff), INTENT(IN) :: ocean_coefficients
    INTEGER :: jk, blockno, je
    INTEGER :: start_edge_index, end_edge_index
    INTEGER :: c1_idx, c1_blk, c2_idx, c2_blk
    INTEGER :: v1_idx, v1_blk, v2_idx, v2_blk
    REAL(KIND = 8) :: veloc_tangential
    INTEGER, POINTER :: edge_levels(:, :)
    TYPE(t_subset_range), POINTER :: edges_in_domain
    TYPE(t_patch), POINTER :: patch_2d
    patch_2d => patch_3d % p_patch_2d(1)
    edges_in_domain => patch_2d % edges % in_domain
    edge_levels => patch_3d % p_patch_1d(1) % dolic_e
    CALL rot_vertex_ocean_3d(patch_3d, vn, p_diag % p_vn_dual, ocean_coefficients, p_diag % vort)
    CALL sync_patch_array_3d_dp_deconiface_150(3, patch_2d, p_diag % vort, lacc = .FALSE.)
    DO blockno = edges_in_domain % start_block, edges_in_domain % end_block
      CALL get_index_range(edges_in_domain, blockno, start_edge_index, end_edge_index)
      DO je = start_edge_index, end_edge_index
        c1_idx = patch_2d % edges % cell_idx(je, blockno, 1)
        c1_blk = patch_2d % edges % cell_blk(je, blockno, 1)
        c2_idx = patch_2d % edges % cell_idx(je, blockno, 2)
        c2_blk = patch_2d % edges % cell_blk(je, blockno, 2)
        v1_idx = patch_2d % edges % vertex_idx(je, blockno, 1)
        v1_blk = patch_2d % edges % vertex_blk(je, blockno, 1)
        v2_idx = patch_2d % edges % vertex_idx(je, blockno, 2)
        v2_blk = patch_2d % edges % vertex_blk(je, blockno, 2)
        DO jk = 1, edge_levels(je, blockno)
          veloc_tangential = DOT_PRODUCT(p_diag % p_vn(c1_idx, jk, c1_blk) % x * ocean_coefficients % averagecellstoedges(je, blockno, 1) + p_diag % p_vn(c2_idx, jk, c2_blk) % x * ocean_coefficients % averagecellstoedges(je, blockno, 2), patch_2d % edges % dual_cart_normal(je, blockno) % x)
          veloc_adv_horz_e(je, jk, blockno) = veloc_tangential * (patch_2d % edges % f_e(je, blockno) + 0.5D0 * (p_diag % vort(v1_idx, jk, v1_blk) + p_diag % vort(v2_idx, jk, v2_blk)))
        END DO
      END DO
    END DO
    DO blockno = edges_in_domain % start_block, edges_in_domain % end_block
      CALL get_index_range(edges_in_domain, blockno, start_edge_index, end_edge_index)
      CALL grad_fd_norm_oce_3d_onblock(p_diag % kin, patch_3d, ocean_coefficients % grad_coeff(:, :, blockno), p_diag % grad(:, :, blockno), start_edge_index, end_edge_index, blockno)
    END DO
    idt_src = 3
    CALL dbg_print_3d_deconiface_151('advHorCgrid: kin energy', p_diag % kin, str_module, idt_src, patch_2d % cells % owned)
    CALL dbg_print_3d_deconiface_152('advHorCgrid: vorticity', p_diag % vort, str_module, idt_src, patch_2d % verts % owned)
    CALL dbg_print_2d_deconiface_153('advHorCgrid: f_e', patch_2d % edges % f_e, str_module, idt_src, patch_2d % edges % owned)
    CALL dbg_print_3d_deconiface_154('advHorCgrid: grad kin en', p_diag % grad, str_module, idt_src, patch_2d % edges % owned)
    CALL dbg_print_3d_deconiface_155('advHorCgrid: veloc_adv_horz_e', veloc_adv_horz_e, str_module, idt_src, patch_2d % edges % owned)
  END SUBROUTINE veloc_adv_horz_mimetic_classiccgrid
  SUBROUTINE veloc_adv_vert_rot(patch_3d, p_diag, ocean_coefficients, veloc_adv_vert_e)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_ocean_types, ONLY: t_hydro_ocean_diag, t_operator_coeff
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_nml, ONLY: n_zlev
    USE mo_math_types, ONLY: t_cartesian_coordinates
    USE mo_grid_subset, ONLY: get_index_range
    USE mo_scalar_product, ONLY: map_cell2edges_3d_mlevels_deconiface_162 => map_cell2edges_3d_mlevels, map_scalar_prismtop2center_onblock, map_vector_center2prismtop_onblock
    USE mo_ocean_math_operators, ONLY: verticaldiv_vector_onfulllevels_on_block
    USE mo_util_dbg_prnt, ONLY: dbg_print, dbg_print_3d_deconiface_163 => dbg_print_3d
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    TYPE(t_hydro_ocean_diag) :: p_diag
    TYPE(t_operator_coeff), INTENT(IN) :: ocean_coefficients
    REAL(KIND = 8), INTENT(INOUT) :: veloc_adv_vert_e(1 : nproma, 1 : n_zlev, 1 : patch_3d % p_patch_2d(1) % nblks_e)
    INTEGER :: jc, jk, blockno
    INTEGER :: start_index, end_index
    TYPE(t_cartesian_coordinates) :: z_adv_u_fulllevels(nproma, n_zlev)
    TYPE(t_cartesian_coordinates) :: vn_halflevels(nproma, n_zlev + 1)
    TYPE(t_cartesian_coordinates) :: z_adv_u_m(nproma, n_zlev, patch_3d % p_patch_2d(1) % alloc_cell_blocks)
    REAL(KIND = 8) :: center_vertical_velocity(nproma, n_zlev)
    TYPE(t_subset_range), POINTER :: all_cells
    TYPE(t_patch), POINTER :: patch_2d
    patch_2d => patch_3d % p_patch_2d(1)
    all_cells => patch_2d % cells % all
    DO blockno = all_cells % start_block, all_cells % end_block
      CALL get_index_range(all_cells, blockno, start_index, end_index)
      CALL map_scalar_prismtop2center_onblock(patch_3d, p_diag % w(:, :, blockno), center_vertical_velocity, blockno, start_index, end_index)
      CALL map_vector_center2prismtop_onblock(patch_3d, p_diag % p_vn(:, :, blockno), vn_halflevels, blockno, start_index, end_index)
      CALL verticaldiv_vector_onfulllevels_on_block(patch_3d, vn_halflevels, z_adv_u_fulllevels, 1, blockno, start_index, end_index)
      DO jc = start_index, end_index
        DO jk = 1, patch_3d % p_patch_1d(1) % dolic_c(jc, blockno)
          z_adv_u_m(jc, jk, blockno) % x = center_vertical_velocity(jc, jk) * z_adv_u_fulllevels(jc, jk) % x
        END DO
        DO jk = patch_3d % p_patch_1d(1) % dolic_c(jc, blockno) + 1, n_zlev
          z_adv_u_m(jc, jk, blockno) % x = 0.0D0
        END DO
      END DO
    END DO
    idt_src = 3
    CALL dbg_print('vn 1%x(1)', p_diag % p_vn(:, 1, :) % x(1), str_module, idt_src, patch_2d % cells % owned)
    CALL dbg_print('vn 1%x(2)', p_diag % p_vn(:, 1, :) % x(2), str_module, idt_src, patch_2d % cells % owned)
    CALL dbg_print('vn 1%x(3)', p_diag % p_vn(:, 1, :) % x(3), str_module, idt_src, patch_2d % cells % owned)
    CALL dbg_print('vn 2%x(1)', p_diag % p_vn(:, 2, :) % x(1), str_module, idt_src, patch_2d % cells % owned)
    CALL dbg_print('vn 2%x(2)', p_diag % p_vn(:, 2, :) % x(2), str_module, idt_src, patch_2d % cells % owned)
    CALL dbg_print('vn 2%x(3)', p_diag % p_vn(:, 2, :) % x(3), str_module, idt_src, patch_2d % cells % owned)
    CALL dbg_print('VertAdvect x(1)', z_adv_u_m(:, :, :) % x(1), str_module, idt_src, patch_2d % cells % owned)
    CALL dbg_print('VertAdvect x(2)', z_adv_u_m(:, :, :) % x(2), str_module, idt_src, patch_2d % cells % owned)
    CALL dbg_print('VertAdvect x(3)', z_adv_u_m(:, :, :) % x(3), str_module, idt_src, patch_2d % cells % owned)
    CALL map_cell2edges_3d_mlevels_deconiface_162(patch_3d, z_adv_u_m, veloc_adv_vert_e, ocean_coefficients)
    idt_src = 3
    CALL dbg_print_3d_deconiface_163('VertMimRot: V.Adv. Final', veloc_adv_vert_e, str_module, idt_src, patch_2d % edges % owned)
  END SUBROUTINE veloc_adv_vert_rot
  SUBROUTINE veloc_adv_vert_mimetic_rot(patch_3d, p_diag, p_op_coeff, veloc_adv_vert_e, lacc)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_ocean_types, ONLY: t_hydro_ocean_diag, t_operator_coeff
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_nml, ONLY: n_zlev
    USE mo_math_types, ONLY: t_cartesian_coordinates
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_grid_subset, ONLY: get_index_range
    USE mo_ocean_math_operators, ONLY: verticalderiv_vec_midlevel_on_block
    USE mo_scalar_product, ONLY: map_cell2edges_3d_mlevels_deconiface_164 => map_cell2edges_3d_mlevels, map_vec_prismtop2center_on_block
    USE mo_util_dbg_prnt, ONLY: dbg_print_3d_deconiface_165 => dbg_print_3d
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    TYPE(t_hydro_ocean_diag) :: p_diag
    TYPE(t_operator_coeff), INTENT(IN) :: p_op_coeff
    REAL(KIND = 8), INTENT(INOUT) :: veloc_adv_vert_e(1 : nproma, 1 : n_zlev, 1 : patch_3d % p_patch_2d(1) % nblks_e)
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: start_level
    INTEGER :: jc, jk, blockno
    INTEGER :: start_index, end_index
    INTEGER :: fin_level
    LOGICAL :: lzacc
    TYPE(t_cartesian_coordinates) :: z_adv_u_i(nproma, n_zlev + 1)
    TYPE(t_cartesian_coordinates) :: z_adv_u_m(nproma, n_zlev, patch_3d % p_patch_2d(1) % alloc_cell_blocks)
    TYPE(t_subset_range), POINTER :: all_cells
    TYPE(t_patch), POINTER :: patch_2d
    patch_2d => patch_3d % p_patch_2d(1)
    all_cells => patch_2d % cells % all
    start_level = 1
    CALL set_acc_host_or_device(lzacc, lacc)
    DO blockno = 1, patch_2d % alloc_cell_blocks
      DO jk = 1, n_zlev
        DO jc = 1, nproma
          z_adv_u_m(jc, jk, blockno) % x(1) = 0.0D0
          z_adv_u_m(jc, jk, blockno) % x(2) = 0.0D0
          z_adv_u_m(jc, jk, blockno) % x(3) = 0.0D0
        END DO
      END DO
    END DO
    DO blockno = all_cells % start_block, all_cells % end_block
      CALL get_index_range(all_cells, blockno, start_index, end_index)
      CALL verticalderiv_vec_midlevel_on_block(patch_3d, p_diag % p_vn(:, :, blockno), z_adv_u_i(:, :), 2, blockno, start_index, end_index, lacc = lzacc)
      DO jc = start_index, end_index
        fin_level = patch_3d % p_patch_1d(1) % dolic_c(jc, blockno)
        IF (fin_level >= 2) THEN
          z_adv_u_i(jc, start_level) % x = - p_diag % w(jc, 1, blockno) * p_diag % p_vn(jc, 1, blockno) % x * patch_3d % p_patch_1d(1) % constantprismcenters_invzdistance(jc, 1, blockno)
          DO jk = start_level + 1, fin_level
            z_adv_u_i(jc, jk) % x = p_diag % w(jc, jk, blockno) * z_adv_u_i(jc, jk) % x
          END DO
          z_adv_u_i(jc, fin_level + 1) % x = 0.0D0
        END IF
      END DO
      CALL map_vec_prismtop2center_on_block(patch_3d, z_adv_u_i, z_adv_u_m(:, :, blockno), blockno, start_index, end_index, lacc = lzacc)
    END DO
    CALL map_cell2edges_3d_mlevels_deconiface_164(patch_3d, z_adv_u_m, veloc_adv_vert_e, p_op_coeff, lacc = lzacc)
    idt_src = 3
    CALL dbg_print_3d_deconiface_165('VertMimRot: V.Adv. Final', veloc_adv_vert_e, str_module, idt_src, patch_2d % edges % owned)
  END SUBROUTINE veloc_adv_vert_mimetic_rot
  SUBROUTINE veloc_adv_vert_mimetic_div(patch_3d, p_diag, ocean_coefficients, veloc_adv_vert_e)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_ocean_types, ONLY: t_hydro_ocean_diag, t_operator_coeff
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_nml, ONLY: n_zlev
    USE mo_math_types, ONLY: t_cartesian_coordinates
    USE mo_fortran_tools, ONLY: init, init_zero_3d_dp_deconiface_166 => init_zero_3d_dp
    USE mo_grid_subset, ONLY: get_index_range
    USE mo_scalar_product, ONLY: map_cell2edges_3d_mlevels_deconiface_167 => map_cell2edges_3d_mlevels
    USE mo_sync, ONLY: sync_patch_array_3d_dp_deconiface_168 => sync_patch_array_3d_dp
    USE mo_util_dbg_prnt, ONLY: dbg_print_3d_deconiface_169 => dbg_print_3d
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    TYPE(t_hydro_ocean_diag) :: p_diag
    TYPE(t_operator_coeff), INTENT(IN) :: ocean_coefficients
    REAL(KIND = 8), INTENT(INOUT) :: veloc_adv_vert_e(1 : nproma, 1 : n_zlev, patch_3d % p_patch_2d(1) % nblks_e)
    INTEGER :: start_level, elev
    INTEGER :: jc, jk, blockno
    INTEGER :: start_index, end_index
    INTEGER :: fin_level
    REAL(KIND = 8), POINTER :: del_zlev_m(:)
    REAL(KIND = 8) :: z_w_diff(nproma, n_zlev - 1, patch_3d % p_patch_2d(1) % alloc_cell_blocks)
    TYPE(t_cartesian_coordinates) :: z_adv_u_i(nproma, n_zlev + 1, patch_3d % p_patch_2d(1) % alloc_cell_blocks)
    TYPE(t_subset_range), POINTER :: all_cells
    TYPE(t_patch), POINTER :: patch_2d
    patch_2d => patch_3d % p_patch_2d(1)
    all_cells => patch_2d % cells % all
    start_level = 1
    elev = n_zlev
    CALL init(z_adv_u_i(1 : nproma, 1 : n_zlev + 1, 1 : patch_2d % alloc_cell_blocks) % x(1), lacc = .FALSE.)
    CALL init(z_adv_u_i(1 : nproma, 1 : n_zlev + 1, 1 : patch_2d % alloc_cell_blocks) % x(2), lacc = .FALSE.)
    CALL init(z_adv_u_i(1 : nproma, 1 : n_zlev + 1, 1 : patch_2d % alloc_cell_blocks) % x(3), lacc = .FALSE.)
    CALL init_zero_3d_dp_deconiface_166(z_w_diff(1 : nproma, 1 : n_zlev - 1, 1 : patch_2d % alloc_cell_blocks), lacc = .FALSE.)
    DO blockno = all_cells % start_block, all_cells % end_block
      CALL get_index_range(all_cells, blockno, start_index, end_index)
      DO jc = start_index, end_index
        fin_level = patch_3d % p_patch_1d(1) % dolic_c(jc, blockno)
        IF (fin_level >= 2) THEN
          del_zlev_m => patch_3d % p_patch_1d(1) % inv_prism_thick_c(jc, :, blockno)
          DO jk = start_level, fin_level - 1
            z_w_diff(jc, jk, blockno) = del_zlev_m(jk) * (p_diag % w(jc, jk, blockno) - p_diag % w(jc, jk + 1, blockno))
          END DO
          jk = 1
          z_adv_u_i(jc, jk, blockno) % x = del_zlev_m(1) * 0.5D0 * (p_diag % w(jc, 1, blockno) * (p_diag % p_vn(jc, 1, blockno) % x + p_diag % p_vn(jc, 1, blockno) % x) - p_diag % w(jc, 2, blockno) * (p_diag % p_vn(jc, 1, blockno) % x + p_diag % p_vn(jc, 2, blockno) % x))
          DO jk = start_level + 1, fin_level - 1
            z_adv_u_i(jc, jk, blockno) % x = del_zlev_m(jk) * 0.5D0 * (p_diag % w(jc, jk, blockno) * (p_diag % p_vn(jc, jk - 1, blockno) % x + p_diag % p_vn(jc, jk, blockno) % x) - p_diag % w(jc, jk + 1, blockno) * (p_diag % p_vn(jc, jk, blockno) % x + p_diag % p_vn(jc, jk + 1, blockno) % x))
          END DO
        END IF
      END DO
    END DO
    CALL map_cell2edges_3d_mlevels_deconiface_167(patch_3d, z_adv_u_i, veloc_adv_vert_e, ocean_coefficients)
    CALL sync_patch_array_3d_dp_deconiface_168(2, patch_2d, veloc_adv_vert_e, lacc = .FALSE.)
    idt_src = 3
    CALL dbg_print_3d_deconiface_169('VertMimDiv: VelAdv Final', veloc_adv_vert_e, str_module, idt_src, patch_2d % edges % owned)
  END SUBROUTINE veloc_adv_vert_mimetic_div
END MODULE mo_ocean_velocity_advection
MODULE mo_ocean_velocity_diffusion
  IMPLICIT NONE
  CHARACTER(LEN = 12) :: str_module = 'oceDiffusion'
  LOGICAL :: eliminate_upper_diag = .TRUE.
  CONTAINS
  SUBROUTINE velocity_diffusion(patch_3d, vn_in, physics_parameters, p_diag, operators_coeff, laplacian_vn_out, lacc)
    USE mo_model_domain, ONLY: t_patch_3d
    USE mo_ocean_physics_types, ONLY: t_ho_params
    USE mo_ocean_types, ONLY: t_hydro_ocean_diag, t_operator_coeff
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_ocean_nml, ONLY: laplacian_form, velocitydiffusion_order
    USE mo_exception, ONLY: finish
    USE mo_util_dbg_prnt, ONLY: dbg_print_3d_deconiface_170 => dbg_print_3d
    TYPE(t_patch_3d), TARGET :: patch_3d
    REAL(KIND = 8) :: vn_in(:, :, :)
    TYPE(t_ho_params) :: physics_parameters
    TYPE(t_hydro_ocean_diag) :: p_diag
    TYPE(t_operator_coeff), INTENT(IN) :: operators_coeff
    REAL(KIND = 8) :: laplacian_vn_out(:, :, :)
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    LOGICAL :: lzacc
    CHARACTER(LEN = *), PARAMETER :: method_name = "velocity_diffusion"
    CALL set_acc_host_or_device(lzacc, lacc)
    IF (velocitydiffusion_order == 1) THEN
      IF (laplacian_form == 2) THEN
        CALL finish(method_name, "form of harmonic Laplacian not recommended")
        CALL veloc_diff_harmonic_div_grad(patch_3d, physics_parameters % harmonicviscosity_coeff, p_diag, operators_coeff, laplacian_vn_out)
      ELSE IF (laplacian_form == 1) THEN
        CALL veloc_diff_harmonic_curl_curl(patch_3d = patch_3d, u_vec_e = vn_in, vort = p_diag % vort, div_coeff = operators_coeff % div_coeff, harmonicdiffusion = laplacian_vn_out, k_h = physics_parameters % harmonicviscosity_coeff, lacc = lzacc)
        CALL dbg_print_3d_deconiface_170('laplacian_vn_out:', laplacian_vn_out, str_module, 4, in_subset = patch_3d % p_patch_2d(1) % edges % owned)
      END IF
    ELSE IF (velocitydiffusion_order == 2 .OR. velocitydiffusion_order == 21) THEN
      IF (laplacian_form == 2) THEN
        CALL veloc_diff_biharmonic_div_grad(patch_3d, physics_parameters, p_diag, operators_coeff, laplacian_vn_out)
      ELSE IF (laplacian_form == 1) THEN
        CALL veloc_diff_biharmonic_curl_curl(patch_3d, physics_parameters, vn_in, p_diag % vort, operators_coeff, laplacian_vn_out, lacc = lzacc)
      END IF
    ELSE IF (velocitydiffusion_order == 0) THEN
      laplacian_vn_out = 0.0D0
    ELSE
      CALL finish(method_name, "unknown VelocityDiffusion_order")
    END IF
  END SUBROUTINE velocity_diffusion
  SUBROUTINE veloc_diff_harmonic_div_grad(patch_3d, grad_coeff, p_diag, operators_coeff, laplacian_vn_out)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d
    USE mo_ocean_types, ONLY: t_hydro_ocean_diag, t_operator_coeff
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_nml, ONLY: n_zlev
    USE mo_math_types, ONLY: t_cartesian_coordinates
    USE mo_ocean_math_operators, ONLY: div_vector_ontriangle, grad_vector
    USE mo_sync, ONLY: sync_patch_array_mult
    USE mo_scalar_product, ONLY: map_cell2edges_3d_mlevels_deconiface_171 => map_cell2edges_3d_mlevels
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    REAL(KIND = 8) :: grad_coeff(:, :, :)
    TYPE(t_hydro_ocean_diag) :: p_diag
    TYPE(t_operator_coeff), INTENT(IN) :: operators_coeff
    REAL(KIND = 8), INTENT(INOUT) :: laplacian_vn_out(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_e)
    INTEGER :: start_level, end_level
    TYPE(t_cartesian_coordinates) :: z_grad_u(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_e)
    TYPE(t_cartesian_coordinates) :: z_div_grad_u(nproma, n_zlev, patch_3d % p_patch_2d(1) % alloc_cell_blocks)
    TYPE(t_patch), POINTER :: patch_2d
    patch_2d => patch_3d % p_patch_2d(1)
    start_level = 1
    end_level = n_zlev
    CALL grad_vector(cellvector = p_diag % p_vn, patch_3d = patch_3d, grad_coeff = grad_coeff, gradvector = z_grad_u)
    CALL div_vector_ontriangle(patch_3d = patch_3d, edgevector = z_grad_u, divvector = z_div_grad_u, div_coeff = operators_coeff % div_coeff)
    CALL sync_patch_array_mult(1, patch_2d, 3, lacc = .FALSE., f3din1 = z_div_grad_u(:, :, :) % x(1), f3din2 = z_div_grad_u(:, :, :) % x(2), f3din3 = z_div_grad_u(:, :, :) % x(3))
    CALL map_cell2edges_3d_mlevels_deconiface_171(patch_3d, z_div_grad_u, laplacian_vn_out, operators_coeff)
  END SUBROUTINE veloc_diff_harmonic_div_grad
  SUBROUTINE veloc_diff_biharmonic_div_grad(patch_3d, physics_parameters, p_diag, operators_coeff, laplacian_vn_out)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_ocean_physics_types, ONLY: t_ho_params
    USE mo_ocean_types, ONLY: t_hydro_ocean_diag, t_operator_coeff
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_nml, ONLY: n_zlev
    USE mo_math_types, ONLY: t_cartesian_coordinates
    USE mo_fortran_tools, ONLY: init_zero_3d_dp_deconiface_172 => init_zero_3d_dp, init_zero_3d_dp_deconiface_173 => init_zero_3d_dp, init_zero_3d_dp_deconiface_174 => init_zero_3d_dp, init_zero_3d_dp_deconiface_175 => init_zero_3d_dp
    USE mo_grid_subset, ONLY: get_index_range
    USE mo_sync, ONLY: sync_patch_array_3d_dp_deconiface_176 => sync_patch_array_3d_dp, sync_patch_array_3d_dp_deconiface_178 => sync_patch_array_3d_dp, sync_patch_array_3d_dp_deconiface_180 => sync_patch_array_3d_dp
    USE mo_ocean_math_operators, ONLY: div_oce_3d_mlevels_deconiface_177 => div_oce_3d_mlevels, div_oce_3d_mlevels_deconiface_179 => div_oce_3d_mlevels, grad_fd_norm_oce_3d
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    TYPE(t_ho_params), INTENT(IN) :: physics_parameters
    TYPE(t_hydro_ocean_diag) :: p_diag
    TYPE(t_operator_coeff), INTENT(IN) :: operators_coeff
    REAL(KIND = 8), INTENT(INOUT) :: laplacian_vn_out(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_e)
    INTEGER :: start_level, end_level
    INTEGER :: level, blockno, edge_index
    INTEGER :: il_c1, ib_c1, il_c2, ib_c2
    INTEGER :: start_edge_index, end_edge_index
    REAL(KIND = 8) :: z_grad_u_normal(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_e)
    REAL(KIND = 8) :: z_grad_u_normal_ptp(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_e)
    REAL(KIND = 8) :: grad_div_e(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_e)
    REAL(KIND = 8) :: div_c(nproma, n_zlev, patch_3d % p_patch_2d(1) % alloc_cell_blocks)
    TYPE(t_cartesian_coordinates) :: z_grad_u(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_e)
    TYPE(t_subset_range), POINTER :: all_edges, edges_in_domain
    TYPE(t_patch), POINTER :: patch_2d
    patch_2d => patch_3d % p_patch_2d(1)
    all_edges => patch_2d % edges % all
    edges_in_domain => patch_2d % edges % in_domain
    start_level = 1
    end_level = n_zlev
    CALL init_zero_3d_dp_deconiface_172(z_grad_u_normal(1 : nproma, 1 : n_zlev, 1 : patch_3d % p_patch_2d(1) % nblks_e), lacc = .FALSE.)
    CALL init_zero_3d_dp_deconiface_173(z_grad_u_normal_ptp(1 : nproma, 1 : n_zlev, 1 : patch_3d % p_patch_2d(1) % nblks_e), lacc = .FALSE.)
    CALL init_zero_3d_dp_deconiface_174(grad_div_e(1 : nproma, 1 : n_zlev, 1 : patch_3d % p_patch_2d(1) % nblks_e), lacc = .FALSE.)
    CALL init_zero_3d_dp_deconiface_175(div_c(1 : nproma, 1 : n_zlev, 1 : patch_3d % p_patch_2d(1) % alloc_cell_blocks), lacc = .FALSE.)
    DO blockno = all_edges % start_block, all_edges % end_block
      CALL get_index_range(all_edges, blockno, start_edge_index, end_edge_index)
      DO edge_index = start_edge_index, end_edge_index
        DO level = start_level, patch_3d % p_patch_1d(1) % dolic_e(edge_index, blockno)
          il_c1 = patch_2d % edges % cell_idx(edge_index, blockno, 1)
          ib_c1 = patch_2d % edges % cell_blk(edge_index, blockno, 1)
          il_c2 = patch_2d % edges % cell_idx(edge_index, blockno, 2)
          ib_c2 = patch_2d % edges % cell_blk(edge_index, blockno, 2)
          z_grad_u(edge_index, level, blockno) % x = (p_diag % p_vn(il_c2, level, ib_c2) % x - p_diag % p_vn(il_c1, level, ib_c1) % x) * patch_2d % edges % inv_dual_edge_length(edge_index, blockno)
          z_grad_u_normal(edge_index, level, blockno) = DOT_PRODUCT(z_grad_u(edge_index, level, blockno) % x, patch_2d % edges % primal_cart_normal(edge_index, blockno) % x)
        END DO
      END DO
    END DO
    CALL sync_patch_array_3d_dp_deconiface_176(2, patch_2d, z_grad_u_normal, lacc = .FALSE.)
    CALL div_oce_3d_mlevels_deconiface_177(z_grad_u_normal, patch_3d, operators_coeff % div_coeff, div_c)
    CALL grad_fd_norm_oce_3d(div_c, patch_3d, operators_coeff % grad_coeff, grad_div_e)
    CALL sync_patch_array_3d_dp_deconiface_178(2, patch_2d, grad_div_e, lacc = .FALSE.)
    CALL div_oce_3d_mlevels_deconiface_179(grad_div_e, patch_3d, operators_coeff % div_coeff, div_c)
    DO blockno = edges_in_domain % start_block, edges_in_domain % end_block
      CALL get_index_range(edges_in_domain, blockno, start_edge_index, end_edge_index)
      DO edge_index = start_edge_index, end_edge_index
        DO level = start_level, patch_3d % p_patch_1d(1) % dolic_e(edge_index, blockno)
          il_c1 = patch_2d % edges % cell_idx(edge_index, blockno, 1)
          ib_c1 = patch_2d % edges % cell_blk(edge_index, blockno, 1)
          il_c2 = patch_2d % edges % cell_idx(edge_index, blockno, 2)
          ib_c2 = patch_2d % edges % cell_blk(edge_index, blockno, 2)
          laplacian_vn_out(edge_index, level, blockno) = - 0.5D0 * physics_parameters % biharmonicviscosity_coeff(edge_index, level, blockno) * (div_c(il_c1, level, ib_c1) + div_c(il_c2, level, ib_c2))
        END DO
      END DO
    END DO
    DO level = 1, n_zlev
      WRITE(*, *) 'Biharmonic divgrad', level, MAXVAL(laplacian_vn_out(:, level, :)), MINVAL(laplacian_vn_out(:, level, :))
    END DO
    CALL sync_patch_array_3d_dp_deconiface_180(2, patch_2d, laplacian_vn_out, lacc = .FALSE.)
  END SUBROUTINE veloc_diff_biharmonic_div_grad
  SUBROUTINE veloc_diff_harmonic_curl_curl(patch_3d, u_vec_e, vort, div_coeff, nabla2_vec_e, harmonicdiffusion, k_h, lacc)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_nml, ONLY: harmonicdiv_weight, harmonicvort_weight, n_zlev
    USE mo_exception, ONLY: finish
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_ocean_math_operators, ONLY: div_oce_3d_mlevels_deconiface_181 => div_oce_3d_mlevels
    USE mo_grid_subset, ONLY: get_index_range
    TYPE(t_patch_3d), TARGET :: patch_3d
    REAL(KIND = 8) :: u_vec_e(:, :, :)
    REAL(KIND = 8) :: vort(:, :, :)
    REAL(KIND = 8) :: div_coeff(:, :, :, :)
    REAL(KIND = 8), OPTIONAL :: nabla2_vec_e(:, :, :)
    REAL(KIND = 8), OPTIONAL :: harmonicdiffusion(:, :, :)
    REAL(KIND = 8), OPTIONAL :: k_h(:, :, :)
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: start_level
    INTEGER :: edge_index, level, blockno
    INTEGER :: start_index, end_index
    LOGICAL :: lzacc
    REAL(KIND = 8) :: z_div_c(nproma, n_zlev, patch_3d % p_patch_2d(1) % alloc_cell_blocks)
    REAL(KIND = 8) :: nabla2(nproma, n_zlev)
    TYPE(t_subset_range), POINTER :: edges_in_domain
    TYPE(t_patch), POINTER :: patch_2d
    patch_2d => patch_3d % p_patch_2d(1)
    edges_in_domain => patch_2d % edges % in_domain
    IF (PRESENT(harmonicdiffusion) .AND. .NOT. PRESENT(k_h)) THEN
      CALL finish('veloc_diff_harmonic_curl_curl', 'present(HarmonicDiffusion) .and. .not. present(k_h)')
    END IF
    IF (.NOT. PRESENT(harmonicdiffusion) .AND. PRESENT(k_h)) THEN
      CALL finish('veloc_diff_harmonic_curl_curl', '.not. present(HarmonicDiffusion) .and. present(k_h)')
    END IF
    start_level = 1
    CALL set_acc_host_or_device(lzacc, lacc)
    CALL div_oce_3d_mlevels_deconiface_181(u_vec_e, patch_3d, div_coeff, z_div_c, subset_range = patch_2d % cells % all, lacc = lzacc)
    DO blockno = edges_in_domain % start_block, edges_in_domain % end_block
      CALL get_index_range(edges_in_domain, blockno, start_index, end_index)
      nabla2(:, :) = 0.0D0
      DO edge_index = start_index, end_index
        DO level = start_level, patch_3d % p_patch_1d(1) % dolic_e(edge_index, blockno)
          nabla2(edge_index, level) = patch_2d % edges % tangent_orientation(edge_index, blockno) * (vort(patch_2d % edges % vertex_idx(edge_index, blockno, 2), level, patch_2d % edges % vertex_blk(edge_index, blockno, 2)) - vort(patch_2d % edges % vertex_idx(edge_index, blockno, 1), level, patch_2d % edges % vertex_blk(edge_index, blockno, 1))) * patch_2d % edges % inv_primal_edge_length(edge_index, blockno) * harmonicvort_weight + (z_div_c(patch_2d % edges % cell_idx(edge_index, blockno, 2), level, patch_2d % edges % cell_blk(edge_index, blockno, 2)) - z_div_c(patch_2d % edges % cell_idx(edge_index, blockno, 1), level, patch_2d % edges % cell_blk(edge_index, blockno, 1))) * patch_2d % edges % inv_dual_edge_length(edge_index, blockno) * harmonicdiv_weight
        END DO
      END DO
      IF (PRESENT(nabla2_vec_e)) THEN
        nabla2_vec_e(:, :, blockno) = nabla2(:, :)
      END IF
      IF (PRESENT(harmonicdiffusion)) THEN
        harmonicdiffusion(:, :, blockno) = nabla2(:, :) * k_h(:, :, blockno)
      END IF
    END DO
  END SUBROUTINE veloc_diff_harmonic_curl_curl
  SUBROUTINE veloc_diff_biharmonic_curl_curl(patch_3d, physics_parameters, u_vec_e, vort, operators_coeff, nabla4_vec_e, lacc)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_ocean_physics_types, ONLY: t_ho_params
    USE mo_ocean_types, ONLY: t_operator_coeff
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_nml, ONLY: biharmonicdiv_weight, biharmonicvort_weight, n_zlev, velocitydiffusion_order
    USE mo_math_types, ONLY: t_cartesian_coordinates
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_sync, ONLY: sync_patch_array_3d_dp_deconiface_182 => sync_patch_array_3d_dp, sync_patch_array_mult_f3din_dp_deconiface_184 => sync_patch_array_mult_f3din_dp
    USE mo_ocean_math_operators, ONLY: div_oce_3d_mlevels_deconiface_183 => div_oce_3d_mlevels, map_edges2vert_3d, rot_vertex_ocean_3d
    USE mo_grid_subset, ONLY: get_index_range
    TYPE(t_patch_3d), TARGET :: patch_3d
    TYPE(t_ho_params) :: physics_parameters
    REAL(KIND = 8) :: u_vec_e(:, :, :)
    REAL(KIND = 8) :: vort(:, :, :)
    TYPE(t_operator_coeff), INTENT(IN) :: operators_coeff
    REAL(KIND = 8) :: nabla4_vec_e(:, :, :)
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: start_level, end_level
    INTEGER :: edge_index, level, blockno
    INTEGER :: start_index, end_index
    REAL(KIND = 8) :: z_div_c(nproma, n_zlev, patch_3d % p_patch_2d(1) % alloc_cell_blocks)
    REAL(KIND = 8) :: z_rot_v(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_v)
    REAL(KIND = 8) :: z_nabla2_e(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_e)
    TYPE(t_cartesian_coordinates) :: p_nabla2_dual(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_v)
    REAL(KIND = 8), DIMENSION(1 : nproma, 1 : n_zlev, 1 : patch_3d % p_patch_2d(1) % nblks_v) :: p_nabla2_dual_x, p_nabla2_dual_y, p_nabla2_dual_z
    TYPE(t_subset_range), POINTER :: edges_in_domain
    TYPE(t_patch), POINTER :: patch_2d
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    patch_2d => patch_3d % p_patch_2d(1)
    edges_in_domain => patch_2d % edges % in_domain
    start_level = 1
    end_level = n_zlev
    CALL veloc_diff_harmonic_curl_curl(patch_3d = patch_3d, u_vec_e = u_vec_e, vort = vort, div_coeff = operators_coeff % div_coeff, nabla2_vec_e = z_nabla2_e, lacc = lzacc)
    CALL sync_patch_array_3d_dp_deconiface_182(2, patch_2d, z_nabla2_e, lacc = lzacc)
    CALL div_oce_3d_mlevels_deconiface_183(z_nabla2_e, patch_3d, operators_coeff % div_coeff, z_div_c, subset_range = patch_2d % cells % all, lacc = lzacc)
    CALL map_edges2vert_3d(patch_2d, z_nabla2_e, operators_coeff % edge2vert_coeff_cc, p_nabla2_dual, lacc = lzacc)
    DO blockno = 1, patch_3d % p_patch_2d(1) % nblks_v
      p_nabla2_dual_x(:, :, blockno) = p_nabla2_dual(:, :, blockno) % x(1)
      p_nabla2_dual_y(:, :, blockno) = p_nabla2_dual(:, :, blockno) % x(2)
      p_nabla2_dual_z(:, :, blockno) = p_nabla2_dual(:, :, blockno) % x(3)
    END DO
    CALL sync_patch_array_mult_f3din_dp_deconiface_184(3, patch_2d, 3, lacc = lzacc, f3din1 = p_nabla2_dual_x, f3din2 = p_nabla2_dual_y, f3din3 = p_nabla2_dual_z)
    DO blockno = 1, patch_3d % p_patch_2d(1) % nblks_v
      p_nabla2_dual(:, :, blockno) % x(1) = p_nabla2_dual_x(:, :, blockno)
      p_nabla2_dual(:, :, blockno) % x(2) = p_nabla2_dual_y(:, :, blockno)
      p_nabla2_dual(:, :, blockno) % x(3) = p_nabla2_dual_z(:, :, blockno)
    END DO
    CALL rot_vertex_ocean_3d(patch_3d, z_nabla2_e, p_nabla2_dual, operators_coeff, z_rot_v, lacc = lzacc)
    DO blockno = edges_in_domain % start_block, edges_in_domain % end_block
      CALL get_index_range(edges_in_domain, blockno, start_index, end_index)
      DO edge_index = start_index, end_index
        DO level = start_level, patch_3d % p_patch_1d(1) % dolic_e(edge_index, blockno)
          nabla4_vec_e(edge_index, level, blockno) = - physics_parameters % biharmonicviscosity_coeff(edge_index, level, blockno) * ((patch_2d % edges % tangent_orientation(edge_index, blockno) * (z_rot_v(patch_2d % edges % vertex_idx(edge_index, blockno, 2), level, patch_2d % edges % vertex_blk(edge_index, blockno, 2)) - z_rot_v(patch_2d % edges % vertex_idx(edge_index, blockno, 1), level, patch_2d % edges % vertex_blk(edge_index, blockno, 1))) * patch_2d % edges % inv_primal_edge_length(edge_index, blockno)) * biharmonicvort_weight + ((z_div_c(patch_2d % edges % cell_idx(edge_index, blockno, 2), level, patch_2d % edges % cell_blk(edge_index, blockno, 2)) - z_div_c(patch_2d % edges % cell_idx(edge_index, blockno, 1), level, patch_2d % edges % cell_blk(edge_index, blockno, 1))) * patch_2d % edges % inv_dual_edge_length(edge_index, blockno) * biharmonicdiv_weight))
        END DO
      END DO
    END DO
    IF (velocitydiffusion_order == 21) THEN
      DO blockno = edges_in_domain % start_block, edges_in_domain % end_block
        CALL get_index_range(edges_in_domain, blockno, start_index, end_index)
        DO edge_index = start_index, end_index
          DO level = start_level, patch_3d % p_patch_1d(1) % dolic_e(edge_index, blockno)
            nabla4_vec_e(edge_index, level, blockno) = nabla4_vec_e(edge_index, level, blockno) + z_nabla2_e(edge_index, level, blockno) * physics_parameters % harmonicviscosity_coeff(edge_index, level, blockno)
          END DO
        END DO
      END DO
    END IF
  END SUBROUTINE veloc_diff_biharmonic_curl_curl
  SUBROUTINE velocity_diffusion_vertical_implicit_onblock(patch_3d, velocity, a_v, operators_coefficients, start_index, end_index, edge_block, lacc)
    USE mo_model_domain, ONLY: t_patch_3d
    USE mo_ocean_types, ONLY: t_operator_coeff
    USE mo_ocean_nml, ONLY: n_zlev
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_run_config, ONLY: dtime
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    REAL(KIND = 8), INTENT(INOUT) :: velocity(:, :)
    REAL(KIND = 8), INTENT(INOUT) :: a_v(:, :)
    TYPE(t_operator_coeff), INTENT(IN), TARGET :: operators_coefficients
    INTEGER, INTENT(IN) :: start_index, end_index, edge_block
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    REAL(KIND = 8) :: inv_prism_thickness(1 : n_zlev), inv_prisms_center_distance(1 : n_zlev)
    REAL(KIND = 8) :: a(1 : n_zlev), b(1 : n_zlev), c(1 : n_zlev)
    REAL(KIND = 8) :: column_velocity(1 : n_zlev)
    REAL(KIND = 8) :: fact(1 : n_zlev)
    INTEGER :: bottom_level
    INTEGER :: edge_index, level
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    DO edge_index = start_index, end_index
      bottom_level = patch_3d % p_patch_1d(1) % dolic_e(edge_index, edge_block)
      IF (bottom_level < 2) CYCLE
      DO level = 1, bottom_level
        inv_prism_thickness(level) = patch_3d % p_patch_1d(1) % inv_prism_thick_e(edge_index, level, edge_block)
        inv_prisms_center_distance(level) = patch_3d % p_patch_1d(1) % inv_prism_center_dist_e(edge_index, level, edge_block)
        column_velocity(level) = velocity(edge_index, level)
      END DO
      a(1) = 0.0D0
      c(1) = - a_v(edge_index, 2) * inv_prism_thickness(1) * inv_prisms_center_distance(2) * dtime
      b(1) = 1.0D0 - c(1)
      DO level = 2, bottom_level - 1
        a(level) = - a_v(edge_index, level) * inv_prism_thickness(level) * inv_prisms_center_distance(level) * dtime
        c(level) = - a_v(edge_index, level + 1) * inv_prism_thickness(level) * inv_prisms_center_distance(level + 1) * dtime
        b(level) = 1.0D0 - a(level) - c(level)
      END DO
      a(bottom_level) = - a_v(edge_index, bottom_level) * inv_prism_thickness(bottom_level) * inv_prisms_center_distance(bottom_level) * dtime
      b(bottom_level) = 1.0D0 - a(bottom_level)
      c(bottom_level) = 0.0D0
      IF (eliminate_upper_diag) THEN
        DO level = bottom_level - 1, 1, - 1
          fact(level) = c(level) / b(level + 1)
          b(level) = b(level) - a(level + 1) * fact(level)
          c(level) = 0.0D0
          column_velocity(level) = column_velocity(level) - fact(level) * column_velocity(level + 1)
        END DO
        velocity(edge_index, 1) = column_velocity(1) / b(1)
        DO level = 2, bottom_level
          velocity(edge_index, level) = (column_velocity(level) - a(level) * velocity(edge_index, level - 1)) / b(level)
        END DO
      ELSE
        DO level = 2, bottom_level
          fact(level) = a(level) / b(level - 1)
          b(level) = b(level) - c(level - 1) * fact(level)
          a(level) = 0.0D0
          column_velocity(level) = column_velocity(level) - fact(level) * column_velocity(level - 1)
        END DO
        velocity(edge_index, bottom_level) = column_velocity(bottom_level) / b(bottom_level)
        DO level = bottom_level - 1, 1, - 1
          velocity(edge_index, level) = (column_velocity(level) - c(level) * velocity(edge_index, level + 1)) / b(level)
        END DO
      END IF
    END DO
  END SUBROUTINE velocity_diffusion_vertical_implicit_onblock
END MODULE mo_ocean_velocity_diffusion
MODULE mo_ocean_ab_timestepping_mimetic
  USE mo_ocean_solve, ONLY: t_ocean_solve
  USE mo_ocean_solve_lhs_type, ONLY: t_primal_flip_flop_lhs, t_surface_height_lhs
  USE mo_ocean_solve_transfer, ONLY: t_subset_transfer, t_trivial_transfer
  IMPLICIT NONE
  TYPE(t_ocean_solve) :: free_sfc_solver
  TYPE(t_ocean_solve) :: free_sfc_solver_comp
  TYPE(t_surface_height_lhs), TARGET :: free_sfc_solver_lhs
  TYPE(t_subset_transfer), TARGET :: free_sfc_solver_trans_sub
  TYPE(t_trivial_transfer), TARGET :: free_sfc_solver_trans_triv
  TYPE(t_ocean_solve) :: inv_mm_solver
  TYPE(t_primal_flip_flop_lhs), TARGET :: inv_mm_solver_lhs
  TYPE(t_trivial_transfer), TARGET :: inv_mm_solver_trans
  CHARACTER(LEN = *), PARAMETER :: str_module = 'oceSTEPmimet'
  INTEGER :: idt_src = 1
  INTEGER, SAVE :: istep = 0
  CONTAINS
  SUBROUTINE init_free_sfc_ab_mimetic(patch_3d, ocean_state, op_coeffs, solvercoeff_sp, lacc)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d
    USE mo_ocean_types, ONLY: t_hydro_ocean_state, t_operator_coeff, t_solvercoeff_singleprecision
    USE mo_ocean_solve_aux, ONLY: ocean_solve_parm_init_deconproc_97 => ocean_solve_parm_init, t_ocean_solve_parm
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_nml, ONLY: l_solver_compare, select_solver, select_transfer, solver_max_iter_per_restart, solver_max_iter_per_restart_sp, solver_max_restart_iterations, solver_tolerance, solver_tolerance_comp, solver_tolerance_sp, use_absolute_solver_tolerance
    USE mo_exception, ONLY: finish
    USE mo_ocean_solve_lhs_type, ONLY: lhs_surface_height_construct_deconproc_98 => lhs_surface_height_construct
    USE mo_ocean_solve_transfer, ONLY: subset_transfer_construct_deconproc_100 => subset_transfer_construct, trivial_transfer_construct_deconproc_101 => trivial_transfer_construct, trivial_transfer_construct_deconproc_99 => trivial_transfer_construct
    USE mo_ocean_solve, ONLY: ocean_solve_construct__t_surface_height_lhs__t_subset_transfer, ocean_solve_construct__t_surface_height_lhs__t_trivial_transfer
    TYPE(t_patch_3d), POINTER, INTENT(IN) :: patch_3d
    TYPE(t_hydro_ocean_state), TARGET, INTENT(INOUT) :: ocean_state
    TYPE(t_operator_coeff), INTENT(IN), TARGET :: op_coeffs
    TYPE(t_solvercoeff_singleprecision), INTENT(IN), TARGET :: solvercoeff_sp
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    TYPE(t_patch), POINTER :: patch_2d
    TYPE(t_ocean_solve_parm) :: par, par_sp
    INTEGER :: trans_mode, sol_type
    LOGICAL :: lzacc
    CHARACTER(LEN = *), PARAMETER :: method_name = 'mo_ocean_ab_timestepping_mimetic:init_free_sfc_ab_mimetic'
    CALL set_acc_host_or_device(lzacc, lacc)
    IF (free_sfc_solver % is_init) RETURN
    patch_2d => patch_3d % p_patch_2d(1)
    CALL ocean_solve_parm_init_deconproc_97(par, 60, 1, 800, patch_2d % cells % in_domain % end_block, patch_2d % alloc_cell_blocks, nproma, patch_2d % cells % in_domain % end_index, solver_tolerance, use_absolute_solver_tolerance)
    par_sp % nidx = (- 1)
    sol_type = 21
    SELECT CASE (select_solver)
    CASE (1)
    CASE (2)
      par % m = solver_max_iter_per_restart
      par % nr = solver_max_restart_iterations
    CASE (3)
      par % nr = solver_max_restart_iterations
      par_sp = par
      par_sp % m = solver_max_iter_per_restart_sp
      par_sp % tol = REAL(solver_tolerance_sp, 8)
      par % m = solver_max_iter_per_restart
    CASE (4)
      sol_type = 22
    CASE (40)
      sol_type = 22
      par % pt = 62
    CASE (9)
      sol_type = 22
      par_sp = par
      par_sp % tol = REAL(solver_tolerance_sp, 8)
    CASE (5)
      sol_type = 22
      par % pt = 61
    CASE (6)
      sol_type = 23
    CASE (7)
      sol_type = 24
      par % m = solver_max_iter_per_restart
      par_sp % nr = solver_max_restart_iterations
    CASE (8)
      sol_type = 25
      par % nr = (par % m + 18) / 19
      par % m = 19
    CASE DEFAULT
      CALL finish(method_name, "Unknown solver")
    END SELECT
    CALL lhs_surface_height_construct_deconproc_98(free_sfc_solver_lhs, patch_3d, ocean_state % p_diag % thick_e, op_coeffs, solvercoeff_sp, lacc = lzacc)
    SELECT CASE (select_transfer)
    CASE (0)
      CALL trivial_transfer_construct_deconproc_99(free_sfc_solver_trans_triv, 11, patch_2d, lacc = lzacc)
      CALL ocean_solve_construct__t_surface_height_lhs__t_trivial_transfer(free_sfc_solver, sol_type, par, par_sp, free_sfc_solver_lhs, free_sfc_solver_trans_triv, lacc = lzacc)
    CASE DEFAULT
      trans_mode = MERGE(71, 70, select_transfer .GT. 0)
      CALL subset_transfer_construct_deconproc_100(free_sfc_solver_trans_sub, 11, patch_2d, ABS(select_transfer), trans_mode, lacc = lzacc)
      CALL ocean_solve_construct__t_surface_height_lhs__t_subset_transfer(free_sfc_solver, sol_type, par, par_sp, free_sfc_solver_lhs, free_sfc_solver_trans_sub, lacc = lzacc)
    END SELECT
    IF (l_solver_compare) THEN
      CALL trivial_transfer_construct_deconproc_101(free_sfc_solver_trans_triv, 11, patch_2d, lacc = lzacc)
      par % tol = solver_tolerance_comp
      par_sp % nidx = (- 1)
      CALL ocean_solve_construct__t_surface_height_lhs__t_trivial_transfer(free_sfc_solver_comp, 24, par, par_sp, free_sfc_solver_lhs, free_sfc_solver_trans_triv, lacc = lzacc)
    END IF
  END SUBROUTINE init_free_sfc_ab_mimetic
  SUBROUTINE solve_free_sfc_ab_mimetic(patch_3d, ocean_state, p_ext_data, p_as, p_oce_sfc, p_phys_param, timestep, op_coeffs, solvercoeff_sp, ret_status, lacc)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_ocean_types, ONLY: t_hydro_ocean_state, t_operator_coeff, t_solvercoeff_singleprecision
    USE mo_ext_data_types, ONLY: t_external_data
    USE mo_ocean_surface_types, ONLY: t_atmos_for_ocean, t_ocean_surface
    USE mo_ocean_physics_types, ONLY: t_ho_params
    USE mo_impl_constants, ONLY: max_char_length
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_util_dbg_prnt, ONLY: dbg_print_2d_deconiface_185 => dbg_print_2d, dbg_print_2d_deconiface_187 => dbg_print_2d, dbg_print_2d_deconiface_189 => dbg_print_2d, dbg_print_3d_deconiface_186 => dbg_print_3d, dbg_print_3d_deconiface_188 => dbg_print_3d, dbg_print_3d_deconiface_193 => dbg_print_3d, debug_print_maxminmean
    USE mo_dynamics_config, ONLY: nnew, nold
    USE mo_ocean_boundcond, ONLY: top_bound_cond_horz_veloc
    USE mo_run_config, ONLY: timers_level
    USE mo_real_timer, ONLY: timer_start, timer_stop
    USE mo_timer, ONLY: timer_ab_expl, timer_ab_rhs4sfc
    USE mo_ocean_initialization, ONLY: is_initial_timestep
    USE mo_ocean_nml, ONLY: createsolvermatrix, l_rigid_lid, l_solver_compare, solver_comp_nsteps, solver_firstguess, solver_tolerance
    USE mo_ocean_math_operators, ONLY: smooth_oncells
    USE mo_grid_subset, ONLY: get_index_range
    USE mo_ocean_solve, ONLY: ocean_solve_dump_matrix_deconproc_104 => ocean_solve_dump_matrix, ocean_solve_solve_deconproc_102 => ocean_solve_solve, ocean_solve_solve_deconproc_103 => ocean_solve_solve
    USE mo_dbg_nml, ONLY: idbg_mxmn
    USE mo_exception, ONLY: message, warning
    USE mo_statistics, ONLY: minmaxmean_2d_inrange_deconiface_190 => minmaxmean_2d_inrange, minmaxmean_2d_inrange_deconiface_191 => minmaxmean_2d_inrange, minmaxmean_2d_inrange_deconiface_194 => minmaxmean_2d_inrange, print_2dvalue_location_deconiface_195 => print_2dvalue_location, print_2dvalue_location_deconiface_196 => print_2dvalue_location
    USE mo_sync, ONLY: sync_patch_array_2d_dp_deconiface_192 => sync_patch_array_2d_dp
    USE mo_mpi, ONLY: work_mpi_barrier
    TYPE(t_patch_3d), POINTER, INTENT(IN) :: patch_3d
    TYPE(t_hydro_ocean_state), TARGET, INTENT(INOUT) :: ocean_state
    TYPE(t_external_data), TARGET, INTENT(IN) :: p_ext_data
    TYPE(t_ocean_surface), INTENT(INOUT) :: p_oce_sfc
    TYPE(t_atmos_for_ocean), INTENT(INOUT) :: p_as
    TYPE(t_ho_params), INTENT(INOUT) :: p_phys_param
    INTEGER, INTENT(IN) :: timestep
    TYPE(t_operator_coeff), INTENT(IN), TARGET :: op_coeffs
    TYPE(t_solvercoeff_singleprecision), INTENT(IN), TARGET :: solvercoeff_sp
    INTEGER :: n_it, n_it_sp, ret_status
    REAL(KIND = 8) :: rn, minmaxmean(3)
    LOGICAL :: l_is_compare_step
    CHARACTER(LEN = max_char_length) :: string
    TYPE(t_subset_range), POINTER :: owned_cells, owned_edges
    TYPE(t_patch), POINTER :: patch_2d
    CHARACTER(LEN = *), PARAMETER :: method_name = 'mo_ocean_ab_timestepping_mimetic:solve_free_sfc_ab_mimetic'
    INTEGER :: jc, blockno
    INTEGER :: startindex, endindex
    TYPE(t_subset_range), POINTER :: all_cells
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    l_is_compare_step = .FALSE.
    patch_2d => patch_3d % p_patch_2d(1)
    owned_cells => patch_2d % cells % owned
    owned_edges => patch_2d % edges % owned
    all_cells => patch_2d % cells % all
    IF (.NOT. free_sfc_solver % is_init) CALL init_free_sfc_ab_mimetic(patch_3d, ocean_state, op_coeffs, solvercoeff_sp, lacc = lzacc)
    ret_status = 0
    CALL dbg_print_2d_deconiface_185('on entry: h-old', ocean_state % p_prog(nold(1)) % h, str_module, 3, in_subset = patch_2d % cells % owned)
    CALL dbg_print_3d_deconiface_186('on entry: vn-old', ocean_state % p_prog(nold(1)) % vn, str_module, 3, in_subset = patch_2d % edges % owned)
    CALL dbg_print_2d_deconiface_187('on entry: h-new', ocean_state % p_prog(nnew(1)) % h, str_module, 2, in_subset = patch_2d % cells % owned)
    CALL dbg_print_3d_deconiface_188('on entry: vn-new', ocean_state % p_prog(nnew(1)) % vn, str_module, 2, in_subset = patch_2d % edges % owned)
    CALL top_bound_cond_horz_veloc(patch_3d, ocean_state, op_coeffs, p_oce_sfc, lacc = lzacc)
    IF (timers_level >= 3) CALL timer_start(timer_ab_expl)
    CALL calculate_explicit_term_ab(patch_3d, ocean_state, p_phys_param, is_initial_timestep(timestep), op_coeffs, p_as, lacc = lzacc)
    IF (timers_level >= 3) CALL timer_stop(timer_ab_expl)
    IF (.NOT. l_rigid_lid) THEN
      IF (timers_level >= 5) CALL timer_start(timer_ab_rhs4sfc)
      CALL fill_rhs4surface_eq_ab(patch_3d, ocean_state, op_coeffs, lacc = lzacc)
      IF (timers_level >= 5) CALL timer_stop(timer_ab_rhs4sfc)
      SELECT CASE (solver_firstguess)
      CASE (1)
        CALL smooth_oncells(patch_3d, ocean_state % p_prog(nold(1)) % h, free_sfc_solver % x_loc_wp, (/0.5D0, 0.5D0/), .FALSE., - 999999.0D0, lacc = lzacc)
      CASE (2)
        DO blockno = all_cells % start_block, all_cells % end_block
          CALL get_index_range(all_cells, blockno, startindex, endindex)
          DO jc = startindex, endindex
            free_sfc_solver % x_loc_wp(jc, blockno) = ocean_state % p_prog(nold(1)) % h(jc, blockno)
          END DO
        END DO
      CASE DEFAULT
        DO blockno = all_cells % start_block, all_cells % end_block
          CALL get_index_range(all_cells, blockno, startindex, endindex)
          DO jc = startindex, endindex
            free_sfc_solver % x_loc_wp(jc, blockno) = 0.0D0
          END DO
        END DO
      END SELECT
      free_sfc_solver % b_loc_wp => ocean_state % p_aux % p_rhs_sfc_eq
      IF (l_solver_compare) THEN
        IF (istep .EQ. 0) l_is_compare_step = .TRUE.
        istep = istep + 1
        IF (istep .GE. solver_comp_nsteps) istep = 0
      END IF
      IF (l_is_compare_step) THEN
        free_sfc_solver_comp % x_loc_wp(:, :) = free_sfc_solver % x_loc_wp(:, :)
        free_sfc_solver_comp % b_loc_wp => free_sfc_solver % b_loc_wp
      END IF
      CALL dbg_print_2d_deconiface_189('bef ocean_solve(' // TRIM(free_sfc_solver % sol_type_name) // '): h-old', ocean_state % p_prog(nold(1)) % h(:, :), str_module, idt_src, in_subset = owned_cells)
      CALL ocean_solve_solve_deconproc_102(free_sfc_solver, n_it, n_it_sp, lacc = lzacc)
      rn = MERGE(free_sfc_solver % res_loc_wp(1), 0.0D0, n_it .NE. 0)
      IF (idbg_mxmn >= 0) THEN
        IF (n_it_sp .NE. - 2) THEN
          WRITE(string, '(2(a,i4),2(a,e28.20),a)') 'SUM of ocean_solve iteration(sp,wp) = (', n_it_sp - 1, ', ', n_it - 1, ') , residual = (', free_sfc_solver % res_loc_wp(1), ', ', rn, ')'
        ELSE
          WRITE(string, '(a,i4,a,e28.20)') 'SUM of ocean_solve iteration =', n_it - 1, ', residual =', rn
        END IF
        CALL message('ocean_solve(' // TRIM(free_sfc_solver % sol_type_name) // '): surface height', TRIM(string))
      END IF
      IF (rn <= solver_tolerance) THEN
        DO blockno = all_cells % start_block, all_cells % end_block
          CALL get_index_range(all_cells, blockno, startindex, endindex)
          DO jc = startindex, endindex
            ocean_state % p_prog(nnew(1)) % h(jc, blockno) = free_sfc_solver % x_loc_wp(jc, blockno)
          END DO
        END DO
        IF (l_is_compare_step) THEN
          CALL ocean_solve_solve_deconproc_103(free_sfc_solver_comp, n_it, n_it_sp, lacc = lzacc)
          rn = MERGE(free_sfc_solver_comp % res_loc_wp(1), 0.0D0, n_it .NE. 0)
          WRITE(string, '(a,i4,a,e28.20)') 'SUM of ocean_solve iteration =', n_it - 1, ', residual =', rn
          CALL message('ocean_solve(' // TRIM(free_sfc_solver_comp % sol_type_name) // '): surface height', TRIM(string))
          free_sfc_solver_comp % x_loc_wp(:, :) = free_sfc_solver % x_loc_wp(:, :) - free_sfc_solver_comp % x_loc_wp(:, :)
          minmaxmean(:) = minmaxmean_2d_inrange_deconiface_190(values = free_sfc_solver_comp % x_loc_wp(:, :), in_subset = owned_cells)
          WRITE(string, "(a,3(e12.3,'  '))") "comparison of solutions: (min/max/mean)", minmaxmean(:)
          CALL message('ocean_solve(' // TRIM(free_sfc_solver_comp % sol_type_name) // '): surface height', TRIM(string))
          free_sfc_solver_comp % x_loc_wp(:, :) = free_sfc_solver_comp % x_loc_wp(:, :) * free_sfc_solver_comp % x_loc_wp(:, :)
          minmaxmean(:) = minmaxmean_2d_inrange_deconiface_191(values = free_sfc_solver_comp % x_loc_wp(:, :), in_subset = owned_cells)
          WRITE(string, "(a,3(e12.3,'  '))") "comparison of solutions (squared): (min/max/mean)", SQRT(minmaxmean(:))
          CALL message('ocean_solve(' // TRIM(free_sfc_solver_comp % sol_type_name) // '): surface height', TRIM(string))
        END IF
        IF (createsolvermatrix) CALL ocean_solve_dump_matrix_deconproc_104(free_sfc_solver, timestep, lacc = lzacc)
        CALL sync_patch_array_2d_dp_deconiface_192(1, patch_2d, ocean_state % p_prog(nnew(1)) % h, lacc = lzacc)
        CALL dbg_print_3d_deconiface_193('vn-new', ocean_state % p_prog(nnew(1)) % vn, str_module, 2, in_subset = owned_edges)
        minmaxmean(:) = minmaxmean_2d_inrange_deconiface_194(values = ocean_state % p_prog(nnew(1)) % h(:, :), in_subset = owned_cells)
        CALL debug_print_maxminmean('h-new after ocean solver', minmaxmean, str_module, 1)
        IF (minmaxmean(1) + patch_3d % p_patch_1d(1) % del_zlev_m(1) <= 0.05D0) THEN
          CALL warning(method_name, "height below min_top_height")
          CALL print_2dvalue_location_deconiface_195(ocean_state % p_prog(nnew(1)) % h(:, :), minmaxmean(1), owned_cells)
          CALL print_2dvalue_location_deconiface_196(ocean_state % p_prog(nnew(2)) % h(:, :), minmaxmean(1), owned_cells)
          CALL work_mpi_barrier
          ret_status = 1
        END IF
      ELSE
        ret_status = 2
        CALL warning(method_name, "NOT YET CONVERGED !!")
      END IF
    END IF
  END SUBROUTINE solve_free_sfc_ab_mimetic
  SUBROUTINE calculate_explicit_term_ab(patch_3d, ocean_state, p_phys_param, is_first_timestep, op_coeffs, p_as, lacc)
    USE mo_model_domain, ONLY: t_patch_3d
    USE mo_ocean_types, ONLY: t_hydro_ocean_state, t_operator_coeff
    USE mo_ocean_physics_types, ONLY: t_ho_params
    USE mo_ocean_surface_types, ONLY: t_atmos_for_ocean
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_run_config, ONLY: timers_level
    USE mo_real_timer, ONLY: timer_start, timer_stop
    USE mo_timer, ONLY: timer_extra1, timer_extra2, timer_extra3, timer_extra4
    USE mo_ocean_velocity_advection, ONLY: veloc_adv_horz_mimetic, veloc_adv_vert_mimetic
    USE mo_dynamics_config, ONLY: nnew, nold
    USE mo_ocean_nml, ONLY: iswm_oce, mass_matrix_inversion_type
    USE mo_ocean_thermodyn, ONLY: calc_internal_press_grad
    USE mo_util_dbg_prnt, ONLY: dbg_print_2d_deconiface_201 => dbg_print_2d, dbg_print_3d_deconiface_197 => dbg_print_3d, dbg_print_3d_deconiface_198 => dbg_print_3d, dbg_print_3d_deconiface_199 => dbg_print_3d, dbg_print_3d_deconiface_200 => dbg_print_3d, dbg_print_3d_deconiface_202 => dbg_print_3d, dbg_print_3d_deconiface_203 => dbg_print_3d, dbg_print_3d_deconiface_204 => dbg_print_3d, dbg_print_3d_deconiface_205 => dbg_print_3d, dbg_print_3d_deconiface_206 => dbg_print_3d, dbg_print_3d_deconiface_207 => dbg_print_3d, dbg_print_3d_deconiface_208 => dbg_print_3d, dbg_print_3d_deconiface_209 => dbg_print_3d, dbg_print_3d_deconiface_210 => dbg_print_3d, dbg_print_3d_deconiface_211 => dbg_print_3d
    USE mo_grid_config, ONLY: n_dom
    USE mo_ocean_velocity_diffusion, ONLY: velocity_diffusion
    TYPE(t_patch_3d), POINTER, INTENT(IN) :: patch_3d
    TYPE(t_hydro_ocean_state), TARGET :: ocean_state
    TYPE(t_ho_params) :: p_phys_param
    LOGICAL, INTENT(IN) :: is_first_timestep
    TYPE(t_operator_coeff), INTENT(IN), TARGET :: op_coeffs
    TYPE(t_atmos_for_ocean), INTENT(INOUT) :: p_as
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    IF (timers_level >= 4) CALL timer_start(timer_extra1)
    IF (is_first_timestep) THEN
      CALL veloc_adv_horz_mimetic(patch_3d, ocean_state % p_prog(nold(1)) % vn, ocean_state % p_prog(nold(1)) % vn, ocean_state % p_diag, ocean_state % p_diag % veloc_adv_horz, op_coeffs, lacc = lzacc)
    ELSE
      CALL veloc_adv_horz_mimetic(patch_3d, ocean_state % p_prog(nold(1)) % vn, ocean_state % p_prog(nnew(1)) % vn, ocean_state % p_diag, ocean_state % p_diag % veloc_adv_horz, op_coeffs, lacc = lzacc)
    END IF
    IF (timers_level >= 4) CALL timer_stop(timer_extra1)
    IF (iswm_oce /= 1) THEN
      IF (timers_level >= 4) CALL timer_start(timer_extra2)
      CALL calc_internal_press_grad(patch_3d, ocean_state % p_diag % rho, ocean_state % p_diag % press_hyd, ocean_state % p_aux % bc_total_top_potential, op_coeffs % grad_coeff, ocean_state % p_diag % press_grad, lacc = lzacc)
      CALL veloc_adv_vert_mimetic(patch_3d, ocean_state % p_diag, op_coeffs, ocean_state % p_diag % veloc_adv_vert, lacc = lzacc)
      IF (timers_level >= 4) CALL timer_stop(timer_extra2)
    ELSE
      ocean_state % p_diag % veloc_adv_vert = 0.0D0
      ocean_state % p_diag % laplacian_vert = 0.0D0
    END IF
    idt_src = 3
    CALL dbg_print_3d_deconiface_197('density', ocean_state % p_diag % rho, str_module, idt_src, in_subset = patch_3d % p_patch_2d(n_dom) % cells % owned)
    CALL dbg_print_3d_deconiface_198('internal pressure', ocean_state % p_diag % press_hyd, str_module, idt_src, in_subset = patch_3d % p_patch_2d(n_dom) % cells % owned)
    CALL dbg_print_3d_deconiface_199('internal press grad', ocean_state % p_diag % press_grad, str_module, idt_src, in_subset = patch_3d % p_patch_2d(n_dom) % edges % owned)
    idt_src = 4
    CALL dbg_print_3d_deconiface_200('kinetic energy', ocean_state % p_diag % kin, str_module, idt_src, in_subset = patch_3d % p_patch_2d(n_dom) % cells % owned)
    IF (timers_level >= 5) CALL timer_start(timer_extra3)
    CALL velocity_diffusion(patch_3d, ocean_state % p_prog(nold(1)) % vn, p_phys_param, ocean_state % p_diag, op_coeffs, ocean_state % p_diag % laplacian_horz, lacc = lzacc)
    IF (timers_level >= 5) CALL timer_stop(timer_extra3)
    IF (timers_level >= 4) CALL timer_start(timer_extra4)
    IF (mass_matrix_inversion_type == 2 .OR. mass_matrix_inversion_type == 1) THEN
      CALL explicit_vn_pred_invert_mass_matrix(patch_3d, ocean_state, op_coeffs, p_phys_param, is_first_timestep)
    ELSE
      CALL explicit_vn_pred(patch_3d, ocean_state, op_coeffs, p_phys_param, is_first_timestep, lacc = lzacc)
    END IF
    IF (timers_level >= 4) CALL timer_stop(timer_extra4)
    idt_src = 3
    idt_src = 4
    CALL dbg_print_2d_deconiface_201('bc_top_vn', ocean_state % p_aux % bc_top_vn, str_module, idt_src, in_subset = patch_3d % p_patch_2d(n_dom) % edges % owned)
    CALL dbg_print_3d_deconiface_202('horizontal advection', ocean_state % p_diag % veloc_adv_horz, str_module, idt_src, in_subset = patch_3d % p_patch_2d(n_dom) % edges % owned)
    CALL dbg_print_3d_deconiface_203('horizontal grad', ocean_state % p_diag % grad, str_module, idt_src, in_subset = patch_3d % p_patch_2d(n_dom) % edges % owned)
    CALL dbg_print_3d_deconiface_204('vertical advection', ocean_state % p_diag % veloc_adv_vert, str_module, idt_src, in_subset = patch_3d % p_patch_2d(n_dom) % edges % owned)
    CALL dbg_print_3d_deconiface_205('VelocDiff: LaPlacHorz', ocean_state % p_diag % laplacian_horz, str_module, idt_src, in_subset = patch_3d % p_patch_2d(n_dom) % edges % owned)
    IF (iswm_oce /= 1) THEN
      CALL dbg_print_3d_deconiface_206('vn_pred', ocean_state % p_diag % vn_pred, str_module, 2, in_subset = patch_3d % p_patch_2d(n_dom) % edges % owned)
    ELSE
      CALL dbg_print_3d_deconiface_207('VelocDiff: LaPlacVert', ocean_state % p_diag % laplacian_vert, str_module, idt_src, in_subset = patch_3d % p_patch_2d(n_dom) % edges % owned)
    END IF
    idt_src = 5
    CALL dbg_print_3d_deconiface_208('vn(nold)', ocean_state % p_prog(nold(1)) % vn, str_module, idt_src, in_subset = patch_3d % p_patch_2d(n_dom) % edges % owned)
    CALL dbg_print_3d_deconiface_209('G_n+1/2 - g_nimd', ocean_state % p_aux % g_nimd, str_module, idt_src, in_subset = patch_3d % p_patch_2d(n_dom) % edges % owned)
    CALL dbg_print_3d_deconiface_210('G_n', ocean_state % p_aux % g_n, str_module, idt_src, in_subset = patch_3d % p_patch_2d(n_dom) % edges % owned)
    CALL dbg_print_3d_deconiface_211('G_n-1', ocean_state % p_aux % g_nm1, str_module, idt_src, in_subset = patch_3d % p_patch_2d(n_dom) % edges % owned)
  END SUBROUTINE calculate_explicit_term_ab
  SUBROUTINE explicit_vn_pred(patch_3d, ocean_state, op_coeffs, p_phys_param, is_first_timestep, lacc)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_ocean_types, ONLY: t_hydro_ocean_state, t_operator_coeff
    USE mo_ocean_physics_types, ONLY: t_ho_params
    USE mo_parallel_config, ONLY: nproma
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_grid_config, ONLY: n_dom
    USE mo_grid_subset, ONLY: get_index_range
    USE mo_ocean_math_operators, ONLY: grad_fd_norm_oce_2d_onblock
    USE mo_dynamics_config, ONLY: nold
    USE mo_ocean_nml, ONLY: ab_beta, iswm_oce, ppscheme_type, vert_mix_type
    USE mo_ocean_pp_scheme, ONLY: icon_pp_edge_vnpredict_scheme
    USE mo_ocean_velocity_diffusion, ONLY: velocity_diffusion_vertical_implicit_onblock
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    TYPE(t_hydro_ocean_state), TARGET :: ocean_state
    TYPE(t_operator_coeff), INTENT(IN) :: op_coeffs
    TYPE(t_ho_params) :: p_phys_param
    LOGICAL, INTENT(IN) :: is_first_timestep
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    REAL(KIND = 8) :: z_gradh_e(nproma)
    TYPE(t_subset_range), POINTER :: edges_in_domain
    INTEGER :: start_edge_index, end_edge_index, blockno, je
    TYPE(t_patch), POINTER :: patch_2d
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    patch_2d => patch_3d % p_patch_2d(n_dom)
    edges_in_domain => patch_3d % p_patch_2d(n_dom) % edges % in_domain
    DO blockno = edges_in_domain % start_block, edges_in_domain % end_block
      CALL get_index_range(edges_in_domain, blockno, start_edge_index, end_edge_index)
      DO je = start_edge_index, end_edge_index
        z_gradh_e(je) = 0.0D0
      END DO
      CALL grad_fd_norm_oce_2d_onblock(ocean_state % p_prog(nold(1)) % h, patch_2d, op_coeffs % grad_coeff(:, 1, blockno), z_gradh_e(:), start_edge_index, end_edge_index, blockno, lacc = lzacc)
      DO je = start_edge_index, end_edge_index
        z_gradh_e(je) = (1.0D0 - ab_beta) * 9.80665D0 * z_gradh_e(je)
      END DO
      CALL calculate_explicit_term_g_n_onblock(patch_3d, ocean_state, is_first_timestep, start_edge_index, end_edge_index, blockno, lacc = lzacc)
      IF (iswm_oce /= 1) THEN
        CALL calculate_explicit_vn_pred_3d_onblock(patch_3d, ocean_state, z_gradh_e(:), start_edge_index, end_edge_index, blockno, lacc = lzacc)
        IF (vert_mix_type == 1 .AND. ppscheme_type == 4) CALL icon_pp_edge_vnpredict_scheme(patch_3d, blockno, start_edge_index, end_edge_index, ocean_state, ocean_state % p_diag % vn_pred(:, :, blockno), lacc = lzacc)
        CALL velocity_diffusion_vertical_implicit_onblock(patch_3d, ocean_state % p_diag % vn_pred(:, :, blockno), p_phys_param % a_veloc_v(:, :, blockno), op_coeffs, start_edge_index, end_edge_index, blockno, lacc = lzacc)
      ELSE
        CALL calculate_explicit_vn_pred_2d_onblock(patch_3d, ocean_state, z_gradh_e(:), start_edge_index, end_edge_index, blockno, lacc = lzacc)
      END IF
    END DO
  END SUBROUTINE explicit_vn_pred
  SUBROUTINE explicit_vn_pred_invert_mass_matrix(patch_3d, ocean_state, op_coeffs, p_phys_param, is_first_timestep)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_ocean_types, ONLY: t_hydro_ocean_state, t_operator_coeff
    USE mo_ocean_physics_types, ONLY: t_ho_params
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_nml, ONLY: ab_beta, iswm_oce, mass_matrix_inversion_type, n_zlev
    USE mo_grid_config, ONLY: n_dom
    USE mo_sync, ONLY: sync_patch_array_3d_dp_deconiface_212 => sync_patch_array_3d_dp
    USE mo_grid_subset, ONLY: get_index_range
    USE mo_ocean_math_operators, ONLY: grad_fd_norm_oce_2d_onblock
    USE mo_dynamics_config, ONLY: nold
    USE mo_ocean_velocity_diffusion, ONLY: velocity_diffusion_vertical_implicit_onblock
    USE mo_scalar_product, ONLY: map_edges2edges_viacell_2d_per_level
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    TYPE(t_hydro_ocean_state), TARGET :: ocean_state
    TYPE(t_operator_coeff), INTENT(IN) :: op_coeffs
    TYPE(t_ho_params) :: p_phys_param
    LOGICAL, INTENT(IN) :: is_first_timestep
    REAL(KIND = 8) :: z_gradh_e(nproma)
    TYPE(t_subset_range), POINTER :: edges_in_domain
    INTEGER :: start_edge_index, end_edge_index, blockno
    TYPE(t_patch), POINTER :: patch_2d
    REAL(KIND = 8) :: z_e(nproma, n_zlev, patch_3d % p_patch_2d(n_dom) % nblks_e)
    patch_2d => patch_3d % p_patch_2d(n_dom)
    edges_in_domain => patch_3d % p_patch_2d(n_dom) % edges % in_domain
    IF (mass_matrix_inversion_type == 1) THEN
      z_e(:, :, :) = ocean_state % p_diag % veloc_adv_horz(:, :, :) + ocean_state % p_diag % veloc_adv_vert(:, :, :)
      WRITE(0, *) 'ADV before:', MAXVAL(z_e(:, 1, :)), MINVAL(z_e(:, 1, :))
      ocean_state % p_diag % veloc_adv_horz = invert_mass_matrix(patch_3d, ocean_state, op_coeffs, z_e)
      CALL sync_patch_array_3d_dp_deconiface_212(2, patch_2d, z_e, lacc = .FALSE.)
      WRITE(0, *) 'ADV after:', MAXVAL(ocean_state % p_diag % veloc_adv_horz(:, 1, :)), MINVAL(ocean_state % p_diag % veloc_adv_horz(:, 1, :))
    END IF
    DO blockno = edges_in_domain % start_block, edges_in_domain % end_block
      CALL get_index_range(edges_in_domain, blockno, start_edge_index, end_edge_index)
      z_gradh_e(:) = 0.0D0
      CALL grad_fd_norm_oce_2d_onblock(ocean_state % p_prog(nold(1)) % h, patch_2d, op_coeffs % grad_coeff(:, 1, blockno), z_gradh_e(:), start_edge_index, end_edge_index, blockno)
      z_gradh_e(start_edge_index : end_edge_index) = (1.0D0 - ab_beta) * 9.80665D0 * z_gradh_e(start_edge_index : end_edge_index)
      CALL calculate_explicit_term_g_n_onblock(patch_3d, ocean_state, is_first_timestep, start_edge_index, end_edge_index, blockno)
      IF (iswm_oce /= 1) THEN
        CALL calculate_explicit_vn_pred_3d_onblock(patch_3d, ocean_state, z_gradh_e(:), start_edge_index, end_edge_index, blockno)
        CALL velocity_diffusion_vertical_implicit_onblock(patch_3d, ocean_state % p_diag % vn_pred(:, :, blockno), p_phys_param % a_veloc_v(:, :, blockno), op_coeffs, start_edge_index, end_edge_index, blockno)
      ELSE
        CALL calculate_explicit_vn_pred_2d_onblock(patch_3d, ocean_state, z_gradh_e(:), start_edge_index, end_edge_index, blockno)
      END IF
    END DO
    IF (mass_matrix_inversion_type == 2) THEN
      WRITE(0, *) 'vn_pred before:', MAXVAL(ocean_state % p_diag % vn_pred(:, 1, :)), MINVAL(ocean_state % p_diag % vn_pred(:, 1, :))
      CALL map_edges2edges_viacell_2d_per_level(patch_3d, ocean_state % p_diag % vn_pred(:, 1, :), op_coeffs, ocean_state % p_diag % vn_pred_ptp(:, 1, :), 1)
      WRITE(0, *) 'vn_pred after:', MAXVAL(ocean_state % p_diag % vn_pred_ptp(:, 1, :)), MINVAL(ocean_state % p_diag % vn_pred_ptp(:, 1, :))
      ocean_state % p_diag % vn_pred = ocean_state % p_diag % vn_pred_ptp
    END IF
  END SUBROUTINE explicit_vn_pred_invert_mass_matrix
  SUBROUTINE calculate_explicit_vn_pred_3d_onblock(patch_3d, ocean_state, z_gradh_e, start_edge_index, end_edge_index, blockno, lacc)
    USE mo_model_domain, ONLY: t_patch_3d
    USE mo_ocean_types, ONLY: t_hydro_ocean_state
    USE mo_parallel_config, ONLY: nproma
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_ocean_nml, ONLY: l_rigid_lid, use_ssh_in_momentum_eq
    USE mo_dynamics_config, ONLY: nold
    USE mo_run_config, ONLY: dtime
    USE mo_ocean_boundcond, ONLY: velocitybottomboundarycondition_onblock
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    TYPE(t_hydro_ocean_state), TARGET :: ocean_state
    REAL(KIND = 8) :: z_gradh_e(nproma)
    INTEGER, INTENT(IN) :: start_edge_index, end_edge_index, blockno
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: je, jk, bottom_level
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    IF (.NOT. l_rigid_lid) THEN
      DO je = start_edge_index, end_edge_index
        DO jk = 1, patch_3d % p_patch_1d(1) % dolic_e(je, blockno)
          ocean_state % p_diag % vn_pred(je, jk, blockno) = ocean_state % p_prog(nold(1)) % vn(je, jk, blockno) + dtime * (ocean_state % p_aux % g_nimd(je, jk, blockno) - z_gradh_e(je))
        END DO
      END DO
    ELSE
      DO je = start_edge_index, end_edge_index
        DO jk = 1, patch_3d % p_patch_1d(1) % dolic_e(je, blockno)
          ocean_state % p_diag % vn_pred(je, jk, blockno) = ocean_state % p_prog(nold(1)) % vn(je, jk, blockno) + dtime * ocean_state % p_aux % g_nimd(je, jk, blockno)
        END DO
      END DO
    END IF
    CALL velocitybottomboundarycondition_onblock(patch_3d, blockno, start_edge_index, end_edge_index, ocean_state % p_prog(nold(1)) % vn(:, :, blockno), ocean_state % p_diag % vn_pred(:, :, blockno), ocean_state % p_aux % bc_bot_vn(:, blockno), lacc = lzacc)
    DO je = start_edge_index, end_edge_index
      IF (patch_3d % p_patch_1d(1) % dolic_e(je, blockno) >= 2) THEN
        IF (use_ssh_in_momentum_eq) THEN
          ocean_state % p_diag % vn_pred(je, 1, blockno) = ocean_state % p_diag % vn_pred(je, 1, blockno) + dtime * ocean_state % p_aux % bc_top_vn(je, blockno) / patch_3d % p_patch_1d(1) % prism_thick_e(je, 1, blockno)
        ELSE
          ocean_state % p_diag % vn_pred(je, 1, blockno) = ocean_state % p_diag % vn_pred(je, 1, blockno) + dtime * ocean_state % p_aux % bc_top_vn(je, blockno) / patch_3d % p_patch_1d(1) % prism_thick_flat_sfc_e(je, 1, blockno)
        END IF
        bottom_level = patch_3d % p_patch_1d(1) % dolic_e(je, blockno)
        ocean_state % p_diag % vn_pred(je, bottom_level, blockno) = ocean_state % p_diag % vn_pred(je, bottom_level, blockno) - dtime * ocean_state % p_aux % bc_bot_vn(je, blockno) / patch_3d % p_patch_1d(1) % prism_thick_flat_sfc_e(je, bottom_level, blockno)
      END IF
    END DO
  END SUBROUTINE calculate_explicit_vn_pred_3d_onblock
  SUBROUTINE calculate_explicit_vn_pred_2d_onblock(patch_3d, ocean_state, z_gradh_e, start_edge_index, end_edge_index, blockno, lacc)
    USE mo_model_domain, ONLY: t_patch_3d
    USE mo_ocean_types, ONLY: t_hydro_ocean_state
    USE mo_parallel_config, ONLY: nproma
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_dynamics_config, ONLY: nold
    USE mo_run_config, ONLY: dtime
    USE mo_ocean_nml, ONLY: iforc_oce
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    TYPE(t_hydro_ocean_state), TARGET :: ocean_state
    REAL(KIND = 8), INTENT(IN) :: z_gradh_e(nproma)
    INTEGER, INTENT(IN) :: start_edge_index, end_edge_index, blockno
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: je, jk
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    DO je = start_edge_index, end_edge_index
      DO jk = 1, patch_3d % p_patch_1d(1) % dolic_e(je, blockno)
        ocean_state % p_diag % vn_pred(je, jk, blockno) = (ocean_state % p_prog(nold(1)) % vn(je, jk, blockno) + dtime * (ocean_state % p_aux % g_nimd(je, jk, blockno) - z_gradh_e(je)))
      END DO
    END DO
    IF (iforc_oce /= 10) THEN
      DO je = start_edge_index, end_edge_index
        DO jk = 1, patch_3d % p_patch_1d(1) % dolic_e(je, blockno)
          ocean_state % p_diag % vn_pred(je, jk, blockno) = (ocean_state % p_diag % vn_pred(je, jk, blockno) + ocean_state % p_aux % bc_top_vn(je, blockno) - ocean_state % p_aux % bc_bot_vn(je, blockno))
        END DO
      END DO
    END IF
  END SUBROUTINE calculate_explicit_vn_pred_2d_onblock
  SUBROUTINE calculate_explicit_term_g_n_onblock(patch_3d, ocean_state, is_first_timestep, start_edge_index, end_edge_index, blockno, lacc)
    USE mo_model_domain, ONLY: t_patch_3d
    USE mo_ocean_types, ONLY: t_hydro_ocean_state
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_ocean_nml, ONLY: ab_const, mass_matrix_inversion_type, n_zlev
    USE mo_parallel_config, ONLY: nproma
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    TYPE(t_hydro_ocean_state), TARGET :: ocean_state
    LOGICAL, INTENT(IN) :: is_first_timestep
    INTEGER, INTENT(IN) :: start_edge_index, end_edge_index, blockno
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: je, jk
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    IF (mass_matrix_inversion_type /= 1) THEN
      DO je = start_edge_index, end_edge_index
        DO jk = 1, patch_3d % p_patch_1d(1) % dolic_e(je, blockno)
          ocean_state % p_aux % g_n(je, jk, blockno) = - ocean_state % p_diag % press_grad(je, jk, blockno) - ocean_state % p_diag % grad(je, jk, blockno) - ocean_state % p_diag % veloc_adv_horz(je, jk, blockno) - ocean_state % p_diag % veloc_adv_vert(je, jk, blockno) + ocean_state % p_diag % laplacian_horz(je, jk, blockno)
        END DO
      END DO
    ELSE IF (mass_matrix_inversion_type == 1) THEN
      DO je = start_edge_index, end_edge_index
        DO jk = 1, patch_3d % p_patch_1d(1) % dolic_e(je, blockno)
          ocean_state % p_aux % g_n(je, jk, blockno) = - ocean_state % p_diag % press_grad(je, jk, blockno) - ocean_state % p_diag % grad(je, jk, blockno) - ocean_state % p_diag % veloc_adv_horz(je, jk, blockno) + ocean_state % p_diag % laplacian_horz(je, jk, blockno)
        END DO
      END DO
    END IF
    IF (is_first_timestep) THEN
      DO jk = 1, n_zlev
        DO je = 1, nproma
          ocean_state % p_aux % g_nimd(je, jk, blockno) = ocean_state % p_aux % g_n(je, jk, blockno)
        END DO
      END DO
    ELSE
      DO je = start_edge_index, end_edge_index
        DO jk = 1, patch_3d % p_patch_1d(1) % dolic_e(je, blockno)
          ocean_state % p_aux % g_nimd(je, jk, blockno) = (1.5D0 + ab_const) * ocean_state % p_aux % g_n(je, jk, blockno) - (0.5D0 + ab_const) * ocean_state % p_aux % g_nm1(je, jk, blockno)
        END DO
      END DO
    END IF
  END SUBROUTINE calculate_explicit_term_g_n_onblock
  SUBROUTINE fill_rhs4surface_eq_ab(patch_3d, ocean_state, op_coeffs, lacc)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d, t_subset_range
    USE mo_ocean_types, ONLY: t_hydro_ocean_state, t_operator_coeff
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_nml, ONLY: ab_gam, iswm_oce, l_edge_based, n_zlev
    USE mo_run_config, ONLY: dtime
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_grid_subset, ONLY: get_index_range
    USE mo_dynamics_config, ONLY: nold
    USE mo_sync, ONLY: sync_patch_array_3d_dp_deconiface_213 => sync_patch_array_3d_dp
    USE mo_scalar_product, ONLY: map_edges2edges_viacell_2d_constz_deconiface_215 => map_edges2edges_viacell_2d_constz, map_edges2edges_viacell_3d_mlev_const_z_deconiface_214 => map_edges2edges_viacell_3d_mlev_const_z
    USE mo_ocean_math_operators, ONLY: div_oce_3d_general_onblock, div_oce_3d_ontriangles_onblock
    USE mo_util_dbg_prnt, ONLY: dbg_print_2d_deconiface_220 => dbg_print_2d, dbg_print_3d_deconiface_216 => dbg_print_3d, dbg_print_3d_deconiface_217 => dbg_print_3d, dbg_print_3d_deconiface_218 => dbg_print_3d, dbg_print_3d_deconiface_219 => dbg_print_3d
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    TYPE(t_hydro_ocean_state), TARGET :: ocean_state
    TYPE(t_operator_coeff), INTENT(IN) :: op_coeffs
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: start_cell_index, end_cell_index
    INTEGER :: start_edge_index, end_edge_index
    INTEGER :: jc, blockno, jk, je, cells_start_block, cells_end_block
    REAL(KIND = 8) :: inv_gdt2
    REAL(KIND = 8) :: z_e(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_e)
    REAL(KIND = 8) :: div_z_depth_int_c(nproma)
    REAL(KIND = 8) :: div_z_c(nproma, n_zlev)
    REAL(KIND = 8) :: z_vn_ab(nproma, n_zlev, patch_3d % p_patch_2d(1) % nblks_e)
    TYPE(t_subset_range), POINTER :: cells_in_domain, edges_in_domain
    TYPE(t_patch), POINTER :: patch_2d
    REAL(KIND = 8), DIMENSION(:, :, :, :), POINTER :: div_coeff
    LOGICAL :: lzacc
    div_coeff => op_coeffs % div_coeff
    patch_2d => patch_3d % p_patch_2d(1)
    cells_in_domain => patch_3d % p_patch_2d(1) % cells % in_domain
    edges_in_domain => patch_3d % p_patch_2d(1) % edges % in_domain
    cells_start_block = cells_in_domain % start_block
    cells_end_block = cells_in_domain % end_block
    inv_gdt2 = 1.0D0 / (9.80665D0 * dtime * dtime)
    CALL set_acc_host_or_device(lzacc, lacc)
    z_vn_ab(:, :, : edges_in_domain % start_block - 1) = 0.0D0
    IF (iswm_oce == 1) THEN
      DO blockno = edges_in_domain % start_block, edges_in_domain % end_block
        CALL get_index_range(edges_in_domain, blockno, start_edge_index, end_edge_index)
        z_vn_ab(:, :, blockno) = 0.0D0
        DO je = start_edge_index, end_edge_index
          DO jk = 1, patch_3d % p_patch_1d(1) % dolic_e(je, blockno)
            z_vn_ab(je, jk, blockno) = ab_gam * ocean_state % p_diag % vn_pred(je, jk, blockno) + (1.0D0 - ab_gam) * ocean_state % p_prog(nold(1)) % vn(je, jk, blockno)
          END DO
        END DO
      END DO
    ELSE
      DO blockno = edges_in_domain % start_block, edges_in_domain % end_block
        CALL get_index_range(edges_in_domain, blockno, start_edge_index, end_edge_index)
        z_vn_ab(:, :, blockno) = 0.0D0
        DO je = start_edge_index, end_edge_index
          DO jk = 1, patch_3d % p_patch_1d(1) % dolic_e(je, blockno)
            z_vn_ab(je, jk, blockno) = ab_gam * ocean_state % p_diag % vn_pred(je, jk, blockno) + (1.0D0 - ab_gam) * ocean_state % p_prog(nold(1)) % vn(je, jk, blockno)
          END DO
        END DO
      END DO
    END IF
    z_vn_ab(:, :, edges_in_domain % end_block + 1 :) = 0.0D0
    IF (l_edge_based) THEN
      z_e(1 : nproma, 1 : n_zlev, 1 : patch_2d % nblks_e) = 0.0D0
      IF (iswm_oce /= 1) THEN
        DO blockno = edges_in_domain % start_block, edges_in_domain % end_block
          CALL get_index_range(edges_in_domain, blockno, start_edge_index, end_edge_index)
          DO je = start_edge_index, end_edge_index
            DO jk = 1, patch_3d % p_patch_1d(1) % dolic_e(je, blockno)
              z_e(je, jk, blockno) = z_vn_ab(je, jk, blockno) * patch_3d % p_patch_1d(1) % prism_thick_e(je, jk, blockno)
            END DO
          END DO
        END DO
      ELSE IF (iswm_oce == 1) THEN
        DO blockno = edges_in_domain % start_block, edges_in_domain % end_block
          CALL get_index_range(edges_in_domain, blockno, start_edge_index, end_edge_index)
          DO je = start_edge_index, end_edge_index
            DO jk = 1, patch_3d % p_patch_1d(1) % dolic_e(je, blockno)
              z_e(je, jk, blockno) = z_vn_ab(je, jk, blockno) * ocean_state % p_diag % thick_e(je, blockno)
            END DO
          END DO
        END DO
      END IF
    ELSE
      CALL sync_patch_array_3d_dp_deconiface_213(2, patch_2d, z_vn_ab, lacc = lzacc)
      IF (iswm_oce /= 1) THEN
        CALL map_edges2edges_viacell_3d_mlev_const_z_deconiface_214(patch_3d, z_vn_ab, op_coeffs, z_e, lacc = lzacc)
      ELSE
        CALL map_edges2edges_viacell_2d_constz_deconiface_215(patch_3d, z_vn_ab(:, 1, :), op_coeffs, z_e(:, 1, :), lacc = lzacc)
      END IF
    END IF
    IF (patch_2d % cells % max_connectivity == 3) THEN
      DO blockno = cells_start_block, cells_end_block
        CALL get_index_range(cells_in_domain, blockno, start_cell_index, end_cell_index)
        CALL div_oce_3d_ontriangles_onblock(z_e, patch_3d, div_coeff, div_z_c, blockno = blockno, start_index = start_cell_index, end_index = end_cell_index, start_level = 1, end_level = n_zlev, lacc = lzacc)
        div_z_depth_int_c(:) = 0.0D0
        DO jc = start_cell_index, end_cell_index
          div_z_depth_int_c(jc) = SUM(div_z_c(jc, 1 : patch_3d % p_patch_1d(1) % dolic_c(jc, blockno)))
        END DO
        ocean_state % p_aux % p_rhs_sfc_eq(:, blockno) = 0.0D0
        DO jc = start_cell_index, end_cell_index
          IF (patch_3d % p_patch_1d(1) % dolic_c(jc, blockno) > 0) THEN
            ocean_state % p_aux % p_rhs_sfc_eq(jc, blockno) = ((ocean_state % p_prog(nold(1)) % h(jc, blockno) - dtime * div_z_depth_int_c(jc)) * inv_gdt2)
          END IF
        END DO
      END DO
    ELSE
      DO blockno = cells_in_domain % start_block, cells_in_domain % end_block
        CALL get_index_range(cells_in_domain, blockno, start_cell_index, end_cell_index)
        CALL div_oce_3d_general_onblock(z_e, patch_3d, op_coeffs % div_coeff, div_z_c, blockno = blockno, start_index = start_cell_index, end_index = end_cell_index, start_level = 1, end_level = n_zlev, lacc = lzacc)
        div_z_depth_int_c(:) = 0.0D0
        DO jc = start_cell_index, end_cell_index
          div_z_depth_int_c(jc) = SUM(div_z_c(jc, 1 : patch_3d % p_patch_1d(1) % dolic_c(jc, blockno)))
        END DO
        ocean_state % p_aux % p_rhs_sfc_eq(:, blockno) = 0.0D0
        DO jc = start_cell_index, end_cell_index
          IF (patch_3d % p_patch_1d(1) % dolic_c(jc, blockno) > 0) THEN
            ocean_state % p_aux % p_rhs_sfc_eq(jc, blockno) = ((ocean_state % p_prog(nold(1)) % h(jc, blockno) - dtime * div_z_depth_int_c(jc)) * inv_gdt2)
          END IF
        END DO
      END DO
    END IF
    idt_src = 3
    CALL dbg_print_3d_deconiface_216('RHS thick_e', patch_3d % p_patch_1d(1) % prism_thick_e, str_module, idt_src, in_subset = patch_3d % p_patch_2d(1) % edges % owned)
    CALL dbg_print_3d_deconiface_217('RHS thick_c', patch_3d % p_patch_1d(1) % prism_thick_c, str_module, idt_src, in_subset = patch_3d % p_patch_2d(1) % cells % owned)
    CALL dbg_print_3d_deconiface_218('RHS z_vn_ab', z_vn_ab, str_module, idt_src, in_subset = patch_3d % p_patch_2d(1) % edges % owned)
    CALL dbg_print_3d_deconiface_219('RHS z_e', z_e, str_module, idt_src, in_subset = patch_3d % p_patch_2d(1) % edges % owned)
    idt_src = 2
    CALL dbg_print_2d_deconiface_220('RHS final', ocean_state % p_aux % p_rhs_sfc_eq, str_module, idt_src, in_subset = patch_3d % p_patch_2d(1) % cells % owned)
  END SUBROUTINE fill_rhs4surface_eq_ab
  FUNCTION invert_mass_matrix(patch_3d, ocean_state, op_coeffs, rhs_e) RESULT(inv_flip_flop_e)
    USE mo_model_domain, ONLY: t_patch, t_patch_3d
    USE mo_ocean_types, ONLY: t_hydro_ocean_state, t_operator_coeff
    USE mo_impl_constants, ONLY: max_char_length
    USE mo_ocean_solve_aux, ONLY: ocean_solve_parm_init_deconproc_107 => ocean_solve_parm_init, t_ocean_solve_parm
    USE mo_ocean_solve_lhs_type, ONLY: lhs_primal_flip_flop_construct_deconproc_105 => lhs_primal_flip_flop_construct, lhs_primal_flip_flop_construct_deconproc_108 => lhs_primal_flip_flop_construct
    USE mo_ocean_solve_transfer, ONLY: trivial_transfer_construct_deconproc_106 => trivial_transfer_construct
    USE mo_ocean_nml, ONLY: massmatrix_solver_tolerance, n_zlev, solver_max_iter_per_restart, solver_max_restart_iterations
    USE mo_ocean_solve, ONLY: ocean_solve_construct__t_primal_flip_flop_lhs__t_triv__1f49ae23, ocean_solve_solve_deconproc_109 => ocean_solve_solve
    USE mo_dbg_nml, ONLY: idbg_mxmn
    USE mo_exception, ONLY: message
    TYPE(t_patch_3d), TARGET, INTENT(IN) :: patch_3d
    TYPE(t_hydro_ocean_state), TARGET :: ocean_state
    TYPE(t_operator_coeff), INTENT(IN), TARGET :: op_coeffs
    REAL(KIND = 8), INTENT(INOUT), TARGET :: rhs_e(:, :, :)
    REAL(KIND = 8) :: inv_flip_flop_e(SIZE(rhs_e, 1), SIZE(rhs_e, 2), SIZE(rhs_e, 3))
    INTEGER :: jk, n_it, n_it_sp
    REAL(KIND = 8) :: rn
    CHARACTER(LEN = max_char_length) :: string
    TYPE(t_patch), POINTER :: patch_2d
    TYPE(t_ocean_solve_parm) :: par, par_sp
    IF (.NOT. inv_mm_solver % is_init) THEN
      CALL lhs_primal_flip_flop_construct_deconproc_105(inv_mm_solver_lhs, patch_3d, op_coeffs, (- 999))
      patch_2d => patch_3d % p_patch_2d(1)
      CALL trivial_transfer_construct_deconproc_106(inv_mm_solver_trans, 12, patch_2d)
      CALL ocean_solve_parm_init_deconproc_107(par, 60, solver_max_restart_iterations, solver_max_iter_per_restart, patch_2d % cells % in_domain % end_block, SIZE(rhs_e, 3), SIZE(rhs_e, 1), patch_2d % edges % in_domain % end_index, massmatrix_solver_tolerance, .TRUE.)
      par_sp % nidx = (- 1)
      CALL ocean_solve_construct__t_primal_flip_flop_lhs__t_triv__1f49ae23(inv_mm_solver, 21, par, par_sp, inv_mm_solver_lhs, inv_mm_solver_trans)
    END IF
    DO jk = 1, n_zlev
      CALL lhs_primal_flip_flop_construct_deconproc_108(inv_mm_solver_lhs, patch_3d, op_coeffs, jk)
      inv_mm_solver % x_loc_wp(:, :) = 0.0D0
      inv_mm_solver % b_loc_wp => rhs_e(:, jk, :)
      CALL ocean_solve_solve_deconproc_109(inv_mm_solver, n_it, n_it_sp)
      rn = MERGE((inv_mm_solver % res_loc_wp(1)), 0.0D0, n_it .GT. 0)
      inv_flip_flop_e(:, jk, :) = inv_mm_solver % x_loc_wp(:, :)
      IF (idbg_mxmn >= 1) THEN
        WRITE(string, '(a,i4,a,e28.20)') 'ocean_restart_gmres iteration =', n_it - 1, ', residual =', rn
        CALL message('invert_mass_matrix', TRIM(string))
      END IF
    END DO
  END FUNCTION invert_mass_matrix
END MODULE mo_ocean_ab_timestepping_mimetic