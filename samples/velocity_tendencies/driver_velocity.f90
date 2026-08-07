! Standalone timing driver for the ICON velocity_tendencies Fortran baseline lanes.
!
! Reads the raw dump written by dump_data.py (manifest.txt + scalars.txt + one Fortran-order
! <name>.bin per array), rebuilds the derived-type arguments with their recorded lower bounds
! (the refinement-control arrays start at -16), calls velocity_tendencies from
! velocity_advection_acc.f90 for WARMUP untimed + REPS timed calls, and prints the sample's
! 9-column CSV rows on stdout.  Everything else goes to stderr so the lane script can grep.
!
! Usage: driver_velocity <dump_dir> <lane> <threads> <reps> <warmup> [verify|dumpref [ref_dir]]
!
! dumpref writes the kernel's write set to ref_dir (default <dump_dir>/ref); verify reads it back
! and fails when any field's max|driver - ref| exceeds 1e-10 * max(1, max|ref|).  Build one serial
! reference with dumpref, then verify every parallel lane against it -- the bound is not zero
! because the lanes build with -march=native, so FMA contraction and autopar reassociation move
! the last couple of digits.  dump_data.py --reference writes the same layout from the DaCe
! kernel, so the same verify works against that oracle.
MODULE velocity_dump_io
  USE iso_fortran_env, ONLY: error_unit
  IMPLICIT NONE
  INTEGER, PARAMETER :: wp = 8
  INTEGER, PARAMETER :: maxrank = 4
CONTAINS

  SUBROUTINE die(msg)
    CHARACTER(LEN=*), INTENT(IN) :: msg
    WRITE (error_unit, '(A)') 'FATAL: '//msg
    STOP 1
  END SUBROUTINE die

  !> dims + lower bounds of one array from manifest.txt; hard-fails on a missing name or wrong rank.
  SUBROUTINE meta_of(dir, name, rank, dims, lbs)
    CHARACTER(LEN=*), INTENT(IN) :: dir, name
    INTEGER, INTENT(IN) :: rank
    INTEGER, INTENT(OUT) :: dims(maxrank), lbs(maxrank)
    CHARACTER(LEN=512) :: line
    CHARACTER(LEN=64) :: nm, dt
    INTEGER :: u, ios, rk
    dims = 1
    lbs = 1
    OPEN (NEWUNIT=u, FILE=TRIM(dir)//'/manifest.txt', STATUS='OLD', ACTION='READ', IOSTAT=ios)
    IF (ios /= 0) CALL die('cannot open '//TRIM(dir)//'/manifest.txt')
    DO
      READ (u, '(A)', IOSTAT=ios) line
      IF (ios /= 0) EXIT
      READ (line, *, IOSTAT=ios) nm, dt, rk
      IF (ios /= 0) CYCLE
      IF (TRIM(nm) /= name) CYCLE
      IF (rk /= rank) CALL die(name//': manifest rank does not match the driver declaration')
      READ (line, *, IOSTAT=ios) nm, dt, rk, dims(1:rk), lbs(1:rk)
      IF (ios /= 0) CALL die(name//': malformed manifest record')
      CLOSE (u)
      RETURN
    END DO
    CLOSE (u)
    CALL die(name//': not in the manifest')
  END SUBROUTINE meta_of

  !> One named scalar from scalars.txt, always as a double (integer callers round).
  REAL(wp) FUNCTION scal(dir, name)
    CHARACTER(LEN=*), INTENT(IN) :: dir, name
    CHARACTER(LEN=256) :: line
    CHARACTER(LEN=64) :: nm, dt
    INTEGER :: u, ios
    OPEN (NEWUNIT=u, FILE=TRIM(dir)//'/scalars.txt', STATUS='OLD', ACTION='READ', IOSTAT=ios)
    IF (ios /= 0) CALL die('cannot open '//TRIM(dir)//'/scalars.txt')
    DO
      READ (u, '(A)', IOSTAT=ios) line
      IF (ios /= 0) EXIT
      READ (line, *, IOSTAT=ios) nm, dt, scal
      IF (ios /= 0) CYCLE
      IF (TRIM(nm) /= name) CYCLE
      CLOSE (u)
      RETURN
    END DO
    CLOSE (u)
    CALL die(name//': not in scalars.txt')
  END FUNCTION scal

  SUBROUTINE open_bin(dir, name, u)
    CHARACTER(LEN=*), INTENT(IN) :: dir, name
    INTEGER, INTENT(OUT) :: u
    INTEGER :: ios
    OPEN (NEWUNIT=u, FILE=TRIM(dir)//'/'//name//'.bin', FORM='UNFORMATTED', ACCESS='STREAM', &
          STATUS='OLD', ACTION='READ', IOSTAT=ios)
    IF (ios /= 0) CALL die(TRIM(dir)//'/'//name//'.bin: cannot open')
  END SUBROUTINE open_bin

  ! The three slurp_* take an explicit-shape rank-1 dummy: sequence association lets one routine
  ! fill an array of any rank, so the loaders below differ only in ALLOCATE.
  SUBROUTINE slurp_r(dir, name, n, a)
    CHARACTER(LEN=*), INTENT(IN) :: dir, name
    INTEGER, INTENT(IN) :: n
    REAL(wp), INTENT(OUT) :: a(n)
    INTEGER :: u, ios
    CALL open_bin(dir, name, u)
    READ (u, IOSTAT=ios) a
    IF (ios /= 0) CALL die(name//'.bin: short read')
    CLOSE (u)
  END SUBROUTINE slurp_r

  SUBROUTINE slurp_i(dir, name, n, a)
    CHARACTER(LEN=*), INTENT(IN) :: dir, name
    INTEGER, INTENT(IN) :: n
    INTEGER, INTENT(OUT) :: a(n)
    INTEGER :: u, ios
    CALL open_bin(dir, name, u)
    READ (u, IOSTAT=ios) a
    IF (ios /= 0) CALL die(name//'.bin: short read')
    CLOSE (u)
  END SUBROUTINE slurp_i

  SUBROUTINE slurp_l(dir, name, n, a)
    CHARACTER(LEN=*), INTENT(IN) :: dir, name
    INTEGER, INTENT(IN) :: n
    LOGICAL, INTENT(OUT) :: a(n)
    INTEGER(KIND=1), ALLOCATABLE :: raw(:)
    INTEGER :: u, ios
    ALLOCATE (raw(n))
    CALL open_bin(dir, name, u)
    READ (u, IOSTAT=ios) raw
    IF (ios /= 0) CALL die(name//'.bin: short read')
    CLOSE (u)
    a = raw /= 0_1
  END SUBROUTINE slurp_l

  SUBROUTINE load_i1(dir, name, a)
    CHARACTER(LEN=*), INTENT(IN) :: dir, name
    INTEGER, ALLOCATABLE, INTENT(OUT) :: a(:)
    INTEGER :: d(maxrank), l(maxrank)
    CALL meta_of(dir, name, 1, d, l)
    ALLOCATE (a(l(1):l(1) + d(1) - 1))
    CALL slurp_i(dir, name, SIZE(a), a)
  END SUBROUTINE load_i1

  SUBROUTINE load_i3(dir, name, a)
    CHARACTER(LEN=*), INTENT(IN) :: dir, name
    INTEGER, ALLOCATABLE, INTENT(OUT) :: a(:, :, :)
    INTEGER :: d(maxrank), l(maxrank)
    CALL meta_of(dir, name, 3, d, l)
    ALLOCATE (a(l(1):l(1) + d(1) - 1, l(2):l(2) + d(2) - 1, l(3):l(3) + d(3) - 1))
    CALL slurp_i(dir, name, SIZE(a), a)
  END SUBROUTINE load_i3

  SUBROUTINE load_l2(dir, name, a)
    CHARACTER(LEN=*), INTENT(IN) :: dir, name
    LOGICAL, ALLOCATABLE, INTENT(OUT) :: a(:, :)
    INTEGER :: d(maxrank), l(maxrank)
    CALL meta_of(dir, name, 2, d, l)
    ALLOCATE (a(l(1):l(1) + d(1) - 1, l(2):l(2) + d(2) - 1))
    CALL slurp_l(dir, name, SIZE(a), a)
  END SUBROUTINE load_l2

  SUBROUTINE load_r2(dir, name, a)
    CHARACTER(LEN=*), INTENT(IN) :: dir, name
    REAL(wp), ALLOCATABLE, INTENT(OUT) :: a(:, :)
    INTEGER :: d(maxrank), l(maxrank)
    CALL meta_of(dir, name, 2, d, l)
    ALLOCATE (a(l(1):l(1) + d(1) - 1, l(2):l(2) + d(2) - 1))
    CALL slurp_r(dir, name, SIZE(a), a)
  END SUBROUTINE load_r2

  SUBROUTINE load_r3(dir, name, a)
    CHARACTER(LEN=*), INTENT(IN) :: dir, name
    REAL(wp), ALLOCATABLE, INTENT(OUT) :: a(:, :, :)
    INTEGER :: d(maxrank), l(maxrank)
    CALL meta_of(dir, name, 3, d, l)
    ALLOCATE (a(l(1):l(1) + d(1) - 1, l(2):l(2) + d(2) - 1, l(3):l(3) + d(3) - 1))
    CALL slurp_r(dir, name, SIZE(a), a)
  END SUBROUTINE load_r3

  ! The kernel's derived types declare these components POINTER, so ALLOCATABLE dummies cannot
  ! bind to them -- hence the loadp_* twins.
  SUBROUTINE loadp_r1(dir, name, a)
    CHARACTER(LEN=*), INTENT(IN) :: dir, name
    REAL(wp), POINTER, INTENT(OUT) :: a(:)
    INTEGER :: d(maxrank), l(maxrank)
    CALL meta_of(dir, name, 1, d, l)
    ALLOCATE (a(l(1):l(1) + d(1) - 1))
    CALL slurp_r(dir, name, SIZE(a), a)
  END SUBROUTINE loadp_r1

  SUBROUTINE loadp_r2(dir, name, a)
    CHARACTER(LEN=*), INTENT(IN) :: dir, name
    REAL(wp), POINTER, INTENT(OUT) :: a(:, :)
    INTEGER :: d(maxrank), l(maxrank)
    CALL meta_of(dir, name, 2, d, l)
    ALLOCATE (a(l(1):l(1) + d(1) - 1, l(2):l(2) + d(2) - 1))
    CALL slurp_r(dir, name, SIZE(a), a)
  END SUBROUTINE loadp_r2

  SUBROUTINE loadp_r3(dir, name, a)
    CHARACTER(LEN=*), INTENT(IN) :: dir, name
    REAL(wp), POINTER, INTENT(OUT) :: a(:, :, :)
    INTEGER :: d(maxrank), l(maxrank)
    CALL meta_of(dir, name, 3, d, l)
    ALLOCATE (a(l(1):l(1) + d(1) - 1, l(2):l(2) + d(2) - 1, l(3):l(3) + d(3) - 1))
    CALL slurp_r(dir, name, SIZE(a), a)
  END SUBROUTINE loadp_r3

  SUBROUTINE loadp_r4(dir, name, a)
    CHARACTER(LEN=*), INTENT(IN) :: dir, name
    REAL(wp), POINTER, INTENT(OUT) :: a(:, :, :, :)
    INTEGER :: d(maxrank), l(maxrank)
    CALL meta_of(dir, name, 4, d, l)
    ALLOCATE (a(l(1):l(1) + d(1) - 1, l(2):l(2) + d(2) - 1, l(3):l(3) + d(3) - 1, l(4):l(4) + d(4) - 1))
    CALL slurp_r(dir, name, SIZE(a), a)
  END SUBROUTINE loadp_r4

  SUBROUTINE snapshot(a, n, buf)
    INTEGER, INTENT(IN) :: n
    REAL(wp), INTENT(IN) :: a(n)
    REAL(wp), ALLOCATABLE, INTENT(OUT) :: buf(:)
    ALLOCATE (buf(n))
    buf = a
  END SUBROUTINE snapshot

  SUBROUTINE restore(a, n, buf)
    INTEGER, INTENT(IN) :: n
    REAL(wp), INTENT(OUT) :: a(n)
    REAL(wp), INTENT(IN) :: buf(n)
    a = buf
  END SUBROUTINE restore

  !> One output as a raw stream, for a reference build to leave behind (mode dumpref).
  SUBROUTINE dump_out(dir, name, a, n)
    CHARACTER(LEN=*), INTENT(IN) :: dir, name
    INTEGER, INTENT(IN) :: n
    REAL(wp), INTENT(IN) :: a(n)
    INTEGER :: u, ios
    OPEN (NEWUNIT=u, FILE=TRIM(dir)//'/'//name//'.bin', FORM='UNFORMATTED', ACCESS='STREAM', &
          STATUS='REPLACE', ACTION='WRITE', IOSTAT=ios)
    IF (ios /= 0) CALL die(TRIM(dir)//'/'//name//'.bin: cannot write (does the directory exist?)')
    WRITE (u) a
    CLOSE (u)
  END SUBROUTINE dump_out

  !> max|a - ref| for one output, normalised by max|ref|; accumulates the worst relative gap.
  SUBROUTINE verify(dir, name, a, n, worst)
    CHARACTER(LEN=*), INTENT(IN) :: dir, name
    INTEGER, INTENT(IN) :: n
    REAL(wp), INTENT(IN) :: a(n)
    REAL(wp), INTENT(INOUT) :: worst
    REAL(wp), ALLOCATABLE :: ref(:)
    REAL(wp) :: dmax, refmax, rel
    INTEGER :: i
    ALLOCATE (ref(n))
    CALL slurp_r(dir, name, n, ref)
    dmax = 0.0_wp
    refmax = 0.0_wp
    DO i = 1, n
      dmax = MAX(dmax, ABS(a(i) - ref(i)))
      refmax = MAX(refmax, ABS(ref(i)))
    END DO
    rel = dmax/MAX(1.0_wp, refmax)
    WRITE (error_unit, '(A,A,3(A,ES12.5))') 'verify ', name, ' max|d|=', dmax, ' max|ref|=', refmax, ' rel=', rel
    worst = MAX(worst, rel)
  END SUBROUTINE verify

END MODULE velocity_dump_io

PROGRAM driver_velocity
  USE iso_fortran_env, ONLY: error_unit, output_unit
  USE velocity_dump_io
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

  !> Relative bound for mode verify; see the file header for why it is not zero.
  REAL(wp), PARAMETER :: verify_tol = 1.0E-10_wp

  TYPE(t_patch), TARGET :: p_patch
  TYPE(t_int_state), TARGET :: p_int
  TYPE(t_nh_prog) :: p_prog
  TYPE(t_nh_metrics) :: p_metrics
  TYPE(t_nh_diag) :: p_diag
  REAL(wp), ALLOCATABLE :: z_w_concorr_me(:, :, :), z_kin_hor_e(:, :, :), z_vt_ie(:, :, :)
  REAL(wp), ALLOCATABLE :: sv_apc(:), sv_cor(:), sv_wadv(:), sv_vt(:), sv_vnie(:), sv_ubc(:), sv_wcc(:), &
                           sv_kin(:), sv_vtie(:), sv_wcme(:)
  REAL(wp), ALLOCATABLE :: vcfl0(:)

  CHARACTER(LEN=4096) :: dump_dir, ref_dir, lane, arg
  CHARACTER(LEN=32) :: mode
  CHARACTER(LEN=512) :: row
  INTEGER :: threads, reps, warmup, rep, ntnd, istep, nblks_e, nlev, nlevp1
  LOGICAL :: lvn_only, ldeepatmo
  REAL(wp) :: dtime, dt_linintp_ubc, ms, worst
  INTEGER(KIND=8) :: t0, t1, rate

  IF (COMMAND_ARGUMENT_COUNT() < 5) THEN
    WRITE (error_unit, '(A)') 'usage: driver_velocity <dump_dir> <lane> <threads> <reps> <warmup> '// &
      '[verify|dumpref [ref_dir]]'
    STOP 2
  END IF
  CALL GET_COMMAND_ARGUMENT(1, dump_dir)
  CALL GET_COMMAND_ARGUMENT(2, lane)
  CALL GET_COMMAND_ARGUMENT(3, arg)
  READ (arg, *) threads
  CALL GET_COMMAND_ARGUMENT(4, arg)
  READ (arg, *) reps
  CALL GET_COMMAND_ARGUMENT(5, arg)
  READ (arg, *) warmup
  CALL GET_COMMAND_ARGUMENT(6, mode)
  CALL GET_COMMAND_ARGUMENT(7, ref_dir)
  IF (LEN_TRIM(ref_dir) == 0) ref_dir = TRIM(dump_dir)//'/ref'

  ! --- config module globals the kernel reads straight off the modules ----------------------
  nproma = NINT(scal(dump_dir, 'nproma'))
  lextra_diffu = NINT(scal(dump_dir, 'lextra_diffu')) /= 0
  lvert_nest = NINT(scal(dump_dir, 'l_vert_nested')) /= 0
  timers_level = 0
  nflatlev = NINT(scal(dump_dir, 'nflatlev_jg'))
  nrdmax = NINT(scal(dump_dir, 'nrdmax_jg'))

  nlev = NINT(scal(dump_dir, 'nlev'))
  nlevp1 = NINT(scal(dump_dir, 'nlevp1'))
  nblks_e = NINT(scal(dump_dir, 'nblks_e'))
  ntnd = NINT(scal(dump_dir, 'ntnd'))
  istep = NINT(scal(dump_dir, 'istep'))
  lvn_only = NINT(scal(dump_dir, 'lvn_only')) /= 0
  ldeepatmo = NINT(scal(dump_dir, 'ldeepatmo')) /= 0
  dtime = scal(dump_dir, 'dtime')
  dt_linintp_ubc = scal(dump_dir, 'dt_linintp_ubc')

  ! p_patch % id indexes the per-domain nflatlev/nrdmax above; the artifact records one domain.
  p_patch%id = 1
  p_patch%n_childdom = 0
  p_patch%nlev = nlev
  p_patch%nlevp1 = nlevp1
  p_patch%nshift = 0
  p_patch%nshift_total = 0
  p_patch%nshift_child = 0
  p_patch%nblks_c = NINT(scal(dump_dir, 'nblks_c'))
  p_patch%nblks_e = nblks_e
  p_patch%nblks_v = NINT(scal(dump_dir, 'nblks_v'))
  p_patch%geometry_info%mean_cell_area = 0.0_wp

  CALL load_i3(dump_dir, 'p_patch_cells_neighbor_idx', p_patch%cells%neighbor_idx)
  CALL load_i3(dump_dir, 'p_patch_cells_neighbor_blk', p_patch%cells%neighbor_blk)
  CALL load_i3(dump_dir, 'p_patch_cells_edge_idx', p_patch%cells%edge_idx)
  CALL load_i3(dump_dir, 'p_patch_cells_edge_blk', p_patch%cells%edge_blk)
  CALL loadp_r2(dump_dir, 'p_patch_cells_area', p_patch%cells%area)
  CALL load_i1(dump_dir, 'p_patch_cells_start_index', p_patch%cells%start_index)
  CALL load_i1(dump_dir, 'p_patch_cells_end_index', p_patch%cells%end_index)
  CALL load_i1(dump_dir, 'p_patch_cells_start_block', p_patch%cells%start_block)
  CALL load_i1(dump_dir, 'p_patch_cells_end_block', p_patch%cells%end_block)
  CALL load_l2(dump_dir, 'p_patch_cells_decomp_info_owner_mask', p_patch%cells%decomp_info%owner_mask)

  CALL load_i3(dump_dir, 'p_patch_edges_cell_idx', p_patch%edges%cell_idx)
  CALL load_i3(dump_dir, 'p_patch_edges_cell_blk', p_patch%edges%cell_blk)
  CALL load_i3(dump_dir, 'p_patch_edges_vertex_idx', p_patch%edges%vertex_idx)
  CALL load_i3(dump_dir, 'p_patch_edges_vertex_blk', p_patch%edges%vertex_blk)
  CALL load_i3(dump_dir, 'p_patch_edges_quad_idx', p_patch%edges%quad_idx)
  CALL load_i3(dump_dir, 'p_patch_edges_quad_blk', p_patch%edges%quad_blk)
  CALL load_r2(dump_dir, 'p_patch_edges_tangent_orientation', p_patch%edges%tangent_orientation)
  CALL load_r2(dump_dir, 'p_patch_edges_inv_primal_edge_length', p_patch%edges%inv_primal_edge_length)
  CALL load_r2(dump_dir, 'p_patch_edges_inv_dual_edge_length', p_patch%edges%inv_dual_edge_length)
  CALL load_r2(dump_dir, 'p_patch_edges_area_edge', p_patch%edges%area_edge)
  CALL load_r2(dump_dir, 'p_patch_edges_f_e', p_patch%edges%f_e)
  CALL load_r2(dump_dir, 'p_patch_edges_fn_e', p_patch%edges%fn_e)
  CALL load_r2(dump_dir, 'p_patch_edges_ft_e', p_patch%edges%ft_e)
  CALL load_i1(dump_dir, 'p_patch_edges_start_index', p_patch%edges%start_index)
  CALL load_i1(dump_dir, 'p_patch_edges_end_index', p_patch%edges%end_index)
  CALL load_i1(dump_dir, 'p_patch_edges_start_block', p_patch%edges%start_block)
  CALL load_i1(dump_dir, 'p_patch_edges_end_block', p_patch%edges%end_block)

  CALL load_i3(dump_dir, 'p_patch_verts_cell_idx', p_patch%verts%cell_idx)
  CALL load_i3(dump_dir, 'p_patch_verts_cell_blk', p_patch%verts%cell_blk)
  CALL load_i3(dump_dir, 'p_patch_verts_edge_idx', p_patch%verts%edge_idx)
  CALL load_i3(dump_dir, 'p_patch_verts_edge_blk', p_patch%verts%edge_blk)
  CALL load_i1(dump_dir, 'p_patch_verts_start_index', p_patch%verts%start_index)
  CALL load_i1(dump_dir, 'p_patch_verts_end_index', p_patch%verts%end_index)
  CALL load_i1(dump_dir, 'p_patch_verts_start_block', p_patch%verts%start_block)
  CALL load_i1(dump_dir, 'p_patch_verts_end_block', p_patch%verts%end_block)

  CALL load_r3(dump_dir, 'p_int_c_lin_e', p_int%c_lin_e)
  CALL load_r3(dump_dir, 'p_int_cells_aw_verts', p_int%cells_aw_verts)
  CALL load_r3(dump_dir, 'p_int_e_bln_c_s', p_int%e_bln_c_s)
  CALL load_r3(dump_dir, 'p_int_geofac_grdiv', p_int%geofac_grdiv)
  CALL load_r3(dump_dir, 'p_int_geofac_n2s', p_int%geofac_n2s)
  CALL load_r3(dump_dir, 'p_int_geofac_rot', p_int%geofac_rot)
  CALL load_r3(dump_dir, 'p_int_rbf_vec_coeff_e', p_int%rbf_vec_coeff_e)

  CALL loadp_r3(dump_dir, 'p_prog_vn', p_prog%vn)
  CALL loadp_r3(dump_dir, 'p_prog_w', p_prog%w)

  CALL loadp_r3(dump_dir, 'p_metrics_coeff1_dwdz', p_metrics%coeff1_dwdz)
  CALL loadp_r3(dump_dir, 'p_metrics_coeff2_dwdz', p_metrics%coeff2_dwdz)
  CALL loadp_r3(dump_dir, 'p_metrics_coeff_gradekin', p_metrics%coeff_gradekin)
  CALL loadp_r3(dump_dir, 'p_metrics_ddqz_z_full_e', p_metrics%ddqz_z_full_e)
  CALL loadp_r3(dump_dir, 'p_metrics_ddqz_z_half', p_metrics%ddqz_z_half)
  CALL loadp_r3(dump_dir, 'p_metrics_ddxn_z_full', p_metrics%ddxn_z_full)
  CALL loadp_r3(dump_dir, 'p_metrics_ddxt_z_full', p_metrics%ddxt_z_full)
  CALL loadp_r3(dump_dir, 'p_metrics_wgtfac_c', p_metrics%wgtfac_c)
  CALL loadp_r3(dump_dir, 'p_metrics_wgtfac_e', p_metrics%wgtfac_e)
  CALL loadp_r3(dump_dir, 'p_metrics_wgtfacq_e', p_metrics%wgtfacq_e)
  CALL loadp_r1(dump_dir, 'p_metrics_deepatmo_gradh_ifc', p_metrics%deepatmo_gradh_ifc)
  CALL loadp_r1(dump_dir, 'p_metrics_deepatmo_gradh_mc', p_metrics%deepatmo_gradh_mc)
  CALL loadp_r1(dump_dir, 'p_metrics_deepatmo_invr_ifc', p_metrics%deepatmo_invr_ifc)
  CALL loadp_r1(dump_dir, 'p_metrics_deepatmo_invr_mc', p_metrics%deepatmo_invr_mc)

  CALL loadp_r3(dump_dir, 'p_diag_vt', p_diag%vt)
  CALL loadp_r3(dump_dir, 'p_diag_vn_ie', p_diag%vn_ie)
  CALL loadp_r3(dump_dir, 'p_diag_vn_ie_ubc', p_diag%vn_ie_ubc)
  CALL loadp_r3(dump_dir, 'p_diag_w_concorr_c', p_diag%w_concorr_c)
  CALL loadp_r4(dump_dir, 'p_diag_ddt_vn_apc_pc', p_diag%ddt_vn_apc_pc)
  CALL loadp_r4(dump_dir, 'p_diag_ddt_vn_cor_pc', p_diag%ddt_vn_cor_pc)
  CALL loadp_r4(dump_dir, 'p_diag_ddt_w_adv_pc', p_diag%ddt_w_adv_pc)
  ! max_vcfl_dyn is a component scalar in the kernel but a length-1 buffer in the dump.
  ALLOCATE (vcfl0(1))
  CALL slurp_r(dump_dir, 'p_diag_max_vcfl_dyn', 1, vcfl0)
  p_diag%ddt_vn_adv_is_associated = NINT(scal(dump_dir, 'ddt_vn_cor_associated')) /= 0
  p_diag%ddt_vn_cor_is_associated = p_diag%ddt_vn_adv_is_associated

  CALL load_r3(dump_dir, 'z_kin_hor_e', z_kin_hor_e)
  CALL load_r3(dump_dir, 'z_vt_ie', z_vt_ie)
  CALL load_r3(dump_dir, 'z_w_concorr_me', z_w_concorr_me)

  ! Write set, restored before every rep so each rep does identical work (run_velocity_perf.OUTPUTS).
  CALL snapshot(p_diag%ddt_vn_apc_pc, SIZE(p_diag%ddt_vn_apc_pc), sv_apc)
  CALL snapshot(p_diag%ddt_vn_cor_pc, SIZE(p_diag%ddt_vn_cor_pc), sv_cor)
  CALL snapshot(p_diag%ddt_w_adv_pc, SIZE(p_diag%ddt_w_adv_pc), sv_wadv)
  CALL snapshot(p_diag%vt, SIZE(p_diag%vt), sv_vt)
  CALL snapshot(p_diag%vn_ie, SIZE(p_diag%vn_ie), sv_vnie)
  CALL snapshot(p_diag%vn_ie_ubc, SIZE(p_diag%vn_ie_ubc), sv_ubc)
  CALL snapshot(p_diag%w_concorr_c, SIZE(p_diag%w_concorr_c), sv_wcc)
  CALL snapshot(z_kin_hor_e, SIZE(z_kin_hor_e), sv_kin)
  CALL snapshot(z_vt_ie, SIZE(z_vt_ie), sv_vtie)
  CALL snapshot(z_w_concorr_me, SIZE(z_w_concorr_me), sv_wcme)

  WRITE (error_unit, '(A,I0,A,I0,A,I0,A,I0)') 'loaded dump: nproma=', nproma, ' nblks_e=', nblks_e, &
    ' nlev=', nlev, ' nlevp1=', nlevp1

  ! The kernel's !$ACC DATA says PRESENT(p_diag, ...) because in ICON solve_nh has already put those
  ! derived types on the device; standalone, this is that step.  Shallow on purpose: nvfortran
  ! cannot deep-copy the POINTER components, and -gpu=mem:managed (the openacc lane's flag) makes
  ! the component data device-accessible without one.  Inert for the non-OpenACC lanes.
  !$ACC ENTER DATA COPYIN(p_patch, p_int, p_prog, p_metrics, p_diag)

  CALL SYSTEM_CLOCK(COUNT_RATE=rate)
  DO rep = -warmup, reps - 1
    CALL restore(p_diag%ddt_vn_apc_pc, SIZE(p_diag%ddt_vn_apc_pc), sv_apc)
    CALL restore(p_diag%ddt_vn_cor_pc, SIZE(p_diag%ddt_vn_cor_pc), sv_cor)
    CALL restore(p_diag%ddt_w_adv_pc, SIZE(p_diag%ddt_w_adv_pc), sv_wadv)
    CALL restore(p_diag%vt, SIZE(p_diag%vt), sv_vt)
    CALL restore(p_diag%vn_ie, SIZE(p_diag%vn_ie), sv_vnie)
    CALL restore(p_diag%vn_ie_ubc, SIZE(p_diag%vn_ie_ubc), sv_ubc)
    CALL restore(p_diag%w_concorr_c, SIZE(p_diag%w_concorr_c), sv_wcc)
    CALL restore(z_kin_hor_e, SIZE(z_kin_hor_e), sv_kin)
    CALL restore(z_vt_ie, SIZE(z_vt_ie), sv_vtie)
    CALL restore(z_w_concorr_me, SIZE(z_w_concorr_me), sv_wcme)
    p_diag%max_vcfl_dyn = vcfl0(1)

    CALL SYSTEM_CLOCK(t0)
    CALL velocity_tendencies(p_prog, p_patch, p_int, p_metrics, p_diag, z_w_concorr_me, z_kin_hor_e, z_vt_ie, &
                             ntnd, istep, lvn_only, dtime, dt_linintp_ubc, ldeepatmo)
    CALL SYSTEM_CLOCK(t1)
    ms = 1000.0_wp*REAL(t1 - t0, wp)/REAL(rate, wp)
    IF (rep >= 0) THEN
      WRITE (row, '(A,I0,A,I0,A,I0,A,I0,A,F0.3,A)') 'velocity_tendencies,loopexch,', nproma, ',', nblks_e, ',', &
        threads, ',', rep, ',', ms, ',r02b05,'//TRIM(lane)
      WRITE (output_unit, '(A)') TRIM(row)
      FLUSH (output_unit)
    END IF
  END DO

  !$ACC EXIT DATA DELETE(p_patch, p_int, p_prog, p_metrics, p_diag)

  vcfl0(1) = p_diag%max_vcfl_dyn
  IF (mode == 'dumpref') THEN
    CALL dump_out(ref_dir, 'p_diag_ddt_vn_apc_pc', p_diag%ddt_vn_apc_pc, SIZE(p_diag%ddt_vn_apc_pc))
    CALL dump_out(ref_dir, 'p_diag_ddt_vn_cor_pc', p_diag%ddt_vn_cor_pc, SIZE(p_diag%ddt_vn_cor_pc))
    CALL dump_out(ref_dir, 'p_diag_ddt_w_adv_pc', p_diag%ddt_w_adv_pc, SIZE(p_diag%ddt_w_adv_pc))
    CALL dump_out(ref_dir, 'p_diag_vt', p_diag%vt, SIZE(p_diag%vt))
    CALL dump_out(ref_dir, 'p_diag_vn_ie', p_diag%vn_ie, SIZE(p_diag%vn_ie))
    CALL dump_out(ref_dir, 'p_diag_vn_ie_ubc', p_diag%vn_ie_ubc, SIZE(p_diag%vn_ie_ubc))
    CALL dump_out(ref_dir, 'p_diag_w_concorr_c', p_diag%w_concorr_c, SIZE(p_diag%w_concorr_c))
    CALL dump_out(ref_dir, 'z_kin_hor_e', z_kin_hor_e, SIZE(z_kin_hor_e))
    CALL dump_out(ref_dir, 'z_vt_ie', z_vt_ie, SIZE(z_vt_ie))
    CALL dump_out(ref_dir, 'z_w_concorr_me', z_w_concorr_me, SIZE(z_w_concorr_me))
    CALL dump_out(ref_dir, 'p_diag_max_vcfl_dyn', vcfl0, 1)
    WRITE (error_unit, '(A)') 'wrote reference outputs to '//TRIM(ref_dir)
  ELSE IF (mode == 'verify') THEN
    worst = 0.0_wp
    CALL verify(ref_dir, 'p_diag_ddt_vn_apc_pc', p_diag%ddt_vn_apc_pc, SIZE(p_diag%ddt_vn_apc_pc), worst)
    CALL verify(ref_dir, 'p_diag_ddt_vn_cor_pc', p_diag%ddt_vn_cor_pc, SIZE(p_diag%ddt_vn_cor_pc), worst)
    CALL verify(ref_dir, 'p_diag_ddt_w_adv_pc', p_diag%ddt_w_adv_pc, SIZE(p_diag%ddt_w_adv_pc), worst)
    CALL verify(ref_dir, 'p_diag_vt', p_diag%vt, SIZE(p_diag%vt), worst)
    CALL verify(ref_dir, 'p_diag_vn_ie', p_diag%vn_ie, SIZE(p_diag%vn_ie), worst)
    CALL verify(ref_dir, 'p_diag_vn_ie_ubc', p_diag%vn_ie_ubc, SIZE(p_diag%vn_ie_ubc), worst)
    CALL verify(ref_dir, 'p_diag_w_concorr_c', p_diag%w_concorr_c, SIZE(p_diag%w_concorr_c), worst)
    CALL verify(ref_dir, 'z_kin_hor_e', z_kin_hor_e, SIZE(z_kin_hor_e), worst)
    CALL verify(ref_dir, 'z_vt_ie', z_vt_ie, SIZE(z_vt_ie), worst)
    CALL verify(ref_dir, 'z_w_concorr_me', z_w_concorr_me, SIZE(z_w_concorr_me), worst)
    CALL verify(ref_dir, 'p_diag_max_vcfl_dyn', vcfl0, 1, worst)
    WRITE (error_unit, '(A,ES12.5,A,ES12.5)') 'verify worst relative gap ', worst, ' vs bound ', verify_tol
    IF (worst > verify_tol) THEN
      WRITE (error_unit, '(A)') 'VERIFY FAILED'
      STOP 1
    END IF
    WRITE (error_unit, '(A)') 'VERIFY OK'
  ELSE IF (LEN_TRIM(mode) > 0) THEN
    CALL die('unknown mode '//TRIM(mode)//' (want verify or dumpref)')
  END IF

END PROGRAM driver_velocity
