!> Differential verification helpers for the ICON ``solve_nh`` dycore.
!>
!> The dycore lowered to an SDFG (``solve_nh_dace_icon``) is checked against
!> the stock Fortran ``solve_nh`` by running BOTH on the SAME input and
!> comparing the prognostic output BIT-FOR-BIT.  A correct lowering matches
!> numpy/Fortran exactly, so any bit difference is a real bug -- there are no
!> tolerances here.
!>
!> The prognostic fields (``vn``/``w``/``rho``/``theta_v``/``exner``) and the
!> transport-prep fluxes are ``POINTER`` components, so intrinsic assignment
!> (``dst = src``) only shallow-copies the pointers -- both operands would then
!> alias the SAME storage and the two runs would clobber each other.  The clone
!> routines therefore allocate FRESH targets and copy the data (a genuine deep
!> copy), leaving the read-only geometry (``metrics``/``int``/``patch``) and the
!> write-before-read ``diag`` scratch shared.
!>
!> ``solve_nh`` never assigns ``prog(nnow)`` (verified), so a clone with an
!> independent ``prog`` is a sufficient reference input; ``prep_adv`` IS an
!> accumulator (``vn_traj = vn_traj + ...``), so it is cloned as well.
!>
!> The module ``use``s only the ICON type modules, so it compiles unchanged
!> against both the pruned single-TU types (standalone e2e) and ICON's real
!> ``mo_nonhydro_types`` / ``mo_prepadv_types`` (binding-swap integration).
module mo_solve_nh_diff
  use iso_fortran_env,   only: error_unit
  use iso_c_binding,     only: i8 => c_int64_t
  use mo_nonhydro_types, only: t_nh_state, t_nh_diag
  use mo_prepadv_types,  only: t_prepare_adv
  implicit none
  private

  integer, parameter :: wp = 8

  public :: clone_state_indep_prog, free_state_clone
  public :: clone_prepadv_indep, free_prepadv_clone
  public :: compare_prog_nnew, compare_prepadv, compare_diag

contains

  !> Deep-copy a POINTER 3D field: allocate a fresh target with the source's
  !> exact bounds and copy the values.  ``d`` is left disassociated when ``s``
  !> is not associated.  Release the fresh target with ``free_ptr3`` -- pointer
  !> allocations are not auto-finalized.
  subroutine clone_ptr3(d, s)
    real(wp), pointer, contiguous, intent(out) :: d(:, :, :)
    real(wp), pointer, contiguous, intent(in)  :: s(:, :, :)
    if (associated(s)) then
      allocate (d(lbound(s, 1):ubound(s, 1), &
                  lbound(s, 2):ubound(s, 2), &
                  lbound(s, 3):ubound(s, 3)))
      d = s
    else
      d => null()
    end if
  end subroutine clone_ptr3

  subroutine free_ptr3(d)
    real(wp), pointer, contiguous, intent(inout) :: d(:, :, :)
    if (associated(d)) then
      deallocate (d)
      d => null()
    end if
  end subroutine free_ptr3

  !> BIT-EXACT comparison of two 3D fields via raw IEEE bit patterns.  A NaN
  !> with identical bits counts as equal (deterministic); any bit difference is
  !> a genuine discrepancy.  Accumulates the differing-element count into
  !> ``ndiff`` and the largest magnitude gap into ``maxabs`` for the report.
  subroutine cmp_ptr3(x, y, name, ndiff, maxabs)
    real(wp), pointer, contiguous, intent(in) :: x(:, :, :), y(:, :, :)
    character(*), intent(in)                  :: name
    integer, intent(inout)                    :: ndiff
    real(wp), intent(inout)                   :: maxabs
    integer  :: i, j, k, nloc
    real(wp) :: d

    if (.not. associated(x) .and. .not. associated(y)) return
    if (.not. associated(x) .or. .not. associated(y)) then
      write (error_unit, '(A)') "  DIFF "//trim(name)//": association mismatch"
      ndiff = ndiff + 1
      return
    end if

    nloc = 0
    do k = lbound(x, 3), ubound(x, 3)
      do j = lbound(x, 2), ubound(x, 2)
        do i = lbound(x, 1), ubound(x, 1)
          if (transfer(x(i, j, k), 1_i8) /= transfer(y(i, j, k), 1_i8)) then
            nloc = nloc + 1
            d = abs(x(i, j, k) - y(i, j, k))
            if (d > maxabs) maxabs = d
          end if
        end do
      end do
    end do
    if (nloc > 0) then
      write (error_unit, '(A,I0,A)') "  DIFF "//trim(name)//": ", nloc, " elements differ (bit-level)"
      ndiff = ndiff + nloc
    end if
  end subroutine cmp_ptr3

  !> Rank-4 analogue of ``clone_ptr3``: deep-copy a POINTER 4D field (the diag
  !> ``ddt_vn_apc_pc`` / ``ddt_vn_cor_pc`` / ``ddt_w_adv_pc`` predictor-corrector
  !> tendencies the velocity SDFG writes).  ``d`` is disassociated when ``s`` is
  !> not associated; release with ``free_ptr4``.
  subroutine clone_ptr4(d, s)
    real(wp), pointer, contiguous, intent(out) :: d(:, :, :, :)
    real(wp), pointer, contiguous, intent(in)  :: s(:, :, :, :)
    if (associated(s)) then
      allocate (d(lbound(s, 1):ubound(s, 1), &
                  lbound(s, 2):ubound(s, 2), &
                  lbound(s, 3):ubound(s, 3), &
                  lbound(s, 4):ubound(s, 4)))
      d = s
    else
      d => null()
    end if
  end subroutine clone_ptr4

  subroutine free_ptr4(d)
    real(wp), pointer, contiguous, intent(inout) :: d(:, :, :, :)
    if (associated(d)) then
      deallocate (d)
      d => null()
    end if
  end subroutine free_ptr4

  !> Rank-4 analogue of ``cmp_ptr3``: BIT-EXACT comparison of two 4D fields via
  !> raw IEEE bit patterns.  Accumulates the differing-element count into
  !> ``ndiff`` and the largest magnitude gap into ``maxabs``.
  subroutine cmp_ptr4(x, y, name, ndiff, maxabs)
    real(wp), pointer, contiguous, intent(in) :: x(:, :, :, :), y(:, :, :, :)
    character(*), intent(in)                  :: name
    integer, intent(inout)                    :: ndiff
    real(wp), intent(inout)                   :: maxabs
    integer  :: i, j, k, l, nloc
    real(wp) :: d

    if (.not. associated(x) .and. .not. associated(y)) return
    if (.not. associated(x) .or. .not. associated(y)) then
      write (error_unit, '(A)') "  DIFF "//trim(name)//": association mismatch"
      ndiff = ndiff + 1
      return
    end if

    nloc = 0
    do l = lbound(x, 4), ubound(x, 4)
      do k = lbound(x, 3), ubound(x, 3)
        do j = lbound(x, 2), ubound(x, 2)
          do i = lbound(x, 1), ubound(x, 1)
            if (transfer(x(i, j, k, l), 1_i8) /= transfer(y(i, j, k, l), 1_i8)) then
              nloc = nloc + 1
              d = abs(x(i, j, k, l) - y(i, j, k, l))
              if (d > maxabs) maxabs = d
            end if
          end do
        end do
      end do
    end do
    if (nloc > 0) then
      write (error_unit, '(A,I0,A)') "  DIFF "//trim(name)//": ", nloc, " elements differ (bit-level)"
      ndiff = ndiff + nloc
    end if
  end subroutine cmp_ptr4

  !> Deep-copy EVERY POINTER field of a ``t_nh_diag`` into ``dst`` so the two
  !> solves own independent diagnostic storage.  The shallow ``dst = src`` in the
  !> caller aliases every ``diag`` pointer; this re-points each to a fresh copy.
  !> The velocity SDFG writes ``ddt_vn_apc_pc`` / ``ddt_vn_cor_pc`` /
  !> ``ddt_w_adv_pc`` (rank-4) and solve_nh writes many of the rank-3 fields, so
  !> comparing ``diag`` (via ``compare_diag``) is what catches a velocity-callback
  !> divergence.  ``clone_ptr3`` / ``clone_ptr4`` no-op on unassociated pointers,
  !> so the ``ddt_vn_*`` fields left ``=> NULL()`` are handled safely.
  subroutine clone_diag_indep(s, d)
    type(t_nh_diag), intent(in)  :: s
    type(t_nh_diag), intent(out) :: d
    ! rank-3 fields
    call clone_ptr3(d%exner_pr,       s%exner_pr)
    call clone_ptr3(d%mass_fl_e,      s%mass_fl_e)
    call clone_ptr3(d%rho_ic,         s%rho_ic)
    call clone_ptr3(d%theta_v_ic,     s%theta_v_ic)
    call clone_ptr3(d%grf_tend_vn,    s%grf_tend_vn)
    call clone_ptr3(d%grf_tend_w,     s%grf_tend_w)
    call clone_ptr3(d%grf_tend_rho,   s%grf_tend_rho)
    call clone_ptr3(d%grf_tend_mflx,  s%grf_tend_mflx)
    call clone_ptr3(d%grf_bdy_mflx,   s%grf_bdy_mflx)
    call clone_ptr3(d%grf_tend_thv,   s%grf_tend_thv)
    call clone_ptr3(d%vn_ie_int,      s%vn_ie_int)
    call clone_ptr3(d%vn_ie_ubc,      s%vn_ie_ubc)
    call clone_ptr3(d%w_int,          s%w_int)
    call clone_ptr3(d%w_ubc,          s%w_ubc)
    call clone_ptr3(d%theta_v_ic_int, s%theta_v_ic_int)
    call clone_ptr3(d%theta_v_ic_ubc, s%theta_v_ic_ubc)
    call clone_ptr3(d%rho_ic_int,     s%rho_ic_int)
    call clone_ptr3(d%rho_ic_ubc,     s%rho_ic_ubc)
    call clone_ptr3(d%mflx_ic_int,    s%mflx_ic_int)
    call clone_ptr3(d%mflx_ic_ubc,    s%mflx_ic_ubc)
    call clone_ptr3(d%vn_incr,        s%vn_incr)
    call clone_ptr3(d%exner_incr,     s%exner_incr)
    call clone_ptr3(d%rho_incr,       s%rho_incr)
    call clone_ptr3(d%vt,             s%vt)
    call clone_ptr3(d%ddt_exner_phy,  s%ddt_exner_phy)
    call clone_ptr3(d%ddt_vn_phy,     s%ddt_vn_phy)
    call clone_ptr3(d%exner_dyn_incr, s%exner_dyn_incr)
    call clone_ptr3(d%vn_ie,          s%vn_ie)
    call clone_ptr3(d%w_concorr_c,    s%w_concorr_c)
    call clone_ptr3(d%mass_fl_e_sv,   s%mass_fl_e_sv)
    call clone_ptr3(d%ddt_vn_dyn,     s%ddt_vn_dyn)
    call clone_ptr3(d%ddt_vn_dmp,     s%ddt_vn_dmp)
    call clone_ptr3(d%ddt_vn_adv,     s%ddt_vn_adv)
    call clone_ptr3(d%ddt_vn_cor,     s%ddt_vn_cor)
    call clone_ptr3(d%ddt_vn_pgr,     s%ddt_vn_pgr)
    call clone_ptr3(d%ddt_vn_phd,     s%ddt_vn_phd)
    call clone_ptr3(d%ddt_vn_iau,     s%ddt_vn_iau)
    call clone_ptr3(d%ddt_vn_ray,     s%ddt_vn_ray)
    call clone_ptr3(d%ddt_vn_grf,     s%ddt_vn_grf)
    ! rank-4 fields
    call clone_ptr4(d%ddt_vn_apc_pc,  s%ddt_vn_apc_pc)
    call clone_ptr4(d%ddt_vn_cor_pc,  s%ddt_vn_cor_pc)
    call clone_ptr4(d%ddt_w_adv_pc,   s%ddt_w_adv_pc)
  end subroutine clone_diag_indep

  subroutine free_diag_clone(d)
    type(t_nh_diag), intent(inout) :: d
    call free_ptr3(d%exner_pr)
    call free_ptr3(d%mass_fl_e)
    call free_ptr3(d%rho_ic)
    call free_ptr3(d%theta_v_ic)
    call free_ptr3(d%grf_tend_vn)
    call free_ptr3(d%grf_tend_w)
    call free_ptr3(d%grf_tend_rho)
    call free_ptr3(d%grf_tend_mflx)
    call free_ptr3(d%grf_bdy_mflx)
    call free_ptr3(d%grf_tend_thv)
    call free_ptr3(d%vn_ie_int)
    call free_ptr3(d%vn_ie_ubc)
    call free_ptr3(d%w_int)
    call free_ptr3(d%w_ubc)
    call free_ptr3(d%theta_v_ic_int)
    call free_ptr3(d%theta_v_ic_ubc)
    call free_ptr3(d%rho_ic_int)
    call free_ptr3(d%rho_ic_ubc)
    call free_ptr3(d%mflx_ic_int)
    call free_ptr3(d%mflx_ic_ubc)
    call free_ptr3(d%vn_incr)
    call free_ptr3(d%exner_incr)
    call free_ptr3(d%rho_incr)
    call free_ptr3(d%vt)
    call free_ptr3(d%ddt_exner_phy)
    call free_ptr3(d%ddt_vn_phy)
    call free_ptr3(d%exner_dyn_incr)
    call free_ptr3(d%vn_ie)
    call free_ptr3(d%w_concorr_c)
    call free_ptr3(d%mass_fl_e_sv)
    call free_ptr3(d%ddt_vn_dyn)
    call free_ptr3(d%ddt_vn_dmp)
    call free_ptr3(d%ddt_vn_adv)
    call free_ptr3(d%ddt_vn_cor)
    call free_ptr3(d%ddt_vn_pgr)
    call free_ptr3(d%ddt_vn_phd)
    call free_ptr3(d%ddt_vn_iau)
    call free_ptr3(d%ddt_vn_ray)
    call free_ptr3(d%ddt_vn_grf)
    call free_ptr4(d%ddt_vn_apc_pc)
    call free_ptr4(d%ddt_vn_cor_pc)
    call free_ptr4(d%ddt_w_adv_pc)
  end subroutine free_diag_clone

  !> Clone ``src`` into ``dst`` with a FULLY INDEPENDENT mutable state.  Shallow
  !> ``dst = src`` first (which aliases every POINTER component -- ``prog(:)``
  !> AND ``diag`` fields), then re-point every prognostic field AND every
  !> ``diag`` field to a fresh deep copy so the reference solve cannot clobber
  !> the SDFG run.  ``metrics`` / ``ref`` / ``p_patch`` / ``p_int`` are READ-ONLY
  !> (solve_nh writes none of them, verified) so they stay shared.
  subroutine clone_state_indep_prog(src, dst)
    type(t_nh_state), intent(in)  :: src
    type(t_nh_state), intent(out) :: dst
    integer :: i
    dst = src
    do i = lbound(src%prog, 1), ubound(src%prog, 1)
      call clone_ptr3(dst%prog(i)%vn,      src%prog(i)%vn)
      call clone_ptr3(dst%prog(i)%w,       src%prog(i)%w)
      call clone_ptr3(dst%prog(i)%rho,     src%prog(i)%rho)
      call clone_ptr3(dst%prog(i)%theta_v, src%prog(i)%theta_v)
      call clone_ptr3(dst%prog(i)%exner,   src%prog(i)%exner)
    end do
    call clone_diag_indep(src%diag, dst%diag)
  end subroutine clone_state_indep_prog

  !> Release the fresh prognostic + diagnostic targets allocated by
  !> ``clone_state_indep_prog`` (the shared ``metrics`` / ``ref`` are untouched).
  subroutine free_state_clone(dst)
    type(t_nh_state), intent(inout) :: dst
    integer :: i
    call free_diag_clone(dst%diag)
    if (.not. allocated(dst%prog)) return
    do i = lbound(dst%prog, 1), ubound(dst%prog, 1)
      call free_ptr3(dst%prog(i)%vn)
      call free_ptr3(dst%prog(i)%w)
      call free_ptr3(dst%prog(i)%rho)
      call free_ptr3(dst%prog(i)%theta_v)
      call free_ptr3(dst%prog(i)%exner)
    end do
  end subroutine free_state_clone

  !> Deep-copy the transport-prep accumulators into an independent ``dst``
  !> (every component is a POINTER, so all four are cloned outright).
  subroutine clone_prepadv_indep(src, dst)
    type(t_prepare_adv), intent(in)  :: src
    type(t_prepare_adv), intent(out) :: dst
    call clone_ptr3(dst%mass_flx_me, src%mass_flx_me)
    call clone_ptr3(dst%mass_flx_ic, src%mass_flx_ic)
    call clone_ptr3(dst%vol_flx_ic,  src%vol_flx_ic)
    call clone_ptr3(dst%vn_traj,     src%vn_traj)
  end subroutine clone_prepadv_indep

  subroutine free_prepadv_clone(dst)
    type(t_prepare_adv), intent(inout) :: dst
    call free_ptr3(dst%mass_flx_me)
    call free_ptr3(dst%mass_flx_ic)
    call free_ptr3(dst%vol_flx_ic)
    call free_ptr3(dst%vn_traj)
  end subroutine free_prepadv_clone

  !> Compare the prognostic output of the two runs at time level ``nnew``
  !> (``vn``/``w``/``rho``/``theta_v``/``exner``).  ``ndiff`` returns the total
  !> differing-element count; 0 == bit-exact.
  subroutine compare_prog_nnew(a, b, nnew, label, ndiff)
    type(t_nh_state), intent(in) :: a, b
    integer, intent(in)          :: nnew
    character(*), intent(in)     :: label
    integer, intent(out)         :: ndiff
    real(wp) :: mx
    ndiff = 0
    mx = 0.0_wp
    call cmp_ptr3(a%prog(nnew)%vn,      b%prog(nnew)%vn,      label//":vn",      ndiff, mx)
    call cmp_ptr3(a%prog(nnew)%w,       b%prog(nnew)%w,       label//":w",       ndiff, mx)
    call cmp_ptr3(a%prog(nnew)%rho,     b%prog(nnew)%rho,     label//":rho",     ndiff, mx)
    call cmp_ptr3(a%prog(nnew)%theta_v, b%prog(nnew)%theta_v, label//":theta_v", ndiff, mx)
    call cmp_ptr3(a%prog(nnew)%exner,   b%prog(nnew)%exner,   label//":exner",   ndiff, mx)
    if (ndiff == 0) then
      write (error_unit, '(A)') "  [diff] "//trim(label)//": prog(nnew) BIT-EXACT {vn,w,rho,theta_v,exner}"
    else
      write (error_unit, '(A,I0,A,ES12.5)') "  [diff] "//trim(label)//": ", ndiff, &
        " prognostic elements differ; max|delta|=", mx
    end if
  end subroutine compare_prog_nnew

  !> Compare the transport-prep fluxes of the two runs.
  subroutine compare_prepadv(a, b, label, ndiff)
    type(t_prepare_adv), intent(in) :: a, b
    character(*), intent(in)        :: label
    integer, intent(out)            :: ndiff
    real(wp) :: mx
    ndiff = 0
    mx = 0.0_wp
    call cmp_ptr3(a%mass_flx_me, b%mass_flx_me, label//":mass_flx_me", ndiff, mx)
    call cmp_ptr3(a%mass_flx_ic, b%mass_flx_ic, label//":mass_flx_ic", ndiff, mx)
    call cmp_ptr3(a%vol_flx_ic,  b%vol_flx_ic,  label//":vol_flx_ic",  ndiff, mx)
    call cmp_ptr3(a%vn_traj,     b%vn_traj,     label//":vn_traj",     ndiff, mx)
    if (ndiff == 0) then
      write (error_unit, '(A)') "  [diff] "//trim(label)//": prep_adv BIT-EXACT {mass_flx_me,mass_flx_ic,vol_flx_ic,vn_traj}"
    else
      write (error_unit, '(A,I0,A,ES12.5)') "  [diff] "//trim(label)//": ", ndiff, &
        " prep_adv elements differ; max|delta|=", mx
    end if
  end subroutine compare_prepadv

  !> Compare EVERY POINTER field of the two runs' ``t_nh_diag`` bit-for-bit.
  !> The velocity callback writes ``ddt_vn_apc_pc`` / ``ddt_vn_cor_pc`` /
  !> ``ddt_w_adv_pc`` and solve_nh writes many of the rank-3 fields, so a
  !> divergence in the velocity SDFG (or any diag-touching lowering) surfaces
  !> here.  ``cmp_ptr3`` / ``cmp_ptr4`` skip fields unassociated in BOTH runs, so
  !> the ``=> NULL()`` ``ddt_vn_*`` fields contribute nothing when unused.
  subroutine compare_diag(a, b, label, ndiff)
    type(t_nh_diag), intent(in) :: a, b
    character(*), intent(in)    :: label
    integer, intent(out)        :: ndiff
    real(wp) :: mx
    ndiff = 0
    mx = 0.0_wp
    call cmp_ptr3(a%exner_pr,       b%exner_pr,       label//":exner_pr",       ndiff, mx)
    call cmp_ptr3(a%mass_fl_e,      b%mass_fl_e,      label//":mass_fl_e",      ndiff, mx)
    call cmp_ptr3(a%rho_ic,         b%rho_ic,         label//":rho_ic",         ndiff, mx)
    call cmp_ptr3(a%theta_v_ic,     b%theta_v_ic,     label//":theta_v_ic",     ndiff, mx)
    call cmp_ptr3(a%grf_tend_vn,    b%grf_tend_vn,    label//":grf_tend_vn",    ndiff, mx)
    call cmp_ptr3(a%grf_tend_w,     b%grf_tend_w,     label//":grf_tend_w",     ndiff, mx)
    call cmp_ptr3(a%grf_tend_rho,   b%grf_tend_rho,   label//":grf_tend_rho",   ndiff, mx)
    call cmp_ptr3(a%grf_tend_mflx,  b%grf_tend_mflx,  label//":grf_tend_mflx",  ndiff, mx)
    call cmp_ptr3(a%grf_bdy_mflx,   b%grf_bdy_mflx,   label//":grf_bdy_mflx",   ndiff, mx)
    call cmp_ptr3(a%grf_tend_thv,   b%grf_tend_thv,   label//":grf_tend_thv",   ndiff, mx)
    call cmp_ptr3(a%vn_ie_int,      b%vn_ie_int,      label//":vn_ie_int",      ndiff, mx)
    call cmp_ptr3(a%vn_ie_ubc,      b%vn_ie_ubc,      label//":vn_ie_ubc",      ndiff, mx)
    call cmp_ptr3(a%w_int,          b%w_int,          label//":w_int",          ndiff, mx)
    call cmp_ptr3(a%w_ubc,          b%w_ubc,          label//":w_ubc",          ndiff, mx)
    call cmp_ptr3(a%theta_v_ic_int, b%theta_v_ic_int, label//":theta_v_ic_int", ndiff, mx)
    call cmp_ptr3(a%theta_v_ic_ubc, b%theta_v_ic_ubc, label//":theta_v_ic_ubc", ndiff, mx)
    call cmp_ptr3(a%rho_ic_int,     b%rho_ic_int,     label//":rho_ic_int",     ndiff, mx)
    call cmp_ptr3(a%rho_ic_ubc,     b%rho_ic_ubc,     label//":rho_ic_ubc",     ndiff, mx)
    call cmp_ptr3(a%mflx_ic_int,    b%mflx_ic_int,    label//":mflx_ic_int",    ndiff, mx)
    call cmp_ptr3(a%mflx_ic_ubc,    b%mflx_ic_ubc,    label//":mflx_ic_ubc",    ndiff, mx)
    call cmp_ptr3(a%vn_incr,        b%vn_incr,        label//":vn_incr",        ndiff, mx)
    call cmp_ptr3(a%exner_incr,     b%exner_incr,     label//":exner_incr",     ndiff, mx)
    call cmp_ptr3(a%rho_incr,       b%rho_incr,       label//":rho_incr",       ndiff, mx)
    call cmp_ptr3(a%vt,             b%vt,             label//":vt",             ndiff, mx)
    call cmp_ptr3(a%ddt_exner_phy,  b%ddt_exner_phy,  label//":ddt_exner_phy",  ndiff, mx)
    call cmp_ptr3(a%ddt_vn_phy,     b%ddt_vn_phy,     label//":ddt_vn_phy",     ndiff, mx)
    call cmp_ptr3(a%exner_dyn_incr, b%exner_dyn_incr, label//":exner_dyn_incr", ndiff, mx)
    call cmp_ptr3(a%vn_ie,          b%vn_ie,          label//":vn_ie",          ndiff, mx)
    call cmp_ptr3(a%w_concorr_c,    b%w_concorr_c,    label//":w_concorr_c",    ndiff, mx)
    call cmp_ptr3(a%mass_fl_e_sv,   b%mass_fl_e_sv,   label//":mass_fl_e_sv",   ndiff, mx)
    call cmp_ptr3(a%ddt_vn_dyn,     b%ddt_vn_dyn,     label//":ddt_vn_dyn",     ndiff, mx)
    call cmp_ptr3(a%ddt_vn_dmp,     b%ddt_vn_dmp,     label//":ddt_vn_dmp",     ndiff, mx)
    call cmp_ptr3(a%ddt_vn_adv,     b%ddt_vn_adv,     label//":ddt_vn_adv",     ndiff, mx)
    call cmp_ptr3(a%ddt_vn_cor,     b%ddt_vn_cor,     label//":ddt_vn_cor",     ndiff, mx)
    call cmp_ptr3(a%ddt_vn_pgr,     b%ddt_vn_pgr,     label//":ddt_vn_pgr",     ndiff, mx)
    call cmp_ptr3(a%ddt_vn_phd,     b%ddt_vn_phd,     label//":ddt_vn_phd",     ndiff, mx)
    call cmp_ptr3(a%ddt_vn_iau,     b%ddt_vn_iau,     label//":ddt_vn_iau",     ndiff, mx)
    call cmp_ptr3(a%ddt_vn_ray,     b%ddt_vn_ray,     label//":ddt_vn_ray",     ndiff, mx)
    call cmp_ptr3(a%ddt_vn_grf,     b%ddt_vn_grf,     label//":ddt_vn_grf",     ndiff, mx)
    call cmp_ptr4(a%ddt_vn_apc_pc,  b%ddt_vn_apc_pc,  label//":ddt_vn_apc_pc",  ndiff, mx)
    call cmp_ptr4(a%ddt_vn_cor_pc,  b%ddt_vn_cor_pc,  label//":ddt_vn_cor_pc",  ndiff, mx)
    call cmp_ptr4(a%ddt_w_adv_pc,   b%ddt_w_adv_pc,   label//":ddt_w_adv_pc",   ndiff, mx)
    if (ndiff == 0) then
      write (error_unit, '(A)') "  [diff] "//trim(label)//": diag BIT-EXACT (all fields)"
    else
      write (error_unit, '(A,I0,A,ES12.5)') "  [diff] "//trim(label)//": ", ndiff, &
        " diag elements differ; max|delta|=", mx
    end if
  end subroutine compare_diag

end module mo_solve_nh_diff
