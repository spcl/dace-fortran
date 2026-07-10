!> Differential verification helpers for ICON's ocean ``solve_free_sfc_ab_mimetic``.
!>
!> The ocean twin of :file:`mo_solve_nh_diff.f90`: the free-surface solve lowered
!> to an SDFG (``solve_free_sfc_dace_icon``) is checked against the stock Fortran
!> body by running BOTH on the SAME input and comparing the mutated state
!> BIT-FOR-BIT.  A correct lowering matches Fortran exactly, so any bit
!> difference is a real bug -- there are no tolerances here.
!>
!> Every mutable field of ``t_hydro_ocean_state`` is a POINTER component (the
!> iconfor DSL expands ``onCells``/``onEdges``/... to ``REAL(wp), POINTER``), so
!> intrinsic assignment (``dst = src``) only shallow-copies the pointers -- both
!> operands would then alias the SAME storage and the two runs would clobber
!> each other.  The clone routines allocate FRESH targets with the source's
!> exact bounds and copy the data.
!>
!> UNLIKE the atmosphere's ``t_nh_state%prog`` (ALLOCATABLE, deep-copied by
!> intrinsic assignment), the ocean's ``p_prog(:)`` is itself a POINTER array:
!> ``dst = src`` leaves ``dst%p_prog`` aliasing ``src%p_prog``, so re-pointing a
!> clone's prog field would clobber the original's slot.  ``clone_ocean_state_
!> indep`` therefore allocates a fresh ``p_prog(:)`` target first.
!>
!> Mutated-member coverage (traced through the real ICON call tree:
!> top_bound_cond_horz_veloc -> calculate_explicit_term_ab [veloc_adv_horz/
!> vert_mimetic, calc_internal_press_grad, velocity_diffusion, explicit_vn_pred]
!> -> fill_rhs4surface_eq_ab -> free_sfc_solver%solve):
!>
!>   * prog(nnew)%h                      -- the solver output (THE prognostic).
!>   * diag: vort, grad, veloc_adv_horz, veloc_adv_vert, laplacian_horz,
!>     laplacian_vert, press_hyd, press_grad, vn_pred, vn_pred_ptp -- written.
!>   * aux: g_n, g_nimd, bc_top_vn, bc_top_u, bc_top_v, bc_top_veloc_cc,
!>     bc_bot_vn, p_rhs_sfc_eq -- written.
!>   * diag: vn_time_weighted, w, w_old, h_e, mass_flx_e, div_mass_flx_c,
!>     ptp_vn + aux g_nm1 -- read-only in THIS call but written by the adjacent
!>     dycore phases (calc_normal_velocity / calc_vert_velocity); cloned +
!>     compared too so the harness stays sound when the wrapper grows to cover
!>     them.
!>
!> Read-only members stay SHARED (both runs read identical values): the
!> geometry (``patch_3D``/``op_coeffs``/``solverCoeff_sp``), ``diag%thick_e`` /
!> ``thick_c`` (captured by POINTER inside the module-level solver lhs at init
!> -- cloning them would be misleading, the lhs would still read the original),
!> ``diag%rho``/``kin``/``zgrad_rho``/``p_vn``/``p_vn_dual`` (computed before
!> the dynamics), and ``aux%bc_total_top_potential``.
!>
!> ``p_phys_param%a_veloc_v`` is the one mutated member that CANNOT be redirected
!> to a clone: the PP scheme (``ICON_PP_Edge_vnPredict_scheme``) writes it
!> through a module-level pointer (``v_params``) inside ``mo_ocean_pp_scheme``,
!> and it TIME-SMOOTHS in place (``a_v = w*a_v + (1-w)*new``) -- if the REF ran
!> after the DUT without help it would read the DUT-updated value as "old" and
!> diverge even for a correct DUT.  The driver therefore snapshots it with
!> ``clone_field3`` before the DUT, parks the DUT's result, restores the
!> pre-call values for the REF, compares the two results, and finally restores
!> the DUT's version so ICON carries on with the DUT state.
module mo_ocean_diff
  use iso_fortran_env, only: error_unit
  use iso_c_binding, only: i8 => c_int64_t
  use mo_ocean_types, only: t_hydro_ocean_state, t_hydro_ocean_diag, t_hydro_ocean_aux
  use mo_math_types, only: t_cartesian_coordinates
  implicit none
  private

  integer, parameter :: wp = 8

  public :: clone_ocean_state_indep, free_ocean_state_clone
  public :: compare_ocean_prog, compare_ocean_diag, compare_ocean_aux
  public :: clone_field3, restore_field3, free_field3, compare_field3
  public :: ocean_diff_enforce

contains

  !> Deep-copy a POINTER 2D field: allocate a fresh target with the source's
  !> exact bounds and copy the values.  ``d`` is left disassociated when ``s``
  !> is not associated.  Release with ``free_ptr2`` -- pointer allocations are
  !> not auto-finalized.  (No CONTIGUOUS on the dummies: the iconfor DSL
  !> declares the fields plain ``POINTER``.)
  subroutine clone_ptr2(d, s)
    real(wp), pointer, intent(out) :: d(:, :)
    real(wp), pointer, intent(in)  :: s(:, :)
    if (associated(s)) then
      allocate (d(lbound(s, 1):ubound(s, 1), &
                  lbound(s, 2):ubound(s, 2)))
      d = s
    else
      d => null()
    end if
  end subroutine clone_ptr2

  subroutine free_ptr2(d)
    real(wp), pointer, intent(inout) :: d(:, :)
    if (associated(d)) then
      deallocate (d)
      d => null()
    end if
  end subroutine free_ptr2

  !> Rank-3 analogue of ``clone_ptr2``.
  subroutine clone_ptr3(d, s)
    real(wp), pointer, intent(out) :: d(:, :, :)
    real(wp), pointer, intent(in)  :: s(:, :, :)
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
    real(wp), pointer, intent(inout) :: d(:, :, :)
    if (associated(d)) then
      deallocate (d)
      d => null()
    end if
  end subroutine free_ptr3

  !> Rank-4 analogue of ``clone_ptr2`` (the prog ``tracer`` collection).
  subroutine clone_ptr4(d, s)
    real(wp), pointer, intent(out) :: d(:, :, :, :)
    real(wp), pointer, intent(in)  :: s(:, :, :, :)
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
    real(wp), pointer, intent(inout) :: d(:, :, :, :)
    if (associated(d)) then
      deallocate (d)
      d => null()
    end if
  end subroutine free_ptr4

  !> Deep-copy a POINTER 2D field of ``t_cartesian_coordinates`` (the aux
  !> ``bc_top_veloc_cc`` the wind-stress boundary condition writes).  Intrinsic
  !> assignment copies the ``%x(3)`` component values.
  subroutine clone_cc2(d, s)
    type(t_cartesian_coordinates), pointer, intent(out) :: d(:, :)
    type(t_cartesian_coordinates), pointer, intent(in)  :: s(:, :)
    if (associated(s)) then
      allocate (d(lbound(s, 1):ubound(s, 1), &
                  lbound(s, 2):ubound(s, 2)))
      d = s
    else
      d => null()
    end if
  end subroutine clone_cc2

  subroutine free_cc2(d)
    type(t_cartesian_coordinates), pointer, intent(inout) :: d(:, :)
    if (associated(d)) then
      deallocate (d)
      d => null()
    end if
  end subroutine free_cc2

  !> BIT-EXACT comparison of two 2D fields via raw IEEE bit patterns.  A NaN
  !> with identical bits counts as equal (deterministic); any bit difference is
  !> a genuine discrepancy.  Accumulates the differing-element count into
  !> ``ndiff`` and the largest magnitude gap into ``maxabs`` for the report.
  subroutine cmp_ptr2(x, y, name, ndiff, maxabs)
    real(wp), pointer, intent(in) :: x(:, :), y(:, :)
    character(*), intent(in)      :: name
    integer, intent(inout)        :: ndiff
    real(wp), intent(inout)       :: maxabs
    integer  :: i, j, nloc
    real(wp) :: d

    if (.not. associated(x) .and. .not. associated(y)) return
    if (.not. associated(x) .or. .not. associated(y)) then
      write (error_unit, '(A)') "  DIFF "//trim(name)//": association mismatch"
      ndiff = ndiff + 1
      return
    end if

    nloc = 0
    do j = lbound(x, 2), ubound(x, 2)
      do i = lbound(x, 1), ubound(x, 1)
        if (transfer(x(i, j), 1_i8) /= transfer(y(i, j), 1_i8)) then
          nloc = nloc + 1
          d = abs(x(i, j) - y(i, j))
          if (d > maxabs) maxabs = d
        end if
      end do
    end do
    if (nloc > 0) then
      write (error_unit, '(A,I0,A)') "  DIFF "//trim(name)//": ", nloc, " elements differ (bit-level)"
      ndiff = ndiff + nloc
    end if
  end subroutine cmp_ptr2

  !> Rank-3 analogue of ``cmp_ptr2``.
  subroutine cmp_ptr3(x, y, name, ndiff, maxabs)
    real(wp), pointer, intent(in) :: x(:, :, :), y(:, :, :)
    character(*), intent(in)      :: name
    integer, intent(inout)        :: ndiff
    real(wp), intent(inout)       :: maxabs
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

  !> Rank-4 analogue of ``cmp_ptr2`` (the prog ``tracer`` collection).
  subroutine cmp_ptr4(x, y, name, ndiff, maxabs)
    real(wp), pointer, intent(in) :: x(:, :, :, :), y(:, :, :, :)
    character(*), intent(in)      :: name
    integer, intent(inout)        :: ndiff
    real(wp), intent(inout)       :: maxabs
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

  !> ``t_cartesian_coordinates`` 2D analogue of ``cmp_ptr2``: compares the
  !> three ``%x`` components of every element bit-for-bit.
  subroutine cmp_cc2(x, y, name, ndiff, maxabs)
    type(t_cartesian_coordinates), pointer, intent(in) :: x(:, :), y(:, :)
    character(*), intent(in)                           :: name
    integer, intent(inout)                             :: ndiff
    real(wp), intent(inout)                            :: maxabs
    integer  :: i, j, c, nloc
    real(wp) :: d

    if (.not. associated(x) .and. .not. associated(y)) return
    if (.not. associated(x) .or. .not. associated(y)) then
      write (error_unit, '(A)') "  DIFF "//trim(name)//": association mismatch"
      ndiff = ndiff + 1
      return
    end if

    nloc = 0
    do j = lbound(x, 2), ubound(x, 2)
      do i = lbound(x, 1), ubound(x, 1)
        do c = 1, 3
          if (transfer(x(i, j)%x(c), 1_i8) /= transfer(y(i, j)%x(c), 1_i8)) then
            nloc = nloc + 1
            d = abs(x(i, j)%x(c) - y(i, j)%x(c))
            if (d > maxabs) maxabs = d
          end if
        end do
      end do
    end do
    if (nloc > 0) then
      write (error_unit, '(A,I0,A)') "  DIFF "//trim(name)//": ", nloc, " elements differ (bit-level)"
      ndiff = ndiff + nloc
    end if
  end subroutine cmp_cc2

  !> Deep-copy the mutated ``t_hydro_ocean_diag`` fields into ``dst``.  The
  !> shallow ``dst = src`` in the caller aliases every ``p_diag`` pointer; this
  !> re-points the fields the AB timestepping writes (plus the adjacent-phase
  !> fields, see module docstring) to fresh copies.  The clone helpers no-op on
  !> unassociated pointers, so config-dependent fields left ``=> NULL()`` are
  !> handled safely.
  subroutine clone_diag_indep(s, d)
    type(t_hydro_ocean_diag), intent(in)    :: s
    type(t_hydro_ocean_diag), intent(inout) :: d
    ! written by solve_free_sfc_ab_mimetic's call tree
    call clone_ptr3(d%vort,             s%vort)
    call clone_ptr3(d%grad,             s%grad)
    call clone_ptr3(d%veloc_adv_horz,   s%veloc_adv_horz)
    call clone_ptr3(d%veloc_adv_vert,   s%veloc_adv_vert)
    call clone_ptr3(d%laplacian_horz,   s%laplacian_horz)
    call clone_ptr3(d%laplacian_vert,   s%laplacian_vert)
    call clone_ptr3(d%press_hyd,        s%press_hyd)
    call clone_ptr3(d%press_grad,       s%press_grad)
    call clone_ptr3(d%vn_pred,          s%vn_pred)
    call clone_ptr3(d%vn_pred_ptp,      s%vn_pred_ptp)
    ! written by the adjacent dycore phases (calc_normal_velocity /
    ! calc_vert_velocity); read-only in THIS call, cloned for forward coverage
    call clone_ptr3(d%vn_time_weighted, s%vn_time_weighted)
    call clone_ptr3(d%w,                s%w)
    call clone_ptr3(d%w_old,            s%w_old)
    call clone_ptr3(d%mass_flx_e,       s%mass_flx_e)
    call clone_ptr3(d%div_mass_flx_c,   s%div_mass_flx_c)
    call clone_ptr3(d%ptp_vn,           s%ptp_vn)
    call clone_ptr2(d%h_e,              s%h_e)
  end subroutine clone_diag_indep

  subroutine free_diag_clone(d)
    type(t_hydro_ocean_diag), intent(inout) :: d
    call free_ptr3(d%vort)
    call free_ptr3(d%grad)
    call free_ptr3(d%veloc_adv_horz)
    call free_ptr3(d%veloc_adv_vert)
    call free_ptr3(d%laplacian_horz)
    call free_ptr3(d%laplacian_vert)
    call free_ptr3(d%press_hyd)
    call free_ptr3(d%press_grad)
    call free_ptr3(d%vn_pred)
    call free_ptr3(d%vn_pred_ptp)
    call free_ptr3(d%vn_time_weighted)
    call free_ptr3(d%w)
    call free_ptr3(d%w_old)
    call free_ptr3(d%mass_flx_e)
    call free_ptr3(d%div_mass_flx_c)
    call free_ptr3(d%ptp_vn)
    call free_ptr2(d%h_e)
  end subroutine free_diag_clone

  !> Deep-copy the mutated ``t_hydro_ocean_aux`` fields: the Adams-Bashforth
  !> term stack (``g_n``/``g_nm1``/``g_nimd``), the surface-equation RHS, and
  !> the velocity boundary conditions the call recomputes.  ``bc_total_top_
  !> potential`` is read-only (tides, computed upstream) and stays shared.
  subroutine clone_aux_indep(s, d)
    type(t_hydro_ocean_aux), intent(in)    :: s
    type(t_hydro_ocean_aux), intent(inout) :: d
    call clone_ptr3(d%g_n,              s%g_n)
    call clone_ptr3(d%g_nm1,            s%g_nm1)
    call clone_ptr3(d%g_nimd,           s%g_nimd)
    call clone_ptr2(d%p_rhs_sfc_eq,     s%p_rhs_sfc_eq)
    call clone_ptr2(d%bc_top_vn,        s%bc_top_vn)
    call clone_ptr2(d%bc_bot_vn,        s%bc_bot_vn)
    call clone_ptr2(d%bc_top_u,         s%bc_top_u)
    call clone_ptr2(d%bc_top_v,         s%bc_top_v)
    call clone_ptr2(d%bc_top_WindStress, s%bc_top_WindStress)
    call clone_cc2(d%bc_top_veloc_cc,   s%bc_top_veloc_cc)
  end subroutine clone_aux_indep

  subroutine free_aux_clone(d)
    type(t_hydro_ocean_aux), intent(inout) :: d
    call free_ptr3(d%g_n)
    call free_ptr3(d%g_nm1)
    call free_ptr3(d%g_nimd)
    call free_ptr2(d%p_rhs_sfc_eq)
    call free_ptr2(d%bc_top_vn)
    call free_ptr2(d%bc_bot_vn)
    call free_ptr2(d%bc_top_u)
    call free_ptr2(d%bc_top_v)
    call free_ptr2(d%bc_top_WindStress)
    call free_cc2(d%bc_top_veloc_cc)
  end subroutine free_aux_clone

  !> Clone ``src`` into ``dst`` with a FULLY INDEPENDENT mutable state.
  !>
  !> Shallow ``dst = src`` first, then repair the aliasing: ``p_prog(:)`` is a
  !> POINTER array (NOT allocatable like the atmosphere's), so the shallow copy
  !> leaves ``dst%p_prog`` pointing at src's slots -- re-pointing a field there
  !> would clobber the original.  A fresh ``p_prog(:)`` target is allocated
  !> (per-slot shallow copy), THEN every prognostic field and every mutated
  !> ``p_diag`` / ``p_aux`` field is re-pointed to a fresh deep copy.
  !>
  !> Stays shared (read-only in the free-surface solve, see module docstring):
  !> ``patch_3D`` / ``operator_coeff`` / ``transport_state`` / the prog
  !> ``tracer_collection`` + ``tracer_ptr`` wrappers (they point into the
  !> ORIGINAL tracers; the tracer transport is not part of this call).
  subroutine clone_ocean_state_indep(src, dst)
    type(t_hydro_ocean_state), intent(in)  :: src
    type(t_hydro_ocean_state), intent(out) :: dst
    integer :: i
    dst = src
    if (associated(src%p_prog)) then
      allocate (dst%p_prog(lbound(src%p_prog, 1):ubound(src%p_prog, 1)))
      do i = lbound(src%p_prog, 1), ubound(src%p_prog, 1)
        dst%p_prog(i) = src%p_prog(i)
        call clone_ptr2(dst%p_prog(i)%h,         src%p_prog(i)%h)
        call clone_ptr2(dst%p_prog(i)%eta_c,     src%p_prog(i)%eta_c)
        call clone_ptr2(dst%p_prog(i)%stretch_c, src%p_prog(i)%stretch_c)
        call clone_ptr3(dst%p_prog(i)%vn,        src%p_prog(i)%vn)
        call clone_ptr4(dst%p_prog(i)%tracer,    src%p_prog(i)%tracer)
      end do
    else
      dst%p_prog => null()
    end if
    call clone_diag_indep(src%p_diag, dst%p_diag)
    call clone_aux_indep(src%p_aux, dst%p_aux)
  end subroutine clone_ocean_state_indep

  !> Release the fresh targets allocated by ``clone_ocean_state_indep``,
  !> including the clone's own ``p_prog(:)`` array (the shared geometry is
  !> untouched).
  subroutine free_ocean_state_clone(dst)
    type(t_hydro_ocean_state), intent(inout) :: dst
    integer :: i
    call free_diag_clone(dst%p_diag)
    call free_aux_clone(dst%p_aux)
    if (.not. associated(dst%p_prog)) return
    do i = lbound(dst%p_prog, 1), ubound(dst%p_prog, 1)
      call free_ptr2(dst%p_prog(i)%h)
      call free_ptr2(dst%p_prog(i)%eta_c)
      call free_ptr2(dst%p_prog(i)%stretch_c)
      call free_ptr3(dst%p_prog(i)%vn)
      call free_ptr4(dst%p_prog(i)%tracer)
    end do
    deallocate (dst%p_prog)
    dst%p_prog => null()
  end subroutine free_ocean_state_clone

  !> Compare the prognostic state of the two runs at time level ``nnew``
  !> (the free-surface solve writes ``h(nnew)``; ``vn``/``tracer``/``eta_c``/
  !> ``stretch_c`` are compared too so a stray write surfaces).  ``ndiff``
  !> returns the total differing-element count; 0 == bit-exact.
  subroutine compare_ocean_prog(a, b, nnew, label, ndiff)
    type(t_hydro_ocean_state), intent(in) :: a, b
    integer, intent(in)                   :: nnew
    character(*), intent(in)              :: label
    integer, intent(out)                  :: ndiff
    real(wp) :: mx
    ndiff = 0
    mx = 0.0_wp
    call cmp_ptr2(a%p_prog(nnew)%h,         b%p_prog(nnew)%h,         label//":h",         ndiff, mx)
    call cmp_ptr3(a%p_prog(nnew)%vn,        b%p_prog(nnew)%vn,        label//":vn",        ndiff, mx)
    call cmp_ptr4(a%p_prog(nnew)%tracer,    b%p_prog(nnew)%tracer,    label//":tracer",    ndiff, mx)
    call cmp_ptr2(a%p_prog(nnew)%eta_c,     b%p_prog(nnew)%eta_c,     label//":eta_c",     ndiff, mx)
    call cmp_ptr2(a%p_prog(nnew)%stretch_c, b%p_prog(nnew)%stretch_c, label//":stretch_c", ndiff, mx)
    if (ndiff == 0) then
      write (error_unit, '(A)') "  [diff] "//trim(label)//": prog(nnew) BIT-EXACT {h,vn,tracer,eta_c,stretch_c}"
    else
      write (error_unit, '(A,I0,A,ES12.5)') "  [diff] "//trim(label)//": ", ndiff, &
        " prognostic elements differ; max|delta|=", mx
    end if
  end subroutine compare_ocean_prog

  !> Compare every cloned ``t_hydro_ocean_diag`` field of the two runs
  !> bit-for-bit.  Fields unassociated in BOTH runs contribute nothing.
  subroutine compare_ocean_diag(a, b, label, ndiff)
    type(t_hydro_ocean_diag), intent(in) :: a, b
    character(*), intent(in)             :: label
    integer, intent(out)                 :: ndiff
    real(wp) :: mx
    ndiff = 0
    mx = 0.0_wp
    call cmp_ptr3(a%vort,             b%vort,             label//":vort",             ndiff, mx)
    call cmp_ptr3(a%grad,             b%grad,             label//":grad",             ndiff, mx)
    call cmp_ptr3(a%veloc_adv_horz,   b%veloc_adv_horz,   label//":veloc_adv_horz",   ndiff, mx)
    call cmp_ptr3(a%veloc_adv_vert,   b%veloc_adv_vert,   label//":veloc_adv_vert",   ndiff, mx)
    call cmp_ptr3(a%laplacian_horz,   b%laplacian_horz,   label//":laplacian_horz",   ndiff, mx)
    call cmp_ptr3(a%laplacian_vert,   b%laplacian_vert,   label//":laplacian_vert",   ndiff, mx)
    call cmp_ptr3(a%press_hyd,        b%press_hyd,        label//":press_hyd",        ndiff, mx)
    call cmp_ptr3(a%press_grad,       b%press_grad,       label//":press_grad",       ndiff, mx)
    call cmp_ptr3(a%vn_pred,          b%vn_pred,          label//":vn_pred",          ndiff, mx)
    call cmp_ptr3(a%vn_pred_ptp,      b%vn_pred_ptp,      label//":vn_pred_ptp",      ndiff, mx)
    call cmp_ptr3(a%vn_time_weighted, b%vn_time_weighted, label//":vn_time_weighted", ndiff, mx)
    call cmp_ptr3(a%w,                b%w,                label//":w",                ndiff, mx)
    call cmp_ptr3(a%w_old,            b%w_old,            label//":w_old",            ndiff, mx)
    call cmp_ptr3(a%mass_flx_e,       b%mass_flx_e,       label//":mass_flx_e",       ndiff, mx)
    call cmp_ptr3(a%div_mass_flx_c,   b%div_mass_flx_c,   label//":div_mass_flx_c",   ndiff, mx)
    call cmp_ptr3(a%ptp_vn,           b%ptp_vn,           label//":ptp_vn",           ndiff, mx)
    call cmp_ptr2(a%h_e,              b%h_e,              label//":h_e",              ndiff, mx)
    if (ndiff == 0) then
      write (error_unit, '(A)') "  [diff] "//trim(label)//": diag BIT-EXACT (all cloned fields)"
    else
      write (error_unit, '(A,I0,A,ES12.5)') "  [diff] "//trim(label)//": ", ndiff, &
        " diag elements differ; max|delta|=", mx
    end if
  end subroutine compare_ocean_diag

  !> Compare every cloned ``t_hydro_ocean_aux`` field of the two runs
  !> bit-for-bit (AB term stack, surface-equation RHS, velocity BCs).
  subroutine compare_ocean_aux(a, b, label, ndiff)
    type(t_hydro_ocean_aux), intent(in) :: a, b
    character(*), intent(in)            :: label
    integer, intent(out)                :: ndiff
    real(wp) :: mx
    ndiff = 0
    mx = 0.0_wp
    call cmp_ptr3(a%g_n,               b%g_n,               label//":g_n",               ndiff, mx)
    call cmp_ptr3(a%g_nm1,             b%g_nm1,             label//":g_nm1",             ndiff, mx)
    call cmp_ptr3(a%g_nimd,            b%g_nimd,            label//":g_nimd",            ndiff, mx)
    call cmp_ptr2(a%p_rhs_sfc_eq,      b%p_rhs_sfc_eq,      label//":p_rhs_sfc_eq",      ndiff, mx)
    call cmp_ptr2(a%bc_top_vn,         b%bc_top_vn,         label//":bc_top_vn",         ndiff, mx)
    call cmp_ptr2(a%bc_bot_vn,         b%bc_bot_vn,         label//":bc_bot_vn",         ndiff, mx)
    call cmp_ptr2(a%bc_top_u,          b%bc_top_u,          label//":bc_top_u",          ndiff, mx)
    call cmp_ptr2(a%bc_top_v,          b%bc_top_v,          label//":bc_top_v",          ndiff, mx)
    call cmp_ptr2(a%bc_top_WindStress, b%bc_top_WindStress, label//":bc_top_WindStress", ndiff, mx)
    call cmp_cc2(a%bc_top_veloc_cc,    b%bc_top_veloc_cc,   label//":bc_top_veloc_cc",   ndiff, mx)
    if (ndiff == 0) then
      write (error_unit, '(A)') "  [diff] "//trim(label)//": aux BIT-EXACT (all cloned fields)"
    else
      write (error_unit, '(A,I0,A,ES12.5)') "  [diff] "//trim(label)//": ", ndiff, &
        " aux elements differ; max|delta|=", mx
    end if
  end subroutine compare_ocean_aux

  !> Snapshot a bare 3D POINTER field (``p_phys_param%a_veloc_v``): the PP
  !> scheme time-smooths it in place through a module pointer the driver cannot
  !> re-point, so the driver saves/restores VALUES instead (see module
  !> docstring).  Same alloc-fresh semantics as ``clone_ptr3``.
  subroutine clone_field3(d, s)
    real(wp), pointer, intent(out) :: d(:, :, :)
    real(wp), pointer, intent(in)  :: s(:, :, :)
    call clone_ptr3(d, s)
  end subroutine clone_field3

  !> Copy ``saved``'s VALUES back into ``dst``'s existing target (no
  !> re-pointing -- the module-level consumers keep their association).
  subroutine restore_field3(dst, saved)
    real(wp), pointer, intent(in) :: dst(:, :, :)
    real(wp), pointer, intent(in) :: saved(:, :, :)
    if (associated(dst) .and. associated(saved)) dst = saved
  end subroutine restore_field3

  subroutine free_field3(d)
    real(wp), pointer, intent(inout) :: d(:, :, :)
    call free_ptr3(d)
  end subroutine free_field3

  !> BIT-EXACT comparison of two bare 3D fields with its own report line.
  subroutine compare_field3(x, y, label, ndiff)
    real(wp), pointer, intent(in) :: x(:, :, :), y(:, :, :)
    character(*), intent(in)      :: label
    integer, intent(out)          :: ndiff
    real(wp) :: mx
    ndiff = 0
    mx = 0.0_wp
    call cmp_ptr3(x, y, label, ndiff, mx)
    if (ndiff == 0) then
      write (error_unit, '(A)') "  [diff] "//trim(label)//": BIT-EXACT"
    else
      write (error_unit, '(A,I0,A,ES12.5)') "  [diff] "//trim(label)//": ", ndiff, &
        " elements differ; max|delta|=", mx
    end if
  end subroutine compare_field3

  !> Optional abort-on-diff: when ``DACE_OCEAN_DIFF_ABORT`` is set to a
  !> non-empty value in the environment, a non-zero ``ndiff`` hard-stops the
  !> run so CI fails at the FIRST divergent call instead of drowning the log.
  !> Default (unset) keeps the report-and-continue behaviour of the
  !> atmosphere driver.
  subroutine ocean_diff_enforce(ndiff, label)
    integer, intent(in)      :: ndiff
    character(*), intent(in) :: label
    character(len=8) :: flag
    integer :: flag_len, stat
    if (ndiff == 0) return
    call get_environment_variable("DACE_OCEAN_DIFF_ABORT", value=flag, length=flag_len, status=stat)
    if (stat /= 0 .or. flag_len == 0) return
    write (error_unit, '(A,I0,A)') "  [diff] "//trim(label)//": ABORTING on ", ndiff, &
      " differing elements (DACE_OCEAN_DIFF_ABORT set)"
    error stop "ocean differential: bit-exact comparison FAILED"
  end subroutine ocean_diff_enforce

end module mo_ocean_diff
