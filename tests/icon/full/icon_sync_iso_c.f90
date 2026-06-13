! ICON sync-patch iso_c wrapper.
!
! Provides ``bind(c)`` entry points that the dycore SDFG calls in place
! of ICON's generic-interface ``sync_patch_array`` / ``_mult`` /
! ``_mult_mixprec`` halo-exchange procedures.  The bridge cannot lower
! ``mo_sync``'s polymorphic ``CLASS(t_comm_pattern)`` dispatch chain, so
! we externalise the call at the iso_c boundary and forward to the
! real Fortran routine here.  The wrapper is intentionally thin:
!
!   * It takes a domain id (1-based) rather than a ``TYPE(t_patch)``
!     descriptor across the C ABI -- ICON's own ``p_patch(:)``
!     module-level array indexes patches by id, and exposing the
!     descriptor would require a flatten-and-rebuild pair on every
!     call.
!   * Arrays come across as ``(int extents..., void* data)`` flat
!     pointers; the wrapper reconstructs a Fortran view via
!     ``c_f_pointer`` and dispatches into the generic.
!   * ``lacc`` arrives as ``c_bool`` (one byte) and casts to
!     ``LOGICAL(c_bool)`` at the call site so ICON's strict
!     ``LOGICAL(4)`` dummies see the right ABI.
!
! Building:
!   gfortran -c -fPIC \
!     -I<ICON_BUILD>/mod \
!     -I<ICON_BUILD>/externals/fortran-support/build/src/mod \
!     -I<ICON_BUILD>/externals/iconmath/build/src/support/mod \
!     icon_sync_iso_c.f90 -o icon_sync_iso_c.o
!   gfortran -shared -fPIC icon_sync_iso_c.o -o libicon_sync_iso_c.so
!
! Each ``bind(c, name=...)`` symbol below corresponds to one of the
! ``keep_external`` registrations in ``test_dycore_from_icon_source.py``.
MODULE icon_sync_iso_c
  USE iso_c_binding, ONLY: c_int, c_double, c_ptr, c_bool, c_f_pointer
  USE mo_kind,          ONLY: dp
  USE mo_model_domain,  ONLY: p_patch
  USE mo_sync,          ONLY: sync_patch_array, sync_patch_array_mult, &
                              sync_patch_array_mult_mixprec
  IMPLICIT NONE
  PUBLIC

CONTAINS

  ! --- Single-field 3D REAL(dp) sync ------------------------------------
  ! Resolves to ``sync_patch_array_3d_dp`` after generic dispatch.
  SUBROUTINE sync_patch_array_3d_dp_c(typ, patch_id, d0, d1, d2, arr_p, lacc) &
       BIND(c, name='sync_patch_array_3d_dp_c')
    INTEGER(c_int), VALUE :: typ
    INTEGER(c_int), VALUE :: patch_id
    INTEGER(c_int), VALUE :: d0, d1, d2
    TYPE(c_ptr),    VALUE :: arr_p
    LOGICAL(c_bool), VALUE :: lacc
    REAL(dp), POINTER :: arr(:,:,:)
    CALL c_f_pointer(arr_p, arr, [INT(d0), INT(d1), INT(d2)])
    CALL sync_patch_array(typ, p_patch(patch_id), arr, &
                          lacc=LOGICAL(lacc, kind=4))
  END SUBROUTINE sync_patch_array_3d_dp_c

  ! --- Two-field mixed-precision sync -----------------------------------
  ! Wraps ``sync_patch_array_mult_mixprec``.  Only the (n_sp=1, n_dp=1)
  ! shape that solve_nh actually exercises is exposed; richer shapes can
  ! be added as needed.
  SUBROUTINE sync_patch_array_mult_mixprec_1sp_1dp_c( &
        typ, patch_id, &
        sp_d0, sp_d1, sp_d2, sp_p, &
        dp_d0, dp_d1, dp_d2, dp_p, &
        lacc) &
       BIND(c, name='sync_patch_array_mult_mixprec_1sp_1dp_c')
    USE mo_kind, ONLY: sp
    INTEGER(c_int), VALUE :: typ
    INTEGER(c_int), VALUE :: patch_id
    INTEGER(c_int), VALUE :: sp_d0, sp_d1, sp_d2
    TYPE(c_ptr),    VALUE :: sp_p
    INTEGER(c_int), VALUE :: dp_d0, dp_d1, dp_d2
    TYPE(c_ptr),    VALUE :: dp_p
    LOGICAL(c_bool), VALUE :: lacc
    REAL(sp), POINTER :: f3din_sp(:,:,:)
    REAL(dp), POINTER :: f3din_dp(:,:,:)
    CALL c_f_pointer(sp_p, f3din_sp, [INT(sp_d0), INT(sp_d1), INT(sp_d2)])
    CALL c_f_pointer(dp_p, f3din_dp, [INT(dp_d0), INT(dp_d1), INT(dp_d2)])
    CALL sync_patch_array_mult_mixprec(typ, p_patch(patch_id), 1, 1, &
                                       f3din1_sp=f3din_sp, &
                                       f3din1_dp=f3din_dp, &
                                       lacc=LOGICAL(lacc, kind=4))
  END SUBROUTINE sync_patch_array_mult_mixprec_1sp_1dp_c

  ! --- Two-field DP sync ------------------------------------------------
  ! Wraps the ``sync_patch_array_mult`` overload that solve_nh uses at
  ! line 1667 (``2, lacc, f3din1=vn, f3din2=z_rho_e``).
  SUBROUTINE sync_patch_array_mult_2_dp_c( &
        typ, patch_id, &
        a_d0, a_d1, a_d2, a_p, &
        b_d0, b_d1, b_d2, b_p, &
        lacc) &
       BIND(c, name='sync_patch_array_mult_2_dp_c')
    INTEGER(c_int), VALUE :: typ
    INTEGER(c_int), VALUE :: patch_id
    INTEGER(c_int), VALUE :: a_d0, a_d1, a_d2
    TYPE(c_ptr),    VALUE :: a_p
    INTEGER(c_int), VALUE :: b_d0, b_d1, b_d2
    TYPE(c_ptr),    VALUE :: b_p
    LOGICAL(c_bool), VALUE :: lacc
    REAL(dp), POINTER :: f3din1(:,:,:), f3din2(:,:,:)
    CALL c_f_pointer(a_p, f3din1, [INT(a_d0), INT(a_d1), INT(a_d2)])
    CALL c_f_pointer(b_p, f3din2, [INT(b_d0), INT(b_d1), INT(b_d2)])
    CALL sync_patch_array_mult(typ, p_patch(patch_id), 2, &
                               f3din1=f3din1, f3din2=f3din2, &
                               lacc=LOGICAL(lacc, kind=4))
  END SUBROUTINE sync_patch_array_mult_2_dp_c

  ! --- Three-field DP sync ----------------------------------------------
  ! solve_nh line 2776: ``sync_patch_array_mult(SYNC_C, p_patch, 3, ...
  ! f3din1=rho, f3din2=theta, f3din3=exner)``.
  SUBROUTINE sync_patch_array_mult_3_dp_c( &
        typ, patch_id, &
        a_d0, a_d1, a_d2, a_p, &
        b_d0, b_d1, b_d2, b_p, &
        c_d0, c_d1, c_d2, c_p, &
        lacc) &
       BIND(c, name='sync_patch_array_mult_3_dp_c')
    INTEGER(c_int), VALUE :: typ
    INTEGER(c_int), VALUE :: patch_id
    INTEGER(c_int), VALUE :: a_d0, a_d1, a_d2
    TYPE(c_ptr),    VALUE :: a_p
    INTEGER(c_int), VALUE :: b_d0, b_d1, b_d2
    TYPE(c_ptr),    VALUE :: b_p
    INTEGER(c_int), VALUE :: c_d0, c_d1, c_d2
    TYPE(c_ptr),    VALUE :: c_p
    LOGICAL(c_bool), VALUE :: lacc
    REAL(dp), POINTER :: f3din1(:,:,:), f3din2(:,:,:), f3din3(:,:,:)
    CALL c_f_pointer(a_p, f3din1, [INT(a_d0), INT(a_d1), INT(a_d2)])
    CALL c_f_pointer(b_p, f3din2, [INT(b_d0), INT(b_d1), INT(b_d2)])
    CALL c_f_pointer(c_p, f3din3, [INT(c_d0), INT(c_d1), INT(c_d2)])
    CALL sync_patch_array_mult(typ, p_patch(patch_id), 3, &
                               f3din1=f3din1, f3din2=f3din2, f3din3=f3din3, &
                               lacc=LOGICAL(lacc, kind=4))
  END SUBROUTINE sync_patch_array_mult_3_dp_c

END MODULE icon_sync_iso_c
