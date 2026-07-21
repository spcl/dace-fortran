! Fortran caller wrapper for the QE us_exx::newdxx_g e2e test.
!
! This harness drives the ACTIVE ``okvan = .TRUE.`` augmentation path with
! ``flag = 'c'`` (the complex / non-gamma case), so the full computational
! body of ``newdxx_g`` runs and accumulates into ``deexx`` -- unlike the
! degenerate ``okvan = .FALSE.`` early-return no-op.
!
! Two BIND(C) entry points, plus stubs for the QE pruner gaps, mirroring
! the structure of ``h_psi_caller.f90`` / ``vexx_bp_k_gpu_caller.f90``:
!
!   init_newdxx_g_state_c  -- one-shot initialisation of the QE module-level
!                             state for a small, fixed, deterministic
!                             ultrasoft problem:
!                               * okvan   = .TRUE.   (take the active path)
!                               * gamma_only = .FALSE. (required for 'c';
!                                 also selects the becphi_c inner branch and
!                                 disables the gstart==2 gamma correction)
!                               * 1 atom (nat=1) of 1 ultrasoft type
!                                 (upf(1)%tvanp=.TRUE.), nh(1)=2 projectors
!                                 -> nkb=2, ngms=4 G-vectors.
!                             Every module array the kernel reads on the 'c'
!                             path (ofsbeta / ijtoh / nh / upf%tvanp / ityp /
!                             tau / omega / eigts1..3 / mill / qgm / nij_type)
!                             is allocated and filled with fixed values, so
!                             the reference and the SDFG-via-binding see
!                             byte-identical pseudopotential state.
!
!   run_newdxx_g_c          -- builds the matching ``fft_type_descriptor``
!                             (nnr / ngm / nl) and forwards the caller's
!                             ``vc`` / ``deexx`` / ``becphi_c`` buffers to
!                             ``newdxx_g`` with ``flag='c'`` and the OPTIONAL
!                             ``becphi_r`` omitted.  ``deexx`` comes back with
!                             the augmentation contribution added.
!
! ``f_tcpu`` / ``f_wall`` are declared EXTERNAL in ``MODULE mytime`` and are
! reached now (the active path runs ``start_clock`` / ``stop_clock``), so the
! 0.0 stubs both resolve the symbols AND keep the clock arithmetic finite.
! ``newdxx_g`` references no FFT generics or BLAS, so none of those are
! stubbed.
!
! Note on c-binding: the QE fixture's pruner emits a stub
! ``MODULE iso_c_binding`` that shadows the intrinsic and only exports
! ``c_int8_t`` / ``c_char`` / ``c_double``.  ``c_int`` is therefore not
! available; we use default ``INTEGER`` (kind 4) instead, which is ABI-
! compatible with C ``int`` on the only platforms QE targets.
!
! The fixed problem dimensions are exported as PARAMETERs so the Python
! side and both wrappers agree without magic numbers:
!   NNR = 4, NGMS = 4, NKB = 2.

SUBROUTINE init_newdxx_g_state_c() &
    BIND(C, NAME="init_newdxx_g_state_c")
  USE kinds, ONLY: dp
  USE uspp, ONLY: okvan, nkb, ofsbeta, ijtoh
  USE control_flags, ONLY: gamma_only
  USE ions_base, ONLY: nat, ityp, tau
  USE cell_base, ONLY: omega
  USE uspp_param, ONLY: nh, upf
  USE gvect, ONLY: eigts1, eigts2, eigts3, mill, gstart
  USE us_exx, ONLY: qgm, nij_type
  IMPLICIT NONE
  INTEGER, PARAMETER :: ngms = 4
  INTEGER :: ig, ih, jh, col

  ! --- branch flags: active 'c' (complex, non-gamma) path ---------------
  okvan = .TRUE.
  gamma_only = .FALSE.
  gstart = 2            ! irrelevant: gamma correction is gated on gamma_only

  ! --- geometry: 1 atom, 1 ultrasoft type, nh=2 projectors --------------
  nat = 1
  nkb = 2               ! = nat * nh(1)
  omega = 2.0_dp

  IF (ALLOCATED(ityp)) DEALLOCATE(ityp)
  ALLOCATE(ityp(1));            ityp(1) = 1
  IF (ALLOCATED(tau)) DEALLOCATE(tau)
  ALLOCATE(tau(3, 1));         tau(:, 1) = [0.5_dp, 0.6_dp, 0.7_dp]
  IF (ALLOCATED(nh)) DEALLOCATE(nh)
  ALLOCATE(nh(1));             nh(1) = 2
  IF (ALLOCATED(ofsbeta)) DEALLOCATE(ofsbeta)
  ALLOCATE(ofsbeta(1));        ofsbeta(1) = 0
  IF (ALLOCATED(upf)) DEALLOCATE(upf)
  ALLOCATE(upf(1));            upf(1)%tvanp = .TRUE.

  ! ijtoh(ih,jh,nt): packed (ih,jh) -> column-within-type, here the full
  ! nh*nh = 4 map so every (ih,jh) addresses a distinct qgm column.
  IF (ALLOCATED(ijtoh)) DEALLOCATE(ijtoh)
  ALLOCATE(ijtoh(2, 2, 1))
  DO ih = 1, 2
    DO jh = 1, 2
      ijtoh(ih, jh, 1) = (ih - 1) * 2 + jh
    END DO
  END DO
  IF (ALLOCATED(nij_type)) DEALLOCATE(nij_type)
  ALLOCATE(nij_type(1));       nij_type(1) = 0

  ! --- structure factors, indexed by mill(d,ig); flat lower-bound-1 grid -
  IF (ALLOCATED(mill)) DEALLOCATE(mill)
  ALLOCATE(mill(3, ngms))
  DO ig = 1, ngms
    mill(:, ig) = ig
  END DO
  IF (ALLOCATED(eigts1)) DEALLOCATE(eigts1)
  IF (ALLOCATED(eigts2)) DEALLOCATE(eigts2)
  IF (ALLOCATED(eigts3)) DEALLOCATE(eigts3)
  ALLOCATE(eigts1(ngms, 1), eigts2(ngms, 1), eigts3(ngms, 1))
  DO ig = 1, ngms
    eigts1(ig, 1) = CMPLX(1.0_dp, 0.10_dp * ig, kind = dp)
    eigts2(ig, 1) = CMPLX(0.9_dp, 0.20_dp * ig, kind = dp)
    eigts3(ig, 1) = CMPLX(0.8_dp, 0.30_dp * ig, kind = dp)
  END DO

  ! --- augmentation form factors qgm(ngms, ncol), ncol = 4 --------------
  IF (ALLOCATED(qgm)) DEALLOCATE(qgm)
  ALLOCATE(qgm(ngms, 4))
  DO col = 1, 4
    DO ig = 1, ngms
      qgm(ig, col) = CMPLX(0.10_dp * ig + 0.01_dp * col, 0.02_dp * ig - 0.03_dp * col, kind = dp)
    END DO
  END DO
END SUBROUTINE init_newdxx_g_state_c


SUBROUTINE run_newdxx_g_c(nnr, ngms, nkb_in, vc, deexx, becphi_c) &
    BIND(C, NAME="run_newdxx_g_c")
  USE kinds, ONLY: dp
  USE fft_types, ONLY: fft_type_descriptor
  USE us_exx, ONLY: newdxx_g
  IMPLICIT NONE
  INTEGER, VALUE :: nnr, ngms, nkb_in
  COMPLEX(KIND=dp), INTENT(INOUT) :: vc(nnr)
  COMPLEX(KIND=dp), INTENT(INOUT) :: deexx(nkb_in)
  COMPLEX(KIND=dp), INTENT(IN) :: becphi_c(nkb_in)
  TYPE(fft_type_descriptor) :: dfftt
  REAL(KIND=dp) :: xk(3), xkq(3)
  INTEGER :: ig

  dfftt%nnr = nnr
  dfftt%ngm = ngms
  ALLOCATE(dfftt%nl(ngms))
  DO ig = 1, ngms
    dfftt%nl(ig) = ig
  END DO
  xk(:) = [0.1_dp, 0.2_dp, 0.3_dp]
  xkq(:) = [0.0_dp, 0.0_dp, 0.0_dp]
  ! flag='c' -> complex (non-gamma) branch; becphi_r omitted (OPTIONAL).
  CALL newdxx_g(dfftt, vc, xkq, xk, 'c', deexx, becphi_c=becphi_c)
  DEALLOCATE(dfftt%nl)
END SUBROUTINE run_newdxx_g_c


! Wall / CPU time stubs.  ``start_clock`` / ``stop_clock`` are defined in
! the fixture and ARE reached on the active path; their bodies call out to
! ``f_wall()`` / ``f_tcpu()`` (real timers, declared EXTERNAL in MODULE
! mytime).  Stub them to 0.0 -- this resolves the otherwise-undefined
! symbols for the ctypes ``dlopen`` and keeps the (ignored) clock totals
! finite.
FUNCTION f_tcpu() RESULT(t)
  USE kinds, ONLY: dp
  IMPLICIT NONE
  REAL(KIND=dp) :: t
  t = 0.0_dp
END FUNCTION f_tcpu

FUNCTION f_wall() RESULT(t)
  USE kinds, ONLY: dp
  IMPLICIT NONE
  REAL(KIND=dp) :: t
  t = 0.0_dp
END FUNCTION f_wall
