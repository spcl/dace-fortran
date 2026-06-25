! Fortran caller wrapper for the QE h_psi_module::h_psi e2e test.
!
! Two BIND(C) entry points, plus stubs for the QE pruner gaps, mirroring
! the structure of ``vexx_bp_k_gpu_caller.f90``:
!
!   init_h_psi_state_c  -- one-shot initialisation of the QE module-level
!                          state for a controlled, deterministic pass
!                          (noncolin=.false., gamma_only=.false.,
!                          use_gpu=.false., real_space=.false., nkb=0,
!                          use_bgrp_in_hpsi=.false., lda_plus_u=.false.,
!                          scissor=.false., lelfield=.false.,
!                          exx_started=.false., ismeta=.false.).  Every
!                          USE'd module scalar / array the kernel touches
!                          on this path is set or allocated to a shape
!                          compatible with the wrapped ``lda / n / m /
!                          npol`` problem.
!
!   run_h_psi_c         -- forwards the caller-supplied ``psi`` / ``hpsi``
!                          buffers to ``h_psi``.  On the controlled path
!                          the trace is:
!                            h_psi: use_bgrp_in_hpsi=.false. -> call h_psi_
!                            h_psi_: kinetic   -> hpsi = g2kin * psi
!                                    h_psi:pot -> gamma_only=.false.,
!                                                 noncolin=.false. ->
!                                                 vloc_psi_k_acc, with
!                                                 vrs=0 the local-potential
!                                                 contribution is exactly 0
!                                    nkb=0     -> no calbec / add_vuspsi
!                                    meta/U/scissor/exx/elfield all skipped
!                          Expected output: hpsi(i,j) = g2kin(i)*psi(i,j)
!                          for i<=n, 0 for n<i<=lda (no other FP touches it).
!
! On the controlled path ``vloc_psi_k_acc`` reaches ``wave_g2r`` /
! ``wave_r2g``, which call the ``fft_interfaces`` generics ``invfft`` /
! ``fwfft``.  The QE pruner emits an empty ``MODULE fft_interfaces``; the
! parse-test restore replaces that with the upstream ``invfft_y`` /
! ``fwfft_y`` specifics but does not follow them into their bodies, so we
! supply those bodies here.  Because the local potential ``vrs`` is set to
! zero, the FFT-roundtrip output is multiplied by 0 BEFORE it is added to
! hpsi, so the no-op FFT stubs leave hpsi == kinetic exactly.
!
! Note on c-binding: the QE fixture's pruner emits a stub
! ``MODULE iso_c_binding`` that shadows the intrinsic and only exports
! ``c_int8_t`` / ``c_char`` / ``c_double``.  ``c_int`` is therefore not
! available; we use default ``INTEGER`` (kind 4) instead, which is ABI-
! compatible with C ``int`` on the only platforms QE targets.

SUBROUTINE init_h_psi_state_c(lda, n, m, npol_in) &
    BIND(C, NAME="init_h_psi_state_c")
  USE kinds, ONLY: dp
  USE noncollin_module, ONLY: noncolin, npol
  USE control_flags, ONLY: gamma_only, use_gpu, scissor, many_fft
  USE mp_bands, ONLY: use_bgrp_in_hpsi
  USE realus, ONLY: real_space
  USE ldau, ONLY: lda_plus_u
  USE bp, ONLY: lelfield
  USE dft_setting_params, ONLY: exx_started, ismeta
  USE uspp, ONLY: nkb
  USE wvfct, ONLY: npwx, current_k, g2kin
  USE klist, ONLY: igk_k
  USE lsda_mod, ONLY: current_spin, nspin
  USE scf, ONLY: vrs
  USE gvect, ONLY: gstart
  USE fft_base, ONLY: dffts
  USE mytime, ONLY: no
  IMPLICIT NONE
  INTEGER, VALUE :: lda, n, m, npol_in
  INTEGER :: i

  ! --- branch flags: pick the simplest deterministic path ---------------
  noncolin = .FALSE.
  npol = npol_in
  gamma_only = .FALSE.
  use_gpu = .FALSE.
  scissor = .FALSE.
  many_fft = 1
  use_bgrp_in_hpsi = .FALSE.
  real_space = .FALSE.
  lda_plus_u = .FALSE.
  lelfield = .FALSE.
  exx_started = .FALSE.
  ismeta = .FALSE.
  nkb = 0
  gstart = 2
  no = .FALSE.

  current_k = 1
  current_spin = 1
  nspin = 1
  npwx = MAX(lda, 1)

  ! --- kinetic-energy prefactor g2kin (deterministic) -------------------
  ALLOCATE (g2kin(npwx))
  DO i = 1, npwx
    g2kin(i) = REAL(i, dp)
  END DO

  ! --- plane-wave index map: identity igk / nl on a flat grid -----------
  ALLOCATE (igk_k(npwx, 1))
  DO i = 1, npwx
    igk_k(i, 1) = i
  END DO

  ! --- local potential vrs == 0 -> vloc contributes exactly 0 -----------
  ALLOCATE (vrs(npwx, nspin));      vrs = 0.0_dp

  ! --- smooth FFT descriptor: flat identity grid, no task groups --------
  dffts%nnr = npwx
  dffts%ngw = npwx
  dffts%has_task_groups = .FALSE.
  ALLOCATE (dffts%nl(npwx))
  DO i = 1, npwx
    dffts%nl(i) = i
  END DO
END SUBROUTINE init_h_psi_state_c


SUBROUTINE run_h_psi_c(lda, n, m, psi, hpsi) &
    BIND(C, NAME="run_h_psi_c")
  USE kinds, ONLY: dp
  USE noncollin_module, ONLY: npol
  USE h_psi_module, ONLY: h_psi
  IMPLICIT NONE
  INTEGER, VALUE :: lda, n, m
  COMPLEX(KIND=dp), INTENT(INOUT) :: psi(lda * npol, m)
  COMPLEX(KIND=dp), INTENT(INOUT) :: hpsi(lda * npol, m)
  CALL h_psi(lda, n, m, psi, hpsi)
END SUBROUTINE run_h_psi_c


! Wall / CPU time stubs.  ``start_clock`` / ``stop_clock`` (and their
! ``_gpu`` variants) are defined in the fixture; their bodies call out to
! ``f_wall()`` / ``f_tcpu()`` (real timers, declared EXTERNAL in MODULE
! mytime).  Stub them to 0.0 -- the controlled path traverses the clock
! entries but the values are ignored.
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


! ``invfft`` / ``fwfft`` specifics.  The pruner emits an empty
! ``MODULE fft_interfaces``; the parse-test restore re-emits the upstream
! ``invfft_y`` / ``fwfft_y`` specifics but does NOT follow them into their
! bodies, so the linker needs these.  They are REACHED on the controlled
! path (vloc_psi_k_acc -> wave_g2r / wave_r2g), but because ``vrs == 0``
! the FFT-roundtrip result is multiplied by 0 before being added to hpsi,
! so a no-op FFT leaves the kinetic-only result intact.
SUBROUTINE fwfft_y(fft_kind, f, dfft, howmany)
  USE fft_types, ONLY: fft_type_descriptor
  USE fft_param, ONLY: DP
  IMPLICIT NONE
  CHARACTER(LEN=*), INTENT(IN) :: fft_kind
  TYPE(fft_type_descriptor), INTENT(IN) :: dfft
  INTEGER, OPTIONAL, INTENT(IN) :: howmany
  COMPLEX(DP) :: f(:)
END SUBROUTINE fwfft_y


SUBROUTINE invfft_y(fft_kind, f, dfft, howmany)
  USE fft_types, ONLY: fft_type_descriptor
  USE fft_param, ONLY: DP
  IMPLICIT NONE
  CHARACTER(LEN=*), INTENT(IN) :: fft_kind
  TYPE(fft_type_descriptor), INTENT(IN) :: dfft
  INTEGER, OPTIONAL, INTENT(IN) :: howmany
  COMPLEX(DP) :: f(:)
END SUBROUTINE invfft_y


! BLAS symbol stubs.  Unlike the ``vexx_bp_k_gpu`` checkpoint, the ``h_psi``
! USE-closure references the external BLAS kernels ``ddot`` / ``dgemm`` /
! ``dgemv`` / ``dger`` / ``zcopy`` / ``zgemm`` / ``zgemv`` directly (via
! add_vuspsi, vhpsi, calbec, ...), so the reference ``.so`` carries those
! symbols undefined.  ``ctypes.CDLL`` binds eagerly (RTLD_NOW), so an
! unresolved symbol fails the ``dlopen`` even when it is never called.  Every
! one of these sits behind a gate never taken on the controlled path
! (``nkb > 0`` / ``lda_plus_u`` / ``okvan`` / exx), so empty no-op stubs that
! exist purely to resolve the symbols are sufficient -- they are never
! entered.  Implicit-interface externals are matched by name only at link
! time, so the dummy-less stubs bind regardless of the call-site arguments.
SUBROUTINE dgemm
END SUBROUTINE dgemm

SUBROUTINE dgemv
END SUBROUTINE dgemv

SUBROUTINE dger
END SUBROUTINE dger

SUBROUTINE zgemm
END SUBROUTINE zgemm

SUBROUTINE zgemv
END SUBROUTINE zgemv

SUBROUTINE zcopy
END SUBROUTINE zcopy

FUNCTION ddot() RESULT(r)
  USE kinds, ONLY: dp
  IMPLICIT NONE
  REAL(KIND=dp) :: r
  r = 0.0_dp
END FUNCTION ddot
