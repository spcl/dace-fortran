! Fortran caller wrapper for the QE exx_bp::vexx_bp_k_gpu e2e test.
!
!   init_vexx_bp_k_gpu_state_c      -- one-shot NO-OP-path module-state init
!                                      (noncolin/okvan/okpaw=.false., negrp=1,
!                                      nqs=0, nibands=[0]): setup and vexxmain
!                                      loops skip, result_sum no-ops, hpsi =
!                                      hpsi_d identity copy (hpsi_out==hpsi_in).
!   init_vexx_bp_k_gpu_perf_state_c -- REAL exchange-path state (below).
!   init_vexx_bp_k_gpu_uspp_state_c -- the real exchange path with okvan and a
!                                      synthetic ultrasoft pseudopotential.
!   run_vexx_bp_k_gpu_c             -- forwards psi/hpsi to vexx_bp_k_gpu with
!                                      the OPTIONAL becpsi omitted.
!   run_vexx_bp_k_gpu_uspp_c        -- the same with becpsi (fill_uspp_becpsi).
!
! fwfft_y/invfft_y: the pruner emits an empty MODULE fft_interfaces; the
! parse-test restore re-adds the interface specifics but not their bodies.
! Never entered on the no-op path.  On the real path the DaCe side replaces
! the call sites with FFT library nodes, so these bodies run only in the
! gfortran reference and compute the same flat DFT (flat_dft_stub).
!
! The pruner's stub MODULE iso_c_binding shadows the intrinsic and lacks
! c_int; default INTEGER is ABI-compatible with C int on QE's platforms.

SUBROUTINE init_vexx_bp_k_gpu_state_c(lda, n, m, npol_in, max_ibands_in) &
    BIND(C, NAME="init_vexx_bp_k_gpu_state_c")
  USE noncollin_module, ONLY: noncolin, npol
  USE mp_exx, ONLY: max_ibands, max_pairs, jblock, nibands, ibands, &
                    iexx_istart, iexx_iend, iexx_istart_d, &
                    all_start, all_end, egrp_pairs
  USE uspp, ONLY: nkb
  USE klist, ONLY: nkstot, xk
  USE wvfct, ONLY: current_k, npwx
  USE cell_base, ONLY: omega
  USE exx_base, ONLY: dfftt, nqs, exxbuff, exxbuff_d, &
                      x_occupation, x_occupation_d, gt, &
                      xkq_collect, index_xk, index_xkq
  USE exx_bp_utils, ONLY: igk_exx, igk_exx_d
  IMPLICIT NONE
  INTEGER, VALUE :: lda, n, m, npol_in, max_ibands_in

  noncolin = .FALSE.
  npol = npol_in
  max_ibands = max_ibands_in
  max_pairs = 1
  jblock = 1

  ALLOCATE (nibands(1));           nibands = 0
  ALLOCATE (ibands(max_ibands, 1)); ibands = 0
  ALLOCATE (iexx_istart(1));       iexx_istart = 0
  ALLOCATE (iexx_iend(1));         iexx_iend = 0
  ALLOCATE (iexx_istart_d(1));     iexx_istart_d = 0
  ALLOCATE (all_start(1));         all_start = 1
  ALLOCATE (all_end(1));           all_end = 0
  ALLOCATE (egrp_pairs(1, 1, 1));  egrp_pairs = 0

  nkb = 0
  nkstot = 1
  xk(:, 1) = 0.0D0
  npwx = MAX(lda, 1)
  current_k = 1
  omega = 1.0D0

  dfftt%ngm = 1
  dfftt%nnr = 1
  ALLOCATE (dfftt%nl(1));          dfftt%nl = 1

  nqs = 0
  ALLOCATE (exxbuff(1, 1, 1));     exxbuff = (0.0D0, 0.0D0)
  ALLOCATE (exxbuff_d(1, 1, 1));   exxbuff_d = (0.0D0, 0.0D0)
  ALLOCATE (x_occupation(1, 1));   x_occupation = 0.0D0
  ALLOCATE (x_occupation_d(1, 1)); x_occupation_d = 0.0D0
  ALLOCATE (gt(3, 1));             gt = 0.0D0
  ALLOCATE (xkq_collect(3, 1));    xkq_collect = 0.0D0
  ALLOCATE (index_xk(1));          index_xk = 1
  ALLOCATE (index_xkq(1, 1));      index_xkq = 1

  ALLOCATE (igk_exx(npwx, 1));     igk_exx = 1
  ALLOCATE (igk_exx_d(npwx, 1));   igk_exx_d = 1
END SUBROUTINE init_vexx_bp_k_gpu_state_c


! REAL exchange-path init (the no-op init above makes the kernel an identity
! copy: nqs=0 kills vexxmain, nibands=0 kills the band loops).
! Colinear non-USPP path: noncolin/okvan/okpaw/tqr/negrp/many_fft keep their
! module defaults.  One-shot per process, no ALLOCATED guards (same
! convention as the no-op init); never mix the two inits.
SUBROUTINE init_vexx_bp_k_gpu_perf_state_c(lda, n, m, npol_in, max_ibands_in, &
                                           nnr, nqs_in, jblock_in) &
    BIND(C, NAME="init_vexx_bp_k_gpu_perf_state_c")
  USE noncollin_module, ONLY: noncolin, npol
  USE mp_exx, ONLY: max_ibands, max_pairs, jblock, nibands, ibands, &
                    iexx_istart, iexx_iend, iexx_istart_d, iexx_start, &
                    all_start, all_end, egrp_pairs
  USE uspp, ONLY: nkb
  USE klist, ONLY: nks, nkstot, xk
  USE wvfct, ONLY: current_k, npwx
  USE cell_base, ONLY: at, omega, tpiba, tpiba2
  USE exx_base, ONLY: dfftt, exxalfa, nqs, exxbuff, exxbuff_d, &
                      g2_convolution, gt, index_xk, index_xkq, &
                      x_occupation, x_occupation_d, xkq_collect
  USE exx_bp, ONLY: coulomb_done, coulomb_fac
  USE exx_bp_utils, ONLY: igk_exx, igk_exx_d
  IMPLICIT NONE
  INTEGER, VALUE :: lda, n, m, npol_in, max_ibands_in, nnr, nqs_in, jblock_in
  INTEGER :: i, ig, iq, ir, jb, ngm, spacing

  IF (m < 1 .OR. m > max_ibands_in .OR. n < 1 .OR. n > lda .OR. &
      nnr < 1 .OR. nqs_in < 1 .OR. jblock_in < 1 .OR. npol_in /= 1) THEN
    STOP 'init_vexx_bp_k_gpu_perf_state_c: bad dims (need 1<=m<=max_ibands, 1<=n<=lda, npol=1)'
  END IF
  ngm = n

  noncolin = .FALSE.
  npol = npol_in
  max_ibands = max_ibands_in
  jblock = jblock_in
  ! Default 0 would make exxbuff column jbnd-all_start+iexx_start hit 0 = OOB.
  iexx_start = 1
  max_pairs = m * m

  ALLOCATE (nibands(1));                nibands = m
  ALLOCATE (ibands(max_ibands, 1));     ibands = 0
  DO i = 1, m
    ibands(i, 1) = i
  END DO
  ! All-pairs exchange: the pair scan then yields jstart=1, jend=m per band.
  ALLOCATE (egrp_pairs(2, max_pairs, 1))
  DO i = 1, m
    DO jb = 1, m
      egrp_pairs(1, (i - 1) * m + jb, 1) = i
      egrp_pairs(2, (i - 1) * m + jb, 1) = jb
    END DO
  END DO
  ALLOCATE (iexx_istart(1));            iexx_istart = 1
  ALLOCATE (iexx_iend(1));              iexx_iend = m
  ALLOCATE (iexx_istart_d(1));          iexx_istart_d = 1
  ALLOCATE (all_start(1));              all_start = 1
  ALLOCATE (all_end(1));                all_end = m

  nkb = 0
  nks = 1
  nkstot = 1
  xk(:, 1) = 0.0D0
  npwx = lda
  current_k = 1
  omega = 1.0D0
  tpiba = 1.0D0
  tpiba2 = 1.0D0
  at = 0.0D0
  at(1, 1) = 1.0D0
  at(2, 2) = 1.0D0
  at(3, 3) = 1.0D0
  exxalfa = 0.25D0

  dfftt%ngm = ngm
  dfftt%nnr = nnr
  ! Evenly spread injective-ish PW -> grid scatter, always in 1..nnr.
  spacing = MAX(nnr / ngm, 1)
  ALLOCATE (dfftt%nl(ngm), dfftt%nl_d(ngm))
  DO ig = 1, ngm
    dfftt%nl(ig) = 1 + MOD((ig - 1) * spacing, nnr)
  END DO
  dfftt%nl_d = dfftt%nl

  nqs = nqs_in
  ALLOCATE (exxbuff(nnr, m, nqs), exxbuff_d(nnr, m, nqs))
  ! Deterministic nonzero fill that differs per grid point, band and q, so a wrong band or q index changes
  ! the result; zeros/denormals could take different FP paths.
  DO iq = 1, nqs
    DO jb = 1, m
      DO ir = 1, nnr
        exxbuff(ir, jb, iq) = CMPLX(0.5D0 + 0.45D0 * DBLE(MOD(ir + 3 * jb + 7 * iq, 31)) / 31.0D0, &
                                    0.25D0 - 0.5D0 * DBLE(MOD(2 * ir + 5 * jb + iq, 17)) / 17.0D0, KIND(0.0D0))
      END DO
    END DO
  END DO
  exxbuff_d = exxbuff
  ALLOCATE (x_occupation(m, nqs), x_occupation_d(m, nqs))
  DO jb = 1, m
    x_occupation(jb, :) = 1.0D0 / DBLE(jb)
  END DO
  x_occupation_d = x_occupation
  ! A distinct k+q per q gives each q its own Coulomb factors; 0.03*iq never cancels gt(1,ig) = 0.1*ig.
  ALLOCATE (xkq_collect(3, nqs));       xkq_collect = 0.0D0
  DO iq = 1, nqs
    xkq_collect(1, iq) = 0.03D0 * DBLE(iq)
  END DO
  ALLOCATE (index_xk(nqs));             index_xk = 1
  ALLOCATE (index_xkq(1, nqs))
  DO iq = 1, nqs
    index_xkq(1, iq) = iq
  END DO
  ! Nonzero g-vectors keep qq > eps_qdiv, so the Coulomb factors stay finite.
  ALLOCATE (gt(3, ngm))
  DO ig = 1, ngm
    gt(1, ig) = 0.1D0 * DBLE(ig)
    gt(2, ig) = 0.0D0
    gt(3, ig) = 0.0D0
  END DO

  ALLOCATE (igk_exx(npwx, 1), igk_exx_d(npwx, 1))
  DO ig = 1, npwx
    igk_exx(ig, 1) = 1 + MOD(ig - 1, ngm)
  END DO
  igk_exx_d = igk_exx

  ! Precompute Coulomb factors on the host and mark them done, so the
  ! in-kernel g2_convolution_all takes its early RETURN (cost stays untimed).
  ALLOCATE (coulomb_fac(ngm, nqs, nks))
  ALLOCATE (coulomb_done(nqs, nks));    coulomb_done = .TRUE.
  DO iq = 1, nqs
    CALL g2_convolution(ngm, gt, xk(:, 1), xkq_collect(:, iq), coulomb_fac(:, iq, 1))
  END DO
END SUBROUTINE init_vexx_bp_k_gpu_perf_state_c


SUBROUTINE run_vexx_bp_k_gpu_c(lda, n, m, psi, hpsi) &
    BIND(C, NAME="run_vexx_bp_k_gpu_c")
  USE kinds, ONLY: dp
  USE noncollin_module, ONLY: npol
  USE mp_exx, ONLY: max_ibands
  USE exx_bp, ONLY: vexx_bp_k_gpu
  IMPLICIT NONE
  INTEGER, VALUE :: lda, n, m
  COMPLEX(KIND=dp), INTENT(INOUT) :: psi(lda * npol, max_ibands)
  COMPLEX(KIND=dp), INTENT(INOUT) :: hpsi(lda * npol, max_ibands)
  CALL vexx_bp_k_gpu(lda, n, m, psi, hpsi)
END SUBROUTINE run_vexx_bp_k_gpu_c


! USPP exchange path: the real exchange-path state above plus a synthetic ultrasoft pseudopotential, so the
! augmentation (qvan_init, addusxx_g, newdxx_g, add_nlxx_pot) runs. Two species, three atoms: species 1 is
! ultrasoft with an s and a p projector (nh = 4), species 2 norm-conserving with two s projectors (nh = 2), so the
! per-species branches take both arms. The tables are smooth deterministic fills, not physical data: they only
! have to be the same on both sides. n > 256 gives addusxx_g/newdxx_g two blocks of 256 plane waves.
SUBROUTINE init_vexx_bp_k_gpu_uspp_state_c(lda, n, m, npol_in, max_ibands_in, &
                                           nnr, nqs_in, jblock_in) &
    BIND(C, NAME="init_vexx_bp_k_gpu_uspp_state_c")
  USE kinds, ONLY: dp
  USE control_flags, ONLY: gamma_only
  USE uspp, ONLY: nkb, okvan, indv, nhtol, nhtolm, ijtoh, ofsbeta, ap, lpx, lpl
  USE uspp_param, ONLY: nsp, upf, nh, nhm, nbetam, lmaxkb, lmaxq
  USE ions_base, ONLY: nat, ityp, tau
  USE gvect, ONLY: g, mill, eigts1, eigts2, eigts3, gstart
  USE fft_base, ONLY: dfftp
  USE qrad_mod, ONLY: tab_qrad
  USE beta_mod, ONLY: nqx, tab_beta
  USE us_exx, ONLY: becxx
  USE exx_base, ONLY: nqs
  IMPLICIT NONE
  INTEGER, VALUE :: lda, n, m, npol_in, max_ibands_in, nnr, nqs_in, jblock_in
  INTEGER, PARAMETER :: nr = 3, nqxq = 400
  INTEGER :: na, nt, ih, jh, ijh, ig, i, ivl, jvl, iq, jb, nb, l
  INTERFACE
    SUBROUTINE init_vexx_bp_k_gpu_perf_state_c(lda, n, m, npol_in, max_ibands_in, nnr, nqs_in, jblock_in) &
        BIND(C, NAME="init_vexx_bp_k_gpu_perf_state_c")
      INTEGER, VALUE :: lda, n, m, npol_in, max_ibands_in, nnr, nqs_in, jblock_in
    END SUBROUTINE
  END INTERFACE

  CALL init_vexx_bp_k_gpu_perf_state_c(lda, n, m, npol_in, max_ibands_in, nnr, nqs_in, jblock_in)
  okvan = .TRUE.
  gamma_only = .FALSE.
  gstart = 2

  nsp = 2
  nat = 3
  ALLOCATE (ityp(nat), tau(3, nat))
  ityp = (/1, 2, 1/)
  DO na = 1, nat
    tau(:, na) = (/0.11D0 * na, 0.07D0 * na * na, -0.05D0 * na/)
  END DO
  ALLOCATE (upf(nsp))
  upf(1)%tvanp = .TRUE.
  upf(2)%tvanp = .FALSE.
  upf(:)%nbeta = 2
  nbetam = 2
  lmaxkb = 1
  lmaxq = 2 * lmaxkb + 1
  ALLOCATE (nh(nsp))
  nh = (/4, 2/)
  nhm = 4
  ALLOCATE (indv(nhm, nsp), nhtol(nhm, nsp), nhtolm(nhm, nsp))
  indv = 1; nhtol = 0; nhtolm = 1
  indv(1:4, 1) = (/1, 2, 2, 2/)
  nhtol(1:4, 1) = (/0, 1, 1, 1/)
  nhtolm(1:4, 1) = (/1, 2, 3, 4/)
  indv(1:2, 2) = (/1, 2/)
  ! ijtoh numbers the pairs ih <= jh as qvan_init does, symmetric in ih, jh
  ALLOCATE (ijtoh(nhm, nhm, nsp))
  ijtoh = 0
  DO nt = 1, nsp
    ijh = 0
    DO ih = 1, nh(nt)
      DO jh = ih, nh(nt)
        ijh = ijh + 1
        ijtoh(ih, jh, nt) = ijh
        ijtoh(jh, ih, nt) = ijh
      END DO
    END DO
  END DO
  ALLOCATE (ofsbeta(nat))
  nkb = 0
  DO na = 1, nat
    ofsbeta(na) = nkb
    nkb = nkb + nh(ityp(na))
  END DO
  ! Two angular-momentum products per pair, each an Ylm index within lmaxq**2
  lpx = 0; lpl = 1; ap = 0.0D0
  DO ivl = 1, 4
    DO jvl = 1, 4
      lpx(ivl, jvl) = 2
      lpl(ivl, jvl, 1) = 1
      lpl(ivl, jvl, 2) = 1 + MOD(ivl + jvl, lmaxq * lmaxq)
      DO i = 1, lmaxq * lmaxq
        ap(i, ivl, jvl) = 0.1D0 * DBLE(i + ivl) / DBLE(jvl + 1)
      END DO
    END DO
  END DO
  ALLOCATE (tab_qrad(nqxq, nbetam * (nbetam + 1) / 2, lmaxq, nsp))
  DO nt = 1, nsp
    DO l = 1, lmaxq
      DO nb = 1, nbetam * (nbetam + 1) / 2
        DO i = 1, nqxq
          tab_qrad(i, nb, l, nt) = COS(0.013D0 * i * nb + 0.2D0 * l) / DBLE(nt + l)
        END DO
      END DO
    END DO
  END DO
  nqx = nqxq
  ALLOCATE (tab_beta(nqx, nbetam, nsp))
  DO nt = 1, nsp
    DO nb = 1, nbetam
      DO i = 1, nqx
        tab_beta(i, nb, nt) = EXP(-0.004D0 * i * nb) / DBLE(nt)
      END DO
    END DO
  END DO

  ! g-vectors small enough that every interpolation stays inside its table (|q| * tpiba / dq < nqxq - 4)
  ALLOCATE (g(3, n), mill(3, n))
  DO ig = 1, n
    mill(:, ig) = (/MOD(ig, 2 * nr + 1) - nr, MOD(ig / 7, 2 * nr + 1) - nr, MOD(ig / 49, 2 * nr + 1) - nr/)
    g(:, ig) = 0.2D0 * DBLE(mill(:, ig)) + (/0.001D0 * ig, 0.0D0, 0.0D0/)
  END DO
  dfftp%nr1 = nr; dfftp%nr2 = nr; dfftp%nr3 = nr
  ALLOCATE (eigts1(-nr:nr, nat), eigts2(-nr:nr, nat), eigts3(-nr:nr, nat))
  DO na = 1, nat
    DO i = -nr, nr
      eigts1(i, na) = CMPLX(COS(0.3D0 * i * na), -SIN(0.3D0 * i * na), KIND=dp)
      eigts2(i, na) = CMPLX(COS(0.2D0 * i + na), -SIN(0.2D0 * i + na), KIND=dp)
      eigts3(i, na) = CMPLX(COS(0.1D0 * i * i - na), -SIN(0.1D0 * i * i - na), KIND=dp)
    END DO
  END DO

  ! becxx(ikq)%k(nkb, m): one projection set per q-point, distinct per projector, band and q
  ALLOCATE (becxx(nqs))
  DO iq = 1, nqs
    ALLOCATE (becxx(iq)%k(nkb, m))
    DO jb = 1, m
      DO i = 1, nkb
        becxx(iq)%k(i, jb) = CMPLX(0.3D0 + 0.1D0 * MOD(i + 2 * jb + iq, 5), 0.05D0 * (i - jb - iq), KIND=dp)
      END DO
    END DO
  END DO
END SUBROUTINE init_vexx_bp_k_gpu_uspp_state_c


! The USPP path's becpsi%k(nkb, m), the same on the gfortran and the DaCe side.
SUBROUTINE fill_uspp_becpsi(becpsi, m)
  USE kinds, ONLY: dp
  USE becmod, ONLY: bec_type
  USE uspp, ONLY: nkb
  IMPLICIT NONE
  TYPE(bec_type), INTENT(INOUT) :: becpsi
  INTEGER, INTENT(IN) :: m
  INTEGER :: i, ib
  IF (ALLOCATED(becpsi%k)) DEALLOCATE (becpsi%k)
  ALLOCATE (becpsi%k(nkb, m))
  DO ib = 1, m
    DO i = 1, nkb
      becpsi%k(i, ib) = CMPLX(0.2D0 - 0.03D0 * MOD(3 * i + ib, 7), 0.1D0 + 0.02D0 * (i + ib), KIND=dp)
    END DO
  END DO
  IF (.NOT. ALLOCATED(becpsi%r)) ALLOCATE (becpsi%r(1, 1))
  becpsi%r = 0.0_dp
  IF (.NOT. ALLOCATED(becpsi%nc)) ALLOCATE (becpsi%nc(1, 1, 1))
  becpsi%nc = (0.0_dp, 0.0_dp)
  becpsi%nbnd = m
END SUBROUTINE fill_uspp_becpsi


! vexx_bp_k_gpu with becpsi present, as the USPP path needs it.
SUBROUTINE run_vexx_bp_k_gpu_uspp_c(lda, n, m, psi, hpsi) &
    BIND(C, NAME="run_vexx_bp_k_gpu_uspp_c")
  USE kinds, ONLY: dp
  USE becmod, ONLY: bec_type
  USE noncollin_module, ONLY: npol
  USE mp_exx, ONLY: max_ibands
  USE exx_bp, ONLY: vexx_bp_k_gpu
  IMPLICIT NONE
  INTEGER, VALUE :: lda, n, m
  COMPLEX(KIND=dp), INTENT(INOUT) :: psi(lda * npol, max_ibands)
  COMPLEX(KIND=dp), INTENT(INOUT) :: hpsi(lda * npol, max_ibands)
  TYPE(bec_type) :: becpsi
  CALL fill_uspp_becpsi(becpsi, max_ibands)
  CALL vexx_bp_k_gpu(lda, n, m, psi, hpsi, becpsi)
END SUBROUTINE run_vexx_bp_k_gpu_uspp_c


! Timer stubs for the fixture's start_clock/stop_clock; values are ignored.
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


SUBROUTINE fwfft_y(fft_kind, f, dfft, howmany)
  USE fft_types, ONLY: fft_type_descriptor
  USE fft_param, ONLY: DP
  IMPLICIT NONE
  CHARACTER(LEN=*), INTENT(IN) :: fft_kind
  TYPE(fft_type_descriptor), INTENT(IN) :: dfft
  INTEGER, OPTIONAL, INTENT(IN) :: howmany
  COMPLEX(DP) :: f(:)
  CALL flat_dft_stub(SIZE(f), f, -1.0_DP, 1.0_DP)
END SUBROUTINE fwfft_y


SUBROUTINE invfft_y(fft_kind, f, dfft, howmany)
  USE fft_types, ONLY: fft_type_descriptor
  USE fft_param, ONLY: DP
  IMPLICIT NONE
  CHARACTER(LEN=*), INTENT(IN) :: fft_kind
  TYPE(fft_type_descriptor), INTENT(IN) :: dfft
  INTEGER, OPTIONAL, INTENT(IN) :: howmany
  COMPLEX(DP) :: f(:)
  CALL flat_dft_stub(SIZE(f), f, 1.0_DP, 1.0_DP / DBLE(SIZE(f)))
END SUBROUTINE invfft_y


! The DFT the DaCe side lowers fwfft/invfft to (emit_library.emit_fft + DaCe's pure FFT/IFFT expansion): one flat
! transform over the whole buffer, ignoring dfft and howmany; forward exp(-2 pi i k n / N) unnormalized, inverse
! exp(+...) scaled by 1/N.  The reference runs this so both sides compute the same exchange on the real path.
SUBROUTINE flat_dft_stub(nn, f, sign, factor)
  USE fft_param, ONLY: DP
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nn
  COMPLEX(DP), INTENT(INOUT) :: f(nn)
  REAL(DP), INTENT(IN) :: sign, factor
  COMPLEX(DP) :: out(nn)
  REAL(DP) :: angle
  INTEGER :: k, j
  DO k = 0, nn - 1
    out(k + 1) = (0.0_DP, 0.0_DP)
    DO j = 0, nn - 1
      angle = 2.0_DP * 3.141592653589793_DP * DBLE(MOD(k * j, nn)) / DBLE(nn)
      out(k + 1) = out(k + 1) + CMPLX(COS(angle), sign * SIN(angle), KIND=DP) * f(j + 1)
    END DO
  END DO
  f = out * factor
END SUBROUTINE flat_dft_stub
