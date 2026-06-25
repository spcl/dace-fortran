MODULE control_flags
  IMPLICIT NONE
  SAVE
  LOGICAL :: gamma_only = .TRUE.
END MODULE control_flags
MODULE fft_param
  INTEGER, PARAMETER :: mpi_comm_null = - 1
  INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(14, 200)
END MODULE fft_param
MODULE fft_types
  USE fft_param, ONLY: dp, mpi_comm_null
  IMPLICIT NONE
  SAVE
  TYPE :: fft_type_descriptor
    INTEGER :: nr1 = 0
    INTEGER :: nr2 = 0
    INTEGER :: nr3 = 0
    INTEGER :: nr1x = 0
    INTEGER :: nr2x = 0
    INTEGER :: nr3x = 0
    LOGICAL :: lpara = .FALSE.
    LOGICAL :: lgamma = .FALSE.
    INTEGER :: root = 0
    INTEGER :: comm = mpi_comm_null
    INTEGER :: comm2 = mpi_comm_null
    INTEGER :: comm3 = mpi_comm_null
    INTEGER :: nproc = 1
    INTEGER :: nproc2 = 1
    INTEGER :: nproc3 = 1
    INTEGER :: mype = 0
    INTEGER :: mype2 = 0
    INTEGER :: mype3 = 0
    INTEGER, ALLOCATABLE :: iproc(:, :), iproc2(:), iproc3(:)
    INTEGER :: my_nr3p = 0
    INTEGER :: my_nr2p = 0
    INTEGER :: my_i0r3p = 0
    INTEGER :: my_i0r2p = 0
    INTEGER, ALLOCATABLE :: nr3p(:)
    INTEGER, ALLOCATABLE :: nr3p_offset(:)
    INTEGER, ALLOCATABLE :: nr2p(:)
    INTEGER, ALLOCATABLE :: nr2p_offset(:)
    INTEGER, ALLOCATABLE :: nr1p(:)
    INTEGER, ALLOCATABLE :: nr1w(:)
    INTEGER :: nr1w_tg
    INTEGER, ALLOCATABLE :: i0r3p(:)
    INTEGER, ALLOCATABLE :: i0r2p(:)
    INTEGER, ALLOCATABLE :: ir1p(:)
    INTEGER, ALLOCATABLE :: indp(:, :)
    INTEGER, ALLOCATABLE :: ir1w(:)
    INTEGER, ALLOCATABLE :: indw(:, :)
    INTEGER, ALLOCATABLE :: ir1w_tg(:)
    INTEGER, ALLOCATABLE :: indw_tg(:)
    INTEGER, POINTER :: ir1p_d(:), ir1w_d(:), ir1w_tg_d(:)
    INTEGER, POINTER :: indp_d(:, :), indw_d(:, :), indw_tg_d(:, :)
    INTEGER, POINTER :: nr1p_d(:), nr1w_d(:), nr1w_tg_d(:)
    INTEGER :: nst
    INTEGER, ALLOCATABLE :: nsp(:)
    INTEGER, ALLOCATABLE :: nsp_offset(:, :)
    INTEGER, ALLOCATABLE :: nsw(:)
    INTEGER, ALLOCATABLE :: nsw_offset(:, :)
    INTEGER, ALLOCATABLE :: nsw_tg(:)
    INTEGER, ALLOCATABLE :: ngl(:)
    INTEGER, ALLOCATABLE :: nwl(:)
    INTEGER :: ngm
    INTEGER :: ngw
    INTEGER, ALLOCATABLE :: iplp(:)
    INTEGER, ALLOCATABLE :: iplw(:)
    INTEGER :: nnp = 0
    INTEGER :: nnr = 0
    INTEGER :: nnr_tg = 0
    INTEGER, ALLOCATABLE :: iss(:)
    INTEGER, ALLOCATABLE :: isind(:)
    INTEGER, ALLOCATABLE :: ismap(:)
    INTEGER, POINTER :: ismap_d(:)
    INTEGER, ALLOCATABLE :: nl(:)
    INTEGER, ALLOCATABLE :: nlm(:)
    INTEGER, POINTER :: nl_d(:)
    INTEGER, POINTER :: nlm_d(:)
    INTEGER, ALLOCATABLE :: tg_snd(:)
    INTEGER, ALLOCATABLE :: tg_rcv(:)
    INTEGER, ALLOCATABLE :: tg_sdsp(:)
    INTEGER, ALLOCATABLE :: tg_rdsp(:)
    LOGICAL :: has_task_groups = .FALSE.
    LOGICAL :: use_pencil_decomposition = .TRUE.
    CHARACTER(LEN = 12) :: rho_clock_label = ' '
    CHARACTER(LEN = 12) :: wave_clock_label = ' '
    INTEGER :: grid_id
    COMPLEX(KIND = dp), ALLOCATABLE, DIMENSION(:) :: aux
  END TYPE
  CONTAINS
END MODULE fft_types
MODULE iso_c_binding
  INTEGER, PARAMETER :: c_int8_t = 1
  INTEGER, PARAMETER :: c_char = c_int8_t
  INTEGER, PARAMETER :: c_double = 8
  CONTAINS
END MODULE iso_c_binding
MODULE kinds
  IMPLICIT NONE
  SAVE
  INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(14, 200)
  CONTAINS
END MODULE kinds
MODULE cell_base
  USE kinds, ONLY: dp
  IMPLICIT NONE
  SAVE
  REAL(KIND = dp) :: omega = 0.0_dp
  CONTAINS
END MODULE cell_base
MODULE constants
  USE kinds, ONLY: dp
  IMPLICIT NONE
  SAVE
  REAL(KIND = dp), PARAMETER :: pi = 3.14159265358979323846_dp
  REAL(KIND = dp), PARAMETER :: tpi = 2.0_dp * pi
END MODULE constants
MODULE gvect
  USE kinds, ONLY: dp
  IMPLICIT NONE
  SAVE
  INTEGER :: gstart = 2
  INTEGER, ALLOCATABLE, TARGET :: mill(:, :)
  COMPLEX(KIND = dp), ALLOCATABLE :: eigts1(:, :)
  COMPLEX(KIND = dp), ALLOCATABLE :: eigts2(:, :), eigts3(:, :)
  CONTAINS
END MODULE gvect
MODULE ions_base
  USE kinds, ONLY: dp
  IMPLICIT NONE
  SAVE
  INTEGER :: nat = 0
  INTEGER, ALLOCATABLE :: ityp(:)
  REAL(KIND = dp), ALLOCATABLE :: tau(:, :)
  CONTAINS
END MODULE ions_base
MODULE nvtx
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE nvtxstartrange(name, id)
    USE iso_c_binding, ONLY: c_char
    CHARACTER(LEN = *, KIND = c_char) :: name
    INTEGER, OPTIONAL :: id
  END SUBROUTINE nvtxstartrange
  SUBROUTINE nvtxendrange
  END SUBROUTINE nvtxendrange
END MODULE nvtx
MODULE upf_kinds
  IMPLICIT NONE
  SAVE
  INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(14, 200)
END MODULE
MODULE pseudo_types
  USE upf_kinds, ONLY: dp
  IMPLICIT NONE
  SAVE
  TYPE :: paw_in_upf
    REAL(KIND = dp), ALLOCATABLE :: ae_rho_atc(:)
    REAL(KIND = dp), ALLOCATABLE :: pfunc(:, :, :), pfunc_rel(:, :, :), ptfunc(:, :, :), aewfc_rel(:, :)
    REAL(KIND = dp), ALLOCATABLE :: ae_vloc(:)
    REAL(KIND = dp), ALLOCATABLE :: oc(:)
    REAL(KIND = dp), ALLOCATABLE :: augmom(:, :, :)
    REAL(KIND = dp) :: raug
    INTEGER :: iraug
    INTEGER :: lmax_aug
    REAL(KIND = dp) :: core_energy
    CHARACTER(LEN = 12) :: augshape
  END TYPE paw_in_upf
  TYPE :: pseudo_upf
    CHARACTER(LEN = 80) :: generated = ' '
    CHARACTER(LEN = 80) :: author = ' '
    CHARACTER(LEN = 80) :: date = ' '
    CHARACTER(LEN = 80) :: comment = ' '
    CHARACTER(LEN = 2) :: psd = ' '
    CHARACTER(LEN = 4) :: typ = ' '
    CHARACTER(LEN = 6) :: rel = ' '
    LOGICAL :: tvanp
    LOGICAL :: tcoulombp
    LOGICAL :: nlcc
    LOGICAL :: is_gth
    LOGICAL :: is_multiproj
    LOGICAL :: with_metagga_info
    CHARACTER(LEN = 25) :: dft
    REAL(KIND = dp) :: zp
    REAL(KIND = dp) :: etotps
    REAL(KIND = dp) :: ecutwfc
    REAL(KIND = dp) :: ecutrho
    CHARACTER(LEN = 11) :: nv
    INTEGER :: lmax
    INTEGER :: lmax_rho
    REAL(KIND = dp), ALLOCATABLE :: vnl(:, :, :)
    INTEGER :: nwfc
    INTEGER :: nbeta
    INTEGER, ALLOCATABLE :: kbeta(:)
    INTEGER :: kkbeta
    INTEGER, ALLOCATABLE :: lll(:)
    REAL(KIND = dp), ALLOCATABLE :: beta(:, :)
    CHARACTER(LEN = 2), ALLOCATABLE :: els(:)
    CHARACTER(LEN = 2), ALLOCATABLE :: els_beta(:)
    INTEGER, ALLOCATABLE :: nchi(:)
    INTEGER, ALLOCATABLE :: lchi(:)
    REAL(KIND = dp), ALLOCATABLE :: oc(:)
    REAL(KIND = dp), ALLOCATABLE :: epseu(:)
    REAL(KIND = dp), ALLOCATABLE :: rcut_chi(:)
    REAL(KIND = dp), ALLOCATABLE :: rcutus_chi(:)
    REAL(KIND = dp), ALLOCATABLE :: chi(:, :)
    REAL(KIND = dp), ALLOCATABLE :: rho_at(:)
    INTEGER :: mesh
    REAL(KIND = dp) :: xmin
    REAL(KIND = dp) :: rmax
    REAL(KIND = dp) :: zmesh
    REAL(KIND = dp) :: dx
    REAL(KIND = dp), ALLOCATABLE :: r(:)
    REAL(KIND = dp), ALLOCATABLE :: rab(:)
    REAL(KIND = dp), ALLOCATABLE :: rho_atc(:)
    INTEGER :: lloc
    REAL(KIND = dp) :: rcloc
    REAL(KIND = dp), ALLOCATABLE :: vloc(:)
    REAL(KIND = dp), ALLOCATABLE :: dion(:, :)
    LOGICAL :: q_with_l
    INTEGER :: nqf
    INTEGER :: nqlc
    REAL(KIND = dp) :: qqq_eps
    REAL(KIND = dp), ALLOCATABLE :: rinner(:)
    REAL(KIND = dp), ALLOCATABLE :: qqq(:, :)
    REAL(KIND = dp), ALLOCATABLE :: qfunc(:, :)
    REAL(KIND = dp), ALLOCATABLE :: qfuncl(:, :, :)
    REAL(KIND = dp), ALLOCATABLE :: qfcoef(:, :, :, :)
    REAL(KIND = dp), ALLOCATABLE :: tau_core(:)
    REAL(KIND = dp), ALLOCATABLE :: tau_atom(:)
    LOGICAL :: has_wfc
    REAL(KIND = dp), ALLOCATABLE :: aewfc(:, :)
    REAL(KIND = dp), ALLOCATABLE :: pswfc(:, :)
    LOGICAL :: has_so
    REAL(KIND = dp), ALLOCATABLE :: rcut(:)
    REAL(KIND = dp), ALLOCATABLE :: rcutus(:)
    REAL(KIND = dp), ALLOCATABLE :: jchi(:)
    REAL(KIND = dp), ALLOCATABLE :: jjj(:)
    INTEGER :: paw_data_format
    LOGICAL :: tpawp
    TYPE(paw_in_upf) :: paw
    LOGICAL :: has_gipaw
    LOGICAL :: paw_as_gipaw
    INTEGER :: gipaw_data_format
    INTEGER :: gipaw_ncore_orbitals
    REAL(KIND = dp), ALLOCATABLE :: gipaw_core_orbital_n(:)
    REAL(KIND = dp), ALLOCATABLE :: gipaw_core_orbital_l(:)
    CHARACTER(LEN = 2), ALLOCATABLE :: gipaw_core_orbital_el(:)
    REAL(KIND = dp), ALLOCATABLE :: gipaw_core_orbital(:, :)
    REAL(KIND = dp), ALLOCATABLE :: gipaw_vlocal_ae(:)
    REAL(KIND = dp), ALLOCATABLE :: gipaw_vlocal_ps(:)
    INTEGER :: gipaw_wfs_nchannels
    CHARACTER(LEN = 2), ALLOCATABLE :: gipaw_wfs_el(:)
    INTEGER, ALLOCATABLE :: gipaw_wfs_ll(:)
    REAL(KIND = dp), ALLOCATABLE :: gipaw_wfs_ae(:, :)
    REAL(KIND = dp), ALLOCATABLE :: gipaw_wfs_rcut(:)
    REAL(KIND = dp), ALLOCATABLE :: gipaw_wfs_rcutus(:)
    REAL(KIND = dp), ALLOCATABLE :: gipaw_wfs_ps(:, :)
    CHARACTER(LEN = 32) :: md5_cksum = 'NOT SET'
  END TYPE pseudo_upf
  CONTAINS
END MODULE pseudo_types
MODULE uspp
  IMPLICIT NONE
  SAVE
  INTEGER :: nkb
  INTEGER, ALLOCATABLE :: ijtoh(:, :, :), ofsbeta(:)
  LOGICAL :: okvan = .FALSE.
  CONTAINS
END MODULE uspp
MODULE uspp_param
  USE pseudo_types, ONLY: pseudo_upf
  IMPLICIT NONE
  SAVE
  TYPE(pseudo_upf), ALLOCATABLE, TARGET :: upf(:)
  INTEGER, ALLOCATABLE :: nh(:)
  CONTAINS
END MODULE uspp_param
MODULE us_exx
  USE kinds, ONLY: dp
  IMPLICIT NONE
  SAVE
  COMPLEX(KIND = dp), ALLOCATABLE :: qgm(:, :)
  INTEGER, ALLOCATABLE :: nij_type(:)
  CONTAINS
  SUBROUTINE newdxx_g(dfftt, vc, xkq, xk, flag, deexx, becphi_r, becphi_c)
    USE fft_types, ONLY: fft_type_descriptor
    USE kinds, ONLY: dp
    USE uspp, ONLY: ijtoh, nkb, ofsbeta, okvan
    USE control_flags, ONLY: gamma_only
    USE ions_base, ONLY: ityp, nat, tau
    USE constants, ONLY: tpi
    USE cell_base, ONLY: omega
    USE uspp_param, ONLY: nh, upf
    USE gvect, ONLY: eigts1, eigts2, eigts3, gstart, mill
    IMPLICIT NONE
    TYPE(fft_type_descriptor), INTENT(IN) :: dfftt
    COMPLEX(KIND = dp), INTENT(IN) :: vc(dfftt % nnr)
    COMPLEX(KIND = dp), INTENT(IN), OPTIONAL :: becphi_c(nkb)
    REAL(KIND = dp), INTENT(IN), OPTIONAL :: becphi_r(nkb)
    COMPLEX(KIND = dp), INTENT(INOUT) :: deexx(nkb)
    REAL(KIND = dp), INTENT(IN) :: xk(3), xkq(3)
    CHARACTER(LEN = 1), INTENT(IN) :: flag
    INTEGER :: ngms, ig, ikb, jkb, ijkb0, ih, jh, na, nt, nij
    REAL(KIND = dp) :: fact
    COMPLEX(KIND = dp), ALLOCATABLE :: auxvc(:), eigqts(:), aux1(:), aux2(:)
    COMPLEX(KIND = dp) :: fp, fm
    REAL(KIND = dp) :: arg
    LOGICAL :: add_complex, add_real, add_imaginary
    INTEGER, PARAMETER :: blocksize = 256
    INTEGER :: iblock, numblock, realblocksize, offset
    IF (.NOT. okvan) RETURN
    ngms = dfftt % ngm
    add_complex = (flag == 'c' .OR. flag == 'C')
    add_real = (flag == 'r' .OR. flag == 'R')
    add_imaginary = (flag == 'i' .OR. flag == 'I')
    IF (.NOT. (add_complex .OR. add_real .OR. add_imaginary)) CALL errore('newdxx_g', 'called with incorrect flag: ' // flag, 1)
    IF (.NOT. gamma_only .AND. (add_real .OR. add_imaginary)) CALL errore('newdxx_g', 'need gamma tricks for this flag: ' // flag, 2)
    IF (gamma_only .AND. add_complex) CALL errore('newdxx_g', 'gamma trick not good for this flag: ' // flag, 3)
    IF ((add_complex .AND. .NOT. PRESENT(becphi_c)) .OR. (add_real .AND. .NOT. PRESENT(becphi_r)) .OR. (add_imaginary .AND. .NOT. PRESENT(becphi_r))) CALL errore('newdxx_g', 'called with incorrect arguments', 2)
    CALL start_clock('newdxx')
    ALLOCATE(auxvc(ngms))
    ALLOCATE(eigqts(nat))
    DO na = 1, nat
      arg = tpi * SUM((xk(:) - xkq(:)) * tau(:, na))
      eigqts(na) = CMPLX(COS(arg), - SIN(arg), kind = dp)
    END DO
    IF (add_complex) THEN
      auxvc(1 : ngms) = vc(dfftt % nl(1 : ngms))
      fact = omega
    ELSE IF (add_real) THEN
      DO ig = 1, ngms
        fp = (vc(dfftt % nl(ig)) + vc(dfftt % nlm(ig))) / 2.0_dp
        fm = (vc(dfftt % nl(ig)) - vc(dfftt % nlm(ig))) / 2.0_dp
        auxvc(ig) = CMPLX(DBLE(fp), AIMAG(fm), kind = dp)
      END DO
      fact = 2.0_dp * omega
    ELSE IF (add_imaginary) THEN
      DO ig = 1, ngms
        fp = (vc(dfftt % nl(ig)) + vc(dfftt % nlm(ig))) / 2.0_dp
        fm = (vc(dfftt % nl(ig)) - vc(dfftt % nlm(ig))) / 2.0_dp
        auxvc(ig) = CMPLX(AIMAG(fp), - DBLE(fm), kind = dp)
      END DO
      fact = 2.0_dp * omega
    END IF
    numblock = (ngms + blocksize - 1) / blocksize
    ALLOCATE(aux1(blocksize), aux2(blocksize))
    DO iblock = 1, numblock
      offset = (iblock - 1) * blocksize
      realblocksize = MIN(ngms - offset, blocksize)
      DO na = 1, nat
        nt = ityp(na)
        IF (upf(nt) % tvanp) THEN
          nij = nij_type(nt)
          ijkb0 = ofsbeta(na)
          aux2(1 : realblocksize) = CONJG(auxvc(offset + 1 : offset + realblocksize)) * eigqts(na) * eigts1(mill(1, offset + 1 : offset + realblocksize), na) * eigts2(mill(2, offset + 1 : offset + realblocksize), na) * eigts3(mill(3, offset + 1 : offset + realblocksize), na)
          DO ih = 1, nh(nt)
            ikb = ijkb0 + ih
            aux1(:) = (0.0_dp, 0.0_dp)
            DO jh = 1, nh(nt)
              jkb = ijkb0 + jh
              IF (gamma_only) THEN
                aux1(1 : realblocksize) = aux1(1 : realblocksize) + becphi_r(jkb) * CONJG(qgm(offset + 1 : offset + realblocksize, nij + ijtoh(ih, jh, nt)))
              ELSE
                aux1(1 : realblocksize) = aux1(1 : realblocksize) + becphi_c(jkb) * CONJG(qgm(offset + 1 : offset + realblocksize, nij + ijtoh(ih, jh, nt)))
              END IF
            END DO
            deexx(ikb) = deexx(ikb) + fact * DOT_PRODUCT(aux2(1 : realblocksize), aux1(1 : realblocksize))
            IF (gamma_only .AND. gstart == 2 .AND. iblock == 1) deexx(ikb) = deexx(ikb) - omega * CONJG(aux2(1)) * aux1(1)
          END DO
        END IF
      END DO
    END DO
    DEALLOCATE(aux2, aux1)
    DEALLOCATE(eigqts, auxvc)
    CALL stop_clock('newdxx')
    RETURN
  END SUBROUTINE newdxx_g
END MODULE us_exx
MODULE util_param
  CHARACTER(LEN = 5), PARAMETER :: crash_file = 'CRASH'
  INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(14, 200)
  INTEGER, PARAMETER :: stdout = 6
END MODULE util_param
MODULE mytime
  USE util_param, ONLY: dp
  USE iso_c_binding, ONLY: c_double
  IMPLICIT NONE
  SAVE
  INTEGER, PARAMETER :: maxclock = 128
  REAL(KIND = dp), PARAMETER :: notrunning = - 1.0_dp
  REAL(KIND = dp) :: cputime(maxclock), t0cpu(maxclock)
  REAL(KIND = dp) :: walltime(maxclock), t0wall(maxclock)
  CHARACTER(LEN = 12) :: clock_label(maxclock)
  INTEGER :: called(maxclock)
  INTEGER :: nclock = 0
  LOGICAL :: no
  INTERFACE
    FUNCTION f_wall() RESULT(t)
      USE iso_c_binding, ONLY: c_double
      REAL(KIND = c_double) :: t
    END FUNCTION f_wall
    FUNCTION f_tcpu() RESULT(t)
      USE iso_c_binding, ONLY: c_double
      REAL(KIND = c_double) :: t
    END FUNCTION f_tcpu
  END INTERFACE
END MODULE mytime
SUBROUTINE errore(calling_routine, message, ierr)
  USE util_param, ONLY: crash_file, stdout
  IMPLICIT NONE
  CHARACTER(LEN = *), INTENT(IN) :: calling_routine, message
  INTEGER, INTENT(IN) :: ierr
  INTEGER :: crashunit
  CHARACTER(LEN = 6) :: cerr
  IF (ierr <= 0) RETURN
  WRITE(cerr, FMT = '(I6)') ierr
  WRITE(UNIT = *, FMT = '(/,1X,78("%"))')
  WRITE(UNIT = *, FMT = '(5X,"Error in routine ",A," (",A,"):")') TRIM(calling_routine), TRIM(ADJUSTL(cerr))
  WRITE(UNIT = *, FMT = '(5X,A)') TRIM(message)
  WRITE(UNIT = *, FMT = '(1X,78("%"),/)')
  WRITE(*, '("     stopping ...")')
  FLUSH(UNIT = stdout)
  OPEN(NEWUNIT = crashunit, FILE = crash_file, POSITION = 'APPEND', STATUS = 'UNKNOWN')
  WRITE(UNIT = crashunit, FMT = '(/,1X,78("%"))')
  WRITE(UNIT = crashunit, FMT = '(5X,"from ",A," : error #",I10)') TRIM(calling_routine), ierr
  WRITE(UNIT = crashunit, FMT = '(5X,A)') TRIM(message)
  WRITE(UNIT = crashunit, FMT = '(1X,78("%"),/)')
  CLOSE(UNIT = crashunit)
  STOP 1
END SUBROUTINE errore
SUBROUTINE start_clock(label)
  USE mytime, ONLY: clock_label, f_tcpu, f_wall, maxclock, nclock, no, notrunning, t0cpu, t0wall
  USE nvtx, ONLY: nvtxstartrange
  USE util_param, ONLY: stdout
  IMPLICIT NONE
  CHARACTER(LEN = *) :: label
  CHARACTER(LEN = 12) :: label_
  INTEGER :: n
  IF (no .AND. (nclock == 1)) RETURN
  label_ = TRIM(label)
  DO n = 1, nclock
    IF (clock_label(n) == label_) THEN
      IF (t0cpu(n) /= notrunning) THEN
      ELSE
        t0cpu(n) = f_tcpu()
        t0wall(n) = f_wall()
        CALL nvtxstartrange(label_, n)
      END IF
      RETURN
    END IF
  END DO
  IF (nclock == maxclock) THEN
    WRITE(stdout, '("start_clock(",A,"): Too many clocks! call ignored")') label
  ELSE
    nclock = nclock + 1
    clock_label(nclock) = label_
    t0cpu(nclock) = f_tcpu()
    t0wall(nclock) = f_wall()
    CALL nvtxstartrange(label_, n)
  END IF
  RETURN
END SUBROUTINE start_clock
SUBROUTINE stop_clock(label)
  USE mytime, ONLY: called, clock_label, cputime, f_tcpu, f_wall, nclock, no, notrunning, t0cpu, t0wall, walltime
  USE util_param, ONLY: stdout
  USE nvtx, ONLY: nvtxendrange
  IMPLICIT NONE
  CHARACTER(LEN = *) :: label
  CHARACTER(LEN = 12) :: label_
  INTEGER :: n
  IF (no) RETURN
  label_ = TRIM(label)
  DO n = 1, nclock
    IF (clock_label(n) == label_) THEN
      IF (t0cpu(n) == notrunning) THEN
        WRITE(stdout, '("stop_clock: clock # ",I2," for ",A12, " not running")') n, label
      ELSE
        cputime(n) = cputime(n) + f_tcpu() - t0cpu(n)
        walltime(n) = walltime(n) + f_wall() - t0wall(n)
        t0cpu(n) = notrunning
        t0wall(n) = notrunning
        called(n) = called(n) + 1
        CALL nvtxendrange
      END IF
      RETURN
    END IF
  END DO
  WRITE(stdout, '("stop_clock: no clock for ",A12," found !")') label
  RETURN
END SUBROUTINE stop_clock