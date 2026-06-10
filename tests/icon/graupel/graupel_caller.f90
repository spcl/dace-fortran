! Fortran caller wrapper for the AES graupel e2e numerical correctness test.
!
! Exposes:
!   init_graupel_inputs_c  -- seed deterministic random inputs for every IN/INOUT
!                              array, into caller-allocated buffers.
!   run_graupel_c          -- call mo_aes_graupel::graupel_run with the prepared
!                              buffers (matches the Fortran-side signature).
!
! The wrapper takes raw C-bound pointers so the harness (ctypes / SDFG bindings)
! can pass the same buffers to both the gfortran reference and the SDFG-built
! version.  Shapes follow graupel_run's documented dimensions: (ie, ke) for
! the gridcell-level fields, (ie) for column-integrated outputs.
!
! All fields use REAL(KIND=8) (= wp in mo_kind).  Integer inputs (nvec, ke,
! ivstart, ivend, kstart) are passed by value as C int.

SUBROUTINE init_graupel_inputs_c(seed, ivec, k_v, dz, t, p, rho, qv, qc, qi, qr, qs, qg, qnc) &
        BIND(C, name="init_graupel_inputs_c")
    USE iso_c_binding
    IMPLICIT NONE
    INTEGER(C_INT), VALUE :: seed, ivec, k_v
    REAL(C_DOUBLE), DIMENSION(ivec, k_v), INTENT(OUT) :: dz, t, p, rho
    REAL(C_DOUBLE), DIMENSION(ivec, k_v), INTENT(OUT) :: qv, qc, qi, qr, qs, qg
    REAL(C_DOUBLE), DIMENSION(ivec), INTENT(OUT) :: qnc

    INTEGER :: i, k
    INTEGER :: s
    REAL(KIND=8) :: r

    ! Mulberry32-style scramble keyed off the seed.  Same as the
    ! velocity e2e test (init_inputs_random_c): deterministic,
    ! identical between reference + SDFG runs because both seed
    ! from this same routine.
    !
    ! Input regime: warm + dry (well above freezing, hydrometeors
    ! at zero).  This keeps the graupel scheme on its NO-OP path
    ! (no condensation, no ice phase transitions, no terminal
    ! velocities) so the e2e numerical compare reduces to "did the
    ! SDFG also produce the no-op outputs?", which is exactly the
    ! right gate at first contact with the kernel.  A non-no-op
    ! path would force opinion on the microphysics regime selection
    ! before the build itself is even validated.
    s = seed
    DO k = 1, k_v
        DO i = 1, ivec
            s = s * 1664525 + 1013904223
            r = REAL(IAND(ISHFT(s, -16), 32767), KIND=8) / 32768.0d0
            dz(i, k) = 100.0d0 + 400.0d0 * r     ! layer thickness (m)
            t(i, k) = 290.0d0                     ! 290 K (well above freezing)
            p(i, k) = 80000.0d0                   ! 800 hPa
            rho(i, k) = 1.0d0                     ! 1 kg/m3
            qv(i, k) = 0.0d0                      ! no vapor
            qc(i, k) = 0.0d0                      ! no cloud
            qi(i, k) = 0.0d0
            qr(i, k) = 0.0d0
            qs(i, k) = 0.0d0
            qg(i, k) = 0.0d0
        END DO
    END DO
    DO i = 1, ivec
        qnc(i) = 1.0d8                            ! cloud number concentration
    END DO
END SUBROUTINE init_graupel_inputs_c


SUBROUTINE run_graupel_c(ivec, k_v, ivs, ive, ks, dt, dz, t, p, rho, &
                          qv, qc, qi, qr, qs, qg, qnc, &
                          prr_gsp, pri_gsp, prs_gsp, prg_gsp, pflx, pre_gsp) &
        BIND(C, name="run_graupel_c")
    USE iso_c_binding
    USE mo_aes_graupel, ONLY: graupel_run
    IMPLICIT NONE

    INTEGER(C_INT), VALUE :: ivec, k_v, ivs, ive, ks
    REAL(C_DOUBLE), VALUE :: dt

    REAL(C_DOUBLE), DIMENSION(ivec, k_v), INTENT(IN) :: dz, p, rho
    REAL(C_DOUBLE), DIMENSION(ivec, k_v), INTENT(INOUT) :: t
    REAL(C_DOUBLE), DIMENSION(ivec, k_v), INTENT(INOUT) :: qv, qc, qi, qr, qs, qg
    REAL(C_DOUBLE), DIMENSION(ivec), INTENT(IN) :: qnc
    REAL(C_DOUBLE), DIMENSION(ivec), INTENT(OUT) :: prr_gsp, pri_gsp, prs_gsp, prg_gsp, pre_gsp
    REAL(C_DOUBLE), DIMENSION(ivec, k_v), INTENT(OUT) :: pflx

    CALL graupel_run(ivec, k_v, ivs, ive, ks, dt, dz, t, p, rho, &
                     qv, qc, qi, qr, qs, qg, qnc, &
                     prr_gsp, pri_gsp, prs_gsp, prg_gsp, pflx, pre_gsp)
END SUBROUTINE run_graupel_c
