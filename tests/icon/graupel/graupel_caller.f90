! Fortran caller wrapper for the AES graupel numerical correctness tests (original and fused layouts).
!
! Exposes run_graupel_c: calls mo_aes_graupel::graupel_run with caller-owned buffers
! (the Python harness fills the inputs, so the gfortran reference and the SDFG get
! byte-identical data).  Shapes follow graupel_run: (ivec, k_v) for the cell fields,
! (ivec) for qnc and the surface rates.
!
! All fields use REAL(KIND=8) (= wp in mo_kind).  Integer inputs (ivec, k_v,
! ivs, ive, ks) are passed by value as C int.

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
