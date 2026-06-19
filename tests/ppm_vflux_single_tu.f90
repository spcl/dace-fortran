MODULE mo_fortran_tools
  USE iso_c_binding, ONLY: c_ptr, c_f_pointer, c_loc, c_null_ptr
  IMPLICIT NONE
  CONTAINS
  PURE SUBROUTINE set_acc_host_or_device(lzacc, lacc)
    LOGICAL, INTENT(OUT) :: lzacc
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    lzacc = .FALSE.
  END SUBROUTINE set_acc_host_or_device
END MODULE mo_fortran_tools
MODULE mo_ocean_nml
  IMPLICIT NONE
  INTEGER :: n_zlev
  NAMELIST /ocean_dynamics_nml/ n_zlev
  CONTAINS
END MODULE mo_ocean_nml
MODULE mo_ocean_types
  TYPE :: t_verticaladvection_ppm_coefficients
    REAL(KIND = 8), POINTER, DIMENSION(:, :) :: cellheightratio_this_tobelow
    REAL(KIND = 8), POINTER, DIMENSION(:, :) :: cellheightratio_this_tothisbelow
    REAL(KIND = 8), POINTER, DIMENSION(:, :) :: cellheight_2xbelow_x_ratiothis_tothisbelow
    REAL(KIND = 8), POINTER, DIMENSION(:, :) :: cellheightratio_this_tothisabovebelow
    REAL(KIND = 8), POINTER, DIMENSION(:, :) :: cellheightratio_2xaboveplusthis_tothisbelow
    REAL(KIND = 8), POINTER, DIMENSION(:, :) :: cellheightratio_2xbelowplusthis_tothisabove
    REAL(KIND = 8), POINTER, DIMENSION(:, :) :: cellheightratio_thisabove_to2xthisplusbelow
    REAL(KIND = 8), POINTER, DIMENSION(:, :) :: cellheightratio_thisbelow_to2xthisplusabove
    REAL(KIND = 8), POINTER, DIMENSION(:, :) :: cellheight_inv_thisabovebelow2below
  END TYPE t_verticaladvection_ppm_coefficients
END MODULE mo_ocean_types
MODULE mo_parallel_config
  IMPLICIT NONE
  INTEGER :: nproma = 0
  CONTAINS
END MODULE mo_parallel_config
MODULE mo_ocean_limiter
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE v_ppm_slimiter_mo_onblock(p_cc, p_face, p_slope, p_face_up, p_face_low, startindex, endindex, cells_nooflevels, lacc)
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_nml, ONLY: n_zlev
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    REAL(KIND = 8), INTENT(IN) :: p_cc(nproma, n_zlev)
    REAL(KIND = 8), INTENT(IN) :: p_face(nproma, n_zlev + 1)
    REAL(KIND = 8), INTENT(IN) :: p_slope(nproma, n_zlev + 1)
    REAL(KIND = 8), INTENT(INOUT) :: p_face_up(nproma, n_zlev)
    REAL(KIND = 8), INTENT(INOUT) :: p_face_low(nproma, n_zlev)
    INTEGER, INTENT(IN) :: startindex, endindex
    INTEGER, INTENT(IN) :: cells_nooflevels(nproma)
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    INTEGER :: nlev
    INTEGER :: firstlevel
    INTEGER :: jc, jk
    INTEGER :: ikp1
    REAL(KIND = 8) :: z_delta
    REAL(KIND = 8) :: z_a6i
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    firstlevel = 1
    nlev = n_zlev
    DO jc = startindex, endindex
      DO jk = firstlevel, cells_nooflevels(jc)
        ikp1 = jk + 1
        z_delta = p_face(jc, ikp1) - p_face(jc, jk)
        z_a6i = 6.0D0 * (p_cc(jc, jk) - 0.5D0 * (p_face(jc, jk) + p_face(jc, ikp1)))
        IF (p_slope(jc, jk) == 0.0D0) THEN
          p_face_up(jc, jk) = p_cc(jc, jk)
          p_face_low(jc, jk) = p_cc(jc, jk)
        ELSE IF (z_delta * z_a6i > z_delta * z_delta) THEN
          p_face_up(jc, jk) = 3.0D0 * p_cc(jc, jk) - 2.0D0 * p_face(jc, ikp1)
          p_face_low(jc, jk) = p_face(jc, ikp1)
        ELSE IF (z_delta * z_a6i < - 1.0D0 * (z_delta * z_delta)) THEN
          p_face_up(jc, jk) = p_face(jc, jk)
          p_face_low(jc, jk) = 3.0D0 * p_cc(jc, jk) - 2.0D0 * p_face(jc, jk)
        ELSE
          p_face_up(jc, jk) = p_face(jc, jk)
          p_face_low(jc, jk) = p_face(jc, ikp1)
        END IF
      END DO
    END DO
  END SUBROUTINE v_ppm_slimiter_mo_onblock
END MODULE mo_ocean_limiter
MODULE mo_ocean_tracer_transport_vert
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE upwind_vflux_ppm_onblock(tracer, w, dtime, vertical_limiter_type, cell_thickeness, cell_invheight, ppmcoeffs, flux_div_vert, startindex, endindex, cells_nooflevels, lacc)
    USE mo_parallel_config, ONLY: nproma
    USE mo_ocean_nml, ONLY: n_zlev
    USE mo_ocean_types, ONLY: t_verticaladvection_ppm_coefficients
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_ocean_limiter, ONLY: v_ppm_slimiter_mo_onblock
    REAL(KIND = 8), INTENT(IN) :: tracer(nproma, n_zlev)
    REAL(KIND = 8), INTENT(IN) :: w(nproma, n_zlev + 1)
    REAL(KIND = 8), INTENT(IN) :: dtime
    REAL(KIND = 8), INTENT(IN) :: cell_thickeness(nproma, n_zlev)
    REAL(KIND = 8), INTENT(IN) :: cell_invheight(nproma, n_zlev)
    TYPE(t_verticaladvection_ppm_coefficients) :: ppmcoeffs
    INTEGER, INTENT(IN) :: vertical_limiter_type
    REAL(KIND = 8), INTENT(INOUT) :: flux_div_vert(nproma, n_zlev)
    INTEGER, INTENT(IN) :: startindex, endindex
    INTEGER, INTENT(IN) :: cells_nooflevels(nproma)
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    REAL(KIND = 8) :: upward_tracer_flux(nproma, n_zlev + 1)
    REAL(KIND = 8) :: z_face(nproma, n_zlev + 1)
    REAL(KIND = 8) :: z_face_up(nproma, n_zlev)
    REAL(KIND = 8) :: z_face_low(nproma, n_zlev)
    REAL(KIND = 8) :: z_lext_1
    REAL(KIND = 8) :: z_lext_2
    REAL(KIND = 8) :: z_cfl_m, z_cfl_p
    REAL(KIND = 8) :: z_slope(nproma, n_zlev + 1)
    REAL(KIND = 8) :: z_slope_u, z_slope_l
    REAL(KIND = 8) :: z_a11, z_a12
    REAL(KIND = 8) :: z_weta_dt
    INTEGER :: firstlevel, secondlevel
    INTEGER :: levelabove, levelbelow, level2below
    INTEGER :: jc, thislevel, cell_levels
    LOGICAL :: lzacc
    firstlevel = 1
    secondlevel = 2
    CALL set_acc_host_or_device(lzacc, lacc)
    DO thislevel = 1, n_zlev + 1
      DO jc = 1, nproma
        z_slope(jc, thislevel) = 0.0D0
        z_face(jc, thislevel) = 0.0D0
      END DO
    END DO
    DO jc = startindex, endindex
      DO thislevel = secondlevel, cells_nooflevels(jc) - 1
        levelabove = thislevel - 1
        levelbelow = thislevel + 1
        z_slope_u = 2.0D0 * (tracer(jc, thislevel) - tracer(jc, levelabove))
        z_slope_l = 2.0D0 * (tracer(jc, levelbelow) - tracer(jc, thislevel))
        IF ((z_slope_u * z_slope_l) > 0.0D0) THEN
          z_slope(jc, thislevel) = (ppmcoeffs % cellheightratio_this_tothisabovebelow(jc, thislevel)) * ((ppmcoeffs % cellheightratio_2xaboveplusthis_tothisbelow(jc, thislevel)) * (tracer(jc, levelbelow) - tracer(jc, thislevel)) + (ppmcoeffs % cellheightratio_2xbelowplusthis_tothisabove(jc, thislevel)) * (tracer(jc, thislevel) - tracer(jc, levelabove)))
          z_slope(jc, thislevel) = SIGN(MIN(ABS(z_slope(jc, thislevel)), ABS(z_slope_u), ABS(z_slope_l)), z_slope(jc, thislevel))
        END IF
      END DO
    END DO
    DO jc = startindex, endindex
      cell_levels = cells_nooflevels(jc)
      DO thislevel = secondlevel, cell_levels - 2
        levelabove = thislevel - 1
        levelbelow = thislevel + 1
        level2below = thislevel + 2
        z_face(jc, levelbelow) = tracer(jc, thislevel) + (ppmcoeffs % cellheightratio_this_tothisbelow(jc, thislevel)) * (tracer(jc, levelbelow) - tracer(jc, thislevel)) + ppmcoeffs % cellheight_inv_thisabovebelow2below(jc, thislevel) * ((ppmcoeffs % cellheight_2xbelow_x_ratiothis_tothisbelow(jc, thislevel)) * (ppmcoeffs % cellheightratio_thisabove_to2xthisplusbelow(jc, thislevel) - ppmcoeffs % cellheightratio_thisbelow_to2xthisplusabove(jc, thislevel)) * (tracer(jc, levelbelow) - tracer(jc, thislevel)) - z_slope(jc, levelbelow) * cell_thickeness(jc, thislevel) * ppmcoeffs % cellheightratio_thisabove_to2xthisplusbelow(jc, thislevel) + z_slope(jc, thislevel) * cell_thickeness(jc, levelbelow) * ppmcoeffs % cellheightratio_thisbelow_to2xthisplusabove(jc, levelbelow))
      END DO
      IF (cells_nooflevels(jc) >= 1) THEN
        z_face(jc, 1) = tracer(jc, 1)
        IF (cell_levels >= 2) THEN
          z_face(jc, 2) = tracer(jc, 1) * (1.0D0 - ppmcoeffs % cellheightratio_this_tobelow(jc, 1)) + (ppmcoeffs % cellheightratio_this_tothisbelow(jc, 1)) * (ppmcoeffs % cellheightratio_this_tobelow(jc, 1) * tracer(jc, 1) + tracer(jc, 2))
        END IF
      END IF
      IF (cells_nooflevels(jc) > 2) THEN
        z_face(jc, cell_levels) = tracer(jc, cell_levels - 1) * (1.0D0 - ppmcoeffs % cellheightratio_this_tobelow(jc, cell_levels - 1)) + (cell_thickeness(jc, cell_levels - 1) / (cell_thickeness(jc, cell_levels - 1) + cell_thickeness(jc, cell_levels))) * (ppmcoeffs % cellheightratio_this_tobelow(jc, cell_levels - 1) * tracer(jc, cell_levels - 1) + tracer(jc, cell_levels))
      END IF
    END DO
    DO thislevel = 1, n_zlev
      DO jc = 1, nproma
        z_face_low(jc, thislevel) = 0.0D0
        z_face_up(jc, thislevel) = 0.0D0
      END DO
    END DO
    IF (vertical_limiter_type == 1) THEN
      CALL v_ppm_slimiter_mo_onblock(tracer(:, :), z_face(:, :), z_slope(:, :), z_face_up, z_face_low, startindex, endindex, cells_nooflevels, lacc = lzacc)
    ELSE
      DO jc = startindex, endindex
        DO thislevel = secondlevel, cells_nooflevels(jc) - 1
          z_face_up(jc, thislevel) = z_face(jc, thislevel)
          z_face_low(jc, thislevel) = z_face(jc, thislevel + 1)
        END DO
      END DO
    END IF
    DO thislevel = 1, n_zlev + 1
      DO jc = 1, nproma
        upward_tracer_flux(jc, thislevel) = 0.0D0
      END DO
    END DO
    DO jc = startindex, endindex
      DO thislevel = secondlevel, cells_nooflevels(jc)
        levelabove = thislevel - 1
        z_a11 = tracer(jc, levelabove) - 0.5D0 * (z_face_low(jc, levelabove) + z_face_up(jc, levelabove))
        z_weta_dt = ABS(w(jc, thislevel)) * dtime
        z_cfl_p = z_weta_dt * cell_invheight(jc, thislevel)
        z_cfl_m = z_weta_dt * cell_invheight(jc, levelabove)
        z_lext_1 = tracer(jc, levelabove) + 0.5D0 * (z_face_low(jc, levelabove) - z_face_up(jc, levelabove)) * (1.0D0 - z_cfl_m) - z_a11 - z_a11 * z_cfl_m * (- 3.0D0 + 2.0D0 * z_cfl_m)
        z_a12 = tracer(jc, thislevel) - 0.5D0 * (z_face_low(jc, thislevel) + z_face_up(jc, thislevel))
        z_lext_2 = tracer(jc, thislevel) - 0.5D0 * (z_face_low(jc, thislevel) - z_face_up(jc, thislevel)) * (1.0D0 - z_cfl_p) - z_a12 + z_a12 * z_cfl_p * (- 3.0D0 + 2.0D0 * z_cfl_p)
        upward_tracer_flux(jc, thislevel) = 0.5D0 * (w(jc, thislevel) * (z_lext_1 + z_lext_2) + ABS(w(jc, thislevel)) * (z_lext_2 - z_lext_1))
      END DO
    END DO
    DO jc = startindex, endindex
      DO thislevel = firstlevel, cells_nooflevels(jc)
        flux_div_vert(jc, thislevel) = upward_tracer_flux(jc, thislevel) - upward_tracer_flux(jc, thislevel + 1)
      END DO
      DO thislevel = cells_nooflevels(jc) + 1, n_zlev
        flux_div_vert(jc, thislevel) = 0.0D0
      END DO
    END DO
  END SUBROUTINE upwind_vflux_ppm_onblock
END MODULE mo_ocean_tracer_transport_vert