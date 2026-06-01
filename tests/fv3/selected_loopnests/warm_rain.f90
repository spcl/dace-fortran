! =======================================================================
! Standalone carve-out of the FV3 GFDL cloud-microphysics `warm_rain`
! kernel (warm-rain autoconversion + accretion + rain evaporation +
! rain sedimentation) for use as a compiler-frontend test.
!
! Extracted verbatim (physics unchanged) from ai2cm/fv3gfs-fortran
!   FV3/gfsphysics/physics/gfdl_cloud_microphys.F90
! warm_rain          : lines 1099-1308
! revap_racc         : lines 1314-1408
! linear_prof        : lines 1417-1468
! check_column       : lines 2790-2811
! implicit_fall      : lines 2819-2882   (use_ppm = .false. default path)
! sedi_heat          : lines 1044-1093   (do_sedi_heat = .true. default)
! wqs2 / qs_tablew   : lines 3807-3844 / 4249-4274 (water-saturation lookup)
!
! All module-global constants warm_rain transitively references are
! inlined here as `real, parameter` using the exact `setupm`/`setup_con`
! derivation expressions (default real kind = real(4), as the original).
! The lagrangian_fall_ppm / cs_profile / cs_limiters PPM machinery is NOT
! included because the default `use_ppm = .false.` selects implicit_fall.
! =======================================================================

module warm_rain_mod

    implicit none

    private
    public :: warm_rain, check_column, implicit_fall, sedi_heat, &
        revap_racc, linear_prof, warm_rain_driver, qsmith_init_w

    ! -----------------------------------------------------------------------
    ! base physical constants (genuine `real, parameter` in the source module)
    ! -----------------------------------------------------------------------

    real, parameter :: grav = 9.80665        !< acceleration due to gravity
    real, parameter :: rdgas = 287.05        !< gas constant for dry air
    real, parameter :: rvgas = 461.50        !< gas constant for water vapor
    real, parameter :: cp_air = 1004.6       !< heat capacity of dry air (cp)
    real, parameter :: hlv = 2.5e6           !< latent heat of evaporation
    real, parameter :: hlf = 3.3358e5        !< latent heat of fusion

    real, parameter :: cp_vap = 4.0 * rvgas  !< 1846.0
    real, parameter :: cv_air = cp_air - rdgas !< 717.55
    real, parameter :: cv_vap = 3.0 * rvgas  !< 1384.5

    real, parameter :: c_ice = 1972.0        !< heat capacity of ice at -15 C
    real, parameter :: c_liq = 4185.5        !< heat capacity of water at 15 C

    real, parameter :: eps = rdgas / rvgas   !< 0.6219934995

    real, parameter :: t_ice = 273.16        !< freezing temperature
    real, parameter :: table_ice = 273.16    !< freezing point for qs table

    real, parameter :: e00 = 611.21          !< ifs sat. vapor pressure at 0 C

    real, parameter :: dc_vap = cp_vap - c_liq !< -2339.5

    real, parameter :: hlv0 = hlv
    real, parameter :: lv0 = hlv0 - dc_vap * t_ice !< 3.13905782e6

    real, parameter :: qrmin = 1.e-8         !< min rain mixing ratio threshold
    real, parameter :: qvmin = 1.e-20        !< min water vapor (treated as zero)
    real, parameter :: qcmin = 1.e-12        !< min cloud condensate

    real, parameter :: vr_min = 1.e-3        !< min fall speed for rain
    real, parameter :: vf_min = 1.e-5        !< min fall speed (ice/snow/graupel)

    real, parameter :: dz_min = 1.e-2        !< correct flipped height

    real, parameter :: sfcrho = 1.2          !< surface air density
    real, parameter :: rhor = 1.e3           !< density of rain water (lin83)

    real, parameter :: dt_fr = 8.            !< homogeneous freezing offset

    ! -----------------------------------------------------------------------
    ! namelist-default scalars referenced (defaults from the module head)
    ! -----------------------------------------------------------------------

    real, parameter :: rthresh = 10.0e-6     !< critical cloud drop radius (m)
    real, parameter :: c_cracw = 0.9         !< rain accretion efficiency
    real, parameter :: alin = 842.0          !< "a" in lin1983
    real, parameter :: tice = 273.16

    real, parameter :: vr_fac = 1.           !< fall-speed tuning (variable path)
    real, parameter :: vr_max = 12.          !< max fall speed for rain

    ! -----------------------------------------------------------------------
    ! control-flow flags (module defaults; resolved for this carve-out)
    !   const_vr     = .false.  -> variable rain fall speed
    !   do_sedi_w    = .false.  -> skip vertical-velocity transport
    !   do_sedi_heat = .true.   -> call sedi_heat
    !   use_ppm      = .false.  -> implicit_fall (not lagrangian_fall_ppm)
    !   irain_f      = 0        -> with-subgrid-variability autoconversion
    !   z_slope_liq  = .true.   -> linear mono slope (linear_prof)
    !   use_ccn      = .false.
    ! -----------------------------------------------------------------------

    logical, parameter :: const_vr = .false.
    logical, parameter :: do_sedi_w = .false.
    logical, parameter :: do_sedi_heat = .true.
    logical, parameter :: use_ppm = .false.
    integer, parameter :: irain_f = 0
    logical, parameter :: z_slope_liq = .true.
    logical, parameter :: use_ccn = .false.
    logical, parameter :: mono_prof = .true.

    ! -----------------------------------------------------------------------
    ! `gfdl_cloud_microphys_init` derived constants
    ! (hydrostatic path: c_air = cp_air, c_vap = cp_vap)
    ! -----------------------------------------------------------------------

    real, parameter :: c_air = cp_air
    real, parameter :: c_vap = cp_vap
    real, parameter :: d0_vap = c_vap - c_liq          !< -2339.5
    real, parameter :: lv00 = hlv0 - d0_vap * t_ice    !< 3.13905782e6

    ! -----------------------------------------------------------------------
    ! `setup_con` derived constants
    ! -----------------------------------------------------------------------

    real, parameter :: t_wfr = tice - 40.0             !< 233.16

    ! -----------------------------------------------------------------------
    ! `setupm` derived constants (autoconversion / accretion / evaporation)
    ! Reproduced as parameter-expressions so the compiler folds them with the
    ! identical real(4) arithmetic the source performs in setupm.
    ! -----------------------------------------------------------------------

    real, parameter :: pie = 4. * atan(1.0)
    real, parameter :: rnzr = 8.0e6                    !< lin83 intercept
    real, parameter :: vdifu = 2.11e-5
    real, parameter :: tcond = 2.36e-2
    real, parameter :: visk = 1.259e-5
    real, parameter :: hltc = 2.5e6
    real, parameter :: gam290 = 1.827363
    real, parameter :: gam380 = 4.694155

    real, parameter :: scm3 = (visk / vdifu) ** (1. / 3.)
    real, parameter :: act2 = pie * rnzr * rhor        !< act(2)

    ! s. klein's formular (eq 16) from am2
    real, parameter :: fac_rc = (4. / 3.) * pie * rhor * rthresh ** 3

    ! cracw = c_cracw * craci, with craci built from act(2)
    real, parameter :: craci = pie * rnzr * alin * gam380 / (4. * act2 ** 0.95)
    real, parameter :: cracw = c_cracw * craci         !< ~2.9448564

    ! revp: five constants for the rain-evaporation process
    real, parameter :: crevp1 = 2. * pie * vdifu * tcond * rvgas * rnzr
    real, parameter :: crevp2 = 0.78 / sqrt(act2)
    real, parameter :: crevp3 = 0.31 * scm3 * gam290 * sqrt(alin / visk) / act2 ** 0.725
    real, parameter :: crevp4 = tcond * rvgas          !< = cssub(4)
    real, parameter :: crevp5 = hltc ** 2 * vdifu

    ! crevp packed into a length-5 array, matching the source `crevp(:)`
    real, parameter :: crevp(5) = (/ crevp1, crevp2, crevp3, crevp4, crevp5 /)

    ! -----------------------------------------------------------------------
    ! water-saturation lookup tables for wqs2 (built by qsmith_init_w)
    ! These are genuine runtime tables in the source (qs_tablew + desw).
    ! -----------------------------------------------------------------------

    integer, parameter :: qs_length = 2621
    real :: tablew(qs_length), desw(qs_length)
    logical :: tables_are_initialized = .false.

contains

    ! =======================================================================
    ! build the water-saturation table `tablew` and its difference `desw`
    ! (port of qs_tablew + the desw step in qsmith_init)
    ! =======================================================================

    subroutine qsmith_init_w

        implicit none

        real :: delt, tmin, tem, fac0, fac1, fac2
        integer :: i

        if (tables_are_initialized) return

        delt = 0.1
        tmin = table_ice - 160.

        do i = 1, qs_length
            tem = tmin + delt * real (i - 1)
            fac0 = (tem - t_ice) / (tem * t_ice)
            fac1 = fac0 * lv0
            fac2 = (dc_vap * log (tem / t_ice) + fac1) / rvgas
            tablew (i) = e00 * exp (fac2)
        enddo

        do i = 1, qs_length - 1
            desw (i) = max (0., tablew (i + 1) - tablew (i))
        enddo
        desw (qs_length) = desw (qs_length - 1)

        tables_are_initialized = .true.

    end subroutine qsmith_init_w

    ! =======================================================================
    ! saturation specific humidity over pure liquid water + analytic dqs/dT
    ! (table form of wqs2)
    ! =======================================================================

    real function wqs2 (ta, den, dqdt)

        implicit none

        real, intent (in) :: ta, den
        real, intent (out) :: dqdt

        real :: es, ap1, tmin
        integer :: it

        tmin = table_ice - 160.

        if (.not. tables_are_initialized) call qsmith_init_w

        ap1 = 10. * dim (ta, tmin) + 1.
        ap1 = min (2621., ap1)
        it = ap1
        es = tablew (it) + (ap1 - it) * desw (it)
        wqs2 = es / (rvgas * ta * den)
        it = ap1 - 0.5
        ! finite diff, del_t = 0.1:
        dqdt = 10. * (desw (it) + (ap1 - it) * (desw (it + 1) - desw (it))) / (rvgas * ta * den)

    end function wqs2

    ! =======================================================================
    ! warm rain: terminal speed, evaporation/accretion, sedimentation,
    ! and autoconversion of cloud water to rain.
    ! =======================================================================

    subroutine warm_rain (dt, ktop, kbot, dp, dz, tz, qv, ql, qr, qi, qs, qg, &
            den, denfac, ccn, c_praut, rh_rain, vtr, r1, m1_rain, w1, h_var)

        implicit none

        integer, intent (in) :: ktop, kbot

        real, intent (in) :: dt !< time step (s)
        real, intent (in) :: rh_rain, h_var

        real, intent (in), dimension (ktop:kbot) :: dp, dz, den
        real, intent (in), dimension (ktop:kbot) :: denfac, ccn, c_praut

        real, intent (inout), dimension (ktop:kbot) :: tz, vtr
        real, intent (inout), dimension (ktop:kbot) :: qv, ql, qr, qi, qs, qg
        real, intent (inout), dimension (ktop:kbot) :: m1_rain, w1

        real, intent (out) :: r1

        real, parameter :: so3 = 7. / 3.

        real, dimension (ktop:kbot) :: dl, dm
        real, dimension (ktop:kbot + 1) :: ze, zt

        real :: sink, dq, qc0, qc
        real :: qden
        real :: zs = 0.
        real :: dt5

        integer :: k

        ! fall velocity constants:

        real, parameter :: vconr = 2503.23638966667
        real, parameter :: normr = 25132741228.7183
        real, parameter :: thr = 1.e-8

        logical :: no_fall

        dt5 = 0.5 * dt

        ! -----------------------------------------------------------------------
        ! terminal speed of rain
        ! -----------------------------------------------------------------------

        m1_rain (:) = 0.

        call check_column (ktop, kbot, qr, no_fall)

        if (no_fall) then
            vtr (:) = vf_min
            r1 = 0.
        else

            ! -----------------------------------------------------------------------
            ! fall speed of rain
            ! -----------------------------------------------------------------------

            if (const_vr) then
                vtr (:) = vr_fac ! ifs_2016: 4.0
            else
                do k = ktop, kbot
                    qden = qr (k) * den (k)
                    if (qr (k) < thr) then
                        vtr (k) = vr_min
                    else
                        vtr (k) = vr_fac * vconr * sqrt (min (10., sfcrho / den (k))) * &
                            exp (0.2 * log (qden / normr))
                        vtr (k) = min (vr_max, max (vr_min, vtr (k)))
                    endif
                enddo
            endif

            ze (kbot + 1) = zs
            do k = kbot, ktop, - 1
                ze (k) = ze (k + 1) - dz (k) ! dz < 0
            enddo

            ! -----------------------------------------------------------------------
            ! evaporation and accretion of rain for the first 1 / 2 time step
            ! -----------------------------------------------------------------------

            ! if (.not. fast_sat_adj) &
            call revap_racc (ktop, kbot, dt5, tz, qv, ql, qr, qi, qs, qg, den, denfac, rh_rain, h_var)

            if (do_sedi_w) then
                do k = ktop, kbot
                    dm (k) = dp (k) * (1. + qv (k) + ql (k) + qr (k) + qi (k) + qs (k) + qg (k))
                enddo
            endif

            ! -----------------------------------------------------------------------
            ! mass flux induced by falling rain
            ! -----------------------------------------------------------------------

            if (use_ppm) then
                zt (ktop) = ze (ktop)
                do k = ktop + 1, kbot
                    zt (k) = ze (k) - dt5 * (vtr (k - 1) + vtr (k))
                enddo
                zt (kbot + 1) = zs - dt * vtr (kbot)

                do k = ktop, kbot
                    if (zt (k + 1) >= zt (k)) zt (k + 1) = zt (k) - dz_min
                enddo
                ! lagrangian_fall_ppm path omitted (use_ppm = .false. by default)
            else
                call implicit_fall (dt, ktop, kbot, ze, vtr, dp, qr, r1, m1_rain)
            endif

            ! -----------------------------------------------------------------------
            ! vertical velocity transportation during sedimentation
            ! -----------------------------------------------------------------------

            if (do_sedi_w) then
                w1 (ktop) = (dm (ktop) * w1 (ktop) + m1_rain (ktop) * vtr (ktop)) / (dm (ktop) - m1_rain (ktop))
                do k = ktop + 1, kbot
                    w1 (k) = (dm (k) * w1 (k) - m1_rain (k - 1) * vtr (k - 1) + m1_rain (k) * vtr (k)) &
                         / (dm (k) + m1_rain (k - 1) - m1_rain (k))
                enddo
            endif

            ! -----------------------------------------------------------------------
            ! heat transportation during sedimentation
            ! -----------------------------------------------------------------------

            if (do_sedi_heat) &
                call sedi_heat (ktop, kbot, dp, m1_rain, dz, tz, qv, ql, qr, qi, qs, qg, c_liq)

            ! -----------------------------------------------------------------------
            ! evaporation and accretion of rain for the remaing 1 / 2 time step
            ! -----------------------------------------------------------------------

            call revap_racc (ktop, kbot, dt5, tz, qv, ql, qr, qi, qs, qg, den, denfac, rh_rain, h_var)

        endif

        ! -----------------------------------------------------------------------
        ! auto - conversion
        ! assuming linear subgrid vertical distribution of cloud water
        ! following lin et al. 1994, mwr
        ! -----------------------------------------------------------------------

        if (irain_f /= 0) then

            ! -----------------------------------------------------------------------
            ! no subgrid varaibility
            ! -----------------------------------------------------------------------

            do k = ktop, kbot
                qc0 = fac_rc * ccn (k)
                if (tz (k) > t_wfr) then
                    if (use_ccn) then
                        ! -----------------------------------------------------------------------
                        ! ccn is formulted as ccn = ccn_surface * (den / den_surface)
                        ! -----------------------------------------------------------------------
                        qc = qc0
                    else
                        qc = qc0 / den (k)
                    endif
                    dq = ql (k) - qc
                    if (dq > 0.) then
                        sink = min (dq, dt * c_praut (k) * den (k) * exp (so3 * log (ql (k))))
                        ql (k) = ql (k) - sink
                        qr (k) = qr (k) + sink
                    endif
                endif
            enddo

        else

            ! -----------------------------------------------------------------------
            ! with subgrid varaibility
            ! -----------------------------------------------------------------------

            call linear_prof (kbot - ktop + 1, ql (ktop), dl (ktop), z_slope_liq, h_var)

            do k = ktop, kbot
                qc0 = fac_rc * ccn (k)
                if (tz (k) > t_wfr + dt_fr) then
                    dl (k) = min (max (1.e-6, dl (k)), 0.5 * ql (k))
                    ! --------------------------------------------------------------------
                    ! as in klein's gfdl am2 stratiform scheme (with subgrid variations)
                    ! --------------------------------------------------------------------
                    if (use_ccn) then
                        ! --------------------------------------------------------------------
                        ! ccn is formulted as ccn = ccn_surface * (den / den_surface)
                        ! --------------------------------------------------------------------
                        qc = qc0
                    else
                        qc = qc0 / den (k)
                    endif
                    dq = 0.5 * (ql (k) + dl (k) - qc)
                    ! --------------------------------------------------------------------
                    ! dq = dl if qc == q_minus = ql - dl
                    ! dq = 0 if qc == q_plus = ql + dl
                    ! --------------------------------------------------------------------
                    if (dq > 0.) then ! q_plus > qc
                        ! --------------------------------------------------------------------
                        ! revised continuous form: linearly decays (with subgrid dl) to zero at qc == ql + dl
                        ! --------------------------------------------------------------------
                        sink = min (1., dq / dl (k)) * dt * c_praut (k) * den (k) * exp (so3 * log (ql (k)))
                        ql (k) = ql (k) - sink
                        qr (k) = qr (k) + sink
                    endif
                endif
            enddo
        endif

    end subroutine warm_rain

    ! =======================================================================
    ! evaporation and accretion of rain
    ! =======================================================================

    subroutine revap_racc (ktop, kbot, dt, tz, qv, ql, qr, qi, qs, qg, den, denfac, rh_rain, h_var)

        implicit none

        integer, intent (in) :: ktop, kbot

        real, intent (in) :: dt ! time step (s)
        real, intent (in) :: rh_rain, h_var

        real, intent (in), dimension (ktop:kbot) :: den, denfac

        real, intent (inout), dimension (ktop:kbot) :: tz, qv, qr, ql, qi, qs, qg

        real, dimension (ktop:kbot) :: lhl, cvm, q_liq, q_sol, lcpk

        real :: dqv, qsat, dqsdt, evap, t2, qden, q_plus, q_minus, sink
        real :: qpz, dq, dqh, tin

        integer :: k

        do k = ktop, kbot

            if (tz (k) > t_wfr .and. qr (k) > qrmin) then

                ! -----------------------------------------------------------------------
                ! define heat capacity and latent heat coefficient
                ! -----------------------------------------------------------------------

                lhl (k) = lv00 + d0_vap * tz (k)
                q_liq (k) = ql (k) + qr (k)
                q_sol (k) = qi (k) + qs (k) + qg (k)
                cvm (k) = c_air + qv (k) * c_vap + q_liq (k) * c_liq + q_sol (k) * c_ice
                lcpk (k) = lhl (k) / cvm (k)

                tin = tz (k) - lcpk (k) * ql (k) ! presence of clouds suppresses the rain evap
                qpz = qv (k) + ql (k)
                qsat = wqs2 (tin, den (k), dqsdt)
                dqh = max (ql (k), h_var * max (qpz, qcmin))
                dqh = min (dqh, 0.2 * qpz) ! new limiter
                dqv = qsat - qv (k) ! use this to prevent super - sat the gird box
                q_minus = qpz - dqh
                q_plus = qpz + dqh

                ! -----------------------------------------------------------------------
                ! qsat must be > q_minus to activate evaporation
                ! qsat must be < q_plus to activate accretion
                ! -----------------------------------------------------------------------

                ! -----------------------------------------------------------------------
                ! rain evaporation
                ! -----------------------------------------------------------------------

                if (dqv > qvmin .and. qsat > q_minus) then
                    if (qsat > q_plus) then
                        dq = qsat - qpz
                    else
                        ! -----------------------------------------------------------------------
                        ! q_minus < qsat < q_plus
                        ! dq == dqh if qsat == q_minus
                        ! -----------------------------------------------------------------------
                        dq = 0.25 * (q_minus - qsat) ** 2 / dqh
                    endif
                    qden = qr (k) * den (k)
                    t2 = tin * tin
                    evap = crevp (1) * t2 * dq * (crevp (2) * sqrt (qden) + crevp (3) * &
                        exp (0.725 * log (qden))) / (crevp (4) * t2 + crevp (5) * qsat * den (k))
                    evap = min (qr (k), dt * evap, dqv / (1. + lcpk (k) * dqsdt))
                    ! -----------------------------------------------------------------------
                    ! alternative minimum evap in dry environmental air
                    ! sink = min (qr (k), dim (rh_rain * qsat, qv (k)) / (1. + lcpk (k) * dqsdt))
                    ! evap = max (evap, sink)
                    ! -----------------------------------------------------------------------
                    qr (k) = qr (k) - evap
                    qv (k) = qv (k) + evap
                    q_liq (k) = q_liq (k) - evap
                    cvm (k) = c_air + qv (k) * c_vap + q_liq (k) * c_liq + q_sol (k) * c_ice
                    tz (k) = tz (k) - evap * lhl (k) / cvm (k)
                endif

                ! -----------------------------------------------------------------------
                ! accretion: pracc
                ! -----------------------------------------------------------------------

                ! if (qr (k) > qrmin .and. ql (k) > 1.e-7 .and. qsat < q_plus) then
                if (qr (k) > qrmin .and. ql (k) > 1.e-6 .and. qsat < q_minus) then
                    sink = dt * denfac (k) * cracw * exp (0.95 * log (qr (k) * den (k)))
                    sink = sink / (1. + sink) * ql (k)
                    ql (k) = ql (k) - sink
                    qr (k) = qr (k) + sink
                endif

            endif ! warm - rain
        enddo

    end subroutine revap_racc

    ! =======================================================================
    ! definition of vertical subgrid variability used for cloud water
    ! autoconversion (ql --> qr); edges: qe == qbar +/- dm
    ! =======================================================================

    subroutine linear_prof (km, q, dm, z_var, h_var)

        implicit none

        integer, intent (in) :: km

        real, intent (in) :: q (km), h_var

        real, intent (out) :: dm (km)

        logical, intent (in) :: z_var

        real :: dq (km)

        integer :: k

        if (z_var) then
            do k = 2, km
                dq (k) = 0.5 * (q (k) - q (k - 1))
            enddo
            dm (1) = 0.

            ! -----------------------------------------------------------------------
            ! use twice the strength of the positive definiteness limiter (lin et al 1994)
            ! -----------------------------------------------------------------------

            do k = 2, km - 1
                dm (k) = 0.5 * min (abs (dq (k) + dq (k + 1)), 0.5 * q (k))
                if (dq (k) * dq (k + 1) <= 0.) then
                    if (dq (k) > 0.) then ! local max
                        dm (k) = min (dm (k), dq (k), - dq (k + 1))
                    else
                        dm (k) = 0.
                    endif
                endif
            enddo
            dm (km) = 0.

            ! -----------------------------------------------------------------------
            ! impose a presumed background horizontal variability that is proportional to the value itself
            ! -----------------------------------------------------------------------

            do k = 1, km
                dm (k) = max (dm (k), qvmin, h_var * q (k))
            enddo
        else
            do k = 1, km
                dm (k) = max (qvmin, h_var * q (k))
            enddo
        endif

    end subroutine linear_prof

    ! =======================================================================
    ! check if any cell in the column has q above the fall threshold
    ! =======================================================================

    subroutine check_column (ktop, kbot, q, no_fall)

        implicit none

        integer, intent (in) :: ktop, kbot

        real, intent (in) :: q (ktop:kbot)

        logical, intent (out) :: no_fall

        integer :: k

        no_fall = .true.

        do k = ktop, kbot
            if (q (k) > qrmin) then
                no_fall = .false.
                exit
            endif
        enddo

    end subroutine check_column

    ! =======================================================================
    ! time-implicit monotonic sedimentation scheme (Shian-Jiann Lin, 2016)
    ! =======================================================================

    subroutine implicit_fall (dt, ktop, kbot, ze, vt, dp, q, precip, m1)

        implicit none

        integer, intent (in) :: ktop, kbot

        real, intent (in) :: dt

        real, intent (in), dimension (ktop:kbot + 1) :: ze

        real, intent (in), dimension (ktop:kbot) :: vt, dp

        real, intent (inout), dimension (ktop:kbot) :: q

        real, intent (out), dimension (ktop:kbot) :: m1

        real, intent (out) :: precip

        real, dimension (ktop:kbot) :: dz, qm, dd

        integer :: k

        do k = ktop, kbot
            dz (k) = ze (k) - ze (k + 1)
            dd (k) = dt * vt (k)
            q (k) = q (k) * dp (k)
        enddo

        ! -----------------------------------------------------------------------
        ! sedimentation: non - vectorizable loop
        ! -----------------------------------------------------------------------

        qm (ktop) = q (ktop) / (dz (ktop) + dd (ktop))
        do k = ktop + 1, kbot
            qm (k) = (q (k) + dd (k - 1) * qm (k - 1)) / (dz (k) + dd (k))
        enddo

        ! -----------------------------------------------------------------------
        ! qm is density at this stage
        ! -----------------------------------------------------------------------

        do k = ktop, kbot
            qm (k) = qm (k) * dz (k)
        enddo

        ! -----------------------------------------------------------------------
        ! output mass fluxes: non - vectorizable loop
        ! -----------------------------------------------------------------------

        m1 (ktop) = q (ktop) - qm (ktop)
        do k = ktop + 1, kbot
            m1 (k) = m1 (k - 1) + q (k) - qm (k)
        enddo
        precip = m1 (kbot)

        ! -----------------------------------------------------------------------
        ! update:
        ! -----------------------------------------------------------------------

        do k = ktop, kbot
            q (k) = qm (k) / dp (k)
        enddo

    end subroutine implicit_fall

    ! =======================================================================
    ! transport of heat in sedimentation (sjl, july 2014)
    ! input q fields are dry mixing ratios, dm is dry air mass
    ! =======================================================================

    subroutine sedi_heat (ktop, kbot, dm, m1, dz, tz, qv, ql, qr, qi, qs, qg, cw)

        implicit none

        integer, intent (in) :: ktop, kbot

        real, intent (in), dimension (ktop:kbot) :: dm, m1, dz, qv, ql, qr, qi, qs, qg

        real, intent (inout), dimension (ktop:kbot) :: tz

        real, intent (in) :: cw ! heat capacity

        real, dimension (ktop:kbot) :: dgz, cvn

        real :: tmp

        integer :: k

        do k = ktop, kbot
            dgz (k) = - 0.5 * grav * dz (k) ! > 0
            cvn (k) = dm (k) * (cv_air + qv (k) * cv_vap + (qr (k) + ql (k)) * &
                c_liq + (qi (k) + qs (k) + qg (k)) * c_ice)
        enddo

        ! -----------------------------------------------------------------------
        ! backward time - implicit upwind transport scheme; dm is dry air mass
        ! -----------------------------------------------------------------------

        k = ktop
        tmp = cvn (k) + m1 (k) * cw
        tz (k) = (tmp * tz (k) + m1 (k) * dgz (k)) / tmp

        do k = ktop + 1, kbot
            tz (k) = ((cvn (k) + cw * (m1 (k) - m1 (k - 1))) * tz (k) + m1 (k - 1) * &
                cw * tz (k - 1) + dgz (k) * (m1 (k - 1) + m1 (k))) / (cvn (k) + cw * m1 (k))
        enddo

    end subroutine sedi_heat

    ! =======================================================================
    ! flat-argument driver: sets ktop = 1, kbot = km and calls warm_rain.
    ! All dummies are plain scalars or explicit-shape real(km) arrays.
    ! =======================================================================

    subroutine warm_rain_driver (km, dt, rh_rain, h_var, dp, dz, tz, qv, ql, qr, &
            qi, qs, qg, den, denfac, ccn, c_praut, vtr, m1_rain, w1, r1)

        implicit none

        integer, intent (in) :: km

        real, intent (in) :: dt, rh_rain, h_var

        real, intent (in), dimension (km) :: dp, dz, den, denfac, ccn, c_praut

        real, intent (inout), dimension (km) :: tz, qv, ql, qr, qi, qs, qg
        real, intent (inout), dimension (km) :: vtr, m1_rain, w1

        real, intent (out) :: r1

        integer :: ktop, kbot

        ktop = 1
        kbot = km

        call warm_rain (dt, ktop, kbot, dp, dz, tz, qv, ql, qr, qi, qs, qg, &
            den, denfac, ccn, c_praut, rh_rain, vtr, r1, m1_rain, w1, h_var)

    end subroutine warm_rain_driver

end module warm_rain_mod
