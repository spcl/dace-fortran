"""Minimal Fortran stand-ins for the ICON types ``mo_solve_nh_diff.f90`` touches, so the
gfortran-only lanes need no ICON build."""

#: ``mo_nonhydro_types`` + ``mo_prepadv_types`` stand-ins.
MIN_STATE_TYPES_F90 = """\
module mo_nonhydro_types
  implicit none
  type :: t_nh_prog
    real(kind=8), pointer, contiguous :: w(:,:,:), vn(:,:,:), rho(:,:,:), exner(:,:,:), theta_v(:,:,:)
  end type t_nh_prog
  ! Same field names + ranks the real ICON t_nh_diag carries, so
  ! mo_solve_nh_diff's clone_diag_indep / compare_diag compile unchanged.
  type :: t_nh_diag
    real(kind=8), pointer, contiguous :: exner_pr(:,:,:) => null(), mass_fl_e(:,:,:) => null(), &
      rho_ic(:,:,:) => null(), theta_v_ic(:,:,:) => null(), grf_tend_vn(:,:,:) => null(), &
      grf_tend_w(:,:,:) => null(), grf_tend_rho(:,:,:) => null(), grf_tend_mflx(:,:,:) => null(), &
      grf_bdy_mflx(:,:,:) => null(), grf_tend_thv(:,:,:) => null(), vn_ie_int(:,:,:) => null(), &
      vn_ie_ubc(:,:,:) => null(), w_int(:,:,:) => null(), w_ubc(:,:,:) => null(), &
      theta_v_ic_int(:,:,:) => null(), theta_v_ic_ubc(:,:,:) => null(), rho_ic_int(:,:,:) => null(), &
      rho_ic_ubc(:,:,:) => null(), mflx_ic_int(:,:,:) => null(), mflx_ic_ubc(:,:,:) => null(), &
      vn_incr(:,:,:) => null(), exner_incr(:,:,:) => null(), rho_incr(:,:,:) => null(), &
      vt(:,:,:) => null(), ddt_exner_phy(:,:,:) => null(), ddt_vn_phy(:,:,:) => null(), &
      exner_dyn_incr(:,:,:) => null(), vn_ie(:,:,:) => null(), w_concorr_c(:,:,:) => null(), &
      mass_fl_e_sv(:,:,:) => null(), ddt_vn_dyn(:,:,:) => null(), ddt_vn_dmp(:,:,:) => null(), &
      ddt_vn_adv(:,:,:) => null(), ddt_vn_cor(:,:,:) => null(), ddt_vn_pgr(:,:,:) => null(), &
      ddt_vn_phd(:,:,:) => null(), ddt_vn_iau(:,:,:) => null(), ddt_vn_ray(:,:,:) => null(), &
      ddt_vn_grf(:,:,:) => null()
    real(kind=8), pointer, contiguous :: ddt_vn_apc_pc(:,:,:,:) => null(), &
      ddt_vn_cor_pc(:,:,:,:) => null(), ddt_w_adv_pc(:,:,:,:) => null()
    ! non-pointer scalar the velocity callback MAX-accumulates across substeps;
    ! value-copied by the shallow ``dst = src`` (NOT re-pointed) and compared.
    real(kind=8) :: max_vcfl_dyn = 0.0d0
  end type t_nh_diag
  type :: t_nh_state
    type(t_nh_prog), allocatable :: prog(:)
    type(t_nh_diag) :: diag
  end type t_nh_state
end module mo_nonhydro_types

module mo_prepadv_types
  implicit none
  type :: t_prepare_adv
    real(kind=8), pointer, contiguous :: mass_flx_me(:,:,:), mass_flx_ic(:,:,:), vol_flx_ic(:,:,:), vn_traj(:,:,:)
  end type t_prepare_adv
end module mo_prepadv_types
"""

#: Geometry stubs only the patched-module compile check needs (diff helpers never reference them).
MIN_GEOMETRY_TYPES_F90 = """\
module mo_model_domain
  implicit none
  type :: t_patch
    integer :: id
  end type t_patch
end module mo_model_domain

module mo_intp_data_strc
  implicit none
  type :: t_int_state
    integer :: id
  end type t_int_state
end module mo_intp_data_strc
"""
