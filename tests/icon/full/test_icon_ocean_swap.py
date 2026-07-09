"""Same-signature differential swap of ICON's ocean
``solve_free_sfc_ab_mimetic`` body for our binding.

The ocean twin of :file:`test_icon_solve_nh_swap.py`.  ICON's call site
(``mo_ocean_ab_timestepping``) stays untouched.  ``solve_free_sfc_ab_mimetic``
in ``mo_ocean_ab_timestepping_mimetic.f90`` becomes a DIFFERENTIAL DRIVER --
the SUBROUTINE signature and the dummy declarations stay byte-for-byte
identical -- that deep-copies the mutable state (``mo_ocean_diff``), forwards
to ``solve_free_sfc_dace_icon`` (the free-standing SDFG wrapper) as the DUT,
runs the original body -- preserved verbatim as ``solve_free_sfc_ref`` -- as
the REF, and compares the two bit-for-bit.

This test pins:

  * The patched signature surface (header + INTENT declarations) is
    byte-identical to the pristine, so ICON's caller is unaffected.
  * The differential driver structure: clone before DUT, DUT before REF,
    REF on the clone, compares + a_veloc_v snapshot dance, ref renamed.
  * The patched file parses through ``gfortran -fsyntax-only`` against
    minimal STAND-IN type modules (the host has no ICON build tree, so
    ICON's own ``.mod`` files are unavailable; the stand-ins carry the real
    member names/ranks/dummy-argument names, cross-checked against the
    submodule source).  ``mo_ocean_diff.f90`` is compiled FIRST in the same
    invocation so its ``.mod`` is on hand -- this also pins the driver's
    calls against the diff module's real interfaces.

The patched body is wrapper-aware but doesn't yet require the SDFG ``.so``
to exist; runtime resolution is a separate concern (mirrors the atmosphere
swap test).
"""
import os
import subprocess
from pathlib import Path

import pytest

from icon.full._fc import (
    FORTRAN_COMPILERS,
    cpp_flag,
    fortran_compiler_flags,
    syntax_check_argv,
)
from icon.full._icon_ocean_patch import (
    OCEAN_WRAPPER_NAME,
    apply_ocean_solve_patch,
    write_patched_ocean_solve,
)

# The syntax check compiles gfortran-flavoured stand-ins (CLASS(*) dummies for
# the solver-construct calls, gfortran ``.mod`` semantics); parametrize
# gfortran-only so the flang/nvfortran slots are never emitted as runtime
# skips -- mirrors ``test_icon_solve_nh_swap.py``.
GFORTRAN_COMPILERS = [p for p in FORTRAN_COMPILERS if "gfortran" in (p.id or "")]

_HERE = Path(__file__).resolve().parent
#: The differential helper module the patched driver ``USE``s.  Compiled ahead
#: of the patched source in the syntax check so its ``.mod`` is on hand.
_DIFF_F90 = _HERE / "mo_ocean_diff.f90"
_ICON_SRC = Path(os.environ.get("ICON_SRC", str(_HERE / "icon-model")))

_PRISTINE = _ICON_SRC / "src" / "ocean" / "dynamics" / "mo_ocean_ab_timestepping_mimetic.f90"
_PRISTINE_BAK = _PRISTINE.with_suffix(".f90.bak")


def _real_source() -> Path:
    return _PRISTINE_BAK if _PRISTINE_BAK.is_file() else _PRISTINE


_HAVE_ICON = _real_source().is_file()

# Every test here reads ICON's real ocean source through the icon-model
# submodule, which only the heavy CI lane checks out -> ``long``.
pytestmark = pytest.mark.long

#: The 11 dummies of ``solve_free_sfc_ab_mimetic``, in call order.  The driver
#: must forward every one of them to the DUT wrapper (a regression that drops
#: one would silently leave it default-initialised on the wrapper side).
_DUMMIES = ("patch_3d", "ocean_state", "p_ext_data", "p_as", "p_oce_sfc", "p_phys_param", "timestep", "op_coeffs",
            "solverCoeff_sp", "ret_status", "lacc")

# ---------------------------------------------------------------------------
# Patch-side tests (no compiler required).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAVE_ICON, reason="icon-model submodule not checked out")
def test_patch_preserves_signature():
    """The patched file's ``SUBROUTINE solve_free_sfc_ab_mimetic(...)`` header
    + the INTENT dummy declarations are byte-for-byte identical to the
    pristine.  A change anywhere in the signature surface would break ICON's
    ``mo_ocean_ab_timestepping`` call site, so we pin it explicitly."""
    pristine = _real_source().read_text()
    patched = apply_ocean_solve_patch(pristine)

    def _signature_surface(src: str) -> list:
        """The header line(s) + every ``INTENT(...)`` dummy declaration in the
        routine.  This is the ABI surface ICON's callers see; the USE the
        patch adds inside the body is internal and intentionally excluded."""
        lines = src.splitlines()
        start = next(i for i, ln in enumerate(lines) if "SUBROUTINE solve_free_sfc_ab_mimetic(" in ln)
        end = start
        while lines[end].rstrip().endswith("&"):
            end += 1
        surface = list(lines[start:end + 1])
        # Pick up every INTENT(...) line inside the routine UP TO the first
        # ``INTERFACE`` block.  After the patch, the wrapper is declared via
        # an inner INTERFACE with its own dummy declarations (the wrapper's
        # c_bool / c_int types, not solve_free_sfc's); those are internal to
        # the routine and not the ABI surface ICON callers see.
        for i in range(end + 1, len(lines)):
            stripped = lines[i].lstrip().upper()
            if stripped.startswith("END SUBROUTINE SOLVE_FREE_SFC_AB_MIMETIC"):
                break
            if stripped.startswith("INTERFACE"):
                break
            if "INTENT" in stripped:
                surface.append(lines[i])
        return surface

    pristine_sig = "\n".join(_signature_surface(pristine))
    patched_sig = "\n".join(_signature_surface(patched))
    assert pristine_sig == patched_sig, ("patched signature drifted from pristine -- ICON callers "
                                         "would see a different surface.\nFirst differing chars:\n"
                                         f"  pristine: {pristine_sig[:200]!r}\n"
                                         f"  patched:  {patched_sig[:200]!r}")


@pytest.mark.skipif(not _HAVE_ICON, reason="icon-model submodule not checked out")
def test_patched_body_calls_wrapper():
    """The driver forwards to ``solve_free_sfc_dace_icon`` with EVERY one of
    the 11 dummies (``lacc`` via the resolved ``lzacc__dace`` cast because the
    original dummy is OPTIONAL and the wrapper's c_bool is not)."""
    patched = apply_ocean_solve_patch(_real_source().read_text())
    assert f"CALL {OCEAN_WRAPPER_NAME}(" in patched
    dut_call = patched[patched.index(f"CALL {OCEAN_WRAPPER_NAME}("):]
    dut_call = dut_call[:dut_call.index("\n\n")]
    for arg in _DUMMIES[:-2]:
        assert arg in dut_call, f"forwarded arg {arg!r} missing from the DUT call"
    # ret_status is forwarded raw (ICON keeps the DUT's status); lacc goes
    # through set_acc_host_or_device + a 1-byte cast.
    assert "ret_status" in dut_call
    assert "LOGICAL(lzacc__dace, kind=1)" in dut_call


@pytest.mark.skipif(not _HAVE_ICON, reason="icon-model submodule not checked out")
def test_differential_driver_injected():
    """The differential patch KEEPS the original body as the bit-exact
    REFERENCE (renamed ``solve_free_sfc_ref``) and injects the driver --
    clone -> a_veloc_v snapshot -> DUT -> park/restore -> REF -> compare ->
    free -- as the new ``solve_free_sfc_ab_mimetic``.  So the patched file
    GROWS by roughly the driver block; it does NOT shrink."""
    pristine = _real_source().read_text()
    patched = apply_ocean_solve_patch(pristine)

    # The original body survives verbatim, renamed, AFTER the driver (ICON's
    # call site resolves ``solve_free_sfc_ab_mimetic``).
    assert "SUBROUTINE solve_free_sfc_ref(" in patched
    assert "END SUBROUTINE solve_free_sfc_ref" in patched

    # The differential harness is injected into the new driver, emitted
    # BEFORE the renamed reference body, with the run order pinned:
    # clone -> snapshot -> DUT -> park+restore -> REF -> compares -> enforce.
    assert "USE mo_ocean_diff" in patched
    order = [
        "CALL clone_ocean_state_indep(ocean_state, oce_ref__dace)",
        "CALL clone_field3(aveloc_pre__dace, p_phys_param%a_veloc_v)",
        f"CALL {OCEAN_WRAPPER_NAME}(",
        "CALL clone_field3(aveloc_dut__dace, p_phys_param%a_veloc_v)",
        "CALL restore_field3(p_phys_param%a_veloc_v, aveloc_pre__dace)",
        "CALL solve_free_sfc_ref(patch_3d, oce_ref__dace,",
        "CALL compare_ocean_prog(ocean_state, oce_ref__dace, nnew(1),",
        "CALL compare_ocean_diag(ocean_state%p_diag, oce_ref__dace%p_diag,",
        "CALL compare_ocean_aux(ocean_state%p_aux, oce_ref__dace%p_aux,",
        "CALL compare_field3(aveloc_dut__dace, p_phys_param%a_veloc_v,",
        "CALL ocean_diff_enforce(",
        "CALL restore_field3(p_phys_param%a_veloc_v, aveloc_dut__dace)",
        "CALL free_ocean_state_clone(oce_ref__dace)",
        "SUBROUTINE solve_free_sfc_ref(",
    ]
    positions = [patched.index(s) for s in order]
    assert positions == sorted(positions), "differential driver statements out of order"

    # The file GROWS by ~the injected driver, NOT shrinks -- and the growth is
    # a small fraction of the file, so the body was not duplicated twice.
    pristine_n = len(pristine.splitlines())
    patched_n = len(patched.splitlines())
    assert patched_n > pristine_n, ("the differential patch keeps the body as the REF, so the file must "
                                    f"GROW: patched={patched_n} pristine={pristine_n}")
    growth = patched_n - pristine_n
    assert growth < pristine_n // 2, (f"patched grew by {growth} lines -- far more than the injected driver; "
                                      "the original body may have been duplicated instead of renamed")


@pytest.mark.skipif(not _HAVE_ICON, reason="icon-model submodule not checked out")
def test_write_patched_ocean_solve(tmp_path: Path):
    """Round-trip the patch through the disk-writing helper."""
    out = tmp_path / "mo_ocean_ab_timestepping_mimetic.f90"
    line_count = write_patched_ocean_solve(_real_source(), out)
    assert out.is_file()
    assert line_count > 200, f"patched file looks truncated ({line_count} lines)"


# ---------------------------------------------------------------------------
# Compile-side test (gfortran + stand-in type modules; no ICON build).
# ---------------------------------------------------------------------------

#: Minimal stand-in modules for the ``-fsyntax-only`` check: EXACTLY the
#: symbols the patched module ``USE``s, with dummy-argument NAMES matching the
#: real ICON interfaces (the module body calls several of them with keyword
#: arguments) and type members cross-checked against the real
#: ``mo_ocean_types.f90`` / ``mo_model_domain.f90``.
_STANDINS = """\
! Minimal stand-in modules for the gfortran -fsyntax-only check of the patched
! mo_ocean_ab_timestepping_mimetic.f90 (no ICON build on the host).  Each
! module carries EXACTLY the symbols the patched module USEs, with dummy-arg
! NAMES matching the real ICON interfaces (the module body calls several of
! them with keyword arguments).
module mo_kind
  implicit none
  public
  integer, parameter :: wp = 8, sp = 4
end module mo_kind

module mo_parallel_config
  implicit none
  public
  integer :: nproma = 8
end module mo_parallel_config

module mo_impl_constants
  implicit none
  public
  integer, parameter :: sea_boundary = -1, max_char_length = 1024, min_dolic = 2
end module mo_impl_constants

module mo_dbg_nml
  implicit none
  public
  integer :: idbg_mxmn = 0, idbg_val = 0
end module mo_dbg_nml

module mo_ocean_nml
  use mo_kind, only: wp
  implicit none
  public
  integer :: n_zlev = 4
  real(wp) :: solver_tolerance = 1.0e-10_wp
  real(wp) :: ab_const = 0.1_wp, ab_beta = 0.6_wp, ab_gam = 0.6_wp
  integer :: iswm_oce = 0, iforc_oce = 0, no_tracer = 2
  logical :: l_rigid_lid = .false., l_edge_based = .false.
  logical :: use_absolute_solver_tolerance = .true.
  integer :: solver_max_restart_iterations = 100, solver_max_iter_per_restart = 26
  real(wp) :: dhdtw_abort = 3.17e-11_wp
  integer :: select_transfer = 0
  integer :: select_solver = 2
  integer, parameter :: select_gmres = 1, select_gmres_r = 2, select_mres = 3
  integer, parameter :: select_gmres_mp_r = 4, select_cg = 5, select_cgo = 6
  integer, parameter :: select_cgj = 7, select_bcgs = 8, select_legacy_gmres = 9
  integer, parameter :: select_cg_mp = 10
  logical :: use_continuity_correction = .true.
  integer :: solver_max_iter_per_restart_sp = 26
  real(wp) :: solver_tolerance_sp = 1.0e-10_wp
  integer, parameter :: No_Forcing = 10
  integer :: MASS_MATRIX_INVERSION_TYPE = 0
  integer, parameter :: MASS_MATRIX_INVERSION_ADVECTION = 2
  integer, parameter :: MASS_MATRIX_INVERSION_ALLTERMS = 1
  real(wp) :: solver_tolerance_comp = 1.0e-16_wp
  integer :: PPscheme_type = 4
  integer, parameter :: PPscheme_ICON_Edge_vnPredict_type = 4
  integer :: solver_FirstGuess = 0
  real(wp) :: MassMatrix_solver_tolerance = 1.0e-11_wp
  logical :: createSolverMatrix = .false., l_solver_compare = .false.
  integer :: solver_comp_nsteps = 100
  logical :: use_ssh_in_momentum_eq = .true.
  integer :: vert_mix_type = 1
  integer, parameter :: vmix_pp = 1
end module mo_ocean_nml

module mo_run_config
  use mo_kind, only: wp
  implicit none
  public
  real(wp) :: dtime = 600.0_wp
  integer :: debug_check_level = 0
end module mo_run_config

module mo_timer
  implicit none
  public
  integer :: timers_level = 0
  integer :: timer_extra1 = 1, timer_extra2 = 2, timer_extra3 = 3, timer_extra4 = 4
  integer :: timer_ab_expl = 5, timer_ab_rhs4sfc = 6
contains
  subroutine timer_start(timer)
    integer, intent(in) :: timer
  end subroutine timer_start
  subroutine timer_stop(timer)
    integer, intent(in) :: timer
  end subroutine timer_stop
end module mo_timer

module mo_dynamics_config
  implicit none
  public
  ! per-domain time-level indices (the real ICON declares nold/nnew(MAX_DOM))
  integer :: nold(2) = [1, 1], nnew(2) = [2, 2]
end module mo_dynamics_config

module mo_physical_constants
  use mo_kind, only: wp
  implicit none
  public
  real(wp), parameter :: grav = 9.80665_wp
end module mo_physical_constants

module mo_ocean_initialization
  implicit none
  public
contains
  logical function is_initial_timestep(timestep)
    integer, intent(in) :: timestep
    is_initial_timestep = timestep == 1
  end function is_initial_timestep
end module mo_ocean_initialization

module mo_math_types
  use mo_kind, only: wp
  implicit none
  public
  type :: t_cartesian_coordinates
    real(wp) :: x(3)
  end type t_cartesian_coordinates
end module mo_math_types

module mo_grid_subset
  implicit none
  public
  type :: t_subset_range
    integer :: start_block = 1, end_block = 1, end_index = 1
  end type t_subset_range
contains
  subroutine get_index_range(subset_range, block_no, start_index, end_index)
    type(t_subset_range), intent(in) :: subset_range
    integer, intent(in) :: block_no
    integer, intent(out) :: start_index, end_index
    start_index = 1
    end_index = 1
  end subroutine get_index_range
end module mo_grid_subset

module mo_model_domain
  use mo_kind, only: wp
  use mo_grid_subset, only: t_subset_range
  implicit none
  public
  type :: t_grid_cells
    type(t_subset_range) :: in_domain, owned, all
    integer :: max_connectivity = 3
    integer, pointer :: edge_idx(:, :, :) => null(), edge_blk(:, :, :) => null()
    real(wp), pointer :: area(:, :) => null()
  end type t_grid_cells
  type :: t_grid_edges
    type(t_subset_range) :: in_domain, owned, all
  end type t_grid_edges
  type :: t_grid_vertices
    type(t_subset_range) :: owned
  end type t_grid_vertices
  type :: t_patch
    type(t_grid_cells) :: cells
    type(t_grid_edges) :: edges
    type(t_grid_vertices) :: verts
    integer :: nblks_e = 1, alloc_cell_blocks = 1
  end type t_patch
  type :: t_patch_1d
    integer, pointer :: dolic_c(:, :) => null(), dolic_e(:, :) => null()
    real(wp), pointer :: prism_thick_c(:, :, :) => null(), prism_thick_e(:, :, :) => null()
    real(wp), pointer :: prism_thick_flat_sfc_e(:, :, :) => null()
    real(wp), pointer :: del_zlev_m(:) => null()
  end type t_patch_1d
  type :: t_patch_3d
    type(t_patch), pointer :: p_patch_2d(:) => null()
    type(t_patch_1d), pointer :: p_patch_1d(:) => null()
    integer, pointer :: lsm_c(:, :, :) => null(), lsm_e(:, :, :) => null()
    real(wp), pointer :: wet_c(:, :, :) => null()
  end type t_patch_3d
end module mo_model_domain

module mo_ocean_types
  use mo_kind, only: wp
  use mo_math_types, only: t_cartesian_coordinates
  implicit none
  public
  type :: t_hydro_ocean_prog
    real(wp), pointer :: h(:, :) => null(), eta_c(:, :) => null(), stretch_c(:, :) => null()
    real(wp), pointer :: vn(:, :, :) => null()
    real(wp), pointer :: tracer(:, :, :, :) => null()
  end type t_hydro_ocean_prog
  type :: t_hydro_ocean_diag
    real(wp), pointer :: vort(:, :, :) => null(), grad(:, :, :) => null(), &
      & veloc_adv_horz(:, :, :) => null(), veloc_adv_vert(:, :, :) => null(), &
      & laplacian_horz(:, :, :) => null(), laplacian_vert(:, :, :) => null(), &
      & press_hyd(:, :, :) => null(), press_grad(:, :, :) => null(), &
      & vn_pred(:, :, :) => null(), vn_pred_ptp(:, :, :) => null(), &
      & vn_time_weighted(:, :, :) => null(), w(:, :, :) => null(), w_old(:, :, :) => null(), &
      & mass_flx_e(:, :, :) => null(), div_mass_flx_c(:, :, :) => null(), &
      & ptp_vn(:, :, :) => null(), rho(:, :, :) => null(), kin(:, :, :) => null()
    real(wp), pointer :: h_e(:, :) => null(), thick_c(:, :) => null(), thick_e(:, :) => null()
  end type t_hydro_ocean_diag
  type :: t_hydro_ocean_aux
    real(wp), pointer :: g_n(:, :, :) => null(), g_nm1(:, :, :) => null(), g_nimd(:, :, :) => null()
    real(wp), pointer :: p_rhs_sfc_eq(:, :) => null(), bc_top_vn(:, :) => null(), &
      & bc_bot_vn(:, :) => null(), bc_top_u(:, :) => null(), bc_top_v(:, :) => null(), &
      & bc_top_WindStress(:, :) => null(), bc_total_top_potential(:, :) => null()
    type(t_cartesian_coordinates), pointer :: bc_top_veloc_cc(:, :) => null()
  end type t_hydro_ocean_aux
  type :: t_hydro_ocean_state
    type(t_hydro_ocean_prog), pointer :: p_prog(:) => null()
    type(t_hydro_ocean_diag) :: p_diag
    type(t_hydro_ocean_aux) :: p_aux
  end type t_hydro_ocean_state
  type :: t_operator_coeff
    real(wp), pointer :: div_coeff(:, :, :, :) => null()
    real(wp), pointer :: grad_coeff(:, :, :) => null()
  end type t_operator_coeff
  type :: t_solverCoeff_singlePrecision
    integer :: dummy = 0
  end type t_solverCoeff_singlePrecision
end module mo_ocean_types

module mo_ext_data_types
  implicit none
  public
  type :: t_external_data
    integer :: dummy = 0
  end type t_external_data
end module mo_ext_data_types

module mo_exception
  implicit none
  public
  character(len=1024) :: message_text = ""
contains
  subroutine message(name, text)
    character(*), intent(in) :: name, text
  end subroutine message
  subroutine warning(name, text)
    character(*), intent(in) :: name, text
  end subroutine warning
  subroutine finish(name, text)
    character(*), intent(in) :: name
    character(*), intent(in), optional :: text
  end subroutine finish
end module mo_exception

module mo_util_dbg_prnt
  use mo_kind, only: wp
  use mo_grid_subset, only: t_subset_range
  implicit none
  public
  interface dbg_print
    module procedure dbg_print_2d
    module procedure dbg_print_3d
  end interface dbg_print
contains
  subroutine dbg_print_2d(description, p_array, place, level, in_subset)
    character(*), intent(in) :: description
    real(wp), intent(in) :: p_array(:, :)
    character(*), intent(in) :: place
    integer, intent(in) :: level
    type(t_subset_range), intent(in), optional :: in_subset
  end subroutine dbg_print_2d
  subroutine dbg_print_3d(description, p_array, place, level, in_subset)
    character(*), intent(in) :: description
    real(wp), intent(in) :: p_array(:, :, :)
    character(*), intent(in) :: place
    integer, intent(in) :: level
    type(t_subset_range), intent(in), optional :: in_subset
  end subroutine dbg_print_3d
  subroutine debug_print_MaxMinMean(description, values, place, level)
    character(*), intent(in) :: description
    real(wp), intent(in) :: values(3)
    character(*), intent(in) :: place
    integer, intent(in) :: level
  end subroutine debug_print_MaxMinMean
end module mo_util_dbg_prnt

module mo_sync
  use mo_kind, only: wp
  use mo_model_domain, only: t_patch
  implicit none
  public
  integer, parameter :: sync_e = 1, sync_c = 2
  interface sync_patch_array
    module procedure sync_patch_array_2d
    module procedure sync_patch_array_3d
  end interface sync_patch_array
contains
  subroutine sync_patch_array_2d(typ, p_patch, arr, lacc)
    integer, intent(in) :: typ
    type(t_patch), intent(in) :: p_patch
    real(wp), intent(inout) :: arr(:, :)
    logical, intent(in), optional :: lacc
  end subroutine sync_patch_array_2d
  subroutine sync_patch_array_3d(typ, p_patch, arr, lacc)
    integer, intent(in) :: typ
    type(t_patch), intent(in) :: p_patch
    real(wp), intent(inout) :: arr(:, :, :)
    logical, intent(in), optional :: lacc
  end subroutine sync_patch_array_3d
  subroutine sync_patch_array_mult(typ, p_patch, nfields, f3din1, f3din2, lacc)
    integer, intent(in) :: typ
    type(t_patch), intent(in) :: p_patch
    integer, intent(in) :: nfields
    real(wp), intent(inout), optional :: f3din1(:, :, :), f3din2(:, :, :)
    logical, intent(in), optional :: lacc
  end subroutine sync_patch_array_mult
end module mo_sync

module mo_ocean_physics_types
  use mo_kind, only: wp
  implicit none
  public
  type :: t_ho_params
    real(wp), pointer :: a_veloc_v(:, :, :) => null()
  end type t_ho_params
end module mo_ocean_physics_types

module mo_ocean_surface_types
  implicit none
  public
  type :: t_ocean_surface
    integer :: dummy = 0
  end type t_ocean_surface
  type :: t_atmos_for_ocean
    integer :: dummy = 0
  end type t_atmos_for_ocean
end module mo_ocean_surface_types

module mo_ocean_boundcond
  use mo_kind, only: wp
  use mo_model_domain, only: t_patch_3d
  use mo_ocean_types, only: t_hydro_ocean_state, t_operator_coeff
  use mo_ocean_surface_types, only: t_ocean_surface
  implicit none
  public
contains
  subroutine top_bound_cond_horz_veloc(patch_3d, ocean_state, p_op_coeff, p_oce_sfc, lacc)
    type(t_patch_3d), pointer, intent(in) :: patch_3d
    type(t_hydro_ocean_state), intent(inout) :: ocean_state
    type(t_operator_coeff), intent(in) :: p_op_coeff
    type(t_ocean_surface) :: p_oce_sfc
    logical, intent(in), optional :: lacc
  end subroutine top_bound_cond_horz_veloc
  subroutine VelocityBottomBoundaryCondition_onBlock(patch_3d, blockNo, start_edge_index, end_edge_index, &
    & vn_old, vn_pred, bc_bot_vn, lacc)
    type(t_patch_3d), pointer, intent(in) :: patch_3d
    integer, intent(in) :: blockNo, start_edge_index, end_edge_index
    real(wp) :: vn_old(:, :), vn_pred(:, :)
    real(wp) :: bc_bot_vn(:)
    logical, intent(in), optional :: lacc
  end subroutine VelocityBottomBoundaryCondition_onBlock
end module mo_ocean_boundcond

module mo_ocean_thermodyn
  use mo_kind, only: wp
  use mo_model_domain, only: t_patch_3d
  implicit none
  public
contains
  subroutine calculate_density(patch_3d, tracer, rho, lacc)
    type(t_patch_3d), pointer, intent(in) :: patch_3d
    real(wp), intent(in) :: tracer(:, :, :, :)
    real(wp), intent(inout) :: rho(:, :, :)
    logical, intent(in), optional :: lacc
  end subroutine calculate_density
  subroutine calc_internal_press_grad(patch_3d, rho, pressure_hyd, bc_total_top_potential, grad_coeff, &
    & press_grad, lacc)
    type(t_patch_3d), pointer, intent(in) :: patch_3d
    real(wp), intent(in) :: rho(:, :, :)
    real(wp), intent(inout) :: pressure_hyd(:, :, :)
    real(wp), intent(in) :: bc_total_top_potential(:, :)
    real(wp), intent(in) :: grad_coeff(:, :, :)
    real(wp), intent(inout) :: press_grad(:, :, :)
    logical, intent(in), optional :: lacc
  end subroutine calc_internal_press_grad
end module mo_ocean_thermodyn

module mo_ocean_pp_scheme
  use mo_kind, only: wp
  use mo_model_domain, only: t_patch_3d
  use mo_ocean_types, only: t_hydro_ocean_state
  implicit none
  public
contains
  subroutine ICON_PP_Edge_vnPredict_scheme(patch_3d, blockNo, start_index, end_index, ocean_state, &
    & vn_predict, lacc)
    type(t_patch_3d), pointer, intent(in) :: patch_3d
    integer, intent(in) :: blockNo, start_index, end_index
    type(t_hydro_ocean_state), target :: ocean_state
    real(wp) :: vn_predict(:, :)
    logical, intent(in), optional :: lacc
  end subroutine ICON_PP_Edge_vnPredict_scheme
end module mo_ocean_pp_scheme

module mo_scalar_product
  use mo_kind, only: wp
  use mo_model_domain, only: t_patch_3d
  use mo_ocean_types, only: t_operator_coeff
  implicit none
  public
  interface map_edges2edges_viacell_3d_const_z
    module procedure map_edges2edges_viacell_3d_const_z_3d
    module procedure map_edges2edges_viacell_3d_const_z_2d
  end interface map_edges2edges_viacell_3d_const_z
contains
  subroutine map_edges2edges_viacell_3d_const_z_3d(patch_3d, in_vn_e, operators_coefficients, out_vn_e, lacc)
    type(t_patch_3d), pointer, intent(in) :: patch_3d
    real(wp), intent(in) :: in_vn_e(:, :, :)
    type(t_operator_coeff), intent(in) :: operators_coefficients
    real(wp), intent(inout) :: out_vn_e(:, :, :)
    logical, intent(in), optional :: lacc
  end subroutine map_edges2edges_viacell_3d_const_z_3d
  subroutine map_edges2edges_viacell_3d_const_z_2d(patch_3d, in_vn_e, operators_coefficients, out_vn_e, lacc)
    type(t_patch_3d), pointer, intent(in) :: patch_3d
    real(wp), intent(in) :: in_vn_e(:, :)
    type(t_operator_coeff), intent(in) :: operators_coefficients
    real(wp), intent(inout) :: out_vn_e(:, :)
    logical, intent(in), optional :: lacc
  end subroutine map_edges2edges_viacell_3d_const_z_2d
  subroutine map_edges2edges_viacell_2D_per_level(patch_3d, in_vn_e, operators_coefficients, out_vn_e, level)
    type(t_patch_3d), pointer, intent(in) :: patch_3d
    real(wp), intent(in) :: in_vn_e(:, :)
    type(t_operator_coeff), intent(in) :: operators_coefficients
    real(wp), intent(inout) :: out_vn_e(:, :)
    integer, intent(in) :: level
  end subroutine map_edges2edges_viacell_2D_per_level
end module mo_scalar_product

module mo_ocean_math_operators
  use mo_kind, only: wp
  use mo_model_domain, only: t_patch, t_patch_3d
  implicit none
  public
contains
  subroutine div_oce_3D_onTriangles_onBlock(vec_e, patch_3D, div_coeff, div_vec_c, &
    & blockNo, start_index, end_index, start_level, end_level, lacc)
    real(wp), intent(in) :: vec_e(:, :, :)
    type(t_patch_3d), pointer, intent(in) :: patch_3D
    real(wp), intent(in) :: div_coeff(:, :, :, :)
    real(wp), intent(inout) :: div_vec_c(:, :)
    integer, intent(in) :: blockNo, start_index, end_index
    integer, intent(in), optional :: start_level, end_level
    logical, intent(in), optional :: lacc
  end subroutine div_oce_3D_onTriangles_onBlock
  subroutine div_oce_3D_general_onBlock(vec_e, patch_3D, div_coeff, div_vec_c, &
    & blockNo, start_index, end_index, start_level, end_level, lacc)
    real(wp), intent(in) :: vec_e(:, :, :)
    type(t_patch_3d), pointer, intent(in) :: patch_3D
    real(wp), intent(in) :: div_coeff(:, :, :, :)
    real(wp), intent(inout) :: div_vec_c(:, :)
    integer, intent(in) :: blockNo, start_index, end_index
    integer, intent(in), optional :: start_level, end_level
    logical, intent(in), optional :: lacc
  end subroutine div_oce_3D_general_onBlock
  subroutine div_oce_3d(vec_e, patch_3D, div_coeff, div_vec_c, opt_start_level, opt_end_level, &
    & subset_range, lacc)
    real(wp), intent(in) :: vec_e(:, :, :)
    type(t_patch_3d), pointer, intent(in) :: patch_3D
    real(wp), intent(in) :: div_coeff(:, :, :, :)
    real(wp), intent(inout) :: div_vec_c(:, :, :)
    integer, intent(in), optional :: opt_start_level, opt_end_level
    integer, intent(in), optional :: subset_range
    logical, intent(in), optional :: lacc
  end subroutine div_oce_3d
  subroutine smooth_onCells(patch_3D, in_value, out_value, smooth_weights, has_missValue, missValue, lacc)
    type(t_patch_3d), pointer, intent(in) :: patch_3D
    real(wp), intent(in) :: in_value(:, :)
    real(wp), intent(inout) :: out_value(:, :)
    real(wp), intent(in) :: smooth_weights(1:2)
    logical, intent(in) :: has_missValue
    real(wp), intent(in) :: missValue
    logical, intent(in), optional :: lacc
  end subroutine smooth_onCells
  subroutine grad_fd_norm_oce_2d_onBlock(psi_c, patch_2D, grad_coeff, grad_norm_psi_e, &
    & start_index, end_index, blockNo, lacc)
    real(wp), intent(in) :: psi_c(:, :)
    type(t_patch), intent(in) :: patch_2D
    real(wp), intent(in) :: grad_coeff(:)
    real(wp), intent(inout) :: grad_norm_psi_e(:)
    integer, intent(in) :: start_index, end_index, blockNo
    logical, intent(in), optional :: lacc
  end subroutine grad_fd_norm_oce_2d_onBlock
  subroutine grad_fd_norm_oce_2d_3d(psi_c, patch_2D, grad_coeff, grad_norm_psi_e, subset_range, lacc)
    real(wp), intent(in) :: psi_c(:, :)
    type(t_patch), intent(in) :: patch_2D
    real(wp), intent(in) :: grad_coeff(:, :)
    real(wp), intent(inout) :: grad_norm_psi_e(:, :)
    integer, intent(in), optional :: subset_range
    logical, intent(in), optional :: lacc
  end subroutine grad_fd_norm_oce_2d_3d
end module mo_ocean_math_operators

module mo_ocean_velocity_advection
  use mo_kind, only: wp
  use mo_model_domain, only: t_patch_3d
  use mo_ocean_types, only: t_hydro_ocean_diag, t_operator_coeff
  implicit none
  public
contains
  subroutine veloc_adv_horz_mimetic(patch_3D, vn_old, vn_new, p_diag, veloc_adv_horz_e, &
    & ocean_coefficients, lacc)
    type(t_patch_3d), pointer, intent(in) :: patch_3D
    real(wp), pointer, intent(inout) :: vn_old(:, :, :), vn_new(:, :, :)
    type(t_hydro_ocean_diag) :: p_diag
    real(wp), pointer, intent(inout) :: veloc_adv_horz_e(:, :, :)
    type(t_operator_coeff), target, intent(in) :: ocean_coefficients
    logical, intent(in), optional :: lacc
  end subroutine veloc_adv_horz_mimetic
  subroutine veloc_adv_vert_mimetic(patch_3D, p_diag, ocean_coefficients, veloc_adv_vert_e, lacc)
    type(t_patch_3d), target, intent(in) :: patch_3D
    type(t_hydro_ocean_diag) :: p_diag
    type(t_operator_coeff), intent(in) :: ocean_coefficients
    real(wp), intent(inout) :: veloc_adv_vert_e(:, :, :)
    logical, intent(in), optional :: lacc
  end subroutine veloc_adv_vert_mimetic
end module mo_ocean_velocity_advection

module mo_ocean_velocity_diffusion
  use mo_kind, only: wp
  use mo_model_domain, only: t_patch_3d
  use mo_ocean_types, only: t_hydro_ocean_diag, t_operator_coeff
  use mo_ocean_physics_types, only: t_ho_params
  implicit none
  public
contains
  subroutine velocity_diffusion(patch_3D, vn_in, physics_parameters, p_diag, operators_coeff, &
    & laplacian_diff, lacc)
    type(t_patch_3d), pointer, intent(in) :: patch_3D
    real(wp), pointer, intent(in) :: vn_in(:, :, :)
    type(t_ho_params) :: physics_parameters
    type(t_hydro_ocean_diag) :: p_diag
    type(t_operator_coeff), intent(in) :: operators_coeff
    real(wp), pointer, intent(inout) :: laplacian_diff(:, :, :)
    logical, intent(in), optional :: lacc
  end subroutine velocity_diffusion
  subroutine velocity_diffusion_vertical_implicit_onBlock(patch_3d, velocity, a_v, &
    & operators_coefficients, start_index, end_index, edge_block, lacc)
    type(t_patch_3d), pointer, intent(in) :: patch_3d
    real(wp), intent(inout) :: velocity(:, :)
    real(wp), intent(inout) :: a_v(:, :)
    type(t_operator_coeff), intent(in) :: operators_coefficients
    integer, intent(in) :: start_index, end_index, edge_block
    logical, intent(in), optional :: lacc
  end subroutine velocity_diffusion_vertical_implicit_onBlock
end module mo_ocean_velocity_diffusion

module mo_grid_config
  implicit none
  public
  integer :: n_dom = 1
end module mo_grid_config

module mo_mpi
  implicit none
  public
contains
  subroutine work_mpi_barrier()
  end subroutine work_mpi_barrier
end module mo_mpi

module mo_statistics
  use mo_kind, only: wp
  use mo_grid_subset, only: t_subset_range
  implicit none
  public
contains
  function global_minmaxmean(values, in_subset) result(minmaxmean)
    real(wp), intent(in) :: values(:, :)
    type(t_subset_range), intent(in), optional :: in_subset
    real(wp) :: minmaxmean(3)
    minmaxmean = 0.0_wp
  end function global_minmaxmean
  subroutine print_value_location(values, value, in_subset)
    real(wp), intent(in) :: values(:, :)
    real(wp), intent(in) :: value
    type(t_subset_range), intent(in), optional :: in_subset
  end subroutine print_value_location
end module mo_statistics

module mo_ocean_solve_aux
  use mo_kind, only: wp
  implicit none
  public
  integer, parameter :: solve_gmres = 1, solve_cg = 2, solve_mres = 3, solve_bcgs = 4
  integer, parameter :: solve_legacy_gmres = 5
  integer, parameter :: solve_precon_none = 0, solve_precon_jac = 1, solve_cg_opt = 2
  integer, parameter :: solve_trans_scatter = 1, solve_trans_compact = 2
  integer, parameter :: solve_cell = 1, solve_edge = 2, solve_invalid = -1
  type :: t_ocean_solve_parm
    integer :: nidx = -1, m = 0, nr = 0, pt = 0
    real(wp) :: tol = 0.0_wp
  contains
    procedure :: init => ocean_solve_parm_init
  end type t_ocean_solve_parm
contains
  subroutine ocean_solve_parm_init(this, pt, nr, m, nblk_a, nblk, nidx, nidx_e, tol, use_atol)
    class(t_ocean_solve_parm), intent(inout) :: this
    integer, intent(in) :: pt, nr, m, nblk_a, nblk, nidx, nidx_e
    real(wp), intent(in) :: tol
    logical, intent(in) :: use_atol
  end subroutine ocean_solve_parm_init
end module mo_ocean_solve_aux

module mo_surface_height_lhs
  use mo_kind, only: wp
  use mo_model_domain, only: t_patch_3d
  use mo_ocean_types, only: t_operator_coeff, t_solverCoeff_singlePrecision
  implicit none
  public
  type :: t_surface_height_lhs
    integer :: dummy = 0
  contains
    procedure :: construct => surface_height_lhs_construct
  end type t_surface_height_lhs
contains
  subroutine surface_height_lhs_construct(this, patch_3d, thick_e, op_coeffs, solverCoeff_sp, lacc)
    class(t_surface_height_lhs), intent(inout) :: this
    type(t_patch_3d), pointer, intent(in) :: patch_3d
    real(wp), pointer, intent(in) :: thick_e(:, :)
    type(t_operator_coeff), target, intent(in) :: op_coeffs
    type(t_solverCoeff_singlePrecision), target, intent(in) :: solverCoeff_sp
    logical, intent(in), optional :: lacc
  end subroutine surface_height_lhs_construct
end module mo_surface_height_lhs

module mo_primal_flip_flop_lhs
  use mo_model_domain, only: t_patch_3d
  use mo_ocean_types, only: t_operator_coeff
  implicit none
  public
  type :: t_primal_flip_flop_lhs
    integer :: dummy = 0
  contains
    procedure :: construct => primal_flip_flop_lhs_construct
  end type t_primal_flip_flop_lhs
contains
  subroutine primal_flip_flop_lhs_construct(this, patch_3d, op_coeffs, level)
    class(t_primal_flip_flop_lhs), intent(inout) :: this
    type(t_patch_3d), pointer, intent(in) :: patch_3d
    type(t_operator_coeff), target, intent(in) :: op_coeffs
    integer, intent(in) :: level
  end subroutine primal_flip_flop_lhs_construct
end module mo_primal_flip_flop_lhs

module mo_ocean_solve_trivial_transfer
  use mo_model_domain, only: t_patch
  implicit none
  public
  type :: t_trivial_transfer
    integer :: dummy = 0
  contains
    procedure :: construct => trivial_transfer_construct
    procedure :: destruct => trivial_transfer_destruct
  end type t_trivial_transfer
contains
  subroutine trivial_transfer_construct(this, sync_type, patch_2d, lacc)
    class(t_trivial_transfer), intent(inout) :: this
    integer, intent(in) :: sync_type
    type(t_patch), target, intent(inout) :: patch_2d
    logical, intent(in), optional :: lacc
  end subroutine trivial_transfer_construct
  subroutine trivial_transfer_destruct(this)
    class(t_trivial_transfer), intent(inout) :: this
  end subroutine trivial_transfer_destruct
end module mo_ocean_solve_trivial_transfer

module mo_ocean_solve_subset_transfer
  use mo_model_domain, only: t_patch
  implicit none
  public
  type :: t_subset_transfer
    integer :: dummy = 0
  contains
    procedure :: construct => subset_transfer_construct
    procedure :: destruct => subset_transfer_destruct
  end type t_subset_transfer
contains
  subroutine subset_transfer_construct(this, sync_type, patch_2d, redfac, mode, lacc)
    class(t_subset_transfer), intent(inout) :: this
    integer, intent(in) :: sync_type
    type(t_patch), target, intent(inout) :: patch_2d
    integer, intent(in) :: redfac, mode
    logical, intent(in), optional :: lacc
  end subroutine subset_transfer_construct
  subroutine subset_transfer_destruct(this)
    class(t_subset_transfer), intent(inout) :: this
  end subroutine subset_transfer_destruct
end module mo_ocean_solve_subset_transfer

module mo_ocean_solve
  use mo_kind, only: wp
  use mo_ocean_solve_aux, only: t_ocean_solve_parm
  implicit none
  public
  type :: t_ocean_solve
    logical :: is_init = .false.
    character(len=64) :: sol_type_name = "standin"
    real(wp), pointer :: x_loc_wp(:, :) => null()
    real(wp), pointer :: b_loc_wp(:, :) => null()
    real(wp), pointer :: res_loc_wp(:) => null()
  contains
    procedure :: construct => ocean_solve_construct
    procedure :: solve => ocean_solve_solve
    procedure :: dump_matrix => ocean_solve_dump_matrix
  end type t_ocean_solve
contains
  subroutine ocean_solve_construct(this, solve_type, par, par_sp, lhs, transfer, lacc)
    class(t_ocean_solve), intent(inout) :: this
    integer, intent(in) :: solve_type
    type(t_ocean_solve_parm), intent(in) :: par, par_sp
    class(*), intent(inout) :: lhs
    class(*), intent(inout) :: transfer
    logical, intent(in), optional :: lacc
  end subroutine ocean_solve_construct
  subroutine ocean_solve_solve(this, niter, niter_sp, lacc)
    class(t_ocean_solve), intent(inout) :: this
    integer, intent(out) :: niter, niter_sp
    logical, intent(in), optional :: lacc
  end subroutine ocean_solve_solve
  subroutine ocean_solve_dump_matrix(this, lprecon, lacc)
    class(t_ocean_solve), intent(inout) :: this
    integer, intent(in) :: lprecon
    logical, intent(in), optional :: lacc
  end subroutine ocean_solve_dump_matrix
end module mo_ocean_solve

module mo_fortran_tools
  implicit none
  public
contains
  subroutine set_acc_host_or_device(lzacc, lacc)
    logical, intent(out) :: lzacc
    logical, intent(in), optional :: lacc
    lzacc = .false.
    if (present(lacc)) lzacc = lacc
  end subroutine set_acc_host_or_device
end module mo_fortran_tools
"""


@pytest.mark.skipif(not _HAVE_ICON, reason="icon-model submodule not checked out")
@pytest.mark.parametrize("fc", GFORTRAN_COMPILERS)
def test_patched_source_parses_through_fortran_compiler(fc, tmp_path: Path):
    """The Fortran compiler accepts the patched file (syntax-only) against the
    stand-in ``.mod`` set + the real ``mo_ocean_diff.f90``.  The patched
    file's INTERFACE block, the forwarding CALL arg order, and the driver's
    use of the diff module's interfaces are what's pinned here; any drift
    breaks ICON's caller or the deep-copy harness.

    The real source is preprocessed (``#include`` of the iconfor DSL +
    timer macros), so the check runs with ``-cpp`` against the submodule's
    ``src/include``."""
    fc_name, fc_path = fc

    out = tmp_path / "mo_ocean_ab_timestepping_mimetic_patched.f90"
    write_patched_ocean_solve(_real_source(), out)
    (tmp_path / "standins.f90").write_text(_STANDINS)

    include_flags = [f"-I{_ICON_SRC}/src/include"]

    # Stand-ins FIRST (they provide every ICON ``.mod`` the patched module
    # reads), then the diff module (compiled against the stand-in types, its
    # ``.mod`` feeds the driver's USE), then the patched source.
    subprocess.check_call([
        fc_path, *syntax_check_argv(fc_name, tmp_path),
        cpp_flag(fc_name), *fortran_compiler_flags(fc_name), *include_flags,
        str(tmp_path / "standins.f90"),
        str(_DIFF_F90),
        str(out)
    ],
                          cwd=str(tmp_path))
