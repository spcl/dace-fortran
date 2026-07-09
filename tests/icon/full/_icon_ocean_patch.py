"""Differential binding-swap patch for ICON's ocean
``mo_ocean_ab_timestepping_mimetic::solve_free_sfc_ab_mimetic``.

The ocean twin of :file:`_icon_solve_nh_patch.py`.  ICON's own call site
(``mo_ocean_ab_timestepping``) is left UNTOUCHED -- it still calls
``solve_free_sfc_ab_mimetic`` with the identical surface (header, 11 dummies,
USE statements).  We rewrite the module so the subroutine becomes a
DIFFERENTIAL DRIVER:

  * the original body is preserved verbatim under the name
    ``solve_free_sfc_ref`` (the REFERENCE solve);
  * the new ``solve_free_sfc_ab_mimetic`` deep-copies the mutable state
    (``mo_ocean_diff``), runs the SDFG solve ``solve_free_sfc_dace_icon``
    (DUT) in place and the stock ``solve_free_sfc_ref`` (REF) on the
    independent copy, and compares the mutated state BIT-FOR-BIT --
    reporting any divergence per call.  ICON then carries on with the DUT
    (``ocean_state``) result.

The mutable fields are ``POINTER`` components (iconfor DSL), and -- unlike the
atmosphere's ALLOCATABLE ``prog(:)`` -- ``p_prog(:)`` is itself a POINTER
array, so the reference state needs a fresh prog array on top of the per-field
deep copies (see :file:`mo_ocean_diff.f90`).

``p_phys_param%a_veloc_v`` is mutated through a module-level pointer inside
``mo_ocean_pp_scheme`` (PP vnPredict time-smoothing, in place) that the driver
cannot re-point; it is snapshot/restored around the two runs instead so both
see the same pre-call viscosity, then compared, then the DUT's version is
reinstated (ICON keeps the DUT state).
"""
import re
from pathlib import Path

#: Free-standing wrapper symbol the SDFG-generated library exports.
#: Naming convention matches the atmosphere's ``solve_nh_dace_icon``.
OCEAN_WRAPPER_NAME = "solve_free_sfc_dace_icon"

#: USE statements the differential driver needs, inserted right after the
#: SUBROUTINE header (before the dummy declarations): the deep-copy / compare
#: helpers.  ``t_hydro_ocean_state`` / ``wp`` / ``nnew`` /
#: ``set_acc_host_or_device`` are already in scope via the original module's
#: own USE block.
_DIFF_USE = [
    "    USE mo_ocean_diff, ONLY: clone_ocean_state_indep, free_ocean_state_clone, &",
    "                             compare_ocean_prog, compare_ocean_diag, compare_ocean_aux, &",
    "                             clone_field3, restore_field3, free_field3, compare_field3, &",
    "                             ocean_diff_enforce",
]

#: The driver's declaration section (local reference state + the free-standing
#: wrapper INTERFACE) followed by its executable body (clone -> DUT -> REF ->
#: compare -> free).  Inserted after the last dummy declaration.  The INTERFACE
#: declares the wrapper with ICON's REAL types (so we do NOT ``USE`` the
#: bindings module's stub-type ``.mod``); the ``LOGICAL(x, kind=1)`` cast hands
#: the C-bool ABI a 1-byte value (ICON's default LOGICAL is 4 bytes), resolved
#: through ``set_acc_host_or_device`` first because ``lacc`` is OPTIONAL here.
_DIFF_BLOCK = """\
    ! DACE DIFFERENTIAL: reference state (independent deep copy) + the SDFG
    ! wrapper interface.  ``solve_free_sfc_ref`` is the original body, renamed.
    TYPE(t_hydro_ocean_state) :: oce_ref__dace
    INTEGER                   :: ret_ref__dace
    INTEGER                   :: ndiff_prog__dace, ndiff_diag__dace, ndiff_aux__dace, ndiff_phys__dace
    LOGICAL                   :: lzacc__dace
    REAL(wp), POINTER         :: aveloc_pre__dace(:, :, :), aveloc_dut__dace(:, :, :)
    INTERFACE
      SUBROUTINE solve_free_sfc_dace_icon(patch_3d, ocean_state, p_ext_data, p_as, &
                                          p_oce_sfc, p_phys_param, timestep, &
                                          op_coeffs, solverCoeff_sp, ret_status, lacc)
        USE iso_c_binding,          ONLY: c_int, c_bool
        USE mo_model_domain,        ONLY: t_patch_3d
        USE mo_ocean_types,         ONLY: t_hydro_ocean_state, t_operator_coeff, t_solverCoeff_singlePrecision
        USE mo_ext_data_types,      ONLY: t_external_data
        USE mo_ocean_surface_types, ONLY: t_ocean_surface, t_atmos_for_ocean
        USE mo_ocean_physics_types, ONLY: t_ho_params
        TYPE(t_patch_3d), POINTER, INTENT(in)                    :: patch_3d
        TYPE(t_hydro_ocean_state), TARGET, INTENT(inout)         :: ocean_state
        TYPE(t_external_data), TARGET, INTENT(in)                :: p_ext_data
        TYPE(t_atmos_for_ocean), INTENT(inout)                   :: p_as
        TYPE(t_ocean_surface), INTENT(inout)                     :: p_oce_sfc
        TYPE(t_ho_params), INTENT(inout)                         :: p_phys_param
        INTEGER(c_int), INTENT(in)                               :: timestep
        TYPE(t_operator_coeff), INTENT(in), TARGET               :: op_coeffs
        TYPE(t_solverCoeff_singlePrecision), INTENT(in), TARGET  :: solverCoeff_sp
        INTEGER(c_int), INTENT(out)                              :: ret_status
        LOGICAL(c_bool), INTENT(in)                              :: lacc
      END SUBROUTINE solve_free_sfc_dace_icon
    END INTERFACE

    ! ``lacc`` is OPTIONAL: resolve it the same way the original body does
    ! before casting to the wrapper's 1-byte c_bool.
    CALL set_acc_host_or_device(lzacc__dace, lacc)

    ! Deep-copy the mutable state so the two solves run independently.
    CALL clone_ocean_state_indep(ocean_state, oce_ref__dace)
    ! a_veloc_v is time-smoothed IN PLACE through mo_ocean_pp_scheme's module
    ! pointer (not re-pointable from here): snapshot the pre-call values so
    ! the REF sees the same "old" viscosity the DUT saw.
    CALL clone_field3(aveloc_pre__dace, p_phys_param%a_veloc_v)

    ! DUT: the SDFG solve, in place on ocean_state (what ICON keeps).
    CALL solve_free_sfc_dace_icon(patch_3d, ocean_state, p_ext_data, p_as, p_oce_sfc, &
                                  p_phys_param, timestep, op_coeffs, solverCoeff_sp, &
                                  ret_status, LOGICAL(lzacc__dace, kind=1))

    ! Park the DUT's a_veloc_v; hand the REF the pre-call values.
    CALL clone_field3(aveloc_dut__dace, p_phys_param%a_veloc_v)
    CALL restore_field3(p_phys_param%a_veloc_v, aveloc_pre__dace)

    ! REF: the stock Fortran solve, on the independent clone.
    CALL solve_free_sfc_ref(patch_3d, oce_ref__dace, p_ext_data, p_as, p_oce_sfc, &
                            p_phys_param, timestep, op_coeffs, solverCoeff_sp, &
                            ret_ref__dace, lacc)

    ! Bit-exact comparison of everything the call mutates: prog(nnew) + the
    ! cloned diag/aux fields + the snapshot-managed a_veloc_v + ret_status.
    CALL compare_ocean_prog(ocean_state, oce_ref__dace, nnew(1), 'solve_free_sfc', ndiff_prog__dace)
    CALL compare_ocean_diag(ocean_state%p_diag, oce_ref__dace%p_diag, 'solve_free_sfc', ndiff_diag__dace)
    CALL compare_ocean_aux(ocean_state%p_aux, oce_ref__dace%p_aux, 'solve_free_sfc', ndiff_aux__dace)
    CALL compare_field3(aveloc_dut__dace, p_phys_param%a_veloc_v, 'solve_free_sfc:a_veloc_v', ndiff_phys__dace)
    IF (ret_ref__dace /= ret_status) THEN
      WRITE (0, '(A,I0,A,I0)') '  DIFF solve_free_sfc:ret_status: dut=', ret_status, ' ref=', ret_ref__dace
      ndiff_phys__dace = ndiff_phys__dace + 1
    END IF
    CALL ocean_diff_enforce(ndiff_prog__dace + ndiff_diag__dace + ndiff_aux__dace + ndiff_phys__dace, &
                            'solve_free_sfc')

    ! ICON carries on with the DUT result, including its a_veloc_v.
    CALL restore_field3(p_phys_param%a_veloc_v, aveloc_dut__dace)
    CALL free_field3(aveloc_pre__dace)
    CALL free_field3(aveloc_dut__dace)
    CALL free_ocean_state_clone(oce_ref__dace)
  END SUBROUTINE solve_free_sfc_ab_mimetic
"""

_SUBR_HEADER_RE = re.compile(r"^(\s+)SUBROUTINE\s+solve_free_sfc_ab_mimetic\s*\(", re.IGNORECASE)
_END_SUBR_RE = re.compile(r"^(\s+)END\s+SUBROUTINE\s+solve_free_sfc_ab_mimetic\b", re.IGNORECASE)
_INTENT_RE = re.compile(r"\bINTENT\b", re.IGNORECASE)


def apply_ocean_solve_patch(pristine_source: str) -> str:
    """Rewrite ``mo_ocean_ab_timestepping_mimetic.f90`` into the differential
    form.

    Walks for ``SUBROUTINE solve_free_sfc_ab_mimetic(...)`` and its matching
    ``END SUBROUTINE``, then emits, in order:

      1. the differential DRIVER -- the original header + declarations up to
         the last ``INTENT`` line (so the call site sees the identical
         surface), plus ``USE mo_ocean_diff`` and the clone / run-both /
         compare / free body; and
      2. the original subroutine verbatim, renamed ``solve_free_sfc_ref``.

    :param pristine_source: the unmodified file contents.
    :returns: the patched source as a single string.
    :raises ValueError: ``solve_free_sfc_ab_mimetic`` could not be located.
    """
    lines = pristine_source.splitlines()
    subr_start = None
    for i, ln in enumerate(lines):
        if _SUBR_HEADER_RE.match(ln):
            subr_start = i
            break
    if subr_start is None:
        raise ValueError("apply_ocean_solve_patch: SUBROUTINE solve_free_sfc_ab_mimetic header not found")
    # Walk past the (possibly multi-line) signature continuations.
    header_end = subr_start
    while lines[header_end].rstrip().endswith("&"):
        header_end += 1
    # Locate the matching END SUBROUTINE.
    end_subr = None
    for i in range(header_end + 1, len(lines)):
        if _END_SUBR_RE.match(lines[i]):
            end_subr = i
            break
    if end_subr is None:
        raise ValueError("apply_ocean_solve_patch: matching END SUBROUTINE solve_free_sfc_ab_mimetic not found")
    # Find the last INTENT declaration inside the body -- the end of the
    # declaration block (ICON declares the OPTIONAL ``lacc`` dummy LAST, after
    # the locals), after which the driver's own decls + body go.
    last_intent = header_end
    for i in range(header_end + 1, end_subr):
        if _INTENT_RE.search(lines[i]):
            last_intent = i

    # (1) the differential driver: original header + USE helpers + the
    #     declaration block (dummies + locals) + clone/run-both/compare/free.
    driver = (lines[subr_start:header_end + 1] + _DIFF_USE + lines[header_end + 1:last_intent + 1] +
              _DIFF_BLOCK.splitlines())

    # (2) the original subroutine, verbatim, renamed to solve_free_sfc_ref.
    ref = list(lines[subr_start:end_subr + 1])
    ref[0] = re.sub(r"(SUBROUTINE\s+)solve_free_sfc_ab_mimetic(\s*\()",
                    r"\1solve_free_sfc_ref\2",
                    ref[0],
                    count=1,
                    flags=re.IGNORECASE)
    ref[-1] = re.sub(r"(END\s+SUBROUTINE\s+)solve_free_sfc_ab_mimetic\b",
                     r"\1solve_free_sfc_ref",
                     ref[-1],
                     count=1,
                     flags=re.IGNORECASE)

    out = lines[:subr_start] + driver + [""] + ref + lines[end_subr + 1:]
    return "\n".join(out) + "\n"


def write_patched_ocean_solve(pristine_path: Path, patched_path: Path):
    """Convenience: read ``pristine_path``, patch, write to ``patched_path``.
    Returns the patched-line count for diagnostic use."""
    patched = apply_ocean_solve_patch(pristine_path.read_text())
    patched_path.write_text(patched)
    return patched.count("\n")
