"""Rewrites ICON's mo_solve_nonhydro::solve_nh into a differential driver: DUT (SDFG) runs in place,
REF (original body, renamed solve_nh_ref) runs on a deep copy, results compared bit-for-bit per call."""
import re
from pathlib import Path

#: Free-standing wrapper symbol the SDFG-generated library exports.
SOLVE_NH_WRAPPER_NAME = "solve_nh_dace_icon"

#: USE statements inserted after the SUBROUTINE header, before IMPLICIT NONE.
_DIFF_USE = [
    "    USE iso_c_binding, ONLY: c_bool",
    "    USE iso_fortran_env, ONLY: error_unit",
    "    USE mo_solve_nh_diff, ONLY: clone_state_indep_prog, free_state_clone, &",
    "                                clone_prepadv_indep, free_prepadv_clone, &",
    "                                compare_prog_nnew, compare_prepadv, compare_diag",
]

#: Driver decls + body (clone -> DUT -> REF -> compare -> free), inserted after the last dummy declaration.
_DIFF_BLOCK = """\
    ! DACE DIFFERENTIAL: reference state (independent deep copy) + the SDFG
    ! wrapper interface.  ``solve_nh_ref`` is the original body, renamed.
    TYPE(t_nh_state)    :: nh_ref__dace
    TYPE(t_prepare_adv) :: prep_ref__dace
    INTEGER             :: ndiff_prog__dace, ndiff_prognow__dace, ndiff_prep__dace, ndiff_diag__dace
    INTEGER             :: ndiff_total__dace
    INTERFACE
      SUBROUTINE solve_nh_dace_icon(p_nh, p_patch, p_int, prep_adv, &
                                    nnow, nnew, &
                                    l_init, l_recompute, lsave_mflx, &
                                    lprep_adv, lclean_mflx, &
                                    idyn_timestep, jstep, dtime, lacc)
        USE iso_c_binding,     ONLY: c_int, c_double, c_bool
        USE mo_model_domain,   ONLY: t_patch
        USE mo_intp_data_strc, ONLY: t_int_state
        USE mo_nonhydro_types, ONLY: t_nh_state
        USE mo_prepadv_types,  ONLY: t_prepare_adv
        TYPE(t_nh_state),    TARGET, INTENT(INOUT) :: p_nh
        TYPE(t_int_state),   TARGET, INTENT(IN)    :: p_int
        TYPE(t_patch),       TARGET, INTENT(INOUT) :: p_patch
        TYPE(t_prepare_adv), TARGET, INTENT(INOUT) :: prep_adv
        INTEGER(c_int),              INTENT(IN)    :: nnow, nnew
        LOGICAL(c_bool),             INTENT(IN)    :: l_init
        LOGICAL(c_bool),             INTENT(IN)    :: l_recompute
        LOGICAL(c_bool),             INTENT(IN)    :: lsave_mflx
        LOGICAL(c_bool),             INTENT(IN)    :: lprep_adv
        LOGICAL(c_bool),             INTENT(IN)    :: lclean_mflx
        INTEGER(c_int),              INTENT(IN)    :: idyn_timestep
        INTEGER(c_int),              INTENT(IN)    :: jstep
        REAL(c_double),              INTENT(IN)    :: dtime
        LOGICAL(c_bool),             INTENT(IN)    :: lacc
      END SUBROUTINE solve_nh_dace_icon
    END INTERFACE

    ! Deep-copy the mutable state so the two dycores run independently.
    CALL clone_state_indep_prog(p_nh, nh_ref__dace)
    CALL clone_prepadv_indep(prep_adv, prep_ref__dace)

    ! DUT: the SDFG dycore, in place on p_nh / prep_adv (what ICON keeps).
    CALL solve_nh_dace_icon(p_nh, p_patch, p_int, prep_adv, &
                            nnow, nnew, &
                            LOGICAL(l_init, kind=1), &
                            LOGICAL(l_recompute, kind=1), &
                            LOGICAL(lsave_mflx, kind=1), &
                            LOGICAL(lprep_adv, kind=1), &
                            LOGICAL(lclean_mflx, kind=1), &
                            idyn_timestep, jstep, dtime, &
                            LOGICAL(lacc, kind=1))

    ! REF: the stock Fortran dycore, on the independent clone.
    CALL solve_nh_ref(nh_ref__dace, p_patch, p_int, prep_ref__dace, &
                      nnow, nnew, l_init, l_recompute, lsave_mflx, &
                      lprep_adv, lclean_mflx, idyn_timestep, jstep, dtime, lacc)

    ! Bit-exact comparison of the prognostic output + transport-prep fluxes +
    ! the FULL diagnostic state (the diff's clone deep-copies every diag field,
    ! so a velocity-callback divergence -- ddt_vn_apc_pc / ddt_vn_cor_pc /
    ! ddt_w_adv_pc -- surfaces in compare_diag, which also checks the
    ! max_vcfl_dyn scalar that callback MAX-accumulates across substeps).
    ! prog(nnow) is compared too: the reference never assigns it (verified), so
    ! any nnow diff means the DUT stomped the time level the NEXT substep reads
    ! -- a corruption the nnew compare alone cannot see.
    CALL compare_prog_nnew(p_nh, nh_ref__dace, nnew, 'solve_nh', ndiff_prog__dace)
    CALL compare_prog_nnew(p_nh, nh_ref__dace, nnow, 'solve_nh:nnow', ndiff_prognow__dace)
    CALL compare_prepadv(prep_adv, prep_ref__dace, 'solve_nh', ndiff_prep__dace)
    CALL compare_diag(p_nh % diag, nh_ref__dace % diag, 'solve_nh', ndiff_diag__dace)

    ! One greppable line per call: 0 == bit-exact across everything compared.
    ndiff_total__dace = ndiff_prog__dace + ndiff_prognow__dace + ndiff_prep__dace + ndiff_diag__dace
    WRITE (error_unit, '(A,I0,A)') "  [diff] solve_nh TOTAL: ", ndiff_total__dace, &
      " differing elements (0 == bit-exact)"

    CALL free_state_clone(nh_ref__dace)
    CALL free_prepadv_clone(prep_ref__dace)
  END SUBROUTINE solve_nh
"""

_SUBR_HEADER_RE = re.compile(r"^(\s+)SUBROUTINE\s+solve_nh\s*\(", re.IGNORECASE)
_END_SUBR_RE = re.compile(r"^(\s+)END\s+SUBROUTINE\s+solve_nh\b", re.IGNORECASE)
_INTENT_RE = re.compile(r"\bINTENT\b", re.IGNORECASE)


def apply_solve_nh_patch(pristine_source: str) -> str:
    """Rewrites mo_solve_nonhydro.f90's solve_nh into the differential driver + solve_nh_ref (original
    body, verbatim). Raises ValueError if solve_nh isn't found."""
    lines = pristine_source.splitlines()
    subr_start = None
    for i, ln in enumerate(lines):
        if _SUBR_HEADER_RE.match(ln):
            subr_start = i
            break
    if subr_start is None:
        raise ValueError("apply_solve_nh_patch: SUBROUTINE solve_nh header not found")
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
        raise ValueError("apply_solve_nh_patch: matching END SUBROUTINE solve_nh not found")
    # last INTENT line marks the end of the dummy declaration block; driver decls/body go after it.
    last_intent = header_end
    for i in range(header_end + 1, end_subr):
        if _INTENT_RE.search(lines[i]):
            last_intent = i

    # (1) differential driver: original header + USE helpers + dummies + clone/run-both/compare/free body.
    driver = (lines[subr_start:header_end + 1] + _DIFF_USE + lines[header_end + 1:last_intent + 1] +
              _DIFF_BLOCK.splitlines())

    # (2) the original subroutine, verbatim, renamed to solve_nh_ref.
    ref = list(lines[subr_start:end_subr + 1])
    ref[0] = re.sub(r"(SUBROUTINE\s+)solve_nh(\s*\()", r"\1solve_nh_ref\2", ref[0], count=1, flags=re.IGNORECASE)
    ref[-1] = re.sub(r"(END\s+SUBROUTINE\s+)solve_nh\b", r"\1solve_nh_ref", ref[-1], count=1, flags=re.IGNORECASE)

    out = lines[:subr_start] + driver + [""] + ref + lines[end_subr + 1:]
    return "\n".join(out) + "\n"


def write_patched_solve_nh(pristine_path: Path, patched_path: Path):
    """Read pristine_path, patch, write to patched_path; returns the patched-line count."""
    patched = apply_solve_nh_patch(pristine_path.read_text())
    patched_path.write_text(patched)
    return patched.count("\n")
