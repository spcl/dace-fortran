"""Same-signature patch for ICON's ``mo_solve_nonhydro::solve_nh``.

Step 4 of the source-derived-bindings plan: ICON's own call site
(``mo_nh_stepping``) is left UNTOUCHED.  We replace only the BODY of
``solve_nh`` -- the SUBROUTINE header, the 14 dummy declarations, the
USE statements -- stay byte-for-byte identical.  After the patch,
ICON's compiler sees the same surface; the patched body just forwards
to ``solve_nh_dace_icon``, the free-standing iso_c wrapper that
re-extracts pointers via ICON's real types and dispatches into our
dycore SDFG's bind-C entry.

This mirrors the patch :file:`run_icon_e2e.sh` applies to
``mo_velocity_advection.f90`` for the velocity SDFG -- same shape,
scaled to solve_nh's 14-arg signature.
"""
import re
from pathlib import Path


#: Free-standing wrapper symbol the SDFG-generated library exports.
#: Naming convention matches velocity's ``velocity_tendencies_dace_icon``.
SOLVE_NH_WRAPPER_NAME = "solve_nh_dace_icon"


#: Patch payload inserted right before ``solve_nh``'s first
#: executable statement (after USE + IMPLICIT NONE + dummy declarations).
#: Builds an explicit INTERFACE block declaring the wrapper with
#: ICON's real types and forwards the 14 args verbatim, with a
#: ``LOGICAL(x, kind=1)`` cast on each LOGICAL dummy so the C-bool
#: ABI receives a 1-byte value (ICON's LOGICAL is 4 bytes by
#: default).
_PATCH_BLOCK = """\
    ! DACE INTEGRATION: dispatch the dycore to the SDFG-generated
    ! implementation in libdycore_wrapper.so.  The INTERFACE block
    ! declares a FREE-STANDING wrapper symbol so we do NOT USE the
    ! bindings module's .mod (its stub types would conflict with
    ! mo_model_domain / mo_nonhydro_types).  The original body is
    ! removed -- recover via mo_solve_nonhydro.f90.bak.
    INTERFACE
      SUBROUTINE solve_nh_dace_icon(p_nh, p_patch, p_int, prep_adv, &
                                    nnow, nnew, &
                                    l_init, l_recompute, lsave_mflx, &
                                    lprep_adv, lclean_mflx, &
                                    idyn_timestep, jstep, dtime, lacc)
        USE iso_c_binding,        ONLY: c_int, c_double, c_bool
        USE mo_model_domain,      ONLY: t_patch
        USE mo_intp_data_strc,    ONLY: t_int_state
        USE mo_nonhydro_types,    ONLY: t_nh_state
        USE mo_prepadv_types,     ONLY: t_prepare_adv
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
    CALL solve_nh_dace_icon(p_nh, p_patch, p_int, prep_adv, &
                            nnow, nnew, &
                            LOGICAL(l_init, kind=1), &
                            LOGICAL(l_recompute, kind=1), &
                            LOGICAL(lsave_mflx, kind=1), &
                            LOGICAL(lprep_adv, kind=1), &
                            LOGICAL(lclean_mflx, kind=1), &
                            idyn_timestep, jstep, dtime, &
                            LOGICAL(lacc, kind=1))
"""


_SUBR_HEADER_RE = re.compile(
    r"^(\s+)SUBROUTINE\s+solve_nh\s*\(", re.IGNORECASE)
_END_SUBR_RE = re.compile(
    r"^(\s+)END\s+SUBROUTINE\s+solve_nh\b", re.IGNORECASE)
_INTENT_RE = re.compile(r"\bINTENT\b", re.IGNORECASE)


def apply_solve_nh_patch(pristine_source: str) -> str:
    """Apply the same-signature patch to ``mo_solve_nonhydro.f90``.

    Walks the source for ``SUBROUTINE solve_nh(...)`` (one line, with
    continuation handling), then for the last ``INTENT`` declaration
    inside its body.  Inserts the patch block immediately after that
    declaration, dropping every line between it and the matching
    ``END SUBROUTINE solve_nh`` (the original body that's now dead
    code after the forwarding ``CALL`` returns).

    :param pristine_source: the unmodified file contents.
    :returns: the patched source as a single string.
    :raises ValueError: ``solve_nh`` could not be located.
    """
    lines = pristine_source.splitlines()
    subr_start = None
    for i, ln in enumerate(lines):
        if _SUBR_HEADER_RE.match(ln):
            subr_start = i
            break
    if subr_start is None:
        raise ValueError("apply_solve_nh_patch: SUBROUTINE solve_nh "
                         "header not found")
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
        raise ValueError(
            "apply_solve_nh_patch: matching END SUBROUTINE solve_nh "
            "not found")
    # Find the last INTENT declaration inside the body.
    last_intent = header_end
    for i in range(header_end + 1, end_subr):
        if _INTENT_RE.search(lines[i]):
            last_intent = i
    # Forwarding call needs the c_bool kind in scope at the SUBROUTINE
    # body level too (the INTERFACE block has its own USE, but the
    # ``LOGICAL(x, kind=1)`` casts at the CALL site reuse a 1-byte
    # logical that flang resolves via the USE).
    extra_use = ["    USE iso_c_binding, ONLY: c_bool"]
    out = (lines[:header_end + 1]
           + extra_use
           + lines[header_end + 1:last_intent + 1]
           + _PATCH_BLOCK.splitlines()
           + lines[end_subr:])
    return "\n".join(out) + "\n"


def write_patched_solve_nh(pristine_path: Path, patched_path: Path):
    """Convenience: read ``pristine_path``, patch, write to
    ``patched_path``.  Returns the patched-line count for diagnostic
    use."""
    patched = apply_solve_nh_patch(pristine_path.read_text())
    patched_path.write_text(patched)
    return patched.count("\n")
