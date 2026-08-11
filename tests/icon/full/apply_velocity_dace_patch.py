#!/usr/bin/env python3
"""Rewrite mo_velocity_advection.f90's velocity_tendencies body to dispatch into
the DaCe wrapper (velocity_tendencies_dace_icon).  Extracted verbatim from
run_icon_e2e.sh's apply_dace_patch heredoc so the e2e script and the sbatch
jobs share one implementation.  Usage: apply_velocity_dace_patch.py <f90-path>
(the caller restores the pristine source from <f90-path>.bak first)."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
lines = p.read_text().splitlines()
subr_start = next(i for i, ln in enumerate(lines)
                  if "SUBROUTINE velocity_tendencies " in ln and "(" in ln)
header_end = subr_start
while lines[header_end].rstrip().endswith("&"):
    header_end += 1
end_subr = next(i for i, ln in enumerate(lines[header_end + 1:],
                                         start=header_end + 1)
                if "END SUBROUTINE velocity_tendencies" in ln)
last_intent = header_end
for i, ln in enumerate(lines[header_end + 1:end_subr], start=header_end + 1):
    if "INTENT" in ln.upper() and "::" in ln:
        last_intent = i

iface_block = [
    "    ! DACE INTEGRATION: dispatch the velocity tendencies kernel to the",
    "    ! SDFG-generated implementation in libvelocity_inner_wrap.so.  The",
    "    ! INTERFACE block declares a FREE-STANDING wrapper symbol so we do",
    "    ! NOT USE the bindings module's .mod (its stub types would conflict",
    "    ! with mo_model_domain / mo_nonhydro_types).  The original body is",
    "    ! removed -- recover via mo_velocity_advection.f90.bak.",
    "    INTERFACE",
    "      SUBROUTINE velocity_tendencies_dace_icon(p_prog, p_patch, p_int, p_metrics, p_diag, &",
    "                                               z_w_concorr_me, z_kin_hor_e, z_vt_ie, &",
    "                                               ntnd, istep, lvn_only, &",
    "                                               dtime, dt_linintp_ubc, ldeepatmo)",
    "        USE iso_c_binding,        ONLY: c_int, c_double, c_bool",
    "        USE mo_model_domain,      ONLY: t_patch",
    "        USE mo_intp_data_strc,    ONLY: t_int_state",
    "        USE mo_nonhydro_types,    ONLY: t_nh_prog, t_nh_metrics, t_nh_diag",
    "        TYPE(t_nh_prog),    INTENT(INOUT), TARGET :: p_prog",
    "        TYPE(t_patch),      INTENT(IN),    TARGET :: p_patch",
    "        TYPE(t_int_state),  INTENT(IN),    TARGET :: p_int",
    "        TYPE(t_nh_metrics), INTENT(INOUT), TARGET :: p_metrics",
    "        TYPE(t_nh_diag),    INTENT(INOUT), TARGET :: p_diag",
    "        REAL(c_double),     INTENT(INOUT), TARGET :: z_w_concorr_me(:,:,:)",
    "        REAL(c_double),     INTENT(INOUT), TARGET :: z_kin_hor_e(:,:,:)",
    "        REAL(c_double),     INTENT(INOUT), TARGET :: z_vt_ie(:,:,:)",
    "        INTEGER(c_int),     INTENT(IN),    TARGET :: ntnd",
    "        INTEGER(c_int),     INTENT(IN),    TARGET :: istep",
    "        LOGICAL(c_bool),    INTENT(IN),    TARGET :: lvn_only",
    "        REAL(c_double),     INTENT(IN),    TARGET :: dtime",
    "        REAL(c_double),     INTENT(IN),    TARGET :: dt_linintp_ubc",
    "        LOGICAL(c_bool),    INTENT(IN),    TARGET :: ldeepatmo",
    "      END SUBROUTINE velocity_tendencies_dace_icon",
    "    END INTERFACE",
    "    CALL velocity_tendencies_dace_icon(p_prog, p_patch, p_int, p_metrics, p_diag, &",
    "                                       z_w_concorr_me, z_kin_hor_e, z_vt_ie, &",
    "                                       ntnd, istep, &",
    "                                       LOGICAL(lvn_only, kind=1), &",
    "                                       dtime, dt_linintp_ubc, &",
    "                                       LOGICAL(ldeepatmo, kind=1))",
    "",
]
extra_top_use = ["    USE iso_c_binding, ONLY: c_bool"]
new = (lines[:header_end + 1]
       + extra_top_use
       + lines[header_end + 1:last_intent + 1]
       + iface_block
       + lines[end_subr:])
p.write_text("\n".join(new) + "\n")
