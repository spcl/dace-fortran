"""Per-member-SoA callback ABI-alignment invariant for ICON's solve_nh ->
velocity_tendencies composition: the outer SDFG marshals shared derived-type args
into one SoA leaf per member (Fortran declaration order); the inner bind_c_shim
must reconstruct the identical sequence, so the C ABI lines up IFF every shared
type has the same member set/order in BOTH extracted single-TUs. Pins that the
union (ATMO_*_UNION_COMPONENTS) is complete by comparing both committed artifacts
member-for-member. Pointer-to-record HANDLE members are skipped by both sides and
excluded here. Pure text parse -- no flang/SDFG build -- runs everywhere.
"""
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OUTER_TU = _HERE / "solve_nonhydro_inlined_single_tu.f90"
_VELOCITY_TU = _HERE / "velocity_advection_inlined_single_tu.f90"

#: Pointer-to-record handles both the marshaller and the shim skip (no SoA leaf).
_SKIP_MEMBERS = {"comm_pat_c", "comm_pat_e"}


def _decl_names(rhs: str) -> list:
    """Component names on one declaration line after ``::``: split entities on
    top-level commas (parens don't separate) and take each leading identifier."""
    names, depth, start = [], 0, 0
    for i, c in enumerate(rhs):
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == "," and depth == 0:
            names.append(rhs[start:i])
            start = i + 1
    names.append(rhs[start:])
    return [m.group(1) for e in names if (m := re.match(r"\s*(\w+)", e))]


def _type_members(tu: Path) -> dict:
    """Map each derived type -> ordered list of component names in ``tu``."""
    types, cur, members = {}, None, []
    for ln in tu.read_text().splitlines():
        if cur is None:
            m = re.match(r"\s*TYPE\s*::\s*(\w+)\s*$", ln, re.I)
            if m:
                cur, members = m.group(1), []
            continue
        if re.match(r"\s*END\s+TYPE\b", ln, re.I):
            types[cur] = members
            cur = None
            continue
        if "::" in ln:
            members.extend(_decl_names(ln.split("::", 1)[1]))
    return types


def test_shared_struct_layouts_match_member_for_member():
    """Every derived type in BOTH extracted TUs carries the identical member set/order
    (excluding skipped pointer handles) -- precondition for the per-member-SoA ABI."""
    outer = _type_members(_OUTER_TU)
    velocity = _type_members(_VELOCITY_TU)
    shared = sorted(set(outer) & set(velocity))
    # The structs that actually cross the callback boundary must be shared.
    for required in ("t_patch", "t_grid_edges", "t_tangent_vectors", "t_nh_prog", "t_nh_diag", "t_nh_metrics",
                     "t_int_state"):
        assert required in shared, f"{required} missing from one of the extracted TUs"

    desyncs = {}
    for t in shared:
        o = [m for m in outer[t] if m not in _SKIP_MEMBERS]
        v = [m for m in velocity[t] if m not in _SKIP_MEMBERS]
        if o != v:
            desyncs[t] = (o, v)

    assert not desyncs, "per-member-SoA ABI desync between the two extracted TUs:\n" + "\n".join(
        f"  {t}:\n    only in solve_nh: {[m for m in o if m not in v]}\n"
        f"    only in velocity: {[m for m in v if m not in o]}" for t, (o, v) in desyncs.items())
