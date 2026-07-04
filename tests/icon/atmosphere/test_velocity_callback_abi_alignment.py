"""Per-member-SoA callback ABI-alignment invariant for the ICON
``solve_nh`` -> ``velocity_tendencies`` composition.

The outer ``solve_nh`` SDFG dispatches ``velocity_tendencies`` as a
``keep_external(c_abi='per_member_soa')`` callback: it marshals each shared
derived-type argument (``t_patch`` / ``t_nh_prog`` / ``t_int_state`` /
``t_nh_metrics`` / ``t_nh_diag``) into one SoA leaf per scalar/array member, and
the inner velocity ``bind_c_shim`` reconstructs the SAME leaves.  Both walk the
type in Fortran declaration order, so the leaf sequences coincide -- and the C
ABI lines up -- IFF every shared derived type has the IDENTICAL member set in the
IDENTICAL order in BOTH extracted single-TUs.

The two TUs are pruned from the SAME ``mo_model_domain`` / ``mo_nonhydro_types``
closure but by DIFFERENT entries, so each keeps its own reads plus the union
members named in :data:`ATMO_VELOCITY_UNION_COMPONENTS` /
:data:`ATMO_SOLVE_NH_UNION_COMPONENTS`.  This test pins that the union is
complete: it parses both committed artifacts and asserts member-for-member
identity for every shared type.  A missing union member (a field ``solve_nh``
reads but ``velocity`` does not, pruned on the velocity side) shows up here as a
desync BEFORE it becomes a silent per-member-SoA mismatch at run time.

Pointer-to-record HANDLE members (``comm_pat_c`` / ``comm_pat_e``) are SKIPPED by
the marshaller AND the shim (no SoA image), so a handle present on only one side
contributes no leaf either way; they are excluded from the comparison.

Pure text parse of two checked-in ``.f90`` artifacts -- no flang / SDFG build --
so it runs everywhere and fails fast on drift.
"""
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OUTER_TU = _HERE / "solve_nonhydro_inlined_single_tu.f90"
_VELOCITY_TU = _HERE / "velocity_advection_inlined_single_tu.f90"

#: Pointer-to-record handles both the marshaller and the shim skip (no SoA leaf).
_SKIP_MEMBERS = {"comm_pat_c", "comm_pat_e"}


def _decl_names(rhs: str) -> list:
    """Component names on one declaration line, after ``::`` -- split the entity
    list on top-level commas (commas inside array-bound / kind parens do not
    separate entities) and take each entity's leading identifier."""
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
    """Every derived type present in BOTH extracted TUs carries the identical
    member set in the identical declaration order (excluding skipped pointer
    handles) -- the precondition for the per-member-SoA velocity callback ABI."""
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
