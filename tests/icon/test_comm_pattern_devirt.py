"""The monomorphisation engine recognises ICON's REAL ``t_comm_pattern``
hierarchy -- the prerequisite for devirtualizing ``sync_patch_array`` in the
atmosphere dynamical core (see ``tests/sync_devirt_mpi_libnode_test.py`` for the
end-to-end lowering on the same pattern in miniature).

ICON's halo exchange dispatches through the abstract ``t_comm_pattern``
(``mo_communication_types``) over two concrete arms -- ``t_comm_pattern_orig``
(``mo_communication_orig``) and ``t_comm_pattern_yaxt`` (``mo_communication_yaxt``).
That is a single-level hierarchy with a closed arm set, which the engine can
``retype`` (pin to the default ``orig``) or ``ladder``, turning ``p_pat%exchange_data_*``
from a ``fir.dispatch`` an SDFG cannot lower into a static call that inlines.

This test confirms the engine plans the REAL hierarchy (not just the synthetic
fixture); the unrelated ``CLASS(*)`` generic containers the comm closure also
pulls in are out of scope of the scoped rejection, so they do not block it.
"""
import re
from pathlib import Path

import pytest

from dace_fortran.inliner.ast_desugaring.monomorphize import analyze, parse_program

_HERE = Path(__file__).resolve().parent
_COMM_DIR = _HERE / "full" / "icon-model" / "src" / "parallel_infrastructure"

pytestmark = pytest.mark.skipif(
    not (_COMM_DIR / "mo_communication_types.f90").is_file(),
    reason="icon-model submodule not checked out; run `git submodule update --init --recursive`")

_CPP = re.compile(r"^\s*#")


def _type_blocks(fname: str) -> str:
    """Extract every ``TYPE ... END TYPE`` block from a real ICON module
    (cpp-stripped) -- enough for the engine to learn the arm set."""
    text = "\n".join(l for l in (_COMM_DIR / fname).read_text().splitlines() if not _CPP.match(l))
    out, depth, buf = [], 0, []
    for line in text.splitlines():
        s = line.strip().upper()
        starts = bool(re.match(r"TYPE\s*,|TYPE\s*::|TYPE\s+[A-Z]", s)) and "END TYPE" not in s
        if depth == 0 and starts:
            depth, buf = 1, [line]
            continue
        if depth > 0:
            buf.append(line)
            if s.startswith("END TYPE"):
                depth = 0
                out.append("\n".join(buf))
    return "\n".join(out)


def test_real_icon_comm_pattern_is_monomorphisable():
    """ICON's real ``t_comm_pattern`` is a single-level ladder base with exactly
    the two concrete arms (``orig`` / ``yaxt``), every deferred binding overridden
    -- so the engine can devirtualize the halo-exchange dispatch."""
    typedefs = "\n".join(
        _type_blocks(f)
        for f in ("mo_communication_types.f90", "mo_communication_orig.f90", "mo_communication_yaxt.f90"))
    prog = parse_program(f"module zz_comm_types\n{typedefs}\nend module\n")

    plans = {p.abstract_base: p for p in analyze(prog, only_bases=["t_comm_pattern"])}
    assert "t_comm_pattern" in plans, "engine did not recognise t_comm_pattern as a dispatch base"
    plan = plans["t_comm_pattern"]
    arms = {a.type_name for a in plan.arms}
    assert arms == {"t_comm_pattern_orig", "t_comm_pattern_yaxt"}, f"unexpected arm set: {arms}"
    # the abstract carries the full exchange_data_* / setup / get_* binding set.
    assert len(plan.deferred) >= 15, f"expected the full deferred-binding set, got {len(plan.deferred)}"
