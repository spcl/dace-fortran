"""Drift guard for the committed DE-POLYMORPHISED ICON-O surface solver.

The single-TU extraction (``test_extract_single_tu.py``) keeps the
``t_ocean_solve`` subsystem EXTERNAL because flang lowers its virtual dispatch
to ``fir.dispatch`` (no SDFG node for a runtime vtable lookup).  This test takes
the opposite route on the same real sources: it runs the static-vtable
monomorphisation engine over them and pins the de-polymorphised output that the
SDFG-lowering stage can consume.

Pure fparser parse + AST rewrite -- NO flang, NO OpenMPI, no compile -- so it is
fast and gated only on the icon-model submodule.  It

  * regenerates the de-polymorphised source and asserts it is byte-identical to
    the committed ``dycore_solver_monomorphized.f90`` (so the saved artifact
    never silently drifts from the engine + the upstream sources), and
  * asserts the de-polymorphisation is real: the locked rewrite statistics, the
    synthesised ``act__tag`` + the per-arm interposer clones, and ZERO surviving
    ``CLASS(t_transfer)`` / ``CLASS(t_lhs_agen)`` declarations.
"""
import re

import pytest

import fparser.two.Fortran2003 as f03
from fparser.two.utils import walk

from icon.ocean._monomorphize_solver import (ARTIFACT, depolymorphize_solver, have_icon_solver, parse_program)

pytestmark = [
    pytest.mark.long,
    pytest.mark.skipif(not have_icon_solver(),
                       reason="icon-model ocean source not checked out; run "
                       "`git submodule update --init --recursive tests/icon/full/icon-model`"),
]

_BACKEND_ARMS = {
    "t_ocean_solve_gmres",
    "t_ocean_solve_cg",
    "t_ocean_solve_cgj",
    "t_ocean_solve_cgo",
    "t_ocean_solve_bicgstab",
    "t_ocean_solve_mres",
    "t_ocean_solve_legacy_gmres",
}
_SHARED_INTERPOSERS = {"solve", "construct", "dump_matrix"}


def _bare_act_dispatches(prog: f03.Program) -> int:
    """Type-bound dispatches still routed through the bare ``%act`` backend slot
    (the polymorphic calls the monomorphisation must eliminate)."""
    n = 0
    for call in walk(prog, f03.Call_Stmt):
        designator = call.children[0]
        if isinstance(designator, f03.Procedure_Designator):
            obj = designator.children[0]
            if isinstance(obj, f03.Data_Ref) and str(obj.children[-1]).lower() == "act":
                n += 1
    return n


def test_committed_artifact_matches_regenerated():
    """Byte-for-byte drift guard: the committed de-polymorphised file must equal
    a fresh run of the monomorphisation engine on the upstream solver."""
    source, _stats = depolymorphize_solver()
    assert ARTIFACT.is_file(), \
        f"no committed artifact {ARTIFACT.name}; run `python {ARTIFACT.parent}/_monomorphize_solver.py`"
    assert source == ARTIFACT.read_text(), \
        (f"{ARTIFACT.name} drifted from the engine + upstream sources; regenerate it via "
         f"`python {ARTIFACT.parent}/_monomorphize_solver.py`")


def test_artifact_is_fully_depolymorphised():
    """The locked rewrite stats + the structural evidence that no virtual
    dispatch survives in the saved artifact."""
    source, stats = depolymorphize_solver()
    # Backend `act` factory expanded once; the 3 shared interposers cloned once
    # per arm (3 x 7); the 2 retype axes' CLASS declarations all rewritten.
    assert stats.components_rewritten == 1
    assert stats.interposers_cloned == len(_SHARED_INTERPOSERS) * len(_BACKEND_ARMS)
    assert stats.declarations_retyped > 0

    # No `%act` dispatch survives in the (re-parsed) output.
    assert _bare_act_dispatches(parse_program(source)) == 0

    # Static type tag synthesised; every interposer cloned for every arm.
    assert "act__tag" in source
    clones = set(re.findall(r"ocean_solve_backend_(\w+?)__(t_ocean_solve_\w+)", source))
    assert {b for b, _ in clones} == _SHARED_INTERPOSERS
    assert {arm for _, arm in clones} == _BACKEND_ARMS

    # Both retype axes' abstract declarations are gone.
    upper = source.upper()
    assert "CLASS(T_TRANSFER)" not in upper
    assert "CLASS(T_LHS_AGEN)" not in upper
