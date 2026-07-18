# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Real-ICON validation for static-vtable monomorphisation.

monomorphize_rewrite_test.py proves each rewrite primitive + the driver
end-to-end (incl. a flang "0 fir.dispatch" check) on small programs shaped
like ICON's three dispatch axes. This module closes the loop on the actual
upstream ICON ocean-solver sources:

  1. the analyzer accepts the real type hierarchy and produces the expected
     three plans (backend/transfer/lhs_agen);
  2. the driver, given the real-ICON spec, collapses the real solver's
     polymorphic dispatch (t_ocean_solve%act factory + ocean_solve_backend_solve
     interposer) to zero, on real bodies.

A compilable self-contained TU for a flang check on the real sources is the
inliner's job and out of scope here. Reads the icon-model submodule (no build), marked `long`.
"""
import os
import re
from pathlib import Path

import pytest

import fparser.two.Fortran2003 as f03
from fparser.two.utils import walk

from dace_fortran.inliner.ast_desugaring.monomorphize import analyze, parse_program
from dace_fortran.inliner.ast_desugaring.monomorphize_rewrite import (AxisSpec, monomorphize, MonomorphizationSpec)

pytestmark = pytest.mark.long

# ICON ocean-solver sources from ICON_SRC (default: in-repo submodule), mirrors tests/conftest.py's icon_build fixture.
_ICON_SRC = Path(os.environ.get("ICON_SRC", str(Path(__file__).resolve().parents[1] / "icon" / "full" / "icon-model")))
_SOLVER = _ICON_SRC / "src" / "ocean" / "math"

if not (_SOLVER / "mo_ocean_solve_backend.f90").is_file():
    pytest.skip(
        "icon-model submodule not checked out (run "
        "`git submodule update --init tests/icon/full/icon-model`)",
        allow_module_level=True)

_CPP = re.compile(r"^\s*#")

# real-ICON monomorphisation spec: backend is a runtime-allocated ladder;
# transfer/lhs-agen are pinned to one concrete type at the construction site, so
# they retype. Hand-written per the locked design; a later pass should auto-generate it.
_BACKEND = "t_ocean_solve_backend"
_REAL_ICON_SPEC = MonomorphizationSpec(axes=[
    AxisSpec(base="t_transfer", strategy="retype", concrete="t_trivial_transfer"),
    AxisSpec(base="t_lhs_agen", strategy="retype", concrete="t_primal_flip_flop_lhs"),
    AxisSpec(base=_BACKEND, strategy="ladder"),
])

# the seven concrete backend solver arms registered to t_ocean_solve_backend.
_BACKEND_ARMS = {
    "t_ocean_solve_gmres",
    "t_ocean_solve_cg",
    "t_ocean_solve_cgj",
    "t_ocean_solve_cgo",
    "t_ocean_solve_bicgstab",
    "t_ocean_solve_mres",
    "t_ocean_solve_legacy_gmres",
}
# the base's shared (non-deferred) interposers, each cloned once per arm.
_SHARED_INTERPOSERS = {"solve", "construct", "dump_matrix"}


def _strip_cpp(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not _CPP.match(line))


def _src(name: str) -> str:
    return _strip_cpp((_SOLVER / name).read_text())


def _extract_types(text: str) -> str:
    """Just the ``TYPE ... END TYPE`` blocks (defs + bindings) -- enough for the
    analyzer and for the rewrite to learn the arm set, without the heavy bodies."""
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


#: modules whose type definitions feed the analyzer (axis bases + every arm +
#: the t_lhs_agen holder); bodies not needed to plan or retype.
_TYPEDEF_MODULES = [
    "mo_ocean_solve_gmres.f90",
    "mo_ocean_solve_cg.f90",
    "mo_ocean_solve_cgj.f90",
    "mo_ocean_solve_cgo.f90",
    "mo_ocean_solve_bicgStab.f90",
    "mo_ocean_solve_minres.f90",
    "mo_ocean_solve_legacy_gmres.f90",
    "mo_ocean_solve_transfer.f90",
    "mo_ocean_solve_transfer_trivial.f90",
    "mo_ocean_solve_transfer_subset.f90",
    "mo_ocean_solve_lhs_type.f90",
    "mo_ocean_solve_lhs_surface_height.f90",
    "mo_ocean_solve_lhs_zstar.f90",
    "mo_ocean_solve_lhs_primal_flip_flop.f90",
    "mo_ocean_solve_lhs.f90",
]


def _typedef_program() -> f03.Program:
    """All axis type definitions in one synthetic module -- for analysis only."""
    typedefs = "\n".join(
        _extract_types(_src(m)) for m in ["mo_ocean_solve_backend.f90", "mo_ocean_solve.f90"] + _TYPEDEF_MODULES)
    return parse_program(f"module zz_icon_typedefs\n{typedefs}\nend module\n")


def _rewrite_program() -> f03.Program:
    """The real container + backend interposer *bodies* (the code the driver
    rewrites) plus the arm/transfer/lhs type defs the plan + retype need."""
    full = [_src("mo_ocean_solve.f90"), _src("mo_ocean_solve_backend.f90")]
    typedefs = "\n".join(_extract_types(_src(m)) for m in _TYPEDEF_MODULES)
    return parse_program("\n".join(full) + f"\nmodule zz_icon_typedefs\n{typedefs}\nend module\n")


def _bare_act_dispatches(prog: f03.Program) -> int:
    """type-bound dispatches still routed through the bare ``%act`` backend slot."""
    n = 0
    for call in walk(prog, f03.Call_Stmt):
        designator = call.children[0]
        if isinstance(designator, f03.Procedure_Designator):
            obj = designator.children[0]
            if isinstance(obj, f03.Data_Ref) and str(obj.children[-1]).lower() == "act":
                n += 1
    return n


def test_analyzer_accepts_real_icon_solver_hierarchy():
    plans = {p.abstract_base.lower(): p for p in analyze(_typedef_program())}
    assert set(plans) == {"t_ocean_solve_backend", "t_transfer", "t_lhs_agen"}

    backend = plans["t_ocean_solve_backend"]
    assert backend.deferred == ["doit_sp", "doit_wp"]
    assert {a.type_name.lower() for a in backend.arms} == _BACKEND_ARMS
    # every arm binds both deferred procedures (else the analyzer would reject)
    for arm in backend.arms:
        assert set(arm.bindings) == {"doit_sp", "doit_wp"}

    assert {a.type_name.lower() for a in plans["t_transfer"].arms} == {"t_trivial_transfer", "t_subset_transfer"}
    assert {a.type_name.lower()
            for a in plans["t_lhs_agen"].arms} == {"t_surface_height_lhs", "t_lhs_zstar", "t_primal_flip_flop_lhs"}


def test_driver_collapses_real_icon_dispatch():
    prog = _rewrite_program()
    # the real solver genuinely dispatches before we touch it
    assert _bare_act_dispatches(prog) > 0
    before = str(prog).upper()
    assert "CLASS(T_TRANSFER)" in before and "CLASS(T_LHS_AGEN)" in before

    stats = monomorphize(prog, _REAL_ICON_SPEC)

    # backend: real act component expanded; 3 shared interposers cloned once per arm (3x7); no %act slot dispatch survives.
    assert stats.components_rewritten == 1
    assert stats.interposers_cloned == len(_SHARED_INTERPOSERS) * len(_BACKEND_ARMS)
    assert _bare_act_dispatches(prog) == 0

    text = str(prog)
    assert "act__tag" in text  # type tag synthesised at the construction site
    clones = set(re.findall(r"ocean_solve_backend_(\w+?)__(t_ocean_solve_\w+)", text))
    assert {b for b, _ in clones} == _SHARED_INTERPOSERS
    assert {arm for _, arm in clones} == _BACKEND_ARMS

    # transfer + lhs_agen: every real CLASS(base) declaration retyped to concrete
    # (no abstract-interface dummies to preserve here); non-trivial count rewritten.
    assert stats.declarations_retyped > 0
    upper = text.upper()
    assert "CLASS(T_TRANSFER)" not in upper
    assert "CLASS(T_LHS_AGEN)" not in upper
