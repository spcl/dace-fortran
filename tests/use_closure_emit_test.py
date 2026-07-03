"""Selective emit: a whole-project ``compile_commands.json`` should lower
only the entry's TU + its transitive ``USE``-closure, not every TU.

ICON's database lists ~900 TUs; emitting all of them to reach one entry is
wasteful and drags in modules flang need never touch.  ``emit(entry=...)``
(and ``build_sdfg_from_project``) restrict the run to the closure.
"""
import json
from pathlib import Path

import pytest

from _util import have_flang
from dace_fortran.emit_hlfir import (_entry_module, _parse_compile_commands, _select_use_closure, emit)

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_entry_module_parsing():
    # ``_entry_module`` parses the enclosing module out of a *mangled* flang
    # symbol (an IR-introspection helper); user-facing entry names are plain.
    assert _entry_module("_QMmo_solve_nonhydroPsolve_nh") == "mo_solve_nonhydro"
    assert _entry_module("_QMmo_xPfoo") == "mo_x"
    assert _entry_module("_QPbar") is None  # free subroutine, no module


def _write(tmp_path, name, body):
    p = tmp_path / f"{name}.f90"
    p.write_text(body)
    return p


def test_select_use_closure_excludes_unreached(tmp_path):
    a = _write(tmp_path, "a", "module a\n  use b\nend module a\n")
    b = _write(tmp_path, "b", "module b\n  use c\nend module b\n")
    c = _write(tmp_path, "c", "module c\nend module c\n")
    d = _write(tmp_path, "d", "module d\nend module d\n")  # unrelated
    parsed = [(p, [], []) for p in (a, b, c, d)]
    closure = {t[0] for t in _select_use_closure(parsed, "a")}
    assert closure == {a, b, c}  # not d


def test_parse_compile_commands_skips_missing_sources(tmp_path):
    # ``bear`` captures cmake's compiler-ABI probe TUs (e.g.
    # ``CMakeFortranCompilerABI.F90``) whose throw-away sources are gone once
    # configure finishes.  A non-existent source can't be emitted and is never
    # part of a project USE-closure, so the parser must drop it -- otherwise the
    # closure scan's ``src.read_text()`` raises ``FileNotFoundError``.
    real = _write(tmp_path, "real", "module real_m\nend module real_m\n")
    gone = tmp_path / "CMakeFortranCompilerABI.F90"  # never created
    cc = tmp_path / "compile_commands.json"
    cc.write_text(
        json.dumps([{
            "directory": str(tmp_path),
            "command": f"flang-new-21 -c {p}",
            "file": str(p)
        } for p in (real, gone)]))
    parsed = _parse_compile_commands(cc)
    assert [t[0] for t in parsed] == [real]  # gone dropped


def test_emit_only_closure_from_compile_commands(tmp_path):
    a = _write(tmp_path, "a", "module a\n  use b\ncontains\n"
               "  subroutine run()\n  end subroutine run\nend module a\n")
    b = _write(tmp_path, "b", "module b\n  use c\nend module b\n")
    c = _write(tmp_path, "c", "module c\nend module c\n")
    d = _write(tmp_path, "d", "module d\nend module d\n")
    cc = tmp_path / "compile_commands.json"
    cc.write_text(
        json.dumps([{
            "directory": str(tmp_path),
            "command": f"flang-new-21 -c {p}",
            "file": str(p)
        } for p in (c, b, a, d)]))
    out = emit(compile_commands=cc, out_dir=tmp_path / "hlfir", entry="run")
    stems = {p.stem for p in out}
    assert stems == {"a", "b", "c"}  # d not emitted
