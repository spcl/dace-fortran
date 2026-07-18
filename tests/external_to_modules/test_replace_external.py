"""Unit + e2e coverage for ``replace_external_with_modules``: rewrites resolvable
``EXTERNAL :: x`` to ``USE <module>, ONLY: x`` (QE rewrite pattern 1).

Runs against three fixtures: external_basic_example.f90 (one EXTERNAL),
external_multiple_example.f90 (three in one line), external_already_used_example.f90
(EXTERNAL + existing USE); utils_mod.f90 defines the referenced procedures."""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from dace_fortran.preprocess import replace_external_with_modules

_HERE = Path(__file__).resolve().parent
_HAVE_FLANG = shutil.which("flang-new-21") is not None


def _read(name: str) -> str:
    return (_HERE / name).read_text()


def _strip_comments(src: str) -> str:
    """Drop full-line ``!`` comments -- the fixture's header explanatorily prints
    the EXTERNAL/USE shapes the test's pattern-presence assertions check for."""
    out = []
    for line in src.splitlines(keepends=True):
        s = line.lstrip()
        if s.startswith("!"):
            continue
        out.append(line)
    return "".join(out)


# ---------------------------------------------------------------------------
# Pattern recognition / rewrite shape
# ---------------------------------------------------------------------------


def test_basic_rewrite_adds_use_and_removes_external():
    """One EXTERNAL becomes one USE; the EXTERNAL line disappears from the rewritten source."""
    src = _read("external_basic_example.f90")
    out = replace_external_with_modules(src, search_dirs=[_HERE])
    assert "USE utils_mod, ONLY: dscale" in out, \
        "expected synthesised USE statement"
    assert not re.search(r"(?im)^\s*EXTERNAL\s*::\s*dscale\s*$", out), \
        "EXTERNAL line should have been deleted"


def test_multiple_names_resolve_into_one_use():
    """Three EXTERNALs in one module collapse into a single USE ... ONLY: a,b,c
    line, not three separate USE statements."""
    src = _read("external_multiple_example.f90")
    out = replace_external_with_modules(src, search_dirs=[_HERE])
    code = _strip_comments(out)
    m = re.search(r"USE\s+utils_mod,\s*ONLY:\s*([^\n]+)", code, re.IGNORECASE)
    assert m, "expected synthesised USE for utils_mod"
    name_list = [n.strip().lower() for n in m.group(1).split(",")]
    assert set(name_list) == {"dscale", "dadd", "dsum"}, \
        f"USE should import all three names, got {name_list}"
    # check code only -- the fixture's header comment mentions EXTERNAL explanatorily
    assert not re.search(r"(?im)^\s*EXTERNAL\b", code), \
        f"EXTERNAL line should have been deleted from code, got:\n{code}"


def test_already_used_module_does_not_duplicate_use():
    """Kernel already USEs utils_mod: EXTERNAL is deleted, no duplicate USE added."""
    src = _read("external_already_used_example.f90")
    out = replace_external_with_modules(src, search_dirs=[_HERE])
    code = _strip_comments(out)
    # Original USE survives, no duplicate added.
    use_lines = re.findall(r"(?im)^\s*USE\s+utils_mod\b[^\n]*$", code)
    assert len(use_lines) == 1, \
        f"expected one USE utils_mod line in code, got {len(use_lines)}: {use_lines}"
    # The EXTERNAL line is gone.
    assert not re.search(r"(?im)^\s*EXTERNAL\b", code), \
        "EXTERNAL line should have been deleted"


def test_unresolvable_external_left_alone(tmp_path):
    """Unresolvable procedure name stays EXTERNAL -- conservative: a missing
    source file isn't papered over."""
    src = """
SUBROUTINE run(out_val)
  IMPLICIT NONE
  REAL(8), INTENT(OUT) :: out_val
  REAL(8) :: not_in_any_module
  EXTERNAL :: not_in_any_module
  out_val = not_in_any_module(1.0d0)
END SUBROUTINE
"""
    out = replace_external_with_modules(src, search_dirs=[tmp_path])
    # No USE was synthesised.
    assert "USE " not in out
    # EXTERNAL line survives.
    assert "EXTERNAL :: not_in_any_module" in out


def test_passthrough_when_no_search_dirs():
    """No search dirs -- input passes through verbatim."""
    src = _read("external_basic_example.f90")
    out = replace_external_with_modules(src)
    assert out == src


def test_idempotent():
    """Second pass over already-rewritten source is a no-op."""
    src = _read("external_basic_example.f90")
    once = replace_external_with_modules(src, search_dirs=[_HERE])
    twice = replace_external_with_modules(once, search_dirs=[_HERE])
    assert once == twice


def test_string_with_external_word_in_it_is_not_rewritten():
    """A character literal containing "EXTERNAL" isn't a declaration -- must be left alone."""
    src = """
SUBROUTINE run(msg)
  IMPLICIT NONE
  CHARACTER(LEN=*), INTENT(IN) :: msg
  PRINT *, "this is an EXTERNAL string"
END SUBROUTINE
"""
    out = replace_external_with_modules(src, search_dirs=[_HERE])
    assert "this is an EXTERNAL string" in out


def test_external_with_no_double_colon_form_is_recognised():
    """Legacy ``EXTERNAL name`` form (no ``::``) is also matched."""
    src = """
SUBROUTINE run(out_val, x, f)
  IMPLICIT NONE
  REAL(8), INTENT(IN) :: x, f
  REAL(8), INTENT(OUT) :: out_val
  REAL(8) :: dscale
  EXTERNAL dscale
  out_val = dscale(x, f)
END SUBROUTINE
"""
    out = replace_external_with_modules(src, search_dirs=[_HERE])
    assert "USE utils_mod, ONLY: dscale" in out
    assert "EXTERNAL dscale" not in out


# ---------------------------------------------------------------------------
# flang-level smoke parse (rewrite is syntactically valid)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAVE_FLANG, reason="flang-new-21 not on PATH")
def test_rewritten_basic_example_parses_under_flang(tmp_path):
    """Rewrite output is valid Fortran flang can lower (kernel + sidecar module staged side by side)."""
    src = _read("external_basic_example.f90")
    rewritten = replace_external_with_modules(src, search_dirs=[_HERE])
    # Stage rewritten kernel + sidecar module + compile module first.
    kernel = tmp_path / "kernel.f90"
    mod = tmp_path / "utils_mod.f90"
    kernel.write_text(rewritten)
    mod.write_text(_read("utils_mod.f90"))
    subprocess.check_call([
        "flang-new-21", "-fc1", "-emit-hlfir", "-fintrinsic-modules-path", "/usr/lib/llvm-21/include/flang",
        str(mod), "-o",
        str(tmp_path / "utils_mod.hlfir")
    ],
                          cwd=tmp_path)
    subprocess.check_call([
        "flang-new-21", "-fc1", "-emit-hlfir", "-fintrinsic-modules-path", "/usr/lib/llvm-21/include/flang",
        str(kernel), "-o",
        str(tmp_path / "kernel.hlfir")
    ],
                          cwd=tmp_path)
    assert (tmp_path / "kernel.hlfir").exists()
