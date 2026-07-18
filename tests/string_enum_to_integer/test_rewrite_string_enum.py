"""Unit + e2e coverage for ``rewrite_string_enum_to_integer`` (pattern 2 of the
dual-pattern QE rewrite): a ``CHARACTER`` enum-style switch (``flag == 'c'``) becomes
``INTEGER`` with a sidecar ``enum_maps`` dict for the Python-boundary string surface.
Covers textual rewrite correctness, the sidecar dict's shape, and a flang smoke parse."""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from dace_fortran.preprocess import rewrite_string_enum_to_integer

_HERE = Path(__file__).resolve().parent
_HAVE_FLANG = shutil.which("flang-new-21") is not None


def _read(name: str) -> str:
    return (_HERE / name).read_text()


def _strip_comments(src: str) -> str:
    out = []
    for line in src.splitlines(keepends=True):
        if line.lstrip().startswith("!"):
            continue
        out.append(line)
    return "".join(out)


# ---------------------------------------------------------------------------
# Basic single-equality form
# ---------------------------------------------------------------------------


def test_basic_signature_becomes_integer():
    """``CHARACTER(LEN=1), INTENT(IN) :: action`` rewrites to
    ``INTEGER, INTENT(IN) :: action``."""
    src = _read("string_enum_basic_example.f90")
    out, enum_maps = rewrite_string_enum_to_integer(src)
    code = _strip_comments(out)
    assert re.search(r"(?im)^\s*INTEGER,\s*INTENT\s*\(\s*IN\s*\)\s*::\s*action\s*$",
                     code), \
        "signature should be integer-typed after rewrite"
    assert not re.search(r"(?im)^\s*CHARACTER.*action\s*$", code), \
        "CHARACTER declaration of action should be gone"


def test_basic_comparisons_become_integer_literals():
    """The three ``action == '<lit>'`` comparisons map to ``== 0``/``== 1``/``== 2``
    (first-appearance order)."""
    src = _read("string_enum_basic_example.f90")
    out, enum_maps = rewrite_string_enum_to_integer(src)
    code = _strip_comments(out)
    # All three integer comparisons present.
    assert re.search(r"action\s*==\s*0", code)
    assert re.search(r"action\s*==\s*1", code)
    assert re.search(r"action\s*==\s*2", code)
    # No string literals left in code comparisons.
    assert not re.search(r"action\s*==\s*'[a-zA-Z]'", code), \
        "string comparison should be gone from code"


def test_basic_enum_map_shape():
    """``enum_maps`` exposes the per-procedure per-arg mapping."""
    src = _read("string_enum_basic_example.f90")
    _, enum_maps = rewrite_string_enum_to_integer(src)
    assert "run" in enum_maps, f"expected 'run' procedure: {list(enum_maps)}"
    assert "action" in enum_maps["run"], \
        f"expected 'action' arg: {list(enum_maps['run'])}"
    m = enum_maps["run"]["action"]
    # Three distinct literals, all lowercase.
    assert set(m) == {"c", "r", "i"}, f"got {m}"
    # Deterministic first-appearance order.
    assert m["c"] == 0
    assert m["r"] == 1
    assert m["i"] == 2


# ---------------------------------------------------------------------------
# Case-insensitive grouping (the QE shape)
# ---------------------------------------------------------------------------


def test_case_insensitive_pairs_collapse_to_one_int():
    """``flag == 'c' .OR. flag == 'C'`` -- both literals map to the SAME integer, giving
    ``flag == 0 .OR. flag == 0`` (later collapsed by the optimiser)."""
    src = _read("string_enum_case_insensitive_example.f90")
    out, enum_maps = rewrite_string_enum_to_integer(src)
    m = enum_maps["run"]["flag"]
    # Three distinct case-insensitive groups, integers 0..2.
    assert set(m) == {"c", "r", "i"}
    assert set(m.values()) == {0, 1, 2}


def test_case_insensitive_comparison_rewrites_both_variants():
    """Both ``'c'`` and ``'C'`` occurrences in the source rewrite
    to the same integer."""
    src = _read("string_enum_case_insensitive_example.f90")
    out, _ = rewrite_string_enum_to_integer(src)
    code = _strip_comments(out)
    # ``flag == 'c' .OR. flag == 'C'`` should become ``flag == 0 .OR. flag == 0``
    assert re.search(r"flag\s*==\s*0\s*\.OR\.\s*flag\s*==\s*0", code,
                     re.IGNORECASE), \
        f"case-insensitive pair should collapse to same int.  got:\n{code}"


# ---------------------------------------------------------------------------
# SELECT CASE form
# ---------------------------------------------------------------------------


def test_select_case_rewrites_case_branches():
    """``SELECT CASE (mode)`` with literal-string ``CASE (...)`` branches has each
    case literal rewritten to its assigned integer."""
    src = _read("string_enum_select_case_example.f90")
    out, enum_maps = rewrite_string_enum_to_integer(src)
    code = _strip_comments(out)
    m = enum_maps["run"]["mode"]
    assert set(m) == {"forward", "backward", "zero"}
    # Each CASE branch references an integer.
    assert re.search(r"CASE\s*\(\s*0\s*\)", code)
    assert re.search(r"CASE\s*\(\s*1\s*\)", code)
    assert re.search(r"CASE\s*\(\s*2\s*\)", code)
    # CASE DEFAULT stays.
    assert "CASE DEFAULT" in code.upper() or "CASE default" in code


# ---------------------------------------------------------------------------
# Pass-through + idempotency
# ---------------------------------------------------------------------------


def test_passthrough_on_kernel_with_no_string_enum():
    """A procedure whose only CHARACTER dummy is never compared against a literal is
    left alone -- might be a real string the bridge passes through."""
    src = """
SUBROUTINE run(out_val, msg)
  IMPLICIT NONE
  CHARACTER(LEN=*), INTENT(IN) :: msg
  REAL(8), INTENT(OUT) :: out_val
  out_val = REAL(LEN(msg), 8)
END SUBROUTINE
"""
    out, enum_maps = rewrite_string_enum_to_integer(src)
    assert out == src
    assert enum_maps == {}


def test_passthrough_on_kernel_with_no_character_dummies():
    """No CHARACTER dummy means there's nothing to consider."""
    src = """
SUBROUTINE run(out_val, x)
  IMPLICIT NONE
  REAL(8), INTENT(IN) :: x
  REAL(8), INTENT(OUT) :: out_val
  out_val = x * 2.0d0
END SUBROUTINE
"""
    out, enum_maps = rewrite_string_enum_to_integer(src)
    assert out == src
    assert enum_maps == {}


def test_idempotent():
    """A second pass over an already-rewritten source finds no
    CHARACTER dummies and is a no-op."""
    src = _read("string_enum_basic_example.f90")
    once, m1 = rewrite_string_enum_to_integer(src)
    twice, m2 = rewrite_string_enum_to_integer(once)
    assert once == twice
    assert m2 == {}  # nothing left to rewrite


# ---------------------------------------------------------------------------
# flang smoke-parse  --  rewrite is syntactically valid
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAVE_FLANG, reason="flang-new-21 not on PATH")
@pytest.mark.parametrize("probe", [
    "string_enum_basic_example.f90",
    "string_enum_case_insensitive_example.f90",
    "string_enum_select_case_example.f90",
])
def test_rewritten_probe_parses_under_flang(probe, tmp_path):
    """flang lowers each rewritten probe to HLFIR without semantic
    errors -- proves the output is valid Fortran."""
    src = _read(probe)
    rewritten, _ = rewrite_string_enum_to_integer(src)
    f = tmp_path / "k.f90"
    f.write_text(rewritten)
    subprocess.check_call([
        "flang-new-21", "-fc1", "-emit-hlfir", "-fintrinsic-modules-path", "/usr/lib/llvm-21/include/flang",
        str(f), "-o",
        str(tmp_path / "k.hlfir")
    ],
                          cwd=tmp_path)
    assert (tmp_path / "k.hlfir").exists()
