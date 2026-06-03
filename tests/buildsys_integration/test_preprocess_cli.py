"""Coverage for the build-system-facing preprocess CLI.

These tests exercise the same code path a cmake ``add_custom_command``
or an autotools pattern rule will invoke: a fresh Python subprocess
running ``python -m dace_fortran.preprocess_cli`` with the input
source on disk and the rewritten output written to a target path.

The CLI is the single integration point between the bridge's
preprocess passes and any external build system; pinning its
behaviour here is the load-bearing contract that
``cmake/DaceFortran.cmake`` and ``autotools/dace_fortran.m4`` rely
on.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run_cli(*argv: str, input_path: Path = None, expect_rc: int = 0) -> tuple:
    """Run the CLI in a fresh subprocess; return (stdout, stderr) on
    success.  ``input_path`` is read into ``--in`` (or use ``--in -``
    for stdin via a separate test)."""
    cmd = [sys.executable, "-m", "dace_fortran.preprocess_cli", *argv]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == expect_rc, (f"CLI rc={res.returncode} (expected {expect_rc})\n"
                                         f"argv={argv}\nstdout={res.stdout}\nstderr={res.stderr}")
    return res.stdout, res.stderr


# ---------------------------------------------------------------------------
# Smoke: --help, version-y probe
# ---------------------------------------------------------------------------


def test_help_exits_zero():
    """``python -m dace_fortran.preprocess_cli --help`` prints usage
    and exits 0 -- the universal "is the entrypoint importable + does
    argparse parse" probe build systems run."""
    stdout, _ = _run_cli("--help")
    assert "preprocess" in stdout.lower()
    assert "--rewrite-external" in stdout
    assert "--rewrite-string-enum" in stdout


# ---------------------------------------------------------------------------
# Pattern 1: --rewrite-external against a sidecar module
# ---------------------------------------------------------------------------

_EXTERNAL_KERNEL = """\
SUBROUTINE run(out_val, x, f)
  IMPLICIT NONE
  REAL(8), INTENT(IN) :: x, f
  REAL(8), INTENT(OUT) :: out_val
  REAL(8) :: dscale
  EXTERNAL :: dscale
  out_val = dscale(x, f)
END SUBROUTINE
"""

_UTILS_MOD = """\
MODULE utils_mod
  IMPLICIT NONE
CONTAINS
  REAL(8) FUNCTION dscale(x, f)
    REAL(8), INTENT(IN) :: x, f
    dscale = x * f
  END FUNCTION dscale
END MODULE utils_mod
"""


def test_rewrite_external_resolves_through_search_dir(tmp_path):
    """A standalone CLI invocation with ``--rewrite-external`` +
    ``--search-dir`` resolves ``EXTERNAL`` to ``USE`` -- mirrors the
    cmake invocation the macro builds."""
    src = tmp_path / "kernel.f90"
    src.write_text(_EXTERNAL_KERNEL)
    sidecar_dir = tmp_path / "utils"
    sidecar_dir.mkdir()
    (sidecar_dir / "utils_mod.f90").write_text(_UTILS_MOD)
    out = tmp_path / "build" / "kernel.preprocessed.f90"

    _run_cli("--rewrite-external", "--search-dir", str(sidecar_dir), "--in", str(src), "--out", str(out))

    rewritten = out.read_text()
    assert "USE utils_mod, ONLY: dscale" in rewritten
    assert "EXTERNAL :: dscale" not in rewritten
    assert "EXTERNAL dscale" not in rewritten


# ---------------------------------------------------------------------------
# Pattern 2: --rewrite-string-enum + sidecar JSON
# ---------------------------------------------------------------------------

_STRING_ENUM_KERNEL = """\
SUBROUTINE run(out_val, action)
  IMPLICIT NONE
  CHARACTER(LEN=1), INTENT(IN) :: action
  REAL(8), INTENT(OUT) :: out_val
  IF (action == 'c') THEN
    out_val = 1.0d0
  ELSE IF (action == 'r') THEN
    out_val = 2.0d0
  ELSE
    out_val = 0.0d0
  END IF
END SUBROUTINE
"""


def test_rewrite_string_enum_writes_sidecar_json(tmp_path):
    """When ``--rewrite-string-enum`` is on and ``--out`` is set, the
    CLI writes the enum-map sidecar ``<out>.enum_maps.json`` next to
    the rewritten source.  Binding-generation downstream consumes it."""
    src = tmp_path / "kernel.f90"
    src.write_text(_STRING_ENUM_KERNEL)
    out = tmp_path / "build" / "kernel.preprocessed.f90"

    _run_cli("--rewrite-string-enum", "--in", str(src), "--out", str(out))

    # Source rewrite landed.
    rewritten = out.read_text()
    assert "INTEGER, INTENT(IN) :: action" in rewritten
    assert "CHARACTER(LEN=1), INTENT(IN) :: action" not in rewritten

    # Sidecar JSON sits next to the rewritten source.
    sidecar = out.with_suffix(out.suffix + ".enum_maps.json")
    assert sidecar.is_file(), f"missing sidecar at {sidecar}"
    data = json.loads(sidecar.read_text())
    assert "run" in data, f"sidecar missing 'run': {data}"
    assert set(data["run"]["action"]) == {"c", "r"}


def test_rewrite_string_enum_no_sidecar_when_kernel_has_no_enum(tmp_path):
    """A kernel without any CHARACTER enum produces an empty
    ``enum_maps`` dict from the pass; the CLI doesn't write a
    sidecar in that case (no information to record)."""
    src = tmp_path / "k.f90"
    src.write_text("""\
SUBROUTINE run(x)
  REAL(8), INTENT(INOUT) :: x
  x = x + 1.0d0
END SUBROUTINE
""")
    out = tmp_path / "build" / "k.preprocessed.f90"
    _run_cli("--rewrite-string-enum", "--in", str(src), "--out", str(out))
    sidecar = out.with_suffix(out.suffix + ".enum_maps.json")
    assert not sidecar.exists(), \
        f"unexpected sidecar at {sidecar}"


# ---------------------------------------------------------------------------
# Default composition --all-defaults
# ---------------------------------------------------------------------------


def test_all_defaults_applies_merge_strip_kind_powers(tmp_path):
    """``--all-defaults`` runs the same canonical mix as
    ``preprocess_fortran_source`` defaults (merge / strip-OpenMP /
    normalize-kind / rewrite-integer-powers)."""
    src = tmp_path / "k.f90"
    src.write_text("""\
SUBROUTINE run(out_val, x)
  IMPLICIT NONE
  REAL(KIND=wp), INTENT(IN) :: x
  REAL(KIND=wp), INTENT(OUT) :: out_val
  !$OMP PARALLEL
  out_val = x**2.0
  !$OMP END PARALLEL
END SUBROUTINE
""")
    out = tmp_path / "build" / "k.preprocessed.f90"
    _run_cli("--all-defaults", "--in", str(src), "--out", str(out))
    rewritten = out.read_text()
    # Kind alias normalised to ``8``.
    assert "KIND=8" in rewritten
    assert "KIND=wp" not in rewritten
    # OpenMP sentinel stripped.
    assert "!$OMP" not in rewritten
    # x**2.0 -> (x*x).
    assert "(x*x)" in rewritten or "x*x" in rewritten


# ---------------------------------------------------------------------------
# stdin / stdout
# ---------------------------------------------------------------------------


def test_stdin_to_stdout_works(tmp_path):
    """``--in -`` reads from stdin; omitting ``--out`` writes to
    stdout.  Useful for pipelines (``cat foo.f90 | cli | xargs flang``)."""
    cmd = [sys.executable, "-m", "dace_fortran.preprocess_cli", "--all-defaults", "--in", "-"]
    res = subprocess.run(cmd,
                         input=_STRING_ENUM_KERNEL.replace("CHARACTER(LEN=1), INTENT(IN) :: action",
                                                           "REAL(KIND=wp), INTENT(IN) :: dummy"),
                         capture_output=True,
                         text=True)
    assert res.returncode == 0
    # Kind alias normalisation happened, and the output came to stdout.
    assert "KIND=8" in res.stdout


# ---------------------------------------------------------------------------
# Argument hygiene
# ---------------------------------------------------------------------------


def test_missing_in_flag_is_an_argument_error():
    """Without ``--in`` argparse exits with rc=2 (its standard
    error code)."""
    cmd = [sys.executable, "-m", "dace_fortran.preprocess_cli", "--all-defaults"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2


def test_warn_when_rewrite_external_without_search_dir(tmp_path):
    """``--rewrite-external`` without any ``--search-dir`` is a
    no-op (no modules to resolve against); the CLI prints a clear
    warning so the user notices the misconfig."""
    src = tmp_path / "k.f90"
    src.write_text("SUBROUTINE k(); END SUBROUTINE\n")
    out = tmp_path / "k.preprocessed.f90"
    _, stderr = _run_cli("--rewrite-external", "--in", str(src), "--out", str(out))
    assert "no-op" in stderr.lower() or "search-dir" in stderr.lower(), \
        f"expected a search-dir warning, got stderr:\n{stderr}"
