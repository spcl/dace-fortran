"""Coverage for the build-system-facing preprocess CLI: a fresh subprocess running
``python -m dace_fortran.preprocess_cli`` -- the single integration point
``cmake/DaceFortran.cmake`` and ``autotools/dace_fortran.m4`` both rely on."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run_cli(*argv: str, input_path: Path = None, expect_rc: int = 0) -> tuple:
    """Run the CLI in a fresh subprocess; returns (stdout, stderr)."""
    cmd = [sys.executable, "-m", "dace_fortran.preprocess_cli", *argv]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == expect_rc, (f"CLI rc={res.returncode} (expected {expect_rc})\n"
                                         f"argv={argv}\nstdout={res.stdout}\nstderr={res.stderr}")
    return res.stdout, res.stderr


# ---------------------------------------------------------------------------
# Smoke: --help, version-y probe
# ---------------------------------------------------------------------------


def test_help_exits_zero():
    """``--help`` prints usage and exits 0 -- the "entrypoint importable + argparse parses" smoke probe."""
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
    """``--rewrite-external`` + ``--search-dir`` resolves ``EXTERNAL`` to ``USE`` --
    mirrors the cmake macro's invocation."""
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
    """``--rewrite-string-enum`` + ``--out`` writes the sidecar
    ``<out>.enum_maps.json`` that binding-generation consumes downstream."""
    src = tmp_path / "kernel.f90"
    src.write_text(_STRING_ENUM_KERNEL)
    out = tmp_path / "build" / "kernel.preprocessed.f90"

    _run_cli("--rewrite-string-enum", "--in", str(src), "--out", str(out))

    rewritten = out.read_text()
    assert "INTEGER, INTENT(IN) :: action" in rewritten
    assert "CHARACTER(LEN=1), INTENT(IN) :: action" not in rewritten

    sidecar = out.with_suffix(out.suffix + ".enum_maps.json")
    assert sidecar.is_file(), f"missing sidecar at {sidecar}"
    data = json.loads(sidecar.read_text())
    assert "run" in data, f"sidecar missing 'run': {data}"
    assert set(data["run"]["action"]) == {"c", "r"}


def test_rewrite_string_enum_no_sidecar_when_kernel_has_no_enum(tmp_path):
    """A kernel without any CHARACTER enum produces an empty ``enum_maps``; the CLI
    writes no sidecar in that case."""
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
    """``--all-defaults`` runs the same canonical mix as ``preprocess_fortran_source``
    (merge / strip-OpenMP / normalize-kind / rewrite-integer-powers)."""
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
    """``--in -`` reads stdin; omitting ``--out`` writes stdout -- pipeline-friendly."""
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
    """Without ``--in`` argparse exits rc=2 (its standard error code)."""
    cmd = [sys.executable, "-m", "dace_fortran.preprocess_cli", "--all-defaults"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2


# ---------------------------------------------------------------------------
# --inplace -- the build-system-free path
# ---------------------------------------------------------------------------


def test_inplace_rewrites_file_in_place(tmp_path):
    """``--inplace`` rewrites the file at its original path -- no cmake/automake
    glue needed."""
    src = tmp_path / "kernel.f90"
    src.write_text(_STRING_ENUM_KERNEL)
    _run_cli("--rewrite-string-enum", "--inplace", "--in", str(src))
    rewritten = src.read_text()
    assert "INTEGER, INTENT(IN) :: action" in rewritten
    assert "CHARACTER(LEN=1), INTENT(IN) :: action" not in rewritten
    sidecar = src.with_name(src.name + ".enum_maps.json")
    assert sidecar.is_file()


def test_inplace_batch_rewrites_every_input(tmp_path):
    """``--inplace`` plus several ``--in`` paths rewrites every one in order --
    processes a whole source tree in one shot."""
    src1 = tmp_path / "k1.f90"
    src2 = tmp_path / "k2.f90"
    src1.write_text(_STRING_ENUM_KERNEL)
    src2.write_text(_STRING_ENUM_KERNEL.replace("action", "mode"))
    _run_cli("--rewrite-string-enum", "--inplace", "--in", str(src1), "--in", str(src2))
    assert "INTEGER, INTENT(IN) :: action" in src1.read_text()
    assert "INTEGER, INTENT(IN) :: mode" in src2.read_text()


def test_inplace_with_backup_suffix_keeps_original(tmp_path):
    """``--backup-suffix .orig`` keeps a copy of each original next to the rewritten file."""
    src = tmp_path / "k.f90"
    orig = _STRING_ENUM_KERNEL
    src.write_text(orig)
    _run_cli("--rewrite-string-enum", "--inplace", "--backup-suffix", ".orig", "--in", str(src))
    rewritten = src.read_text()
    assert "INTEGER" in rewritten
    backup = src.with_name("k.f90.orig")
    assert backup.is_file()
    assert backup.read_text() == orig


def test_inplace_noop_when_no_pass_changes_source(tmp_path):
    """A no-op input is left untouched, including mtime -- so incremental rebuilds
    don't fire spuriously."""
    src = tmp_path / "k.f90"
    src.write_text("SUBROUTINE k(); END SUBROUTINE\n")
    orig_mtime = src.stat().st_mtime
    import time
    time.sleep(1.05)  # past the FS mtime granularity
    _run_cli("--rewrite-string-enum", "--inplace", "--in", str(src))
    assert src.stat().st_mtime == orig_mtime, \
        "no-op rewrite should preserve mtime"


def test_inplace_and_out_are_mutually_exclusive(tmp_path):
    """``--inplace`` and ``--out`` together is a usage error -- ambiguous semantics."""
    src = tmp_path / "k.f90"
    src.write_text("SUBROUTINE k(); END SUBROUTINE\n")
    out = tmp_path / "out.f90"
    cmd = [
        sys.executable, "-m", "dace_fortran.preprocess_cli", "--all-defaults", "--inplace", "--in",
        str(src), "--out",
        str(out)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "mutually exclusive" in res.stderr.lower() or \
           "mutually exclusive" in res.stdout.lower()


def test_warn_when_rewrite_external_without_search_dir(tmp_path):
    """``--rewrite-external`` without ``--search-dir`` is a no-op; the CLI warns so
    the user notices the misconfig."""
    src = tmp_path / "k.f90"
    src.write_text("SUBROUTINE k(); END SUBROUTINE\n")
    out = tmp_path / "k.preprocessed.f90"
    _, stderr = _run_cli("--rewrite-external", "--in", str(src), "--out", str(out))
    assert "no-op" in stderr.lower() or "search-dir" in stderr.lower(), \
        f"expected a search-dir warning, got stderr:\n{stderr}"
