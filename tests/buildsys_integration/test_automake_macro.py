"""E2e smoke for autotools/dace_fortran.m4: stages a project, runs
aclocal -> autoconf -> automake -> configure -> make, asserts the preprocess rule fires.
Skipped when autoconf/automake/aclocal aren't on PATH."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_HAVE_AUTOCONF = shutil.which("autoconf") is not None
_HAVE_AUTOMAKE = shutil.which("automake") is not None
_HAVE_ACLOCAL = shutil.which("aclocal") is not None
_HAVE_MAKE = shutil.which("make") is not None
_REPO_ROOT = Path(__file__).resolve().parents[2]
_M4_DIR = _REPO_ROOT / "autotools"

pytestmark = pytest.mark.skipif(not (_HAVE_AUTOCONF and _HAVE_AUTOMAKE and _HAVE_ACLOCAL and _HAVE_MAKE),
                                reason="autoconf / automake / aclocal / make required")


def _write_project(tmp_path: Path) -> Path:
    """Stage a minimal automake project that exercises the macro."""
    # configure.ac pulls in the macro; AM_INIT_AUTOMAKE([foreign]) skips GNU boilerplate (NEWS/AUTHORS/ChangeLog/README).
    (tmp_path / "configure.ac").write_text(f"""\
AC_INIT([dace_fortran_automake_smoke], [0.1])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE([foreign no-dependencies subdir-objects])
AC_PROG_INSTALL

# Force a specific Python so the smoke test doesn't depend on PATH
# ordering.
DACE_FORTRAN_PYTHON={sys.executable}
export DACE_FORTRAN_PYTHON
AC_SUBST([DACE_FORTRAN_PYTHON])

DACE_FORTRAN_PREPROCESS

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
""")
    # vendored macro: symlinks the in-repo copy into m4/ instead of pulling from an installed share tree.
    (tmp_path / "m4").mkdir()
    (tmp_path / "m4" / "dace_fortran.m4").symlink_to(_M4_DIR / "dace_fortran.m4")
    # The aux dir automake's missing-tool / install-sh helpers go in.
    (tmp_path / "build-aux").mkdir()

    # source tree matches the cmake test's kernel + sidecar module so assertions stay shape-equivalent.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "kernel.f90").write_text("""\
SUBROUTINE run(out_val, x, f)
  IMPLICIT NONE
  REAL(KIND=wp), INTENT(IN) :: x, f
  REAL(KIND=wp), INTENT(OUT) :: out_val
  REAL(KIND=wp) :: dscale
  EXTERNAL :: dscale
  out_val = dscale(x, f)
END SUBROUTINE
""")
    (tmp_path / "utils").mkdir()
    (tmp_path / "utils" / "utils_mod.f90").write_text("""\
MODULE utils_mod
  IMPLICIT NONE
CONTAINS
  REAL(KIND=8) FUNCTION dscale(x, f)
    REAL(KIND=8), INTENT(IN) :: x, f
    dscale = x * f
  END FUNCTION dscale
END MODULE utils_mod
""")

    # Makefile.am includes the rules file and remaps kernel.f90 via the helper macro; no compile
    # step, so the default target is a marker file depending on the rewritten output.
    rules_mk = (_REPO_ROOT / "autotools" / "dace_fortran.mk")
    (tmp_path / "Makefile.am").write_text(f"""\
include {rules_mk}

DACE_FORTRAN_PASSES      = all_defaults rewrite_external
DACE_FORTRAN_SEARCH_DIRS = $(srcdir)/utils

preprocessed_sources = $(call dace_fortran_preprocess, src/kernel.f90)

all-local: preprocessed.stamp

preprocessed.stamp: $(preprocessed_sources)
\t@touch $@

CLEANFILES = preprocessed.stamp
""")
    return tmp_path


def _autoreconf(proj: Path):
    """aclocal+autoconf+automake -> configure + Makefile.in; -I m4 lets aclocal find dace_fortran.m4."""
    subprocess.check_call(["aclocal", "--install", "-I", "m4"], cwd=str(proj))
    subprocess.check_call(["autoconf"], cwd=str(proj))
    subprocess.check_call(["automake", "--add-missing", "--copy", "--foreign"], cwd=str(proj))


def test_configure_succeeds(tmp_path):
    """./configure succeeds -- DACE_FORTRAN_PREPROCESS macro expanded cleanly into the generated script."""
    proj = _write_project(tmp_path)
    _autoreconf(proj)
    res = subprocess.run(["./configure"], cwd=str(proj), capture_output=True, text=True)
    assert res.returncode == 0, \
        f"configure failed:\nstdout={res.stdout}\nstderr={res.stderr}"
    # Sanity: the configure log mentions the macro's probe.
    assert "checking whether" in res.stdout and "dace_fortran" in res.stdout, \
        f"configure didn't run the dace_fortran probe; stdout:\n{res.stdout}"


def test_make_runs_preprocess_and_emits_sources(tmp_path):
    """make produces the rewritten .preprocessed.f90; same content assertions as the cmake counterpart
    (kind-alias rewrite + EXTERNAL resolution)."""
    proj = _write_project(tmp_path)
    _autoreconf(proj)
    subprocess.check_call(["./configure"], cwd=str(proj), stdout=subprocess.DEVNULL)
    res = subprocess.run(["make"], cwd=str(proj), capture_output=True, text=True)
    assert res.returncode == 0, \
        f"make failed:\nstdout={res.stdout}\nstderr={res.stderr}"
    # default DACE_FORTRAN_BUILD_DIR = $(top_builddir)/dace_fortran_preprocessed; in-tree top_builddir = '.'.
    out = proj / "dace_fortran_preprocessed" / "src" / "kernel.preprocessed.f90"
    assert out.is_file(), \
        f"no preprocessed kernel at {out}; tree:\n" + \
        "\n".join(str(p.relative_to(proj)) for p in proj.rglob("*"))
    rewritten = out.read_text()
    # ``KIND=wp`` -> ``KIND=8``.
    assert "KIND=8" in rewritten
    assert "KIND=wp" not in rewritten
    # ``EXTERNAL :: dscale`` -> ``USE utils_mod, ONLY: dscale``.
    assert "USE utils_mod, ONLY: dscale" in rewritten
    assert "EXTERNAL :: dscale" not in rewritten


def test_make_is_incremental_on_unchanged_source(tmp_path):
    """Second make with no changes does not re-preprocess (mtime stays put)."""
    proj = _write_project(tmp_path)
    _autoreconf(proj)
    subprocess.check_call(["./configure"], cwd=str(proj), stdout=subprocess.DEVNULL)
    subprocess.check_call(["make"], cwd=str(proj), stdout=subprocess.DEVNULL)
    out = proj / "dace_fortran_preprocessed" / "src" / "kernel.preprocessed.f90"
    mtime1 = out.stat().st_mtime
    # sleep past 1s: some filesystems round mtime to whole seconds.
    import time
    time.sleep(1.05)
    subprocess.check_call(["make"], cwd=str(proj), stdout=subprocess.DEVNULL)
    mtime2 = out.stat().st_mtime
    assert mtime1 == mtime2, \
        f"unchanged source was re-preprocessed; mtime1={mtime1} mtime2={mtime2}"
