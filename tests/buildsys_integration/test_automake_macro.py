"""End-to-end smoke for ``autotools/dace_fortran.m4``.

Stages a tiny self-contained autotools project, runs the standard
``aclocal -> autoconf -> automake -> configure -> make`` chain, and
asserts the preprocess rule fires and produces the rewritten
sources.

Skipped when ``autoconf`` / ``automake`` / ``aclocal`` aren't on
the PATH (typical on minimal CI images).
"""
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
    # ``configure.ac`` -- pulls in the m4 macro and invokes it.
    # ``AM_INIT_AUTOMAKE([foreign])`` keeps automake from demanding
    # GNU-project boilerplate (NEWS / AUTHORS / ChangeLog / README).
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
    # Vendored macro -- a project would normally pull the file from
    # the installed dace-fortran share tree, but for the test we
    # symlink the in-repo copy directly into ``m4/``.
    (tmp_path / "m4").mkdir()
    (tmp_path / "m4" / "dace_fortran.m4").symlink_to(_M4_DIR / "dace_fortran.m4")
    # The aux dir automake's missing-tool / install-sh helpers go in.
    (tmp_path / "build-aux").mkdir()

    # Source tree.  The kernel + sidecar module match the cmake test
    # so the assertions can stay shape-equivalent.
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

    # ``Makefile.am`` -- include the rules file then use the helper
    # macro to remap each kernel.f90 to its preprocessed sibling.
    # No compile step (we just want the preprocess rule to fire), so
    # the project's default target is a marker file that depends on
    # the rewritten output.
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
    """Run aclocal + autoconf + automake to materialise configure +
    Makefile.in.  ``--include`` adds the project's m4/ directory so
    ``aclocal`` picks up dace_fortran.m4."""
    subprocess.check_call(["aclocal", "--install", "-I", "m4"], cwd=str(proj))
    subprocess.check_call(["autoconf"], cwd=str(proj))
    subprocess.check_call(["automake", "--add-missing", "--copy", "--foreign"], cwd=str(proj))


def test_configure_succeeds(tmp_path):
    """``./configure`` succeeds -- the DACE_FORTRAN_PREPROCESS macro
    expanded cleanly into the generated ``configure`` script."""
    proj = _write_project(tmp_path)
    _autoreconf(proj)
    res = subprocess.run(["./configure"], cwd=str(proj), capture_output=True, text=True)
    assert res.returncode == 0, \
        f"configure failed:\nstdout={res.stdout}\nstderr={res.stderr}"
    # Sanity: the configure log mentions the macro's probe.
    assert "checking whether" in res.stdout and "dace_fortran" in res.stdout, \
        f"configure didn't run the dace_fortran probe; stdout:\n{res.stdout}"


def test_make_runs_preprocess_and_emits_sources(tmp_path):
    """``make`` invokes the pattern rule and produces the rewritten
    ``.preprocessed.f90`` under the build dir.  Same content
    assertions as the cmake counterpart -- composed passes produce
    the kind-alias rewrite + the EXTERNAL resolution."""
    proj = _write_project(tmp_path)
    _autoreconf(proj)
    subprocess.check_call(["./configure"], cwd=str(proj), stdout=subprocess.DEVNULL)
    res = subprocess.run(["make"], cwd=str(proj), capture_output=True, text=True)
    assert res.returncode == 0, \
        f"make failed:\nstdout={res.stdout}\nstderr={res.stderr}"
    # Default DACE_FORTRAN_BUILD_DIR per the m4 macro:
    # $(top_builddir)/dace_fortran_preprocessed.  In-tree configure
    # makes top_builddir = ``.``.
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
    """A second ``make`` invocation with no changes does NOT
    re-preprocess (the file's mtime stays put)."""
    proj = _write_project(tmp_path)
    _autoreconf(proj)
    subprocess.check_call(["./configure"], cwd=str(proj), stdout=subprocess.DEVNULL)
    subprocess.check_call(["make"], cwd=str(proj), stdout=subprocess.DEVNULL)
    out = proj / "dace_fortran_preprocessed" / "src" / "kernel.preprocessed.f90"
    mtime1 = out.stat().st_mtime
    # Sleep past 1s so any mtime change is observable on filesystems
    # that round to whole seconds.
    import time
    time.sleep(1.05)
    subprocess.check_call(["make"], cwd=str(proj), stdout=subprocess.DEVNULL)
    mtime2 = out.stat().st_mtime
    assert mtime1 == mtime2, \
        f"unchanged source was re-preprocessed; mtime1={mtime1} mtime2={mtime2}"
