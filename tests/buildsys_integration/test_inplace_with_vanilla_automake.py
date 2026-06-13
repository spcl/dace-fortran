"""End-to-end proof of the user's "skip the build-system glue
entirely" path:

  1. Stage a vanilla automake Fortran-library project (Fortran
     sources + Makefile.am + configure.ac).  The project knows
     NOTHING about dace-fortran: no m4 macro, no .mk include,
     no DACE_FORTRAN_* anywhere.
  2. Run ``python -m dace_fortran.preprocess_cli --inplace`` over
     every source to apply the bridge's rewrites in place.
  3. Run the user's existing autotools chain --
     ``aclocal -> autoreconf -> ./configure -> make`` -- against
     the now-rewritten sources.
  4. Assert a real ``libfoo.so`` is produced AND that it loads /
     exports the rewritten kernel's symbol.

If this passes, the answer to the user's question -- "can we just
use a Python script that replaces the file with our generated
header, and we still compile internally?" -- is yes,
unambiguously, with no autotools glue at all.
"""
import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_NEEDED = ("gfortran", "autoreconf", "aclocal", "automake", "autoconf", "libtoolize", "make")
_HAVE_TOOLCHAIN = all(shutil.which(t) is not None for t in _NEEDED)

pytestmark = pytest.mark.skipif(not _HAVE_TOOLCHAIN, reason=f"missing one of: {', '.join(_NEEDED)}")

# Vanilla project sources -- no dace-fortran-specific anything.
_CONFIGURE_AC = """\
AC_INIT([dace_fortran_inplace_smoke], [0.1])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE([foreign no-dependencies subdir-objects])
LT_INIT
AC_PROG_FC
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
"""

# A no-Libtool ``.la`` library; build a plain ``libfoo.so`` so the
# test loads with ctypes without libtool wrappers in the way.
_MAKEFILE_AM = """\
AM_FCFLAGS = -fPIC -J$(builddir)

lib_LTLIBRARIES = libfoo.la

# ``utils/utils_mod.f90`` first so its .mod file exists when
# ``kernel.f90`` compiles.  ``subdir-objects`` (set in
# AM_INIT_AUTOMAKE) handles the subdir layout transparently.
libfoo_la_SOURCES = utils/utils_mod.f90 kernel.f90
libfoo_la_LDFLAGS = -avoid-version -module -shared
"""

# Two rewrites composed in one shot:
#   * Kind alias ``wp`` -> ``8``      (normalize-kind)
#   * EXTERNAL ``dscale`` -> ``USE``  (rewrite-external)
# The sidecar module sits in src/utils/ where ``--search-dir`` finds it.
_KERNEL = """\
SUBROUTINE foo_run(out_val, x, f) BIND(C, NAME="foo_run")
  USE iso_c_binding
  IMPLICIT NONE
  REAL(c_double), INTENT(IN) :: x, f
  REAL(c_double), INTENT(OUT) :: out_val
  REAL(KIND=wp) :: dscale
  EXTERNAL :: dscale
  out_val = dscale(x, f) + 1.0d0
END SUBROUTINE
"""

_UTILS_MOD = """\
MODULE utils_mod
  USE iso_c_binding
  IMPLICIT NONE
CONTAINS
  REAL(c_double) FUNCTION dscale(x, f)
    REAL(c_double), INTENT(IN) :: x, f
    dscale = x * f
  END FUNCTION dscale
END MODULE utils_mod
"""


def _stage_project(tmp_path: Path) -> Path:
    """Materialise the no-glue automake project on disk."""
    (tmp_path / "build-aux").mkdir()
    (tmp_path / "m4").mkdir()
    (tmp_path / "configure.ac").write_text(_CONFIGURE_AC)
    (tmp_path / "Makefile.am").write_text(_MAKEFILE_AM)
    (tmp_path / "kernel.f90").write_text(_KERNEL)
    (tmp_path / "utils").mkdir()
    (tmp_path / "utils" / "utils_mod.f90").write_text(_UTILS_MOD)
    return tmp_path


def _autoreconf(proj: Path):
    """Standard autotools bootstrap.  ``autoreconf -fvi`` does
    libtoolize + aclocal + autoconf + automake in one shot."""
    subprocess.check_call(["autoreconf", "-fvi"], cwd=str(proj), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def _preprocess_inplace(proj: Path):
    """Run the dace-fortran CLI with ``--inplace`` over the kernel
    source.  This is the ENTIRE bridge-side integration: one
    command, no automake glue, no m4 macro.
    """
    inputs = list(proj.glob("*.f90"))  # kernel.f90 (utils/ stays unprocessed)
    cmd = [
        sys.executable, "-m", "dace_fortran.preprocess_cli", "--all-defaults", "--rewrite-external", "--search-dir",
        str(proj / "utils"), "--inplace"
    ]
    for f in inputs:
        cmd.extend(["--in", str(f)])
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, \
        f"preprocess CLI failed:\nstdout={res.stdout}\nstderr={res.stderr}"


def test_inplace_then_vanilla_automake_chain_produces_so(tmp_path):
    """The bottom-line proof: preprocess in place, then run
    ``aclocal -> autoreconf -> configure -> make`` exactly as the
    user would for a project that's never heard of dace-fortran.
    A real ``libfoo.so`` lands in the build tree."""
    proj = _stage_project(tmp_path)

    # Step 1: preprocess in place.  No autotools touched yet.
    _preprocess_inplace(proj)

    # Confirm the rewrites landed BEFORE configure runs.  This is
    # the "your build system never saw the original" condition.
    rewritten = (proj / "kernel.f90").read_text()
    assert "USE utils_mod, ONLY: dscale" in rewritten
    assert "EXTERNAL :: dscale" not in rewritten
    # ``REAL(KIND=wp) :: dscale`` was deleted alongside the EXTERNAL
    # (the function-result type declaration is redundant once dscale
    # is use-associated).  The kind alias ``wp`` therefore disappears
    # via line deletion, not via normalize-kind's substitution -- the
    # bottom-line contract is that no unresolved ``wp`` survives.
    code_lines = "\n".join(line for line in rewritten.splitlines() if not line.lstrip().startswith("!"))
    assert "wp" not in code_lines.lower(), \
        f"kind alias wp should be resolved; got code:\n{code_lines}"

    # Step 2-4: vanilla autotools chain.  No dace-fortran flags.
    _autoreconf(proj)
    res = subprocess.run(["./configure"], cwd=str(proj), capture_output=True, text=True)
    assert res.returncode == 0, \
        f"configure failed:\nstdout={res.stdout}\nstderr={res.stderr}"
    # Plain ``make`` -- the Makefile.am lists both
    # ``utils/utils_mod.f90`` and ``kernel.f90`` in
    # ``libfoo_la_SOURCES``, in module-first order.
    res = subprocess.run(["make"], cwd=str(proj), capture_output=True, text=True)
    assert res.returncode == 0, \
        f"make failed:\nstdout={res.stdout[-2000:]}\nstderr={res.stderr[-2000:]}"

    # Step 5: assert the .so is on disk and the kernel symbol is in it.
    candidates = list(proj.glob(".libs/libfoo.so*")) + \
                  list(proj.glob("libfoo.so*"))
    so_paths = [p for p in candidates if p.is_file()]
    assert so_paths, \
        f"no libfoo.so produced; tree:\n" + \
        "\n".join(str(p.relative_to(proj)) for p in proj.rglob("*"))

    # Bonus: load the .so and call into the rewritten kernel.
    # ctypes resolves utils_mod's dscale symbol from the same .so
    # (libtool linked it in).
    lib = ctypes.CDLL(str(so_paths[0]))
    foo_run = lib.foo_run
    foo_run.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    foo_run.restype = None
    out = ctypes.c_double(0.0)
    x = ctypes.c_double(2.0)
    f = ctypes.c_double(3.0)
    foo_run(ctypes.byref(out), ctypes.byref(x), ctypes.byref(f))
    # dscale(2.0, 3.0) = 6.0; foo_run returns 6.0 + 1.0 = 7.0.
    assert abs(out.value - 7.0) < 1e-12, \
        f"foo_run returned {out.value} (expected 7.0)"


def test_inplace_only_no_op_when_kernel_already_canonical(tmp_path):
    """A kernel that's already canonical is left untouched by
    ``--inplace``.  Use a kernel that doesn't need any rewrites at
    all -- the two-pass composition of merge-modules with
    rewrite-external isn't naturally idempotent (rewrite-external
    adds a ``USE`` statement merge-modules would inline on a second
    pass), but a kernel with no rewrites to do should be a no-op
    on every invocation."""
    proj = tmp_path
    canon = proj / "k.f90"
    canon.write_text("""\
SUBROUTINE k(x)
  REAL(8), INTENT(INOUT) :: x
  x = x + 1.0d0
END SUBROUTINE
""")
    cmd = [sys.executable, "-m", "dace_fortran.preprocess_cli", "--all-defaults", "--inplace", "--in", str(canon)]
    subprocess.check_call(cmd)
    first_mtime = canon.stat().st_mtime

    # Sleep past 1s and re-run; mtime should NOT change.
    import time
    time.sleep(1.05)
    subprocess.check_call(cmd)
    second_mtime = canon.stat().st_mtime
    assert first_mtime == second_mtime, \
        f"canonical kernel was re-written; mtimes {first_mtime} -> {second_mtime}"
