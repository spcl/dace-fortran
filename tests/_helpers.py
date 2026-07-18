"""Shared helpers for verbatim ports from ``f2dace/dev:tests/fortran/`` -- reference build, arg routing, strict-xfail marker."""

import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

#: Retry only the f2py reference build: under heavy parallel load a fork/exec ENOMEM
#: can transiently fail gfortran, unrelated to the source.  Never retry the SDFG build.
_F2PY_BUILD_ATTEMPTS = 3

#: An rc=0 f2py build can still miss a routine (crackfortran drop under swap-thrash),
#: surfacing later as a cryptic AttributeError.  Detect at import and rebuild under a
#: fresh extension name (re-importing a same-named CPython ext SIGABRTs at teardown).
_F2PY_IMPORT_ATTEMPTS = 3


def f2py_build_with_retry(cmd, *, cwd, mod_name, env=None):
    """Retry ``f2py -c`` on transient fork/exec ENOMEM (not a source error); raises
    ``RuntimeError`` on a genuine compile failure.  Reference-build only -- never the
    SDFG build.  Shared retry policy with ``f2py_compile`` in ``_util``."""
    cwd = Path(cwd)
    for attempt in range(1, _F2PY_BUILD_ATTEMPTS + 1):
        proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
        if proc.returncode == 0:
            return
        if attempt == _F2PY_BUILD_ATTEMPTS:
            raise RuntimeError(f"f2py reference build for {mod_name!r} failed after "
                               f"{_F2PY_BUILD_ATTEMPTS} attempts (rc={proc.returncode}).\n"
                               f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
        # drop half-written extension; back off for the resource spike to clear
        for stale in cwd.glob(f"{mod_name}*.so"):
            stale.unlink()
        time.sleep(2 * attempt)


def _f2py_routine_present(mod, name: str) -> bool:
    """True if ``name`` is wrapped on ``mod`` -- checks both top-level (free
    subroutine) and per-module namespace (module subroutine) placement."""
    if hasattr(mod, name):
        return True
    for nm in vars(mod):
        if nm.startswith("_"):
            continue
        ns = getattr(mod, nm)
        if callable(getattr(ns, name, None)):
            return True
    return False


def f2py_build_and_import(src_file, *, out_dir, mod_name, only=None, extra_args=(), env=None):
    """Build ``src_file`` via ``numpy.f2py -c`` and import it.  Hardened against two
    heavy-parallel-load failure modes: fork/exec ENOMEM (retried by
    :func:`f2py_build_with_retry`) and an rc=0 but incomplete module missing an
    ``only``-named routine (rebuilt under a fresh extension name).  Raises
    ``RuntimeError`` naming the missing routine rather than leaking ``AttributeError``."""
    out_dir = Path(out_dir)
    base = [sys.executable, "-m", "numpy.f2py", "-c", str(src_file)]
    only_args = ["only:", *only, ":"] if only else []
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    expected = tuple(only) if only else ()
    incomplete = None
    for attempt in range(1, _F2PY_IMPORT_ATTEMPTS + 1):
        ext_name = mod_name if attempt == 1 else f"{mod_name}_r{attempt}"
        cmd = [*base, "-m", ext_name, "--quiet", *extra_args, *only_args]
        f2py_build_with_retry(cmd, cwd=out_dir, mod_name=ext_name, env=env)
        __import__(ext_name)
        mod = sys.modules[ext_name]
        missing = [s for s in expected if not _f2py_routine_present(mod, s)]
        if not missing:
            return mod
        wrapped = sorted(n for n in vars(mod) if not n.startswith("_"))
        incomplete = (ext_name, missing, wrapped)
        for stale in out_dir.glob(f"{ext_name}*.so"):
            stale.unlink()
    raise RuntimeError(f"f2py reference {mod_name!r} built (rc=0) but is missing requested "
                       f"routine(s) {incomplete[1]} after {_F2PY_IMPORT_ATTEMPTS} attempts; "
                       f"wrapped names = {incomplete[2]}")


def f2py(src_text: str, out_dir: Path, mod_name: str):
    """Compile ``src_text`` via ``numpy.f2py`` and return the imported module.  Skips
    if gfortran/meson missing; retries on transient resource-exhaustion (see
    ``_F2PY_BUILD_ATTEMPTS``)."""
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not available")
    if shutil.which("meson") is None:
        pytest.skip("meson not available (f2py backend on Python>=3.12)")
    out_dir.mkdir(parents=True, exist_ok=True)
    src_file = out_dir / f"{mod_name}.f90"
    src_file.write_text(src_text)
    return f2py_build_and_import(src_file, out_dir=out_dir, mod_name=mod_name)


def sdfg_call_args(sdfg, int_values: dict) -> dict:
    """Route each int in ``int_values`` to a plain int or length-1 int32 array per the
    SDFG's Scalar-vs-Array classification.  Mirrors the helper in
    ``icon/selected_loopnests/test_sdfg_equivalence.py``."""
    from dace.data import Scalar
    arglist = sdfg.arglist()
    out = {}
    for k, v in int_values.items():
        desc = arglist.get(k)
        if desc is None or isinstance(desc, Scalar):
            out[k] = v
        else:
            out[k] = np.array([v], dtype=np.int32)
    return out


def xfail(reason: str, *, strict: bool = True):
    """Uniform strict-xfail marker -- a silent xpass fires so flipped-green tests get
    visibly un-marked."""
    return pytest.mark.xfail(strict=strict, reason=reason)
