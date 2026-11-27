"""Shared helpers for verbatim ports from ``f2dace/dev:tests/fortran/`` -- reference build, arg routing, strict-xfail marker."""

import importlib.util
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from tests import _cache

#: Retry only the f2py reference build: under heavy parallel load a fork/exec ENOMEM
#: can transiently fail gfortran, unrelated to the source.  Never retry the SDFG build.
_F2PY_BUILD_ATTEMPTS = 3

#: An rc=0 f2py build can still miss a routine (crackfortran drop under swap-thrash),
#: surfacing later as a cryptic AttributeError.  Detect at import and rebuild under a
#: fresh extension name (re-importing a same-named CPython ext SIGABRTs at teardown).
_F2PY_IMPORT_ATTEMPTS = 3

#: Environment variables whose value changes what f2py/meson builds; part of the reference cache key.
_F2PY_ENV = ("FFLAGS", "FCFLAGS", "F90FLAGS", "CFLAGS", "LDFLAGS", "CC", "FC")


def f2py_build_with_retry(cmd, *, cwd, mod_name, env=None):
    """Retry ``f2py -c`` on transient fork/exec ENOMEM (not a source error); raises
    ``RuntimeError`` on a genuine compile failure.  Reference-build only -- never the
    SDFG build.  Shared retry policy with ``f2py_compile`` in ``_util``."""
    cwd = Path(cwd)
    for attempt in range(1, _F2PY_BUILD_ATTEMPTS + 1):
        proc = subprocess.run(cmd, check=False, cwd=cwd, env=env, capture_output=True, text=True)
        if proc.returncode == 0:
            return
        if attempt == _F2PY_BUILD_ATTEMPTS:
            raise RuntimeError(
                f"f2py reference build for {mod_name!r} failed after "
                f"{_F2PY_BUILD_ATTEMPTS} attempts (rc={proc.returncode}).\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
            )
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


def import_extension(out_dir: Path, name: str) -> ModuleType:
    """Load the extension module ``name`` built into ``out_dir`` from its file, leaving ``sys.path`` alone: a
    search-path entry would let every later import of a same-named module in this process resolve to this test's
    build."""
    filename = f"{name}{sysconfig.get_config_var('EXT_SUFFIX')}"
    # meson usually drops the extension in its cwd, occasionally only in its build directory.
    built = [out_dir / filename] if (out_dir / filename).is_file() else sorted(out_dir.rglob(filename))
    if not built:
        raise RuntimeError(f"f2py reported success but no {filename} exists under {out_dir}")
    spec = importlib.util.spec_from_file_location(name, built[0])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def f2py_build_and_import(src_files, *, out_dir, mod_name, only=None, extra_args=(), env=None):
    """Build ``src_files`` (one path or several) via ``numpy.f2py -c`` and import the module.  Hardened against two
    heavy-parallel-load failure modes: fork/exec ENOMEM (retried by :func:`f2py_build_with_retry`) and an rc=0 but
    incomplete module missing an ``only``-named routine (rebuilt under a fresh extension name).  Raises
    ``RuntimeError`` naming the missing routine rather than leaking ``AttributeError``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [src_files] if isinstance(src_files, (str, Path)) else list(src_files)
    base = [sys.executable, "-m", "numpy.f2py", "-c", *map(str, sources)]
    only_args = ["only:", *only, ":"] if only else []
    expected = tuple(only) if only else ()
    incomplete = None
    # The reference is a pure function of its source, flags and toolchain: reuse an earlier build of the same inputs.
    filename = f"{mod_name}{sysconfig.get_config_var('EXT_SUFFIX')}"
    key = _cache.digest(
        *(Path(s).read_bytes() for s in sources),
        repr((mod_name, tuple(extra_args), only_args, [(env or {}).get(v) for v in _F2PY_ENV])),
        sys.version,
        np.__version__,
        _cache.tool_version("gfortran"),
        _cache.tool_version("gcc"),
    )
    hit = _cache.entry("f2py", key)
    if hit is not None:
        shutil.copyfile(hit, out_dir / filename)
        mod = import_extension(out_dir, mod_name)
        if all(_f2py_routine_present(mod, s) for s in expected):
            return mod
        (out_dir / filename).unlink()
    for attempt in range(1, _F2PY_IMPORT_ATTEMPTS + 1):
        ext_name = mod_name if attempt == 1 else f"{mod_name}_r{attempt}"
        cmd = [*base, "-m", ext_name, "--quiet", *extra_args, *only_args]
        f2py_build_with_retry(cmd, cwd=out_dir, mod_name=ext_name, env=env)
        mod = import_extension(out_dir, ext_name)
        missing = [s for s in expected if not _f2py_routine_present(mod, s)]
        if not missing:
            if ext_name == mod_name:
                _cache.store("f2py", key, next(out_dir.rglob(filename)))
            return mod
        wrapped = sorted(n for n in vars(mod) if not n.startswith("_"))
        incomplete = (ext_name, missing, wrapped)
        for stale in out_dir.glob(f"{ext_name}*.so"):
            stale.unlink()
    raise RuntimeError(
        f"f2py reference {mod_name!r} built (rc=0) but is missing requested "
        f"routine(s) {incomplete[1]} after {_F2PY_IMPORT_ATTEMPTS} attempts; "
        f"wrapped names = {incomplete[2]}"
    )


def f2py(src_text: str, out_dir: Path, mod_name: str):
    """Compile ``src_text`` via ``numpy.f2py`` and return the imported module.  Fails
    if gfortran/meson missing; retries on transient resource-exhaustion (see
    ``_F2PY_BUILD_ATTEMPTS``)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    src_file = out_dir / f"{mod_name}.f90"
    src_file.write_text(src_text)
    return f2py_build_and_import(src_file, out_dir=out_dir, mod_name=mod_name)


def sdfg_call_args(sdfg, int_values: dict) -> dict:
    """Route each int in ``int_values`` to a plain int or length-1 int32 array per the
    SDFG's Scalar-vs-Array classification."""
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
