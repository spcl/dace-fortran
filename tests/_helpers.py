"""Shared helpers for verbatim ports from ``f2dace/dev:tests/fortran/``.

These mirror the helpers used in the FaCe-native ports
(``baseline_*_test.py``) so a single canonical
implementation is reused across every ported file:

- ``_f2py(src, out_dir, mod_name)``    --  gfortran-backed reference build.
- ``_sdfg_call_args(sdfg, int_vals)``  --  route int args to scalar vs
  length-1-Array based on what ``sdfg.arglist()`` classifies them as.
- ``_xfail(reason, *, strict=True)``   --  uniform strict-xfail marker.
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

#: The ``f2py`` reference build shells out to gfortran/meson/ninja.  Under
#: heavy parallel load -- xdist ``-n`` workers, and in practice a second
#: pytest session running concurrently -- on a memory-constrained host the
#: kernel swaps hard and a child ``fork``/``exec`` for the compiler can
#: transiently fail (``ENOMEM``), surfacing as a ``CalledProcessError`` that
#: has nothing to do with the Fortran source.  Retry the *reference* build a
#: few times: a genuine, reproducible compile error still fails every attempt
#: and surfaces, while a one-off resource hiccup no longer masquerades as a
#: kernel regression.  Only the deterministic reference build is retried --
#: never the SDFG build or the numerical comparison.
_F2PY_BUILD_ATTEMPTS = 3

#: A *successful* (rc=0) f2py build can still emit an extension that is
#: missing the requested routine: under the same swap-thrash pressure that
#: causes the ENOMEM fork failures above, crackfortran can drop a routine
#: (in practice the most complex signature -- e.g. a ``DIMENSION(n, lev+1)``
#: dummy) yet f2py still exits 0 with an importable, *incomplete* module.
#: That surfaces much later as a cryptic ``AttributeError: module '<m>' has
#: no attribute '<sub>'`` in the calling test -- a flake indistinguishable
#: from a real bug.  Detect it at import time and rebuild under a FRESH
#: extension name (re-importing a same-named CPython extension in one
#: process is the f2py-teardown SIGABRT hazard) within this budget.
_F2PY_IMPORT_ATTEMPTS = 3


def f2py_build_with_retry(cmd, *, cwd, mod_name, env=None):
    """Run an ``f2py -c`` build *command* with bounded retry on transient
    resource-exhaustion (a ``fork``/``exec`` ENOMEM under swap thrash, not a
    source error).  Drops any half-written extension and backs off between
    attempts; a genuine, reproducible compile error still fails every attempt
    and raises ``RuntimeError`` with the full compiler diagnostic.  Safe ONLY
    for the deterministic *reference* build -- never the SDFG build or the
    numerical comparison (retrying those would mask real regressions).

    Single source of truth for the retry policy: shared by ``f2py`` here and
    ``f2py_compile`` in ``_util`` so the two reference-build paths can never
    drift (the un-hardened second path was the cause of a safety-net flake)."""
    cwd = Path(cwd)
    for attempt in range(1, _F2PY_BUILD_ATTEMPTS + 1):
        proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
        if proc.returncode == 0:
            return
        if attempt == _F2PY_BUILD_ATTEMPTS:
            raise RuntimeError(f"f2py reference build for {mod_name!r} failed after "
                               f"{_F2PY_BUILD_ATTEMPTS} attempts (rc={proc.returncode}).\n"
                               f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
        # Drop any half-written extension and back off so the resource
        # spike (swap thrash / fork ENOMEM) can clear before retrying.
        for stale in cwd.glob(f"{mod_name}*.so"):
            stale.unlink()
        time.sleep(2 * attempt)


def _f2py_routine_present(mod, name: str) -> bool:
    """True if ``name`` is wrapped on the imported f2py module.  A *free*
    subroutine lands as a top-level attribute (``mod.name``); a *module*
    subroutine nests one level under its Fortran module's namespace object
    (``mod.<fmodule>.name``), so check both."""
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
    """Build ``src_file`` via ``numpy.f2py -c`` and import the resulting
    extension, returning the imported module.

    Hardens the reference build against two distinct heavy-parallel-load
    failure modes, neither of which is a real source error:

    * transient ``fork``/``exec`` ENOMEM (rc != 0) -- handled per-build by
      :func:`f2py_build_with_retry`; and
    * an rc=0 but *incomplete* module that is missing the routine(s) named
      in ``only`` (see ``_F2PY_IMPORT_ATTEMPTS``) -- detected here and
      rebuilt under a fresh extension name.

    ``only`` is the tuple of routine names f2py should wrap (and that this
    function then verifies are present on the imported module); ``extra_args``
    are invariant f2py flags (e.g. ``--f90flags=...``) carried on every
    attempt.  A genuinely un-wrappable routine fails every attempt and raises
    a clear ``RuntimeError`` naming the missing routine and what *was*
    wrapped, instead of leaking an ``AttributeError`` into the caller.
    """
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
    """Compile ``src_text`` as a Python extension via ``numpy.f2py`` and
    return the imported module.  Skips the test if gfortran or meson is
    not installed.  The reference build is retried on transient
    resource-exhaustion failures (see ``_F2PY_BUILD_ATTEMPTS``)."""
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not available")
    if shutil.which("meson") is None:
        pytest.skip("meson not available (f2py backend on Python>=3.12)")
    out_dir.mkdir(parents=True, exist_ok=True)
    src_file = out_dir / f"{mod_name}.f90"
    src_file.write_text(src_text)
    return f2py_build_and_import(src_file, out_dir=out_dir, mod_name=mod_name)


def sdfg_call_args(sdfg, int_values: dict) -> dict:
    """Route each integer arg in ``int_values`` to either a plain int or
    a length-1 numpy int32 array, depending on whether the SDFG
    descriptor classifies it as a Scalar/symbol or a length-1 Array.
    Mirrors the helper in ``icon/selected_loopnests/test_sdfg_equivalence.py``.
    """
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
    """Uniform strict-xfail marker  --  any silent xpass should fire so
    flipped-green tests get a deliberate, visible un-marking."""
    return pytest.mark.xfail(strict=strict, reason=reason)
