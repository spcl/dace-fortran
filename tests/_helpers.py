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
            raise RuntimeError(
                f"f2py reference build for {mod_name!r} failed after "
                f"{_F2PY_BUILD_ATTEMPTS} attempts (rc={proc.returncode}).\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
        # Drop any half-written extension and back off so the resource
        # spike (swap thrash / fork ENOMEM) can clear before retrying.
        for stale in cwd.glob(f"{mod_name}*.so"):
            stale.unlink()
        time.sleep(2 * attempt)


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
    cmd = [sys.executable, "-m", "numpy.f2py", "-c",
           str(src_file), "-m", mod_name, "--quiet"]
    f2py_build_with_retry(cmd, cwd=out_dir, mod_name=mod_name)
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    __import__(mod_name)
    return sys.modules[mod_name]


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
