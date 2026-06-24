# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Inliner + end-to-end numerical-correctness tests over the vendored Fortran
LULESH (``tests/lulesh/``; GPL-v3 third-party fixture -- see ``NOTICE.md``).

Two things are exercised on real LULESH source:

* **Whole-program faithfulness, both backends.** The fparser inliner
  (:func:`inline_to_ast`, ``optimize=False``) and the regex merge engine
  (:func:`merge_used_modules`) each fold the two-file project
  (``lulesh.f90`` driver + ``lulesh_comp_kernels.f90`` module) into one
  self-contained TU that keeps the ``PROGRAM`` and the kernels.  The driver
  calls the g77 extension ``rand``/``srand``; the inliner recognises those as
  external builtins (see ``cleanup.EXTERNAL_BUILTIN_PROCEDURES``) instead of
  asserting, so the program survives the merge with the calls intact.

* **e2e numerical correctness, both merge engines.** The pure
  ``CalcElemVolumeDerivative`` kernel (which calls ``VoluDer`` eight times) is
  pruned + inlined out of the 3.3k-line kernels module, built into an SDFG via
  both merge engines, and checked element-wise against a gfortran reference on
  Park-Miller-seeded inputs (the LULESH ``rand()`` ``minstd`` LCG).
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from dace_fortran.fparser_inliner import inline_to_ast
from dace_fortran.preprocess import merge_used_modules

_HERE = Path(__file__).parent
_KERNELS = _HERE / "lulesh_comp_kernels.f90"
_DRIVER = _HERE / "lulesh.f90"
_ENTRY = "calcelemvolumederivative"


def _have(tool: str) -> bool:
    return shutil.which(tool) is not None


def _park_miller(seed: int, n: int) -> np.ndarray:
    """LULESH-faithful pseudo-random reals in ``[0, 1)``.

    This is exactly the generator gfortran's ``rand()``/``irand()`` use -- the
    Park-Miller "minimal standard" LCG ``seed = 16807*seed mod (2**31 - 1)``
    (verified bit-exact against gfortran for the integer sequence).  Using it
    here means the e2e inputs are reproducible and ``rand``-faithful without
    depending on a compiler-specific intrinsic.
    """
    s = seed & 0x7FFFFFFF
    out = np.empty(n)
    for i in range(n):
        s = (16807 * s) % 2147483647
        out[i] = s / 2147483646.0
    return out


# --------------------------------------------------------------------------- #
# Whole-program inliner faithfulness (both backends)
# --------------------------------------------------------------------------- #
def test_regex_backend_merges_whole_lulesh():
    """The regex merge engine folds driver + kernels module into one TU."""
    merged = merge_used_modules(_DRIVER.read_text(), search_dirs=[_HERE])
    lo = merged.lower()
    assert "program lulesh" in lo
    assert "module lulesh_comp_kernels" in lo
    # The USE-d module must be placed before the program that depends on it.
    assert lo.index("module lulesh_comp_kernels") < lo.index("program lulesh")
    for kernel in ("calcelemvolumederivative", "voluder", "integratestressforelems"):
        assert kernel in lo, kernel
    # The g77 `rand`/`srand` extension calls pass through the textual merge.
    assert "rand(" in lo and "srand" in lo


def test_fparser_backend_inlines_whole_lulesh():
    """The fparser inliner merges the two-file project and -- thanks to the
    ``rand`` builtin recognition -- keeps the ``PROGRAM`` that calls it."""
    merged = inline_to_ast([_KERNELS, _DRIVER], None, optimize=False, expand_cpp=True).tofortran()
    lo = merged.lower()
    assert "program lulesh" in lo
    for kernel in ("calcelemvolumederivative", "voluder"):
        assert kernel in lo, kernel
    # `rand()` was reclassified as an external call (no AssertionError) and kept.
    assert "rand(" in lo


def test_fparser_recognises_rand_family():
    """Focused guard for ``cleanup.EXTERNAL_BUILTIN_PROCEDURES``: a kernel that
    seeds and draws from the g77 PRNG inlines without raising 'cannot find
    rand'."""
    src = """
subroutine seedy(n, out)
  implicit none
  integer, intent(in) :: n
  integer, intent(out) :: out(n)
  integer :: i
  call srand(0)
  do i = 1, n
    out(i) = mod(irand(), 100)
  end do
end subroutine seedy
"""
    merged = inline_to_ast({"seedy.f90": src}, "seedy", optimize=False, include_builtins=False).tofortran().lower()
    assert "srand" in merged
    assert "irand" in merged


# --------------------------------------------------------------------------- #
# e2e numerical correctness on a real LULESH kernel (both merge engines)
# --------------------------------------------------------------------------- #
def _prune_kernel() -> str:
    """Prune + inline ``CalcElemVolumeDerivative`` (+ ``VoluDer``) out of the
    full kernels module into a small self-contained TU."""
    return inline_to_ast([_KERNELS], _ENTRY, optimize=False, expand_cpp=True).tofortran()


def _f2py_reference(tu_text: str, out_dir: Path, mod: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{mod}.f90").write_text(tu_text)
    subprocess.check_call(
        [
            sys.executable, "-m", "numpy.f2py", "-c", f"{mod}.f90", "-m", mod, "--quiet",
            "--f90flags=-O0 -fno-fast-math -ffp-contract=off -ffree-line-length-none"
        ],
        cwd=out_dir,
    )
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    return __import__(mod)


@pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")
@pytest.mark.parametrize("engine", ["regex", "fparser"])
def test_calcelemvolumederivative_e2e(tmp_path: Path, engine: str):
    """SDFG of the inlined kernel matches the gfortran reference element-wise."""
    if not (_have("gfortran") and _have("meson")):
        pytest.skip("gfortran + meson needed for the f2py reference")

    import re
    merged = _prune_kernel()
    module_name = re.search(r"(?im)^\s*MODULE\s+(\w+)", merged).group(1).lower()

    # Park-Miller-seeded node coordinates in [-1, 1), identical on both sides.
    coords = _park_miller(2718281, 24) * 2.0 - 1.0
    x = np.asfortranarray(coords[0:8].copy())
    y = np.asfortranarray(coords[8:16].copy())
    z = np.asfortranarray(coords[16:24].copy())

    ref = getattr(_f2py_reference(merged, tmp_path / "ref", "lulref"), module_name)
    dvdx_r = np.zeros(8, order="F")
    dvdy_r = np.zeros(8, order="F")
    dvdz_r = np.zeros(8, order="F")
    ref.calcelemvolumederivative(dvdx_r, dvdy_r, dvdz_r, x, y, z)

    sdfg = build_sdfg(merged, tmp_path / f"sdfg_{engine}", name="cevd", entry=_ENTRY, merge_engine=engine).build()
    dvdx = np.zeros(8, order="F")
    dvdy = np.zeros(8, order="F")
    dvdz = np.zeros(8, order="F")
    sdfg(dvdx=dvdx, dvdy=dvdy, dvdz=dvdz, x=x.copy(order="F"), y=y.copy(order="F"), z=z.copy(order="F"))

    np.testing.assert_allclose(dvdx, dvdx_r, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dvdy, dvdy_r, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dvdz, dvdz_r, rtol=1e-12, atol=1e-12)
