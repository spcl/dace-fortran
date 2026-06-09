"""NAS Parallel Benchmark LU -- end-to-end numerical correctness.

Companion to :mod:`test_lu_multi_file_build` (build-only) and
:mod:`test_lu_single_file_build` (build-only).  Where those validate
that the bridge produces an SDFG with at least one LU kernel
referenced, this one drives the SDFG to completion and compares the
SSOR solver's residual norms (``rsdnm(5)``) against a gfortran
reference compiled from the same source.

NPB LU configuration ('Class S' parameters)
-------------------------------------------

``nx0=ny0=nz0=12``, ``itmax=50``, ``dt=0.5``, ``omega=1.2``,
``tolrsd=1e-8``.  Small enough that the full solver runs in under a
second on each side, large enough that all five residual components
are non-trivially affected by the SSOR sweep.

Test layout
-----------

``test_lu_reference_runs`` -- gfortran reference smoke check.  The
upstream ``dolu()`` entry is parameterless and writes its result to
LU's module-level state; ``lu_caller.f90`` adds three BIND(C) entry
points (``init_lu_c`` / ``run_dolu_c`` / ``get_rsdnm_c``) so a
ctypes-loaded ``.so`` can configure the solver, run it, and read
``rsdnm`` back.  After 50 SSOR steps on the 12^3 grid the expected
norms are about ``[1.6e-2, 2.2e-3, 1.5e-3, 1.5e-3, 3.4e-2]`` -- the
test asserts all five are finite and strictly positive (the gfortran
build / link / config-init path is the gate; the exact values just
pin that the solver did something).

``test_lu_numerical_correctness`` -- full reference-vs-SDFG element-
wise comparison.  Currently ``xfail-strict-false`` until the bridge
closes two gaps surfaced by exercising the SDFG run:

* ``dt`` (the SSOR time step) is read inside ``ssor`` but does not
  appear in the SDFG's ``arglist`` / ``symbols`` / ``arrays``.  Some
  upstream simplification drops it before the bindings layer sees
  it, so the SDFG behaves as if ``dt=0`` and the SSOR sweep is a
  no-op.
* Inner-subroutine iteration variables (``blts_i`` / ``buts_i`` /
  ``jacld_j`` / ``jacu_i`` / ``ssor_k`` / ...) leak into the
  top-level SDFG's ``free_symbols``.  They should be local to their
  containing NestedSDFGs.

Until both are fixed the SDFG-side ``rsdnm`` diverges from the
reference by ~6 orders of magnitude; the comparison stays xfail.
The reference-side test pins the harness so when those bridge fixes
land the comparison auto-flips.
"""
import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from _util import have_flang
from dace_fortran import build_sdfg_from_files

_HERE = Path(__file__).resolve().parent
_LU = _HERE / "lu.F90"
_USE = _HERE / "useapplu.F90"
_CALLER = _HERE / "lu_caller.f90"
_ENTRY = "_QMuseappluPcall_dolu"

# NPB Class S (the smallest class) -- 12^3 grid, 50 SSOR steps.  Picked
# so the full solver runs in under a second on each side; large enough
# that all five residual components see non-trivial changes through the
# SSOR sweep.
_NX0 = _NY0 = _NZ0 = 12
_ITMAX = 50
_DT = 0.5
_OMEGA = 1.2
_TOLRSD = 1.0e-08

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _compile_reference(tmp_path):
    """gfortran-compile ``lu.F90`` + ``useapplu.F90`` + ``lu_caller.f90``
    into a ``.so`` and return ready-to-call ctypes bindings.

    :returns: ``(init, run, get_rsdnm)`` -- three callables bound with
        ``argtypes`` / ``restype`` for the BIND(C) wrappers.
    """
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran required for the reference build")

    libpath = tmp_path / "liblu_ref.so"
    subprocess.check_call([
        "gfortran",
        "-shared",
        "-fPIC",
        "-O0",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-ffree-line-length-none",
        str(_LU),
        str(_USE),
        str(_CALLER),
        "-o",
        str(libpath),
    ],
                          cwd=str(tmp_path))
    lib = ctypes.CDLL(str(libpath))

    init = lib.init_lu_c
    init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
    init.restype = None

    run = lib.run_dolu_c
    run.argtypes = []
    run.restype = None

    get_rsdnm = lib.get_rsdnm_c
    get_rsdnm.argtypes = [ctypes.c_void_p]
    get_rsdnm.restype = None

    return init, run, get_rsdnm


def _run_reference(tmp_path):
    """Compile + init + run the gfortran reference and read ``rsdnm``."""
    init, run, get_rsdnm = _compile_reference(tmp_path)
    init(_NX0, _NY0, _NZ0, _ITMAX, _DT)
    run()
    rsdnm = np.zeros(5, dtype=np.float64, order="F")
    get_rsdnm(rsdnm.ctypes.data)
    return rsdnm


def _build_sdfg(tmp_path):
    """Build the LU SDFG via the multi-file bridge entry point.

    Belt-and-braces: explicitly pin the CPU compile flags here even
    though ``tests/conftest.py`` sets the same defaults at session
    start.  E2E numerical correctness tests that compare against a
    strict gfortran reference (``-O0 -fno-fast-math
    -ffp-contract=off``) must use the same flags on the SDFG side to
    avoid spurious FMA / reassociation drift; making the dependence
    explicit in the test guards against silent behaviour changes if
    a future commit weakens or removes the conftest defaults.
    """
    import dace
    dace.Config.set("compiler",
                    "cpu",
                    "args",
                    value=("-fPIC -Wall -Wextra -O0 -fno-fast-math -ffp-contract=off "
                           "-Wno-unused-parameter -Wno-unused-label"))
    return build_sdfg_from_files(
        [_LU, _USE],
        entry=_ENTRY,
        name="npb_lu",
        out_dir=tmp_path / "build",
    )


def _run_sdfg(sdfg):
    """Allocate zero-initialised kwargs from ``sdfg.arglist()``, seed
    the NPB Class S configuration scalars, call the SDFG, return the
    realised ``rsdnm`` buffer.
    """
    kw = {}
    for name, desc in sdfg.arglist().items():
        shape = tuple(int(s) for s in desc.shape)
        is_float = desc.dtype.as_numpy_dtype() == np.float64
        kw[name] = np.zeros(shape, dtype=np.float64 if is_float else np.int32, order='F')
    # Seed the NPB Class S configuration scalars (each is a (1,)-Array
    # on the SDFG side -- the bridge surfaces module-level scalars as
    # one-element arrays).  ``inorm`` is a free symbol so the bridge
    # resolves it via the kwarg name.
    for name, value in (('nx0', _NX0), ('ny0', _NY0), ('nz0', _NZ0), ('itmax', _ITMAX),
                        ('omega', _OMEGA), ('dt', _DT)):
        if name in kw:
            kw[name][...] = value
    if 'tolrsd' in kw:
        kw['tolrsd'][...] = _TOLRSD
    if 'inorm' in sdfg.symbols:
        kw['inorm'] = np.int32(_ITMAX)
    sdfg(**kw)
    return kw['rsdnm']


@pytest.mark.long
def test_lu_reference_runs(tmp_path):
    """gfortran reference compiles, initialises, and runs to completion.

    After 50 SSOR steps on a 12^3 grid the five residual norms should
    all be finite and strictly positive (a no-op solver would leave
    them at the BSS-zero default).  Exact values:
    ``rsdnm ~= [1.6e-2, 2.2e-3, 1.5e-3, 1.5e-3, 3.4e-2]``.
    """
    rsdnm = _run_reference(tmp_path)
    assert np.all(np.isfinite(rsdnm)), f"rsdnm has non-finite entries: {rsdnm}"
    assert np.all(rsdnm > 0.0), f"rsdnm has non-positive entries: {rsdnm}"
    # Sanity-pin the magnitude (within a wide band) so an upstream
    # silent change of the SSOR solver is caught.  Reference run is
    # bit-exact on every machine we test -- ``-fno-fast-math`` and
    # ``-ffp-contract=off`` are passed at compile time -- so the band
    # only has to be wide enough to absorb the integer-vs-float kind
    # promotion noise in the NPB ``parameter`` initialisers.
    expected = np.array([1.62e-2, 2.20e-3, 1.52e-3, 1.50e-3, 3.43e-2])
    np.testing.assert_allclose(rsdnm, expected, rtol=1e-2, err_msg=f"rsdnm drifted: got {rsdnm}")


@pytest.mark.long
@pytest.mark.xfail(strict=False,
                   reason=("After the 7-commit dt-flow chain (a724cf0 -> dc35458) the "
                           "structural gaps are closed: dt appears in the SDFG arglist as "
                           "a non-transient (1,)-Array, 266 tasklets read it via "
                           "``_in_dt`` connectors, the d/a/b/c matrices receive jacld's "
                           "writes (no longer dropped by walkSCFBeforeRegion), and the "
                           "free_symbols leaks closed.  The remaining divergence -- "
                           "rsdnm ~ 1e5 vs reference ~1e-2, ~6 orders -- is rooted in "
                           "the SSOR iteration loop being effectively a no-op: ``istep`` "
                           "has only 11 mentions in the SDFG JSON and ``itmax`` only 7, "
                           "vs the >100 expected for a 50-iter loop that reads/writes "
                           "rsd / d-matrix per step.  The do istep loop is likely "
                           "collapsed, peeled to a single iteration, or its iterations "
                           "share state with each other in a way that effectively makes "
                           "later iterations no-ops.  Likely root: the bridge's "
                           "interstate-edge sequencing across the do istep loop body's "
                           "many call sites (rhs/l2norm/jacld+blts/jacu+buts/u-update) "
                           "doesn't carry the updated u/rsd values forward.\n\nWAS: \n"
                           "(1) ``dt`` is "
                           "read inside ``ssor`` but does not appear in the SDFG's "
                           "arglist / symbols / arrays (some upstream simplification "
                           "drops it before bindings see it), so the SDFG behaves as if "
                           "``dt=0`` and the SSOR sweep is a no-op.  (2) Inner-subroutine "
                           "iteration variables (``blts_i`` / ``buts_i`` / ``jacld_j`` / "
                           "``jacu_i`` / ``ssor_k`` / ...) leak into the top-level SDFG's "
                           "free_symbols when they should be local to their containing "
                           "NestedSDFGs.  Until both close the SDFG-side rsdnm diverges "
                           "from the reference by ~6 orders of magnitude.  The reference "
                           "side is pinned by ``test_lu_reference_runs``; when these fixes "
                           "land the comparison auto-flips."))
def test_lu_numerical_correctness(tmp_path):
    """End-to-end gfortran-reference vs SDFG element-wise rsdnm match.

    Both sides run the NPB Class S configuration through ``dolu`` and
    compare the five SSOR residual norms with tight tolerance.  The
    test reads ``rsdnm`` from the SDFG kwargs dict (the bridge
    surfaces module-level state as Array kwargs) and from the
    gfortran reference's accessor wrapper (``get_rsdnm_c``).
    """
    rsdnm_ref = _run_reference(tmp_path)
    sdfg = _build_sdfg(tmp_path)
    rsdnm_sdfg = _run_sdfg(sdfg)
    np.testing.assert_allclose(rsdnm_sdfg, rsdnm_ref, rtol=1e-10, atol=1e-12)
