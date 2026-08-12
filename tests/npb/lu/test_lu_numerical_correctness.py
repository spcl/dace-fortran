"""NAS Parallel Benchmark LU -- end-to-end numerical correctness.

Companion to :mod:`test_lu_multi_file_build` / :mod:`test_lu_single_file_build`
(build-only); this one drives the SDFG to completion and compares the SSOR
solver's residual norms (``rsdnm(5)``) against a gfortran reference.

Class S params: ``nx0=ny0=nz0=12``, ``itmax=50``, ``dt=0.5``, ``omega=1.2``,
``tolrsd=1e-8`` -- small enough to run in under a second, large enough that all
five residual components move under the SSOR sweep.

``test_lu_reference_runs`` is a gfortran smoke check (``lu_caller.f90`` adds
BIND(C) entry points around the parameterless ``dolu()``).
``test_lu_numerical_correctness`` is the full element-wise DUT-vs-reference
compare; was xfail (~34% error) from a reused-scalar WAR/WAW hazard in the
``rhs`` viscous flux (``tmp = rho_i(i); ...; tmp = rho_i(i-1)`` collapsed onto
one DaCe scalar with no intra-state ordering).  Fixed by
``emit_cfg._scalar_reassign_in_state``; see ``tests/scalar_reuse_war_test.py``
for the minimal reproducer.
"""
import ctypes
import shutil
import subprocess
from pathlib import Path

import dace.data as dace_data
import numpy as np
import pytest

from _util import have_flang
from dace_fortran import build_sdfg_from_files

_HERE = Path(__file__).resolve().parent
_LU = _HERE / "lu.F90"
_USE = _HERE / "useapplu.F90"
_CALLER = _HERE / "lu_caller.f90"
_ENTRY = "useapplu::call_dolu"

# NPB Class S: 12^3 grid, 50 SSOR steps -- under a second, still exercises all five residual components.
_NX0 = _NY0 = _NZ0 = 12
_ITMAX = 50
_DT = 0.5
_OMEGA = 1.2
_TOLRSD = 1.0e-08

pytestmark = pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH")


def _compile_reference(tmp_path):
    """gfortran-compile the LU sources into a ``.so``; returns ``(init, run, get_rsdnm)`` ctypes-bound callables."""
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

    Belt-and-braces: pins the same CPU flags conftest.py sets at session start
    (-O0 -fno-fast-math -ffp-contract=off) explicitly, so a strict gfortran
    comparison can't silently drift if a future commit weakens the defaults.
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
    """Allocate zero-init kwargs from ``sdfg.arglist()``, seed the NPB Class S
    config scalars, call the SDFG, return the realised ``rsdnm`` buffer."""
    kw = {}
    for name, desc in sdfg.arglist().items():
        np_dtype = np.float64 if desc.dtype.as_numpy_dtype() == np.float64 else np.int32
        # Most module-level scalars surface as (1,)-Arrays, but the ones that are also free
        # symbols (itmax, inorm) surface as true Scalars -- handing those an array raises
        # "Passing an array to a scalar". Branch on the descriptor rather than on a name list,
        # so a future symbol promotion does not silently reintroduce the mismatch.
        if isinstance(desc, dace_data.Scalar):
            kw[name] = np_dtype(0)
        else:
            kw[name] = np.zeros(tuple(int(s) for s in desc.shape), dtype=np_dtype, order='F')

    def seed(name, value):
        if name not in kw:
            return
        if isinstance(kw[name], np.ndarray):
            kw[name][...] = value
        else:
            kw[name] = type(kw[name])(value)

    for name, value in (('nx0', _NX0), ('ny0', _NY0), ('nz0', _NZ0), ('itmax', _ITMAX), ('omega', _OMEGA), ('dt', _DT),
                        ('inorm', _ITMAX), ('tolrsd', _TOLRSD)):
        seed(name, value)
    sdfg(**kw)
    return kw['rsdnm']


def test_lu_reference_runs(tmp_path):
    """gfortran reference compiles, initialises, and runs to completion; all
    five residual norms finite and strictly positive (a no-op solver would leave BSS-zero)."""
    rsdnm = _run_reference(tmp_path)
    assert np.all(np.isfinite(rsdnm)), f"rsdnm has non-finite entries: {rsdnm}"
    assert np.all(rsdnm > 0.0), f"rsdnm has non-positive entries: {rsdnm}"
    # Sanity-pin the magnitude (wide band) to catch a silent SSOR solver change.
    # Reference is bit-exact across machines (-fno-fast-math/-ffp-contract=off at
    # compile time); band only needs to absorb NPB parameter-initialiser kind-promotion noise.
    expected = np.array([1.62e-2, 2.20e-3, 1.52e-3, 1.50e-3, 3.43e-2])
    np.testing.assert_allclose(rsdnm, expected, rtol=1e-2, err_msg=f"rsdnm drifted: got {rsdnm}")


def test_lu_numerical_correctness(tmp_path):
    """End-to-end gfortran-reference vs SDFG element-wise ``rsdnm`` match, tight tolerance.

    Was xfail (~34% error) from the WAR/WAW scalar-reuse hazard described in the
    module docstring; now bit-exact at every itmax.
    """
    rsdnm_ref = _run_reference(tmp_path)
    sdfg = _build_sdfg(tmp_path)
    rsdnm_sdfg = _run_sdfg(sdfg)
    np.testing.assert_allclose(rsdnm_sdfg, rsdnm_ref, rtol=1e-10, atol=1e-12)
