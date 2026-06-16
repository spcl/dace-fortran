"""End-to-end numerical correctness for ICON's AES graupel scheme.

Compiles ``mo_aes_graupel + mo_aes_thermo + mo_kind +
mo_physical_constants`` as a gfortran reference (via the
``graupel_caller.f90`` wrapper that exposes C-bound init / run
entry points), then builds the SAME source through the bridge as
an SDFG, runs both with identical seeded random inputs, and
compares the prognostic + diagnostic outputs element-wise.

Pattern mirrors ``tests/icon/full/test_velocity_full.py``:

  * single source on both sides -> no spec divergence,
  * single init routine -> byte-identical input buffers,
  * raw-pointer reference call vs DaCe numpy-arg dispatch ->
    identical Fortran-side ABI from both runs.

Status: ``xfail(strict=False)`` -- the SDFG build itself currently
hits ``InvalidSDFGEdgeError`` during validation (memlet
dimensionality mismatch downstream of the graupel multi-file
inline).  This test scaffolds the harness so when the validation
gap closes the test auto-flips to passing; until then the failure
points at the validation error rather than silently absent.

Random-input determinism:  ``init_graupel_inputs_c`` is a small
Fortran routine in ``graupel_caller.f90`` that runs a Mulberry32-
style scramble keyed off the integer seed.  Same routine seeds
both the reference and SDFG sides, so byte-identical buffers feed
both kernels.
"""
import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest

from _util import have_flang
from dace_fortran import build_sdfg_from_files

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_HERE = Path(__file__).resolve().parent
_AES = _HERE / "aes_graupel"

_GRAUPEL_SOURCES = [
    _AES / "mo_aes_graupel.f90",
    _AES / "mo_aes_thermo.f90",
    _AES / "mo_kind.f90",
    _AES / "mo_physical_constants.f90",
]

_CALLER = _HERE / "graupel_caller.f90"

_ENTRY = "graupel_run"


def _compile_reference(out_dir: Path) -> ctypes.CDLL:
    """Build the multi-file gfortran reference (graupel + 3 helpers +
    the ``BIND(C)`` caller wrapper) into a single ``.so`` and return
    the ctypes handle.  Sources are compiled to objects in dependency
    order (``mo_kind`` -> ``mo_physical_constants`` -> ``mo_aes_thermo``
    -> ``mo_aes_graupel`` -> ``graupel_caller``) so each USE statement
    finds the ``.mod`` file produced by an earlier step.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    so_path = out_dir / "libgraupel_ref.so"
    flags = ["-O0", "-fno-fast-math", "-ffp-contract=off", "-fPIC"]
    # Dependency-ordered list (leaf modules first).
    ordered = [
        _AES / "mo_kind.f90",
        _AES / "mo_physical_constants.f90",
        _AES / "mo_aes_thermo.f90",
        _AES / "mo_aes_graupel.f90",
        _CALLER,
    ]
    objects = []
    for src in ordered:
        obj = out_dir / (src.stem + ".o")
        # Use ``cwd=out_dir`` so gfortran writes ``.mod`` files there
        # and finds them on subsequent steps.  Avoid ``-J`` because
        # other tests may leave a flang-compiled
        # ``iso_c_binding.mod`` in shared paths (TMPDIR etc.) that
        # gfortran rejects with "not a GNU Fortran module file".
        subprocess.run(["gfortran", *flags, "-c", str(src), "-o", str(obj)], check=True, cwd=str(out_dir))
        objects.append(str(obj))
    subprocess.run(["gfortran", "-shared", "-fPIC", "-o", str(so_path), *objects], check=True, cwd=str(out_dir))
    return ctypes.CDLL(str(so_path))


@pytest.mark.long
def test_aes_graupel_e2e_numerical(tmp_path):
    """``graupel_run`` reference vs SDFG: element-wise compare of
    every INOUT prognostic + every OUT diagnostic for seeded random
    inputs."""
    ivec, k_v = 4, 8
    ivs, ive, ks = 1, ivec, 1
    dt = 30.0
    seed = 42

    lib = _compile_reference(tmp_path / "ref")
    init = lib.init_graupel_inputs_c
    init.restype = None
    init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, *([ctypes.c_void_p] * 10), ctypes.c_void_p]
    run = lib.run_graupel_c
    run.restype = None
    # 11 inputs (dz, t, p, rho, qv, qc, qi, qr, qs, qg, qnc) + 6 outputs
    # (prr_gsp, pri_gsp, prs_gsp, prg_gsp, pflx, pre_gsp) = 17 array args.
    # The previous ``* 16`` was off-by-one and the missing trailing arg
    # corrupted the stack, surfacing as a SIGSEGV inside the gfortran
    # reference before the kernel body even ran.
    run.argtypes = [ctypes.c_int] * 5 + [ctypes.c_double] + [ctypes.c_void_p] * 17

    # Reference buffers.
    f64_2d = lambda: np.zeros((ivec, k_v), dtype=np.float64, order='F')
    f64_1d = lambda: np.zeros((ivec, ), dtype=np.float64, order='F')

    bufs_ref = {
        'dz': f64_2d(),
        't': f64_2d(),
        'p': f64_2d(),
        'rho': f64_2d(),
        'qv': f64_2d(),
        'qc': f64_2d(),
        'qi': f64_2d(),
        'qr': f64_2d(),
        'qs': f64_2d(),
        'qg': f64_2d(),
        'qnc': f64_1d(),
    }

    # Seed both sides from the same source-of-truth.
    init(ctypes.c_int(seed), ctypes.c_int(ivec), ctypes.c_int(k_v),
         *[bufs_ref[k].ctypes.data for k in ('dz', 't', 'p', 'rho', 'qv', 'qc', 'qi', 'qr', 'qs', 'qg', 'qnc')])

    # Outputs.
    pflx_ref = f64_2d()
    prr_ref = f64_1d()
    pri_ref = f64_1d()
    prs_ref = f64_1d()
    prg_ref = f64_1d()
    pre_ref = f64_1d()

    run(ctypes.c_int(ivec), ctypes.c_int(k_v), ctypes.c_int(ivs), ctypes.c_int(ive), ctypes.c_int(ks),
        ctypes.c_double(dt), bufs_ref['dz'].ctypes.data, bufs_ref['t'].ctypes.data, bufs_ref['p'].ctypes.data,
        bufs_ref['rho'].ctypes.data, bufs_ref['qv'].ctypes.data, bufs_ref['qc'].ctypes.data, bufs_ref['qi'].ctypes.data,
        bufs_ref['qr'].ctypes.data, bufs_ref['qs'].ctypes.data, bufs_ref['qg'].ctypes.data, bufs_ref['qnc'].ctypes.data,
        prr_ref.ctypes.data, pri_ref.ctypes.data, prs_ref.ctypes.data, prg_ref.ctypes.data, pflx_ref.ctypes.data,
        pre_ref.ctypes.data)

    # SDFG side.  XFAILs at build until the validation gap closes.
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg_from_files(_GRAUPEL_SOURCES, entry=_ENTRY, name="graupel_run", out_dir=sdfg_dir / "build")

    # Re-seed identical inputs for the SDFG side (Fortran INOUT
    # arrays were mutated by the reference call).
    bufs_sdfg = {k: np.zeros_like(v) for k, v in bufs_ref.items()}
    init(ctypes.c_int(seed), ctypes.c_int(ivec), ctypes.c_int(k_v),
         *[bufs_sdfg[k].ctypes.data for k in ('dz', 't', 'p', 'rho', 'qv', 'qc', 'qi', 'qr', 'qs', 'qg', 'qnc')])

    pflx_sdfg = f64_2d()
    prr_sdfg = f64_1d()
    pri_sdfg = f64_1d()
    prs_sdfg = f64_1d()
    prg_sdfg = f64_1d()
    pre_sdfg = f64_1d()

    sdfg(nvec=np.int32(ivec),
         ke=np.int32(k_v),
         ivstart=np.int32(ivs),
         ivend=np.int32(ive),
         kstart=np.int32(ks),
         dt=np.float64(dt),
         dz=bufs_sdfg['dz'],
         t=bufs_sdfg['t'],
         p=bufs_sdfg['p'],
         rho=bufs_sdfg['rho'],
         qv=bufs_sdfg['qv'],
         qc=bufs_sdfg['qc'],
         qi=bufs_sdfg['qi'],
         qr=bufs_sdfg['qr'],
         qs=bufs_sdfg['qs'],
         qg=bufs_sdfg['qg'],
         qnc=bufs_sdfg['qnc'],
         prr_gsp=prr_sdfg,
         pri_gsp=pri_sdfg,
         prs_gsp=prs_sdfg,
         prg_gsp=prg_sdfg,
         pflx=pflx_sdfg,
         pre_gsp=pre_sdfg)

    # Prognostics (INOUT) -- compare post-step values.
    for nm in ('t', 'qv', 'qc', 'qi', 'qr', 'qs', 'qg'):
        np.testing.assert_allclose(bufs_sdfg[nm],
                                   bufs_ref[nm],
                                   rtol=1e-10,
                                   atol=1e-12,
                                   err_msg=f"prognostic {nm} drifted")
    # Diagnostics (OUT).
    np.testing.assert_allclose(pflx_sdfg, pflx_ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(prr_sdfg, prr_ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(pri_sdfg, pri_ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(prs_sdfg, prs_ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(prg_sdfg, prg_ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(pre_sdfg, pre_ref, rtol=1e-10, atol=1e-12)
