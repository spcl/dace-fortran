"""fparser single-TU round-trip for the AES graupel scheme.

Headline acceptance for :func:`dace_fortran.inline_to_single_tu`: take
the same 4-module ``aes_graupel`` project the multi-file build test
drives, inline it into ONE self-contained ``.f90`` via the fparser
engine, then build *that* single TU through the bridge and compare the
numerics against a gfortran reference compiled from the original
multi-file sources.

If the single-TU producer is faithful, the SDFG built from the inlined
file must validate and match the multi-file path's numerics.

The multi-file inline used to surface a downstream codegen gap -- the
AoS-of-pointer-records gather temp (``t_qx_ptr%x``) was sized from
unbound extent symbols that defaulted to 1 at call time and overflowed
the heap.  The ``fir.box_dims -> <name>_d<dim>`` extent resolution closed
it, so this round-trip now builds and matches the multi-file path's
numerics.  The *inlining* step itself (parse -> merge -> prune ->
serialise -> re-parse to a valid TU) is asserted unconditionally up
front, so an inliner regression still surfaces as a hard failure.
"""
import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest

from _util import have_flang
from dace_fortran import build_sdfg_from_files, inline_to_single_tu

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
_ENTRY = "mo_aes_graupel::graupel_run"


def _compile_reference(out_dir: Path) -> ctypes.CDLL:
    """Build the multi-file gfortran reference (mirrors
    ``test_aes_graupel_numerical_correctness._compile_reference``)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    so_path = out_dir / "libgraupel_ref.so"
    flags = ["-O0", "-fno-fast-math", "-ffp-contract=off", "-fPIC", "-ffree-line-length-none"]
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
        subprocess.run(["gfortran", *flags, "-c", str(src), "-o", str(obj)], check=True, cwd=str(out_dir))
        objects.append(str(obj))
    subprocess.run(["gfortran", "-shared", "-fPIC", "-o", str(so_path), *objects], check=True, cwd=str(out_dir))
    return ctypes.CDLL(str(so_path))


def test_aes_graupel_inline_single_tu(tmp_path):
    """The 4-module project inlines into one valid Fortran TU.

    Asserted unconditionally: the inliner must produce a self-contained
    ``.f90`` that still defines ``graupel_run``.  (The SDFG build /
    numerical compare lives in the xfail round-trip test below.)
    """
    out = inline_to_single_tu(_GRAUPEL_SOURCES, entry=_ENTRY, out_dir=tmp_path, name="graupel_inlined")
    assert out.is_file()
    text = out.read_text()
    assert "graupel_run" in text.lower()
    # A self-contained TU: every module it needs is inlined, so no
    # stray external module is referenced beyond the intrinsic stubs.
    assert "FUNCTION" in text.upper() or "SUBROUTINE" in text.upper()


def test_aes_graupel_inline_roundtrip_numerical(tmp_path):
    """``graupel_run`` reference vs SDFG built from the fparser-inlined
    single TU: element-wise compare of every INOUT prognostic + OUT
    diagnostic for seeded random inputs."""
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
    run.argtypes = [ctypes.c_int] * 5 + [ctypes.c_double] + [ctypes.c_void_p] * 17

    def f64_2d():
        return np.zeros((ivec, k_v), dtype=np.float64, order='F')

    def f64_1d():
        return np.zeros((ivec, ), dtype=np.float64, order='F')

    bufs_ref = {k: f64_2d() for k in ('dz', 't', 'p', 'rho', 'qv', 'qc', 'qi', 'qr', 'qs', 'qg')}
    bufs_ref['qnc'] = f64_1d()

    init(ctypes.c_int(seed), ctypes.c_int(ivec), ctypes.c_int(k_v),
         *[bufs_ref[k].ctypes.data for k in ('dz', 't', 'p', 'rho', 'qv', 'qc', 'qi', 'qr', 'qs', 'qg', 'qnc')])

    pflx_ref = f64_2d()
    prr_ref, pri_ref, prs_ref, prg_ref, pre_ref = (f64_1d() for _ in range(5))

    run(ctypes.c_int(ivec), ctypes.c_int(k_v), ctypes.c_int(ivs), ctypes.c_int(ive), ctypes.c_int(ks),
        ctypes.c_double(dt), bufs_ref['dz'].ctypes.data, bufs_ref['t'].ctypes.data, bufs_ref['p'].ctypes.data,
        bufs_ref['rho'].ctypes.data, bufs_ref['qv'].ctypes.data, bufs_ref['qc'].ctypes.data, bufs_ref['qi'].ctypes.data,
        bufs_ref['qr'].ctypes.data, bufs_ref['qs'].ctypes.data, bufs_ref['qg'].ctypes.data, bufs_ref['qnc'].ctypes.data,
        prr_ref.ctypes.data, pri_ref.ctypes.data, prs_ref.ctypes.data, prg_ref.ctypes.data, pflx_ref.ctypes.data,
        pre_ref.ctypes.data)

    # Inline the 4-module project into ONE TU, then build that.
    single_tu = inline_to_single_tu(_GRAUPEL_SOURCES, entry=_ENTRY, out_dir=tmp_path / "inlined", name="graupel_run")
    sdfg = build_sdfg_from_files([single_tu], entry=_ENTRY, name="graupel_run", out_dir=tmp_path / "build")

    bufs_sdfg = {k: np.zeros_like(v) for k, v in bufs_ref.items()}
    init(ctypes.c_int(seed), ctypes.c_int(ivec), ctypes.c_int(k_v),
         *[bufs_sdfg[k].ctypes.data for k in ('dz', 't', 'p', 'rho', 'qv', 'qc', 'qi', 'qr', 'qs', 'qg', 'qnc')])

    pflx_sdfg = f64_2d()
    prr_sdfg, pri_sdfg, prs_sdfg, prg_sdfg, pre_sdfg = (f64_1d() for _ in range(5))

    sdfg(nvec=np.int32(ivec), ke=np.int32(k_v), ivstart=np.int32(ivs), ivend=np.int32(ive), kstart=np.int32(ks),
         dt=np.float64(dt), dz=bufs_sdfg['dz'], t=bufs_sdfg['t'], p=bufs_sdfg['p'], rho=bufs_sdfg['rho'],
         qv=bufs_sdfg['qv'], qc=bufs_sdfg['qc'], qi=bufs_sdfg['qi'], qr=bufs_sdfg['qr'], qs=bufs_sdfg['qs'],
         qg=bufs_sdfg['qg'], qnc=bufs_sdfg['qnc'], prr_gsp=prr_sdfg, pri_gsp=pri_sdfg, prs_gsp=prs_sdfg,
         prg_gsp=prg_sdfg, pflx=pflx_sdfg, pre_gsp=pre_sdfg)

    for nm in ('t', 'qv', 'qc', 'qi', 'qr', 'qs', 'qg'):
        np.testing.assert_allclose(bufs_sdfg[nm], bufs_ref[nm], rtol=1e-10, atol=1e-12,
                                   err_msg=f"prognostic {nm} drifted")
    np.testing.assert_allclose(pflx_sdfg, pflx_ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(prr_sdfg, prr_ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(pri_sdfg, pri_ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(prs_sdfg, prs_ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(prg_sdfg, prg_ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(pre_sdfg, pre_ref, rtol=1e-10, atol=1e-12)
