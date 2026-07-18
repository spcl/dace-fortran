"""E2e SDFG-vs-f2py verification for CLOUDSC microphysics loopnests (ECMWF
cloud-scheme benchmark).  Unlike ``icon/selected_loopnests`` these kernels are
bare flat subroutines -- SDFG-vs-f2py numerical equivalence is the only
correctness check (no struct-typed cross-check).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

_HERE = Path(__file__).resolve().parent

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _f2py_build(src_text: str, out_dir: Path, mod_name: str):
    """f2py-compile ``src_text`` as ``mod_name`` into ``out_dir`` and
    return the imported Python module.  Skips if the toolchain is missing."""
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not available")
    if shutil.which("meson") is None:
        pytest.skip("meson not available (f2py backend on Python>=3.12)")
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / f"{mod_name}.f90"
    src.write_text(src_text)
    # meson reads FFLAGS not --f90flags; f2py's single-line SUBROUTINE decl exceeds
    # gfortran's 132-col limit -- lift the cap (append, don't clobber).
    env = {**os.environ, "FFLAGS": (os.environ.get("FFLAGS", "") + " -ffree-line-length-none").strip()}
    subprocess.check_call(
        [sys.executable, "-m", "numpy.f2py", "-c",
         str(src), "-m", mod_name, "--quiet"],
        cwd=out_dir,
        env=env,
    )
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    __import__(mod_name)
    return sys.modules[mod_name]


def _sdfg_from_src(src: str, tmp: Path, name: str):
    """Build an SDFG via HLFIR bridge with ``hlfir-propagate-shapes`` (matches icon's
    convention).  Entry stays bare/auto-resolved -- wrapping the kernel in a module would
    re-expose it under f2py as ``ref.<module>.<proc>`` and break sibling tests that call
    ``ref.<proc>(...)`` directly."""
    tmp.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, tmp, name=name, pipeline="hlfir-propagate-shapes").build()


def _sdfg_call_args(sdfg, int_values: dict) -> dict:
    """Route each int arg to a plain int (Scalar) or length-1 numpy array (Array),
    matching the SDFG's classification.  Mirrors the icon/selected_loopnests helper."""
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


def test_cloudsc_lu_solver_sdfg_matches_f2py(tmp_path: Path):
    """LU forward+back substitute for the microphysics species block -- linear-algebra
    triple-nested loop, nclv-bounded."""
    src = (_HERE / "cloudsc_lu_solver.f90").read_text()
    ref = _f2py_build(src, tmp_path / "ref", "lu_solver_ref")
    sdfg = _sdfg_from_src(src, tmp_path / "sdfg", name="lu_solver_microphysics")

    rng = np.random.default_rng(101)
    klon, nclv = 8, 5
    kidia, kfdia = 1, klon

    # Diagonal-dominant random matrix so the LU back-substitute is stable.
    zqlhs = rng.standard_normal((klon, nclv, nclv)).astype(np.float64)
    for d in range(nclv):
        zqlhs[:, d, d] += 10.0  # diagonal dominance
    zqlhs = np.asfortranarray(zqlhs)
    zqxn = np.asfortranarray(rng.standard_normal((klon, nclv)).astype(np.float64))

    zqlhs_ref = np.array(zqlhs, order="F")
    zqxn_ref = np.array(zqxn, order="F")
    zqlhs_sdfg = np.array(zqlhs, order="F")
    zqxn_sdfg = np.array(zqxn, order="F")

    # f2py drops klon / nclv (auto-derived from array shapes).
    ref.lu_solver_microphysics(kidia, kfdia, zqlhs_ref, zqxn_ref)

    kw = dict(zqlhs=zqlhs_sdfg, zqxn=zqxn_sdfg)
    kw.update(_sdfg_call_args(sdfg, dict(kidia=kidia, kfdia=kfdia, klon=klon, nclv=nclv)))
    sdfg(**kw)

    np.testing.assert_allclose(zqlhs_sdfg, zqlhs_ref, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(zqxn_sdfg, zqxn_ref, atol=1e-10, rtol=1e-10)


def test_cloudsc_saturation_sdfg_matches_f2py(tmp_path: Path):
    """Saturation-pressure computation across (klon, klev): reads T+P, writes seven
    outputs, pure elementwise (no loop-carried dependence)."""
    src = (_HERE / "cloudsc_saturation_calculation.f90").read_text()
    ref = _f2py_build(src, tmp_path / "ref", "saturation_ref")
    sdfg = _sdfg_from_src(src, tmp_path / "sdfg", name="compute_saturation_values")

    rng = np.random.default_rng(102)
    klon, klev = 16, 8
    kidia, kfdia = 1, klon

    # Realistic atmospheric temperature [200, 300] K + pressure [1e3, 1e5] Pa.
    ztp1 = np.asfortranarray(200.0 + 100.0 * rng.random((klon, klev), dtype=np.float64))
    pap = np.asfortranarray(1e3 + 1e5 * rng.random((klon, klev), dtype=np.float64))

    # CLOUDSC default physical constants -- same numbers on both backends.
    consts = dict(rtt=273.16,
                  retv=0.608,
                  r2es=611.21,
                  r3les=17.502,
                  r3ies=22.587,
                  r4les=32.19,
                  r4ies=-0.7,
                  rtice=250.0,
                  rtwat=273.16,
                  rtwat_rtice_r=1.0 / (273.16 - 250.0))

    outs_sdfg = {
        k: np.zeros((klon, klev), order="F")
        for k in ("zfoealfa", "zfoeewmt", "zqsmix", "zfoeew", "zqsice", "zfoeeliqt", "zqsliq")
    }

    # f2py returns INTENT(OUT) arrays as a tuple; only IN/scalars positional; klon/klev auto-derived.
    out_tuple = ref.compute_saturation_values(kidia, kfdia, ztp1, pap, **consts)
    outs_ref = dict(zip(("zfoealfa", "zfoeewmt", "zqsmix", "zfoeew", "zqsice", "zfoeeliqt", "zqsliq"), out_tuple))

    kw = dict(ztp1=ztp1, pap=pap, **outs_sdfg, **consts)
    kw.update(_sdfg_call_args(sdfg, dict(kidia=kidia, kfdia=kfdia, klon=klon, klev=klev)))
    sdfg(**kw)

    for k in outs_ref:
        np.testing.assert_allclose(outs_sdfg[k], outs_ref[k], atol=1e-10, rtol=1e-10, err_msg=f"output {k} differs")


def test_cloudsc_autoconversion_snow_sdfg_matches_f2py(tmp_path: Path):
    """Snow autoconversion: writes ``zsnowaut(jl)`` + INOUT contribution to
    ``zsolqb(jl, ncldqs, ncldqi)``; ``laericeauto`` (0/1) toggles the aerosol-coupled
    formulation."""
    src = (_HERE / "cloudsc_autoconversion_snow.f90").read_text()
    ref = _f2py_build(src, tmp_path / "ref", "autoconv_snow_ref")
    sdfg = _sdfg_from_src(src, tmp_path / "sdfg", name="autoconversion_snow")

    rng = np.random.default_rng(103)
    # NCLV = array bound (5 species); ncldqs/ncldqi = runtime indices into it (snow=4,
    # ice=2) -- don't confuse the two or zsolqb's shape ends up wrong.
    klon, nclv = 8, 5
    ncldqs, ncldqi = 4, 2
    kidia, kfdia = 1, klon

    ztp1 = np.asfortranarray(250.0 + 30.0 * rng.random((klon, ), dtype=np.float64))
    zicecld = np.asfortranarray(1e-5 + 1e-4 * rng.random((klon, ), dtype=np.float64))
    pnice = np.asfortranarray(1e3 + 1e4 * rng.random((klon, ), dtype=np.float64))
    consts = dict(rtt=273.16, rlcritsnow=0.5e-4, rsnowlin1=1e-3, rsnowlin2=0.018, rnice=1.0, ptsphy=600.0, zepsec=1e-14)
    laericeauto = 1

    zsolqb_ref = np.zeros((klon, nclv, nclv), order="F")
    zsolqb_sdfg = np.zeros_like(zsolqb_ref, order="F")

    # f2py: zsnowaut (OUT) -> return; zsolqb (INOUT) -> positional.
    zsnowaut_ref = ref.autoconversion_snow(kidia,
                                           kfdia,
                                           ztp1,
                                           zicecld,
                                           pnice,
                                           zsolqb_ref,
                                           rtt=consts["rtt"],
                                           rlcritsnow=consts["rlcritsnow"],
                                           rsnowlin1=consts["rsnowlin1"],
                                           rsnowlin2=consts["rsnowlin2"],
                                           rnice=consts["rnice"],
                                           ptsphy=consts["ptsphy"],
                                           zepsec=consts["zepsec"],
                                           laericeauto=laericeauto,
                                           ncldqs=ncldqs,
                                           ncldqi=ncldqi)
    zsnowaut_sdfg = np.zeros((klon, ), order="F")

    kw = dict(ztp1=ztp1,
              zicecld=zicecld,
              pnice=pnice,
              zsolqb=zsolqb_sdfg,
              zsnowaut=zsnowaut_sdfg,
              laericeauto=laericeauto,
              **consts)
    kw.update(_sdfg_call_args(sdfg, dict(kidia=kidia, kfdia=kfdia, klon=klon, nclv=nclv, ncldqs=ncldqs, ncldqi=ncldqi)))
    sdfg(**kw)

    np.testing.assert_allclose(zsnowaut_sdfg, zsnowaut_ref, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(zsolqb_sdfg, zsolqb_ref, atol=1e-12, rtol=1e-10)


def test_cloudsc_ice_supersat_sdfg_matches_f2py(tmp_path: Path):
    """Ice supersaturation adjustment: branchy elementwise body with in/out updates to
    ``zsolqa``, ``zsolac``, ``zqxfg``."""
    src = (_HERE / "cloudsc_ice_supersaturation_adjustment.f90").read_text()
    ref = _f2py_build(src, tmp_path / "ref", "ice_supersat_ref")
    sdfg = _sdfg_from_src(src, tmp_path / "sdfg", name="ice_supersaturation_adjustment")

    rng = np.random.default_rng(104)
    klon, nclv = 8, 5
    kidia, kfdia = 1, klon
    ncldql, ncldqi, ncldqv = 1, 2, 3
    nssopt = 1

    ztp1 = np.asfortranarray(220.0 + 30.0 * rng.random((klon, ), dtype=np.float64))
    za = np.asfortranarray(rng.random((klon, ), dtype=np.float64))
    zqx_ncldqv = np.asfortranarray(1e-5 + 1e-3 * rng.random((klon, ), dtype=np.float64))
    zqsice = np.asfortranarray(1e-5 + 1e-3 * rng.random((klon, ), dtype=np.float64))
    zcorqsice = np.asfortranarray(1e-5 + 1e-3 * rng.random((klon, ), dtype=np.float64))
    zfokoop = np.asfortranarray(rng.random((klon, ), dtype=np.float64))
    consts = dict(rtt=273.16, ramin=1e-12, rthomo=235.0, rkooptau=1e-4, ptsphy=600.0, zepsec=1e-14)

    zsolqa_ref = np.asfortranarray(rng.random((klon, nclv, nclv), dtype=np.float64) * 1e-3)
    zsolac_ref = np.asfortranarray(rng.random((klon, ), dtype=np.float64))
    zqxfg_ref = np.asfortranarray(rng.random((klon, nclv), dtype=np.float64) * 1e-3)
    zsolqa_sdfg = np.array(zsolqa_ref, order="F")
    zsolac_sdfg = np.array(zsolac_ref, order="F")
    zqxfg_sdfg = np.array(zqxfg_ref, order="F")

    # f2py positional: kidia, kfdia, ztp1, za, zqx_ncldqv, zqsice, zcorqsice, zfokoop,
    #                  zsolqa, zsolac, zqxfg, rtt, ramin, rthomo, nssopt, rkooptau, ptsphy, zepsec,
    #                  ncldql, ncldqi, ncldqv   ([klon, nclv] auto-derived)
    ref.ice_supersaturation_adjustment(kidia, kfdia, ztp1, za, zqx_ncldqv, zqsice, zcorqsice, zfokoop, zsolqa_ref,
                                       zsolac_ref, zqxfg_ref, consts["rtt"], consts["ramin"], consts["rthomo"], nssopt,
                                       consts["rkooptau"], consts["ptsphy"], consts["zepsec"], ncldql, ncldqi, ncldqv)

    kw = dict(ztp1=ztp1,
              za=za,
              zqx_ncldqv=zqx_ncldqv,
              zqsice=zqsice,
              zcorqsice=zcorqsice,
              zfokoop=zfokoop,
              zsolqa=zsolqa_sdfg,
              zsolac=zsolac_sdfg,
              zqxfg=zqxfg_sdfg,
              **consts)
    kw.update(
        _sdfg_call_args(
            sdfg,
            dict(kidia=kidia,
                 kfdia=kfdia,
                 klon=klon,
                 nclv=nclv,
                 ncldql=ncldql,
                 ncldqi=ncldqi,
                 ncldqv=ncldqv,
                 nssopt=nssopt)))
    sdfg(**kw)

    np.testing.assert_allclose(zsolqa_sdfg, zsolqa_ref, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(zsolac_sdfg, zsolac_ref, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(zqxfg_sdfg, zqxfg_ref, atol=1e-12, rtol=1e-10)


def test_cloudsc_rain_evap_sdfg_matches_f2py(tmp_path: Path):
    """Rain-evaporation (Abel-Boutle 2012): branchy elementwise update to
    ``zsolqa(jl, ncldqv, ncldqr)`` + scalars; LOGICAL local + many physical constants."""
    src = (_HERE / "cloudsc_rain_evaporation_abel_boutle.f90").read_text()
    ref = _f2py_build(src, tmp_path / "ref", "rain_evap_ref")
    sdfg = _sdfg_from_src(src, tmp_path / "sdfg", name="rain_evaporation_abel_boutle")

    rng = np.random.default_rng(105)
    klon, nclv = 8, 5
    kidia, kfdia = 1, klon
    ncldqv, ncldqr = 3, 4

    ztp1 = np.asfortranarray(250.0 + 40.0 * rng.random((klon, ), dtype=np.float64))
    zqx_ncldqv = np.asfortranarray(1e-5 + 1e-3 * rng.random((klon, ), dtype=np.float64))
    za = np.asfortranarray(rng.random((klon, ), dtype=np.float64))
    zqsliq = np.asfortranarray(1e-5 + 1e-3 * rng.random((klon, ), dtype=np.float64))
    zqxfg_ncldqr = np.asfortranarray(1e-6 + 1e-4 * rng.random((klon, ), dtype=np.float64))
    zcovptot = np.asfortranarray(rng.random((klon, ), dtype=np.float64))
    zcovpclr = np.asfortranarray(rng.random((klon, ), dtype=np.float64))
    zcovpmax = np.asfortranarray(rng.random((klon, ), dtype=np.float64))
    zrho = np.asfortranarray(1.0 + 0.3 * rng.random((klon, ), dtype=np.float64))
    pap = np.asfortranarray(1e4 + 1e5 * rng.random((klon, ), dtype=np.float64))
    consts = dict(rtt=273.16,
                  rv=461.5,
                  rd=287.04,
                  rprecrhmax=1.0,
                  rcovpmin=1e-3,
                  rdensref=1.0,
                  ptsphy=600.0,
                  zepsec=1e-14,
                  rcl_fac1=1.0,
                  rcl_fac2=1.0,
                  rcl_cdenom1=1.0,
                  rcl_cdenom2=1.0,
                  rcl_cdenom3=1.0,
                  rcl_ka273=2.4e-2,
                  rcl_const1r=1.0,
                  rcl_const2r=1.0,
                  rcl_const3r=1.0,
                  rcl_const4r=1.0)

    zsolqa_ref = np.asfortranarray(rng.random((klon, nclv, nclv), dtype=np.float64) * 1e-4)
    zevap_ref = np.zeros((klon, ), order="F")
    zsolqa_sdfg = np.array(zsolqa_ref, order="F")
    zevap_sdfg = np.zeros_like(zevap_ref, order="F")
    zcovptot_ref = np.array(zcovptot, order="F")
    zcovpclr_ref = np.array(zcovpclr, order="F")
    zqxfg_ref = np.array(zqxfg_ncldqr, order="F")
    zcovptot_sdfg = np.array(zcovptot, order="F")
    zcovpclr_sdfg = np.array(zcovpclr, order="F")
    zqxfg_sdfg = np.array(zqxfg_ncldqr, order="F")

    # f2py positional (zevap_out is OUT, returned):
    # kidia, kfdia, ztp1, zqx_ncldqv, za, zqsliq, zqxfg_ncldqr, zcovptot, zcovpclr, zcovpmax,
    # zrho, pap, zsolqa, rtt, rv, rd, rprecrhmax, rcovpmin, rdensref, ptsphy, zepsec,
    # rcl_fac1, rcl_fac2, rcl_cdenom1, rcl_cdenom2, rcl_cdenom3, rcl_ka273,
    # rcl_const1r, rcl_const2r, rcl_const3r, rcl_const4r, ncldqv, ncldqr  ([klon, nclv] auto)
    zevap_ref = ref.rain_evaporation_abel_boutle(
        kidia, kfdia, ztp1, zqx_ncldqv, za, zqsliq, zqxfg_ref, zcovptot_ref, zcovpclr_ref, zcovpmax, zrho, pap,
        zsolqa_ref, consts["rtt"], consts["rv"], consts["rd"], consts["rprecrhmax"], consts["rcovpmin"],
        consts["rdensref"], consts["ptsphy"], consts["zepsec"], consts["rcl_fac1"], consts["rcl_fac2"],
        consts["rcl_cdenom1"], consts["rcl_cdenom2"], consts["rcl_cdenom3"], consts["rcl_ka273"], consts["rcl_const1r"],
        consts["rcl_const2r"], consts["rcl_const3r"], consts["rcl_const4r"], ncldqv, ncldqr)
    zevap_sdfg = np.zeros((klon, ), order="F")

    kw = dict(ztp1=ztp1,
              zqx_ncldqv=zqx_ncldqv,
              za=za,
              zqsliq=zqsliq,
              zqxfg_ncldqr=zqxfg_sdfg,
              zcovptot=zcovptot_sdfg,
              zcovpclr=zcovpclr_sdfg,
              zcovpmax=zcovpmax,
              zrho=zrho,
              pap=pap,
              zsolqa=zsolqa_sdfg,
              zevap_out=zevap_sdfg,
              **consts)
    kw.update(_sdfg_call_args(sdfg, dict(kidia=kidia, kfdia=kfdia, klon=klon, nclv=nclv, ncldqv=ncldqv, ncldqr=ncldqr)))
    sdfg(**kw)

    # rho**0.78 amplifies libm-vs-C++-codegen rounding diffs; ~1e-7 relative drift is
    # expected, tighter tolerance would false-positive on benign ulp differences.
    rt, at = 1e-6, 1e-8
    np.testing.assert_allclose(zsolqa_sdfg, zsolqa_ref, atol=at, rtol=rt)
    np.testing.assert_allclose(zevap_sdfg, zevap_ref, atol=at, rtol=rt)
    np.testing.assert_allclose(zcovptot_sdfg, zcovptot_ref, atol=at, rtol=rt)
    np.testing.assert_allclose(zcovpclr_sdfg, zcovpclr_ref, atol=at, rtol=rt)
    np.testing.assert_allclose(zqxfg_sdfg, zqxfg_ref, atol=at, rtol=rt)


def test_cloudsc_full_microphysics_solve_sdfg_matches_f2py(tmp_path: Path):
    """Full Section-5.2.2 solver: ZQLHS construction + ZQXN RHS assembly + LU
    factorization + back-substitution + ZEPSEC clip into vapor.  Bigger scope than
    ``cloudsc_lu_solver`` (adds the LHS/RHS assembly) -- suspected source of the 1-9 ulp
    ``ZQXN`` drift at JK=NCLDTOP=15 in cloudsc_full (xfail).  If this passes at
    ``rtol=atol=1e-14`` while cloudsc_full still diverges, the bug is further upstream in
    JK-loop-carried state; if this also diverges, the bug is in this assembly+solve
    combination.
    """
    src = (_HERE / "cloudsc_full_microphysics_solve.f90").read_text()
    ref = _f2py_build(src, tmp_path / "ref", "full_solve_ref")
    sdfg = _sdfg_from_src(src, tmp_path / "sdfg", name="full_microphysics_solve")

    rng = np.random.default_rng(106)
    klon, klev, nclv = 8, 137, 5  # match cloudsc_full (KLEV=137, NCLV=5)
    ncldqv = 5
    kidia, kfdia = 1, klon
    jk_idx = 15  # NCLDTOP -- pick the same JK at which cloudsc_full first diverges
    zepsec = 1e-14

    # ZFALLSINK in [0,1) + small ZSOLQB/ZSOLQA keep the matrix near-identity (LU
    # stable); ZQX in [1e-5,1e-2] = plausible mass-mixing ratios.
    zfallsink = np.asfortranarray(rng.random((klon, nclv), dtype=np.float64))
    zsolqb = np.asfortranarray(rng.random((klon, nclv, nclv), dtype=np.float64) * 1e-2)
    zsolqa = np.asfortranarray(rng.random((klon, nclv, nclv), dtype=np.float64) * 1e-2)
    zqx = np.asfortranarray(1e-5 + 1e-2 * rng.random((klon, klev, nclv), dtype=np.float64))

    zqlhs_ref = np.zeros((klon, nclv, nclv), order="F")
    zqxn_ref = np.zeros((klon, nclv), order="F")
    zqlhs_sdfg = np.array(zqlhs_ref, order="F")
    zqxn_sdfg = np.array(zqxn_ref, order="F")

    # jk_idx is a runtime INDEX (not the bound); klon/klev/nclv auto-derived.  ZQXN is
    # INTENT(OUT) -> returned.  Positional: kidia, kfdia, ncldqv, jk_idx, zfallsink,
    # zsolqa, zsolqb, zqx, zqlhs, zepsec
    zqxn_ref = ref.full_microphysics_solve(kidia, kfdia, ncldqv, jk_idx, zfallsink, zsolqa, zsolqb, zqx, zqlhs_ref,
                                           zepsec)

    kw = dict(zfallsink=zfallsink,
              zsolqa=zsolqa,
              zsolqb=zsolqb,
              zqx=zqx,
              zqlhs=zqlhs_sdfg,
              zqxn=zqxn_sdfg,
              zepsec=zepsec)
    kw.update(
        _sdfg_call_args(sdfg,
                        dict(kidia=kidia, kfdia=kfdia, klon=klon, klev=klev, nclv=nclv, ncldqv=ncldqv, jk_idx=jk_idx)))
    sdfg(**kw)

    # ulp-level tolerance: values <1 so 1ulp~2e-16; 1e-14 catches ~50ulp+.
    # cloudsc_lu_solver uses 1e-10 (many more sequential ops).
    np.testing.assert_allclose(zqlhs_sdfg, zqlhs_ref, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(zqxn_sdfg, zqxn_ref, atol=1e-14, rtol=1e-14)


def test_cloudsc_jk_precip_chain_sdfg_matches_f2py(tmp_path: Path):
    """Multi-JK precip chain: JK=NCLDTOP..KLEV loop over ZRHO -> ZFALLSINK -> ZPFPLSX ->
    ZQPRETOT -> ZCOVPTOT (line-3608 3-way multiply, line-3614 NCLDQR+NCLDQS sum,
    max-overlap update).  Skips the LU solver (takes ZQXN + ZFALLSINK as inputs) to
    bisect JK-loop-carried state from the LU-solver portion of cloudsc_full's
    divergence.  First loopnest here with a multi-iteration loop-carried JK loop --
    others are single-iteration.
    """
    src = (_HERE / "cloudsc_jk_precip_chain.f90").read_text()
    ref = _f2py_build(src, tmp_path / "ref", "jk_precip_ref")
    sdfg = _sdfg_from_src(src, tmp_path / "sdfg", name="jk_precip_chain")

    rng = np.random.default_rng(107)
    # matches cloudsc_full's geometry so the bug-reproducing JK iteration count is the same.
    klon, klev, nclv = 1, 137, 5
    ncldqr, ncldqs = 3, 4
    ncldtop = 15
    kidia, kfdia = 1, klon
    rd, rg = 287.0597, 9.80665
    ptsphy = 1800.0  # 30-min physics timestep
    zepsec = 1e-14

    # Pressure increases with depth; T tapers near surface.  ZA random in [0,1].
    pap = np.asfortranarray(
        np.linspace(2e4, 1e5, klev)[None, :].repeat(klon, axis=0) + 100.0 * rng.standard_normal((klon, klev)))
    paph = np.asfortranarray(np.linspace(1.5e4, 1.05e5, klev + 1)[None, :].repeat(klon, axis=0))
    ztp1 = np.asfortranarray(220.0 + 60.0 * rng.random((klon, klev), dtype=np.float64))
    za = np.asfortranarray(rng.random((klon, klev), dtype=np.float64))

    # ZQXN/ZFALLSINK small, sometimes near zero -- exercises IF (ZQPRETOT < ZEPSEC) both sides.
    zqxn = np.asfortranarray(1e-7 * rng.random((klon, klev, nclv), dtype=np.float64))
    zfallsink_in = np.asfortranarray(0.5 + 0.4 * rng.random((klon, klev, nclv), dtype=np.float64))

    zpfplsx_ref = np.zeros((klon, klev + 1, nclv), order="F")
    zqpretot_ref = np.zeros((klon, klev), order="F")
    zcovptot_ref = np.zeros((klon, klev), order="F")
    zpfplsx_sdfg = np.zeros_like(zpfplsx_ref, order="F")
    zqpretot_sdfg = np.zeros_like(zqpretot_ref, order="F")
    zcovptot_sdfg = np.zeros_like(zcovptot_ref, order="F")

    # f2py positional, returns the 3 OUTs as a tuple; klon/klev/nclv auto-derived.
    zpfplsx_ref, zqpretot_ref, zcovptot_ref = ref.jk_precip_chain(kidia, kfdia, ncldqr, ncldqs, ncldtop, pap, paph,
                                                                  ztp1, za, zqxn, zfallsink_in, rd, rg, ptsphy, zepsec)

    kw = dict(pap=pap,
              paph=paph,
              ztp1=ztp1,
              za=za,
              zqxn=zqxn,
              zfallsink_in=zfallsink_in,
              zpfplsx=zpfplsx_sdfg,
              zqpretot=zqpretot_sdfg,
              zcovptot=zcovptot_sdfg,
              rd=rd,
              rg=rg,
              ptsphy=ptsphy,
              zepsec=zepsec)
    kw.update(
        _sdfg_call_args(
            sdfg,
            dict(kidia=kidia,
                 kfdia=kfdia,
                 klon=klon,
                 klev=klev,
                 nclv=nclv,
                 ncldqr=ncldqr,
                 ncldqs=ncldqs,
                 ncldtop=ncldtop)))
    sdfg(**kw)

    # ulp tolerance: if this passes, the JK precip carry is bit-correct and cloudsc_full's
    # drift is upstream (ZFALLSINK/ZQXN); if it fails, the bug is in this chain.
    np.testing.assert_allclose(zpfplsx_sdfg, zpfplsx_ref, atol=1e-14, rtol=1e-14, err_msg="ZPFPLSX diverges")
    np.testing.assert_allclose(zqpretot_sdfg, zqpretot_ref, atol=1e-14, rtol=1e-14, err_msg="ZQPRETOT diverges")
    np.testing.assert_allclose(zcovptot_sdfg, zcovptot_ref, atol=1e-14, rtol=1e-14, err_msg="ZCOVPTOT diverges")


def test_cloudsc_pow_kernel_sdfg_matches_f2py(tmp_path: Path):
    """Minimal ``x ** non_integer_exponent`` reproducer for cloudsc 4.5a rain evap
    (``(RDENSREF/ZRHO) ** 0.4``-shaped).  Hypothesis: libm pow() with a non-integer
    exponent isn't bit-exact between gfortran and DaCe-emitted C++; pins the magnitude
    of that drift in isolation.
    """
    src = (_HERE / "cloudsc_pow_kernel.f90").read_text()
    ref = _f2py_build(src, tmp_path / "ref", "pow_kernel_ref")
    sdfg = _sdfg_from_src(src, tmp_path / "sdfg", name="pow_kernel")

    rng = np.random.default_rng(108)
    n = 32
    x = np.asfortranarray(0.3 + 1.2 * rng.random(n, dtype=np.float64))
    exponent = 0.78  # cloudsc 4.5a uses 0.78 and 0.4

    y_ref = ref.pow_kernel(x, exponent)
    y_sdfg = np.zeros(n, dtype=np.float64, order="F")
    kw = dict(x=x, exponent_val=exponent, y=y_sdfg)
    kw.update(_sdfg_call_args(sdfg, dict(n=n)))
    sdfg(**kw)

    np.testing.assert_allclose(y_sdfg,
                               y_ref,
                               atol=1e-14,
                               rtol=1e-14,
                               err_msg="pow_kernel: SDFG x**0.78 diverges from gfortran x**0.78")


def test_cloudsc_zsolqa_accumulator_sdfg_matches_f2py(tmp_path: Path):
    """Minimal ``ZSOLQA = ZSOLQA + EVAP`` accumulator reproducer (CLOUDSC 4.5a/4.5b's
    ``A(i,j) = A(i,j) + expr`` pattern, dozens of occurrences).  Hypothesis: the bridge
    lowers this as a WCR (atomic-add) edge instead of explicit read+add+write, so
    accumulation order can drift from gfortran's strict left-to-right.  Failure at
    rtol=atol=1e-14 confirms WCR-vs-RMW; fix is explicit read+tasklet+write for
    single-producer accumulation.
    """
    src = (_HERE / "cloudsc_zsolqa_accumulator.f90").read_text()
    ref = _f2py_build(src, tmp_path / "ref", "zsolqa_accum_ref")
    sdfg = _sdfg_from_src(src, tmp_path / "sdfg", name="zsolqa_accumulator")

    rng = np.random.default_rng(109)
    n, nclv = 16, 5
    ncldqv, ncldqr, ncldqs = 5, 3, 4
    kidia, kfdia = 1, n

    zevap = np.asfortranarray(rng.standard_normal(n).astype(np.float64) * 1e-4)
    zsnowsrc = np.asfortranarray(rng.standard_normal(n).astype(np.float64) * 1e-4)
    zsolqa_ref = np.asfortranarray(rng.standard_normal((n, nclv, nclv)).astype(np.float64) * 1e-3)
    zsolqa_sdfg = np.array(zsolqa_ref, order="F")

    ref.zsolqa_accumulator(kidia, kfdia, ncldqv, ncldqr, ncldqs, zevap, zsnowsrc, zsolqa_ref)

    kw = dict(zevap=zevap, zsnowsrc=zsnowsrc, zsolqa=zsolqa_sdfg)
    kw.update(
        _sdfg_call_args(sdfg, dict(n=n,
                                   nclv=nclv,
                                   kidia=kidia,
                                   kfdia=kfdia,
                                   ncldqv=ncldqv,
                                   ncldqr=ncldqr,
                                   ncldqs=ncldqs)))
    sdfg(**kw)

    np.testing.assert_allclose(zsolqa_sdfg,
                               zsolqa_ref,
                               atol=1e-14,
                               rtol=1e-14,
                               err_msg="zsolqa_accumulator: SDFG += diverges from gfortran "
                               "(suggests WCR-instead-of-RMW lowering)")


def test_cloudsc_int_pow_kernel_sdfg_matches_f2py(tmp_path: Path):
    """Integer-exponent power: y2=x**2, y3=x**3 (cloudsc 4.5b SNOW evap's
    ``ZTP1**2``/``ZTP1**3``).  Bridge may lower as repeated-multiply, libm pow(d,d/i), or
    ipow() -- a lowering mismatch vs gfortran's intrinsic gives different rounding even
    under ``-O0 -fno-fast-math``.
    """
    src = (_HERE / "cloudsc_int_pow.f90").read_text()
    ref = _f2py_build(src, tmp_path / "ref", "int_pow_ref")
    sdfg = _sdfg_from_src(src, tmp_path / "sdfg", name="int_pow_kernel")

    rng = np.random.default_rng(110)
    n = 32
    x = np.asfortranarray(220.0 + 80.0 * rng.random(n, dtype=np.float64))

    y2_ref, y3_ref = ref.int_pow_kernel(x)
    y2_sdfg = np.zeros(n, dtype=np.float64, order="F")
    y3_sdfg = np.zeros(n, dtype=np.float64, order="F")
    kw = dict(x=x, y2=y2_sdfg, y3=y3_sdfg)
    kw.update(_sdfg_call_args(sdfg, dict(n=n)))
    sdfg(**kw)

    np.testing.assert_allclose(y2_sdfg, y2_ref, atol=1e-14, rtol=1e-14, err_msg="int_pow x**2 diverges")
    np.testing.assert_allclose(y3_sdfg, y3_ref, atol=1e-14, rtol=1e-14, err_msg="int_pow x**3 diverges")


def test_cloudsc_zterm2_kernel_sdfg_matches_f2py(tmp_path: Path):
    """Verbatim cloudsc.F90 line 3304 ZTERM2 (4.5b SNOW evap, IEVAPSNOW=1): bare-default-real
    literals ``0.65``/``0.5`` combined with runtime power ``ZPR02**RCL_CONST4S``.
    """
    src = (_HERE / "cloudsc_zterm2_kernel.f90").read_text()
    ref = _f2py_build(src, tmp_path / "ref", "zterm2_ref")
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src,
                      tmp_path / "sdfg",
                      name="zterm2_kernel",
                      pipeline="hlfir-propagate-shapes",
                      entry="zterm2_kernel").build()

    rng = np.random.default_rng(120)
    n = 32
    zpr02 = np.asfortranarray(1e-3 + 1e-2 * rng.random(n, dtype=np.float64))
    zcorrfac = np.asfortranarray(0.8 + 0.4 * rng.random(n, dtype=np.float64))
    zrho = np.asfortranarray(0.3 + 1.2 * rng.random(n, dtype=np.float64))
    zcorrfac2 = np.asfortranarray(0.5 + 0.5 * rng.random(n, dtype=np.float64))
    rcl_const3s, rcl_const4s, rcl_const5s, rcl_const6s = 1.5, 0.5, 0.65, 2.0

    zterm2_ref = ref.zterm2_kernel(zpr02, zcorrfac, zrho, zcorrfac2, rcl_const3s, rcl_const4s, rcl_const5s, rcl_const6s)
    zterm2_sdfg = np.zeros(n, dtype=np.float64, order="F")
    kw = dict(zpr02=zpr02,
              zcorrfac=zcorrfac,
              zrho=zrho,
              zcorrfac2=zcorrfac2,
              rcl_const3s=rcl_const3s,
              rcl_const4s=rcl_const4s,
              rcl_const5s=rcl_const5s,
              rcl_const6s=rcl_const6s,
              zterm2=zterm2_sdfg)
    kw.update(_sdfg_call_args(sdfg, dict(n=n)))
    sdfg(**kw)

    np.testing.assert_allclose(zterm2_sdfg,
                               zterm2_ref,
                               atol=1e-14,
                               rtol=1e-14,
                               err_msg="ZTERM2 compound expression diverges")


def test_cloudsc_zbeta_kernel_sdfg_matches_f2py(tmp_path: Path):
    """Verbatim extract of cloudsc.F90 lines 3158-3161 ZBETA expression
    (4.5a RAIN evap Abel-Boutle).  All literals properly _JPRB-suffixed.
    Deeply nested compound: ``(0.5/zqsliq) * ztp1**2 * zesatliq *
    const1r * (zcorr2/zevap_denom) * (0.78/zlambda**const4r +
    const2r*(zrho*zfallcorr)**0.5 / (zcorr2**0.5*zlambda**const3r))``.
    """
    src = (_HERE / "cloudsc_zterm2_kernel.f90").read_text()  # same file
    ref = _f2py_build(src, tmp_path / "ref", "zbeta_ref")
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src,
                      tmp_path / "sdfg",
                      name="zbeta_kernel",
                      pipeline="hlfir-propagate-shapes",
                      entry="zbeta_kernel").build()

    rng = np.random.default_rng(121)
    n = 32
    zqsliq = np.asfortranarray(1e-5 + 1e-3 * rng.random(n, dtype=np.float64))
    ztp1 = np.asfortranarray(240.0 + 50.0 * rng.random(n, dtype=np.float64))
    zesatliq = np.asfortranarray(1e2 + 1e3 * rng.random(n, dtype=np.float64))
    zcorr2 = np.asfortranarray(0.5 + 0.5 * rng.random(n, dtype=np.float64))
    zevap_denom = np.asfortranarray(1.0 + 1.0 * rng.random(n, dtype=np.float64))
    zlambda = np.asfortranarray(1.0 + 10.0 * rng.random(n, dtype=np.float64))
    zrho = np.asfortranarray(0.3 + 1.2 * rng.random(n, dtype=np.float64))
    zfallcorr = np.asfortranarray(0.5 + 1.0 * rng.random(n, dtype=np.float64))
    rcl_const1r, rcl_const2r, rcl_const3r, rcl_const4r = 1.0, 0.3, 0.5, 0.4

    zbeta_ref = ref.zbeta_kernel(zqsliq, ztp1, zesatliq, zcorr2, zevap_denom, zlambda, zrho, zfallcorr, rcl_const1r,
                                 rcl_const2r, rcl_const3r, rcl_const4r)
    zbeta_sdfg = np.zeros(n, dtype=np.float64, order="F")
    kw = dict(zqsliq=zqsliq,
              ztp1=ztp1,
              zesatliq=zesatliq,
              zcorr2=zcorr2,
              zevap_denom=zevap_denom,
              zlambda=zlambda,
              zrho=zrho,
              zfallcorr=zfallcorr,
              rcl_const1r=rcl_const1r,
              rcl_const2r=rcl_const2r,
              rcl_const3r=rcl_const3r,
              rcl_const4r=rcl_const4r,
              zbeta=zbeta_sdfg)
    kw.update(_sdfg_call_args(sdfg, dict(n=n)))
    sdfg(**kw)

    np.testing.assert_allclose(zbeta_sdfg,
                               zbeta_ref,
                               atol=1e-14,
                               rtol=1e-14,
                               err_msg="ZBETA compound expression diverges")


def test_cloudsc_zaplusb_kernel_sdfg_matches_f2py(tmp_path: Path):
    """Verbatim cloudsc.F90 line 3295 ZAPLUSB (4.5b SNOW evap): ``ZTP1**3`` integer
    exponent + 3-term FMA-chain ``RCL_APB1*ZVPICE - RCL_APB2*ZVPICE*ZTP1 +
    PAP*RCL_APB3*ZTP1**3``.
    """
    src = (_HERE / "cloudsc_zterm2_kernel.f90").read_text()
    ref = _f2py_build(src, tmp_path / "ref", "zaplusb_ref")
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src,
                      tmp_path / "sdfg",
                      name="zaplusb_kernel",
                      pipeline="hlfir-propagate-shapes",
                      entry="zaplusb_kernel").build()

    rng = np.random.default_rng(122)
    n = 32
    zvpice = np.asfortranarray(1e-3 + 1e-1 * rng.random(n, dtype=np.float64))
    ztp1 = np.asfortranarray(240.0 + 50.0 * rng.random(n, dtype=np.float64))
    pap = np.asfortranarray(2e4 + 8e4 * rng.random(n, dtype=np.float64))
    rcl_apb1, rcl_apb2, rcl_apb3 = 0.46e0, 1.6e-3, 4.9e-6

    zaplusb_ref = ref.zaplusb_kernel(zvpice, ztp1, pap, rcl_apb1, rcl_apb2, rcl_apb3)
    zaplusb_sdfg = np.zeros(n, dtype=np.float64, order="F")
    kw = dict(zvpice=zvpice,
              ztp1=ztp1,
              pap=pap,
              rcl_apb1=rcl_apb1,
              rcl_apb2=rcl_apb2,
              rcl_apb3=rcl_apb3,
              zaplusb=zaplusb_sdfg)
    kw.update(_sdfg_call_args(sdfg, dict(n=n)))
    sdfg(**kw)

    np.testing.assert_allclose(zaplusb_sdfg,
                               zaplusb_ref,
                               atol=1e-14,
                               rtol=1e-14,
                               err_msg="ZAPLUSB compound expression diverges")
