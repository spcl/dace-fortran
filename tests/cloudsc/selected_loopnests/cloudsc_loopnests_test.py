"""End-to-end tests for CloudSC physics loopnests (Vectra paper artifacts):
autoconversion_snow, ice_supersaturation_adjustment, lu_solver_microphysics,
rain_evaporation_abel_boutle, compute_saturation_values.

Each kernel: ``..._builds`` (bridge parses -> valid SDFG) and ``..._numerical``
(matches gfortran/f2py reference at KLON=KLEV=32, NCLV=5, seeded RNG inputs).

Sources are the clean cloudsc_*.f90 variants (no BIND(C)/ISO_C_BINDING/
SYSTEM_CLOCK) that both the bridge and f2py consume directly. Skips if the
artifacts dir is missing.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_LOOPNESTS_DIR = Path(__file__).parent


def _kernel_source(name: str) -> str:
    """Read a CloudSC kernel source (name = file basename minus 'cloudsc_' prefix).

    Copied from the Vectra paper artifacts and patched: normalised
    ``IF (laericeauto)`` -> ``IF (laericeauto /= 0)`` since flang rejects a
    bare-LOGICAL guard on an INTEGER(KIND=4) declared var.
    """
    src = _LOOPNESTS_DIR / f"cloudsc_{name}.f90"
    if not src.is_file():
        pytest.skip(f"missing kernel source: {src}")
    return src.read_text()


def _build(src: str, tmp: Path, *, name: str, entry: str | None = None):
    sdfg_dir = tmp / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, sdfg_dir, name=name, entry=entry).build()


# sweep sizes: small enough for speed, large enough to avoid masking vectorisation/unrolling bugs
KLON = 32
KLEV = 32
NBLKS = 5

# microphysics species count + named indices into [1..NCLV]
NCLV = 5
NCLDQV = 1
NCLDQL = 2
NCLDQI = 3
NCLDQR = 4
NCLDQS = 5

# ===========================================================================
# 1. autoconversion_snow  --  temperature- and ice-content-gated snow rate
# ===========================================================================


def test_cloudsc_autoconversion_snow_builds(tmp_path: Path):
    """The bridge parses the kernel and produces a valid SDFG."""
    src = _kernel_source("autoconversion_snow")
    sdfg = _build(src, tmp_path, name="autoconversion_snow", entry="autoconversion_snow")
    sdfg.validate()


def test_cloudsc_autoconversion_snow_numerical(tmp_path: Path):
    src = _kernel_source("autoconversion_snow")
    rng = np.random.default_rng(42)

    ZTP1 = np.asfortranarray(rng.uniform(220.0, 300.0, KLON))
    ZICECLD = np.asfortranarray(rng.uniform(0.0, 1e-3, KLON))
    PNICE = np.asfortranarray(rng.uniform(1e3, 1e5, KLON))
    # ZSOLQB shaped (KLON, NCLV, NCLV) so f2py infers klon/nclv from shape;
    # ncldqs/ncldqi are runtime indices into that dim, not bounds.
    ZSOLQB = np.asfortranarray(np.zeros((KLON, NCLV, NCLV)))

    consts = dict(rtt=273.16,
                  rlcritsnow=1.0e-4,
                  rsnowlin1=1.0e-3,
                  rsnowlin2=0.025,
                  rnice=1.0e4,
                  ptsphy=1800.0,
                  zepsec=1.0e-12,
                  laericeauto=1)

    mod = f2py_compile(src, tmp_path / "ref", "autoconv_snow_ref")
    ZSOLQB_ref = ZSOLQB.copy(order="F")
    # f2py positional sig: (kidia, kfdia, ztp1, zicecld, pnice, zsolqb, rtt,
    # ..., laericeauto, ncldqs, ncldqi); klon/nclv inferred from array shapes.
    ZSNOWAUT_ref = mod.autoconversion_snow(1, KLON, ZTP1, ZICECLD, PNICE, ZSOLQB_ref, consts["rtt"],
                                           consts["rlcritsnow"], consts["rsnowlin1"], consts["rsnowlin2"],
                                           consts["rnice"], consts["ptsphy"], consts["zepsec"], consts["laericeauto"],
                                           NCLDQS, NCLDQI)

    sdfg = _build(src, tmp_path, name="autoconversion_snow", entry="autoconversion_snow")
    ZSNOWAUT = np.zeros(KLON, dtype=np.float64, order="F")
    sdfg(kidia=1,
         kfdia=KLON,
         klon=KLON,
         nclv=NCLV,
         ztp1=ZTP1,
         zicecld=ZICECLD,
         pnice=PNICE,
         zsolqb=ZSOLQB,
         zsnowaut=ZSNOWAUT,
         ncldqs=NCLDQS,
         ncldqi=NCLDQI,
         **consts)

    np.testing.assert_allclose(ZSNOWAUT, ZSNOWAUT_ref, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(ZSOLQB, ZSOLQB_ref, rtol=1e-12, atol=1e-15)


# ===========================================================================
# 2. ice_supersaturation_adjustment
# ===========================================================================


def test_cloudsc_ice_supersaturation_adjustment_builds(tmp_path: Path):
    src = _kernel_source("ice_supersaturation_adjustment")
    sdfg = _build(src, tmp_path, name="ice_supersaturation_adjustment", entry="ice_supersaturation_adjustment")
    sdfg.validate()


def test_cloudsc_ice_supersaturation_adjustment_numerical(tmp_path: Path):
    src = _kernel_source("ice_supersaturation_adjustment")
    rng = np.random.default_rng(43)

    ZTP1 = np.asfortranarray(rng.uniform(220.0, 300.0, KLON))
    ZA = np.asfortranarray(rng.uniform(0.0, 1.0, KLON))
    ZQX_NCLDQV = np.asfortranarray(rng.uniform(0.0, 1e-3, KLON))
    ZQSICE = np.asfortranarray(rng.uniform(1e-6, 1e-3, KLON))
    ZCORQSICE = np.asfortranarray(rng.uniform(0.5, 1.5, KLON))
    ZFOKOOP = np.asfortranarray(rng.uniform(0.0, 1.0, KLON))
    ZSOLQA = np.asfortranarray(np.zeros((KLON, NCLV, NCLV)))
    ZSOLAC = np.zeros(KLON, dtype=np.float64, order="F")
    ZQXFG = np.asfortranarray(np.zeros((KLON, NCLV)))

    consts = dict(rtt=273.16, ramin=1.0e-6, rthomo=235.0, nssopt=1, rkooptau=1.0e4, ptsphy=1800.0, zepsec=1.0e-12)

    mod = f2py_compile(src, tmp_path / "ref", "ice_supersat_ref")
    ZSOLQA_ref = ZSOLQA.copy(order="F")
    ZSOLAC_ref = ZSOLAC.copy(order="F")
    ZQXFG_ref = ZQXFG.copy(order="F")
    # f2py: (kidia, kfdia, ztp1, za, zqx_ncldqv, zqsice, zcorqsice, zfokoop,
    # zsolqa, zsolac, zqxfg, rtt, ramin, rthomo, nssopt, rkooptau, ptsphy,
    # zepsec, ncldql, ncldqi, ncldqv, [klon, nclv])
    mod.ice_supersaturation_adjustment(1, KLON, ZTP1, ZA, ZQX_NCLDQV, ZQSICE, ZCORQSICE, ZFOKOOP, ZSOLQA_ref,
                                       ZSOLAC_ref, ZQXFG_ref, consts["rtt"], consts["ramin"], consts["rthomo"],
                                       consts["nssopt"], consts["rkooptau"], consts["ptsphy"], consts["zepsec"], NCLDQL,
                                       NCLDQI, NCLDQV)

    sdfg = _build(src, tmp_path, name="ice_supersaturation_adjustment", entry="ice_supersaturation_adjustment")
    ZSOLQA_sd = ZSOLQA.copy(order="F")
    ZSOLAC_sd = ZSOLAC.copy(order="F")
    ZQXFG_sd = ZQXFG.copy(order="F")
    sdfg(kidia=1,
         kfdia=KLON,
         klon=KLON,
         nclv=NCLV,
         ncldql=NCLDQL,
         ncldqi=NCLDQI,
         ncldqv=NCLDQV,
         ztp1=ZTP1,
         za=ZA,
         zqx_ncldqv=ZQX_NCLDQV,
         zqsice=ZQSICE,
         zcorqsice=ZCORQSICE,
         zfokoop=ZFOKOOP,
         zsolqa=ZSOLQA_sd,
         zsolac=ZSOLAC_sd,
         zqxfg=ZQXFG_sd,
         **consts)

    np.testing.assert_allclose(ZSOLAC_sd, ZSOLAC_ref, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(ZQXFG_sd, ZQXFG_ref, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(ZSOLQA_sd, ZSOLQA_ref, rtol=1e-10, atol=1e-15)


# ===========================================================================
# 3. lu_solver_microphysics  --  column-wise dense LU
# ===========================================================================


def test_cloudsc_lu_solver_builds(tmp_path: Path):
    src = _kernel_source("lu_solver")
    sdfg = _build(src, tmp_path, name="lu_solver_microphysics", entry="lu_solver_microphysics")
    sdfg.validate()


def test_cloudsc_lu_solver_numerical(tmp_path: Path):
    src = _kernel_source("lu_solver")
    rng = np.random.default_rng(44)

    # diagonally-dominant per-column (NCLV x NCLV) matrices -- no pivoting needed, no near-singular columns.
    ZQLHS = np.zeros((KLON, NCLV, NCLV), order="F")
    for jl in range(KLON):
        m = rng.uniform(-0.1, 0.1, (NCLV, NCLV))
        for k in range(NCLV):
            m[k, k] = NCLV + rng.uniform(0.5, 1.5)
        ZQLHS[jl, :, :] = m
    ZQLHS = np.asfortranarray(ZQLHS)
    ZQXN = np.asfortranarray(rng.uniform(-1.0, 1.0, (KLON, NCLV)))

    mod = f2py_compile(src, tmp_path / "ref", "lu_ref")
    ZQLHS_ref = ZQLHS.copy(order="F")
    ZQXN_ref = ZQXN.copy(order="F")
    # f2py: (kidia, kfdia, zqlhs, zqxn, [klon, nclv])
    mod.lu_solver_microphysics(1, KLON, ZQLHS_ref, ZQXN_ref)

    sdfg = _build(src, tmp_path, name="lu_solver_microphysics", entry="lu_solver_microphysics")
    ZQLHS_sd = ZQLHS.copy(order="F")
    ZQXN_sd = ZQXN.copy(order="F")
    sdfg(kidia=1, kfdia=KLON, klon=KLON, nclv=NCLV, zqlhs=ZQLHS_sd, zqxn=ZQXN_sd)

    np.testing.assert_allclose(ZQLHS_sd, ZQLHS_ref, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(ZQXN_sd, ZQXN_ref, rtol=1e-10, atol=1e-13)


# ===========================================================================
# 4. rain_evaporation_abel_boutle
# ===========================================================================


def test_cloudsc_rain_evaporation_builds(tmp_path: Path):
    src = _kernel_source("rain_evaporation_abel_boutle")
    sdfg = _build(src, tmp_path, name="rain_evaporation_abel_boutle", entry="rain_evaporation_abel_boutle")
    sdfg.validate()


def test_cloudsc_rain_evaporation_numerical(tmp_path: Path):
    src = _kernel_source("rain_evaporation_abel_boutle")
    rng = np.random.default_rng(45)

    ZTP1 = np.asfortranarray(rng.uniform(220.0, 300.0, KLON))
    ZQX_NCLDQV = np.asfortranarray(rng.uniform(0.0, 1e-3, KLON))
    ZA = np.asfortranarray(rng.uniform(0.0, 1.0, KLON))
    ZQSLIQ = np.asfortranarray(rng.uniform(1e-6, 1e-3, KLON))
    ZQXFG_NCLDQR = np.asfortranarray(rng.uniform(0.0, 1e-4, KLON))
    ZCOVPTOT = np.asfortranarray(rng.uniform(0.0, 1.0, KLON))
    ZCOVPCLR = np.asfortranarray(rng.uniform(0.0, 1.0, KLON))
    ZCOVPMAX = np.asfortranarray(rng.uniform(0.0, 1.0, KLON))
    ZRHO = np.asfortranarray(rng.uniform(0.5, 1.5, KLON))
    PAP = np.asfortranarray(rng.uniform(5e4, 1e5, KLON))
    ZSOLQA = np.asfortranarray(np.zeros((KLON, NCLV, NCLV)))
    ZEVAP_OUT = np.zeros(KLON, dtype=np.float64, order="F")

    consts = dict(rtt=273.16,
                  rv=461.51,
                  rd=287.06,
                  rprecrhmax=0.7,
                  rcovpmin=0.01,
                  rdensref=1.0,
                  ptsphy=1800.0,
                  zepsec=1.0e-12,
                  rcl_fac1=1.0,
                  rcl_fac2=0.0,
                  rcl_cdenom1=2.5e6,
                  rcl_cdenom2=2.4e-2,
                  rcl_cdenom3=4.6e-7,
                  rcl_ka273=2.4e-2,
                  rcl_const1r=1.0,
                  rcl_const2r=0.5,
                  rcl_const3r=0.5,
                  rcl_const4r=0.5)

    mod = f2py_compile(src, tmp_path / "ref", "rain_evap_ref")
    ZSOLQA_ref = ZSOLQA.copy(order="F")
    ZQXFG_ref = ZQXFG_NCLDQR.copy(order="F")
    ZCOVPTOT_ref = ZCOVPTOT.copy(order="F")
    ZCOVPCLR_ref = ZCOVPCLR.copy(order="F")
    # f2py: zevap_out = rain_evap(...); klon/nclv inferred from arrays; ZEVAP_OUT is intent(out) -> return.
    ZEVAP_OUT_ref = mod.rain_evaporation_abel_boutle(
        1, KLON, ZTP1, ZQX_NCLDQV, ZA, ZQSLIQ, ZQXFG_ref, ZCOVPTOT_ref, ZCOVPCLR_ref, ZCOVPMAX, ZRHO, PAP, ZSOLQA_ref,
        consts["rtt"], consts["rv"], consts["rd"], consts["rprecrhmax"], consts["rcovpmin"], consts["rdensref"],
        consts["ptsphy"], consts["zepsec"], consts["rcl_fac1"], consts["rcl_fac2"], consts["rcl_cdenom1"],
        consts["rcl_cdenom2"], consts["rcl_cdenom3"], consts["rcl_ka273"], consts["rcl_const1r"], consts["rcl_const2r"],
        consts["rcl_const3r"], consts["rcl_const4r"], NCLDQV, NCLDQR)

    sdfg = _build(src, tmp_path, name="rain_evaporation_abel_boutle", entry="rain_evaporation_abel_boutle")
    ZSOLQA_sd = ZSOLQA.copy(order="F")
    ZQXFG_sd = ZQXFG_NCLDQR.copy(order="F")
    ZCOVPTOT_sd = ZCOVPTOT.copy(order="F")
    ZCOVPCLR_sd = ZCOVPCLR.copy(order="F")
    ZEVAP_OUT_sd = ZEVAP_OUT.copy(order="F")
    sdfg(kidia=1,
         kfdia=KLON,
         klon=KLON,
         nclv=NCLV,
         ncldqv=NCLDQV,
         ncldqr=NCLDQR,
         ztp1=ZTP1,
         zqx_ncldqv=ZQX_NCLDQV,
         za=ZA,
         zqsliq=ZQSLIQ,
         zqxfg_ncldqr=ZQXFG_sd,
         zcovptot=ZCOVPTOT_sd,
         zcovpclr=ZCOVPCLR_sd,
         zcovpmax=ZCOVPMAX,
         zrho=ZRHO,
         pap=PAP,
         zsolqa=ZSOLQA_sd,
         zevap_out=ZEVAP_OUT_sd,
         **consts)

    np.testing.assert_allclose(ZEVAP_OUT_sd, ZEVAP_OUT_ref, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(ZSOLQA_sd, ZSOLQA_ref, rtol=1e-10, atol=1e-15)


# ===========================================================================
# 5. compute_saturation_values  --  saturation pressure / mixing ratio
# ===========================================================================


def test_cloudsc_saturation_calculation_builds(tmp_path: Path):
    src = _kernel_source("saturation_calculation")
    sdfg = _build(src, tmp_path, name="compute_saturation_values", entry="compute_saturation_values")
    sdfg.validate()


def test_cloudsc_saturation_calculation_numerical(tmp_path: Path):
    src = _kernel_source("saturation_calculation")
    rng = np.random.default_rng(46)

    ZTP1 = np.asfortranarray(rng.uniform(220.0, 300.0, (KLON, KLEV)))
    PAP = np.asfortranarray(rng.uniform(5e4, 1e5, (KLON, KLEV)))

    consts = dict(rtt=273.16,
                  retv=0.6078,
                  r2es=611.21,
                  r3les=17.502,
                  r3ies=22.587,
                  r4les=32.19,
                  r4ies=-0.7,
                  rtice=250.16,
                  rtwat=273.16,
                  rtwat_rtice_r=1.0 / (273.16 - 250.16))

    mod = f2py_compile(src, tmp_path / "ref", "sat_calc_ref")
    # f2py: 7 intent(out) arrays return as a tuple.
    (zfoealfa_r, zfoeewmt_r, zqsmix_r, zfoeew_r, zqsice_r, zfoeeliqt_r,
     zqsliq_r) = mod.compute_saturation_values(1, KLON, ZTP1, PAP, consts["rtt"], consts["retv"], consts["r2es"],
                                               consts["r3les"], consts["r3ies"], consts["r4les"], consts["r4ies"],
                                               consts["rtice"], consts["rtwat"], consts["rtwat_rtice_r"])
    out_ref_arrays = dict(zfoealfa=zfoealfa_r,
                          zfoeewmt=zfoeewmt_r,
                          zqsmix=zqsmix_r,
                          zfoeew=zfoeew_r,
                          zqsice=zqsice_r,
                          zfoeeliqt=zfoeeliqt_r,
                          zqsliq=zqsliq_r)

    sdfg = _build(src, tmp_path, name="compute_saturation_values", entry="compute_saturation_values")
    out_sd = {
        k: np.asfortranarray(np.zeros((KLON, KLEV)))
        for k in ("zfoealfa", "zfoeewmt", "zqsmix", "zfoeew", "zqsice", "zfoeeliqt", "zqsliq")
    }
    sdfg(kidia=1, kfdia=KLON, klon=KLON, klev=KLEV, ztp1=ZTP1, pap=PAP, **out_sd, **consts)

    for k in out_sd:
        np.testing.assert_allclose(out_sd[k],
                                   out_ref_arrays[k],
                                   rtol=1e-10,
                                   atol=1e-13,
                                   err_msg=f"output {k!r} differs")
