"""FV3 GFDL cloud-microphysics ``warm_rain`` kernel as an end-to-end bridge test.

A self-contained carve-out of ``warm_rain`` (warm-rain autoconversion +
accretion, rain evaporation, and rain sedimentation) from
``fv3gfs-fortran`` (``FV3/gfsphysics/physics/gfdl_cloud_microphys.F90``):
the module-global tuning constants are inlined as ``parameter`` s and the
water-saturation lookup table is built inside the entry, so a single
``warm_rain.f90`` is self-contained -- no derived-type dummies, allocatables,
or host init.  Mirrors the CLOUDSC full integration test: the SAME source
runs through the bridge and through f2py, on one deterministic column, and the
outputs are compared array-by-array.

``warm_rain`` is real(4); the bridge widens reals, so the comparison is loose.

The module-scope saturation tables (``tablew`` / ``desw``) and the lazy-init
flag (``tables_are_initialized``) are kernel-WRITTEN module globals, so the
bridge surfaces them as inout args the caller binds (a real host would bind
them from the module; the test passes the module defaults and the kernel's
``qsmith_init_w`` fills them on the first call).
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_HERE = Path(__file__).parent
_KM = 40
_ENTRY = "_QMwarm_rain_modPwarm_rain_driver"
# Length of the water-saturation lookup tables (``tablew`` / ``desw``),
# the module ``parameter qs_length`` in the source.
_QS_LENGTH = 2621

# Names of the arguments warm_rain mutates (intent inout/out), in signature
# order after the scalars -- these are what we compare between the two paths.
_INOUT = ("tz", "qv", "ql", "qr", "qi", "qs", "qg", "vtr", "m1_rain", "w1")
_IN = ("dp", "dz", "den", "denfac", "ccn", "c_praut")


def _column():
    """Deterministic, physically-plausible single-column inputs (real(4))."""
    k = np.arange(1, _KM + 1, dtype=np.float32)
    a = {
        "qv": 0.010 * (1.0 - 0.02 * k),
        "ql": 0.0010 * np.sin(0.1 * k) ** 2,
        "qr": 0.0010 * np.sin(0.1 * k) ** 2,
        "qi": np.zeros(_KM),
        "qs": np.zeros(_KM),
        "qg": np.zeros(_KM),
        "tz": 290.0 + 5.0 * np.sin(0.2 * k),
        "den": 1.2 - 0.7 * (k - 1) / (_KM - 1),
        "dp": 1000.0 + 50.0 * k,
        "ccn": np.full(_KM, 100.0),
        "c_praut": np.full(_KM, 1.0e-3),
        "vtr": np.zeros(_KM),
        "m1_rain": np.zeros(_KM),
        "w1": np.ones(_KM),
    }
    a["denfac"] = np.sqrt(1.2 / a["den"])
    a["dz"] = -(a["dp"] / (a["den"] * 9.80665))
    return {name: v.astype(np.float32) for name, v in a.items()}


def test_fv3_warm_rain(tmp_path):
    src = (_HERE / "warm_rain.f90").read_text()

    sdfg = build_sdfg(src, tmp_path / "sdfg", name="warm_rain", entry=_ENTRY).build()
    ref = f2py_compile(src, tmp_path / "ref", f"fv3_ref_{tmp_path.name}")

    scalars = dict(dt=np.float32(150.0), rh_rain=np.float32(0.7), h_var=np.float32(0.2))
    base = _column()

    # f2py reference (module subroutine).  f2py makes ``km`` optional
    # (derived from ``dp``'s shape) and returns the intent(out) ``r1``; the
    # intent(inout) arrays are updated in place.  Pass everything by keyword.
    rkw = {n: base[n].copy() for n in (*_IN, *_INOUT)}
    r1_ref = ref.warm_rain_mod.warm_rain_driver(dt=scalars["dt"], rh_rain=scalars["rh_rain"],
                                                h_var=scalars["h_var"], **rkw)

    # Bridge SDFG (km is a free symbol; inout/out arrays passed by name).
    # The module-scope saturation tables (``tablew`` / ``desw``) and the
    # lazy-init flag (``tables_are_initialized``) are kernel-WRITTEN module
    # globals, so the bridge surfaces them as inout args the caller supplies
    # (a real host binds them from the module; here we pass the module
    # defaults: empty tables + ``.false.``, and the kernel's ``qsmith_init_w``
    # fills them on this first call, matching the f2py module state).
    skw = {n: base[n].copy() for n in (*_IN, *_INOUT)}
    r1_out = np.zeros(1, dtype=np.float32)
    sdfg(km=np.int32(_KM), dt=scalars["dt"], rh_rain=scalars["rh_rain"],
         h_var=scalars["h_var"], r1=r1_out,
         tablew=np.zeros(_QS_LENGTH, dtype=np.float32, order='F'),
         desw=np.zeros(_QS_LENGTH, dtype=np.float32, order='F'),
         tables_are_initialized=np.array([False]),
         **skw)

    np.testing.assert_allclose(r1_out[0], r1_ref, rtol=1e-4, atol=1e-6)
    for name in _INOUT:
        np.testing.assert_allclose(skw[name], rkw[name], rtol=1e-4, atol=1e-6,
                                   err_msg=f"mismatch in {name}")
