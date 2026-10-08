"""Shared harness for the AES graupel tests: gfortran reference build, SDFG/reference call wrappers and deterministic physical input columns.

Both graupel variants (the original Muphys layout under ``aes_graupel/`` and the per-column fused layout under ``aes_graupel_fused/``) export ``mo_aes_graupel::graupel_run`` with the same interface, so one C-bound caller (``graupel_caller.f90``) and one set of input columns serve both.
"""

import ctypes
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AES = HERE / "aes_graupel"
FUSED = HERE / "aes_graupel_fused"
CALLER = HERE / "graupel_caller.f90"
ENTRY = "mo_aes_graupel::graupel_run"

# Dependency modules shared by both variants; the graupel module itself differs.
DEP_SOURCES = [AES / "mo_kind.f90", AES / "mo_physical_constants.f90", AES / "mo_aes_thermo.f90"]
ORIGINAL_SOURCE = AES / "mo_aes_graupel.f90"
FUSED_SOURCE = FUSED / "graupel.f90"

IN_2D = ("dz", "p", "rho")
INOUT_2D = ("t", "qv", "qc", "qi", "qr", "qs", "qg")
OUT_1D = ("prr_gsp", "pri_gsp", "prs_gsp", "prg_gsp", "pre_gsp")
# Argument order of ``run_graupel_c`` after the integer/dt scalars.
_ARRAY_ORDER = (
    "dz",
    "t",
    "p",
    "rho",
    "qv",
    "qc",
    "qi",
    "qr",
    "qs",
    "qg",
    "qnc",
    "prr_gsp",
    "pri_gsp",
    "prs_gsp",
    "prg_gsp",
    "pflx",
    "pre_gsp",
)
# Fields written by graupel_run (everything the comparison must cover).
RESULT_FIELDS = (*INOUT_2D, "pflx", *OUT_1D)

TMELT = 273.15
RD = 287.04
RV = 461.51


@dataclass(frozen=True)
class Config:
    """Scalar arguments of one ``graupel_run`` call."""

    ivstart: int
    ivend: int
    kstart: int
    dt: float = 30.0


def compile_reference(out_dir: Path, graupel_sources: list[Path]) -> ctypes.CDLL:
    """Build the multi-file gfortran reference into one ``.so`` and bind ``run_graupel_c``.

    Sources compile in dependency order with ``cwd=out_dir`` so every USE finds the previous ``.mod``; ``-J`` is avoided because other tests may leave flang-built ``iso_c_binding.mod`` in shared temp paths that gfortran rejects. ``-O0 -ffp-contract=off`` keeps the reference free of FMA contraction.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    so_path = out_dir / "libgraupel_ref.so"
    flags = ["-O0", "-fno-fast-math", "-ffp-contract=off", "-fPIC", "-ffree-line-length-none"]
    objects = []
    for src in [*DEP_SOURCES, *graupel_sources, CALLER]:
        obj = out_dir / (src.stem + ".o")
        subprocess.run(["gfortran", *flags, "-c", str(src), "-o", str(obj)], check=True, cwd=str(out_dir))
        objects.append(str(obj))
    subprocess.run(["gfortran", "-shared", "-fPIC", "-o", str(so_path), *objects], check=True, cwd=str(out_dir))
    lib = ctypes.CDLL(str(so_path))
    lib.run_graupel_c.restype = None
    lib.run_graupel_c.argtypes = [ctypes.c_int] * 5 + [ctypes.c_double] + [ctypes.c_void_p] * len(_ARRAY_ORDER)
    return lib


def zero_outputs(fields: dict[str, np.ndarray]) -> None:
    """Zero the OUT arrays: ``pflx`` is only written for ``k >= MINVAL(kmin)`` so untouched entries must start identical on both sides."""
    for name in ("pflx", *OUT_1D):
        fields[name][...] = 0.0


def run_reference(lib: ctypes.CDLL, fields: dict[str, np.ndarray], cfg: Config) -> None:
    nvec, ke = fields["t"].shape
    lib.run_graupel_c(
        nvec,
        ke,
        cfg.ivstart,
        cfg.ivend,
        cfg.kstart,
        ctypes.c_double(cfg.dt),
        *[fields[n].ctypes.data for n in _ARRAY_ORDER],
    )


def sdfg_args(fields: dict[str, np.ndarray], cfg: Config) -> dict[str, np.ndarray | np.generic]:
    """Keyword arguments of one SDFG ``graupel_run`` call; the arrays are ``fields``' own buffers."""
    nvec, ke = fields["t"].shape
    return {
        "nvec": np.int32(nvec),
        "ke": np.int32(ke),
        "ivstart": np.int32(cfg.ivstart),
        "ivend": np.int32(cfg.ivend),
        "kstart": np.int32(cfg.kstart),
        "dt": np.float64(cfg.dt),
        **{n: fields[n] for n in _ARRAY_ORDER},
    }


def run_sdfg(sdfg, fields: dict[str, np.ndarray], cfg: Config) -> None:
    sdfg(**sdfg_args(fields, cfg))


def copy_fields(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {n: a.copy(order="F") for n, a in fields.items()}


def _sat_ratio_qv(temp: np.ndarray, rho: np.ndarray, ratio: float, over_ice: bool) -> np.ndarray:
    """Vapour content at ``ratio`` times saturation (Tetens fit; only used to pose inputs, the scheme computes its own saturation)."""
    a, b = (22.46, 0.55) if over_ice else (17.269, 35.86)
    es = 611.2 * np.exp(a * (temp - TMELT) / (temp - b))
    return ratio * es / (rho * RV * temp)


SCENARIOS = ("cold_ice_snow", "mixed_phase_rain", "melting_layer", "warm_rain_evap", "warm_dry_noop")


def physical_columns(ke: int = 20, dz0: float = 250.0, repeats: int = 1) -> dict[str, np.ndarray]:
    """One column per scenario (column ``iv`` runs ``SCENARIOS[iv % len(SCENARIOS)]``; ``repeats`` tiles the set), level 1 at the top and ``ke`` at the surface.

    A standard-lapse temperature profile (6.5 K/km) with a surface temperature per scenario sets the freezing level; hydrometeors are placed in layer bands (1-based ``k``) chosen so that each process family has material to act on. ``qnc`` differs per column so the ``qnc(ivstart)``-for-every-column quirk is observable.
    """
    nvec = len(SCENARIOS) * repeats
    f = {n: np.zeros((nvec, ke), order="F") for n in (*IN_2D, *INOUT_2D, "pflx")}
    f["qnc"] = np.linspace(1.0e8, 3.0e8, nvec)
    for n in OUT_1D:
        f[n] = np.zeros(nvec)
    k = np.arange(1, ke + 1)
    for iv in range(nvec):
        name = SCENARIOS[iv % len(SCENARIOS)]
        z = (ke - k + 0.5) * dz0
        tsfc = {
            "cold_ice_snow": 262.0,
            "mixed_phase_rain": 283.0,
            "melting_layer": 288.0,
            "warm_rain_evap": 295.0,
            "warm_dry_noop": 290.0,
        }[name]
        temp = tsfc - 6.5e-3 * z
        p = 100000.0 * np.exp(-z / 8000.0)
        rho = p / (RD * temp)
        f["dz"][iv] = dz0 * (1.0 + 0.1 * np.sin(k))  # mild, deterministic thickness variation
        f["t"][iv], f["p"][iv], f["rho"][iv] = temp, p, rho

        def band(lo, hi):
            return (k >= lo) & (k <= hi)

        if name == "cold_ice_snow":
            f["qv"][iv] = _sat_ratio_qv(temp, rho, 0.9, True)
            sup = band(2, 9)  # supersaturated vapour, no ice yet: heterogeneous nucleation (T < tmelt - 25)
            f["qv"][iv, sup] = _sat_ratio_qv(temp, rho, 1.3, True)[sup]
            f["qi"][iv, band(8, 16)] = 3.0e-5
            f["qs"][iv, band(8, ke)] = 1.5e-4
            f["qg"][iv, band(12, ke)] = 5.0e-5
        elif name == "mixed_phase_rain":
            f["qv"][iv] = _sat_ratio_qv(temp, rho, 1.0, False)
            f["qc"][iv, band(5, 12)] = 5.0e-4
            f["qr"][iv, band(7, ke)] = 2.0e-4
            f["qi"][iv, band(5, 10)] = 2.0e-5
            f["qs"][iv, band(5, 12)] = 1.0e-4
        elif name == "melting_layer":
            f["qv"][iv] = _sat_ratio_qv(temp, rho, 0.95, False)
            f["qs"][iv, band(6, 16)] = 2.0e-4
            f["qg"][iv, band(6, 16)] = 1.0e-4
            f["qr"][iv, band(12, ke)] = 5.0e-5
        elif name == "warm_rain_evap":
            f["qv"][iv] = _sat_ratio_qv(temp, rho, 0.5, False)
            f["qc"][iv, band(10, 14)] = 4.0e-4
            f["qr"][iv, band(10, ke)] = 3.0e-4
        else:  # warm_dry_noop: hydrometeors and vapour zero, T far above freezing
            pass
    return f


def assert_match(ref: dict[str, np.ndarray], got: dict[str, np.ndarray], rtol: float, atol: float) -> None:
    """Every INOUT and OUT array of the SDFG run matches the gfortran reference."""
    for name in RESULT_FIELDS:
        np.testing.assert_allclose(got[name], ref[name], rtol=rtol, atol=atol, err_msg=f"{name} differs from gfortran")


def assert_families_fire(inputs: dict[str, np.ndarray], ref: dict[str, np.ndarray]) -> None:
    """Each process family leaves a non-trivial footprint in the reference result of ``physical_columns(repeats=1)`` run over the full range, so a no-op SDFG cannot match it."""
    d = {n: ref[n] - inputs[n] for n in INOUT_2D}
    col = SCENARIOS.index

    cold = col("cold_ice_snow")
    # deposition + heterogeneous nucleation: vapour drawn down, ice phase grows, latent heating
    assert d["qv"][cold].min() < -1e-6
    assert d["qi"][cold, 1:7].min() > 1e-7  # nucleated ice in the ice-free supersaturated upper levels
    assert d["t"][cold].max() > 1e-3

    mixed = col("mixed_phase_rain")
    # cloud depletion (riming/accretion) and rain freezing to graupel in mixed-phase cloud, with latent heating
    assert d["qc"][mixed].min() < -2e-5
    assert d["qr"][mixed].min() < -1e-4 and d["qg"][mixed].max() > 1e-4
    assert d["t"][mixed].max() > 0.05

    melt = col("melting_layer")
    above_freezing = inputs["t"][melt] > TMELT
    # snow/graupel melt to rain at T > tmelt, cooling the layer
    assert (d["qs"][melt] + d["qg"][melt])[above_freezing].min() < -1e-6
    assert d["qr"][melt][above_freezing].max() > 1e-6
    assert d["t"][melt][above_freezing].min() < -1e-3

    warm = col("warm_rain_evap")
    # rain evaporates in sub-saturated warm air: vapour up, temperature down, rain down
    assert d["qv"][warm].max() > 1e-6
    assert d["qr"][warm].min() < -1e-6
    assert d["t"][warm].min() < -1e-3

    # sedimentation reaches the surface (k = ke) with the right species and an energy flux
    active = [cold, mixed, melt, warm]
    assert ref["pflx"][active, -1].min() > 0.0
    assert ref["pri_gsp"][cold] + ref["prs_gsp"][cold] + ref["prg_gsp"][cold] > 0.0
    assert ref["prr_gsp"][warm] > 0.0 and ref["prr_gsp"][melt] > 0.0
    assert (ref["pre_gsp"][active] != 0.0).all()

    noop = col("warm_dry_noop")
    for n in d:
        assert not d[n][noop].any(), f"no-op column changed {n}"
    assert not ref["pflx"][noop].any()
    assert not any(ref[n][noop] for n in OUT_1D)


def random_state(rng: np.random.Generator) -> tuple[dict[str, np.ndarray], Config]:
    """Random but bounded atmospheric state and call configuration (any species mix, T 215-300 K, random ``ivstart/ivend/kstart``, ``dt``), for fuzzing the whole scheme."""
    nvec, ke = int(rng.integers(1, 9)), int(rng.integers(1, 25))
    z = (ke - np.arange(1, ke + 1) + 0.5) * 300.0
    f = {n: np.zeros((nvec, ke), order="F") for n in (*IN_2D, *INOUT_2D, "pflx")}
    f["qnc"] = rng.uniform(1e7, 5e8, nvec)
    for n in OUT_1D:
        f[n] = np.zeros(nvec)
    f["t"][:] = rng.uniform(215.0, 300.0, (nvec, 1)) + rng.normal(0.0, 3.0, (nvec, ke))
    f["p"][:] = 1e5 * np.exp(-z / 8000.0) * rng.uniform(0.9, 1.1, (nvec, ke))
    f["rho"][:] = f["p"] / (RD * f["t"])
    f["dz"][:] = rng.uniform(50.0, 600.0, (nvec, ke))
    for name, scale in (("qv", 1e-2), ("qc", 1e-3), ("qi", 1e-4), ("qr", 1e-3), ("qs", 5e-4), ("qg", 5e-4)):
        f[name][:] = scale * rng.random((nvec, ke)) * (rng.random((nvec, ke)) < 0.6)
    ivstart = int(rng.integers(1, nvec + 1))
    cfg = Config(
        ivstart, int(rng.integers(ivstart, nvec + 1)), int(rng.integers(1, ke + 1)), float(rng.choice([10, 30, 120]))
    )
    return f, cfg
