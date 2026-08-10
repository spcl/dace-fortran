"""Load the dwarf-p-cloudsc input.h5 into the exact get_inputs_physical contract (keys, dtypes, F-order shapes)."""
import functools
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]


@functools.lru_cache(maxsize=None, typed=True)
def registries():
    """tests/ is not an installed package; same sys.path pattern as tests/conftest.py and run_cloudsc_perf.py."""
    if str(REPO / "tests") not in sys.path:
        sys.path.insert(0, str(REPO / "tests"))
    from cloudsc.full import _registries
    return _registries


# Run geometry / port-structural scalars describe THIS run, not the deck: never overridden from it.
RUN_GEOMETRY_PARAMETERS = frozenset({
    'KLON', 'KLEV', 'NBLOCKS', 'NGPTOT', 'NGPTOTG', 'NPROMA', 'NGPBLKS', 'NUMOMP', 'KFLDX', 'NBETA', 'NCLV', 'NCLDQI',
    'NCLDQL', 'NCLDQR', 'NCLDQS', 'NCLDQV'
})

# Tolerates 7-significant-digit registry literals (RD=287.0597 vs the deck's 287.0596736665907).
LITERAL_RTOL = 1.0e-6


def read_h5_datasets(h5_path: Path) -> dict:
    """Read every dataset of the plain-HDF5 deck once, as numpy arrays."""
    try:
        import h5py  # deferred: optional dependency, declared under the 'samples' extra
    except ImportError:
        h5py = None
    if h5py is not None:
        with h5py.File(h5_path, "r") as f:
            return {name: np.asarray(f[name]) for name in f}
    try:
        import netCDF4  # deferred: read-only fallback, netCDF4 opens plain-HDF5 files
    except ImportError as exc:
        raise ImportError("no HDF5 reader for the dwarf-p-cloudsc deck: install h5py "
                          "(pip install 'dace-fortran[samples]'); netCDF4 also works as a read-only fallback") from exc
    ds = netCDF4.Dataset(str(h5_path), "r")
    try:
        ds.set_auto_mask(False)
        return {name: np.asarray(var[...]) for name, var in ds.variables.items()}
    finally:
        ds.close()


def find_dataset(by_lower: dict, name: str):
    """Case-insensitive lookup: flat name first, then the YRECLDP_ namelist prefix."""
    v = by_lower.get(name.lower())
    if v is None:
        v = by_lower.get("yrecldp_" + name.lower())
    return v


def scalar_like(registry_value, raw: np.ndarray):
    """Coerce a length-1 dataset to the registry value's kind; the kind picks the bridge ABI routing."""
    v = raw.ravel()[0]
    if isinstance(registry_value, np.bool_):
        return np.bool_(v != 0)
    if isinstance(registry_value, (int, np.integer)) and not isinstance(registry_value, bool):
        return int(v)
    return float(v)


def tile_columns(src: np.ndarray, klon: int, nblocks: int) -> np.ndarray:
    """Dwarf cyclic expansion: gridpoint g = ib*klon + il reads source column g mod KLON_SRC."""
    klon_src = src.shape[-1]
    grid = (np.arange(klon)[:, None] + klon * np.arange(nblocks)[None, :]) % klon_src
    gathered = src[..., grid]
    # C-order (d_n,...,d_1,KLON_SRC) is the F-order array (KLON_SRC,d_1,...,d_n); reverse leading axes.
    m = src.ndim
    axes = (m - 1, ) + tuple(range(m - 2, -1, -1)) + (m, )
    return np.array(np.transpose(gathered, axes), order='F')


def crosscheck_constants(by_lower: dict) -> tuple:
    """Split registry-vs-deck constant mismatches into (core yomcst/yoethf, YRECLDP tunables) triples."""
    core, tunables = [], []
    for name, ref in registries()._PHYSICAL_CONSTANTS.items():
        flat = by_lower.get(name.lower())
        if flat is not None:
            bucket, h5v = core, float(flat.ravel()[0])
        else:
            tun = by_lower.get("yrecldp_" + name.lower())
            if tun is None:
                raise KeyError(f"constant {name} missing from the dwarf input deck")
            bucket, h5v = tunables, float(tun.ravel()[0])
        if not math.isclose(float(ref), h5v, rel_tol=LITERAL_RTOL, abs_tol=0.0):
            bucket.append((name, float(ref), h5v))
    return core, tunables


def format_mismatches(rows: list) -> str:
    """One aligned 'name registry-value deck-value' line per mismatch."""
    return "\n".join(f"  {name:24s} registry={reg:<24.17g} h5={h5v:.17g}" for name, reg, h5v in rows)


def build_inputs(datasets: dict, klon: int, nblocks: int) -> dict:
    """Raw deck datasets -> the get_inputs_physical contract dict (no constants cross-check here)."""
    reg = registries()
    params = reg.parameters
    if params['KLON'] != klon or params['NBLOCKS'] != nblocks:
        raise ValueError(f"registry shapes were baked for KLON={params['KLON']}, NBLOCKS={params['NBLOCKS']}: "
                         f"export CLOUDSC_KLON={klon} CLOUDSC_NBLOCKS={nblocks} before the first registries import")
    by_lower = {k.lower(): v for k, v in datasets.items()}
    if int(by_lower['klev'].ravel()[0]) != params['KLEV']:
        raise ValueError("KLEV mismatch between the deck and the registry")

    inp = {}
    for name in reg.program_parameters:
        base = params[name]
        raw = None if name in RUN_GEOMETRY_PARAMETERS else find_dataset(by_lower, name)
        inp[name] = base if raw is None else scalar_like(base, raw)

    for name in reg.program_inputs:
        info: Any = reg.data[name]  # registry values are tuple-or-[shape, dtype], untyped by design
        shape, dtype = (tuple(info[0]), info[1]) if isinstance(info, list) else (tuple(info), np.float64)
        if name in ('RBETA', 'RBETAP1'):
            # Port signature takes a scalar, not the deck's (NBETA+1,) table; NSSOPT=1 never reads it.
            inp[name] = 0.5
        elif shape == (0, ):
            raw = find_dataset(by_lower, name)
            if raw is None:
                raise KeyError(f"scalar {name} missing from the dwarf input deck")
            inp[name] = float(raw.ravel()[0])
        else:
            raw = find_dataset(by_lower, name)
            if raw is None:
                # Absent from the deck (o3/u/v tendencies, tendency_loc): dwarf zero-initializes them.
                arr = np.zeros(shape, dtype=dtype, order='F')
            else:
                arr = tile_columns(raw, klon, nblocks)
                if arr.dtype != dtype:
                    arr = np.asfortranarray(arr.astype(dtype))
            if arr.shape != shape:
                raise ValueError(f"{name}: tiled shape {arr.shape} != registry shape {shape}")
            inp[name] = arr
    return inp


def load_dwarf_inputs(h5_path, klon: int, nblocks: int) -> dict:
    """Read the deck once, cross-check constants (core mismatch = hard fail), build the contract dict."""
    datasets = read_h5_datasets(Path(h5_path))
    by_lower = {k.lower(): v for k, v in datasets.items()}
    core_bad, tunable_bad = crosscheck_constants(by_lower)
    if core_bad:
        raise ValueError("dwarf deck disagrees with the registry on universal yomcst/yoethf constants "
                         "(loader mapping or regime bug, refusing to run):\n" + format_mismatches(core_bad))
    inp = build_inputs(datasets, klon, nblocks)
    if tunable_bad:
        params = registries().parameters
        switches = [
            f"{n} {params[n]}->{inp[n]}" for n in registries().program_parameters
            if n not in RUN_GEOMETRY_PARAMETERS and inp[n] != params[n]
        ]
        lines = [
            "=" * 100,
            f"DWARF vs SYNTHETIC REGIME DIVERGENCE: {len(tunable_bad)} YRECLDP tunables differ from the synthetic",
            "registry (deck wins -- this run IS the dwarf regime):",
            format_mismatches(tunable_bad),
            "switches/indices overridden from the deck: " + ", ".join(switches),
            "kept from the port, NOT the deck: KFLDX=1 (deck 0), NBETA=NBLOCKS (deck 100), RBETA/RBETAP1 scalar 0.5",
            "=" * 100,
        ]
        print("\n".join(lines), file=sys.stderr, flush=True)
    return inp
