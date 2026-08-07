"""R02B05 serde data -> re-blocked compressed .npz keyed by the harness mesh_buffers flat names."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from velocity_data_build import vd

# Harness refin-ctrl window, values from tests/icon/ocean/_ocean_e2e.py
# (_REFIN_CTRL_LBOUND / _REFIN_CTRL_EXTENT); binding shims declare extent 33.
REFIN_LBOUND = -16
REFIN_EXTENT = 33
REFIN_NAMES = tuple(f"p_patch_{space}_{end}_{part}" for space in ("cells", "edges", "verts") for end in ("start", "end")
                    for part in ("index", "block"))

SCALAR_META = ("nproma", "nblks_c", "nblks_e", "nblks_v", "valid_cells", "valid_edges", "valid_verts", "istep",
               "lvn_only", "ntnd", "ldeepatmo", "lextra_diffu", "dtime", "dt_linintp_ubc")


def embed_refin(arr: np.ndarray, lbound: int) -> np.ndarray:
    """Place the recorded (lbound..lbound+n-1) window into the harness 33-slot window, clamp-replicating outside."""
    lo = lbound - REFIN_LBOUND
    if lo < 0 or lo + arr.shape[0] > REFIN_EXTENT:
        raise ValueError(f"refin window lbound={lbound} extent={arr.shape[0]} exceeds [{REFIN_LBOUND}, "
                         f"{REFIN_LBOUND + REFIN_EXTENT - 1}]")
    out = np.empty(REFIN_EXTENT, dtype=np.int32)
    out[:lo] = arr[0]
    out[lo:lo + arr.shape[0]] = arr
    out[lo + arr.shape[0]:] = arr[-1]
    return out


def convert(data_dir: Path, timestep: int, nproma: int) -> tuple[dict, dict]:
    loaded = vd.load(str(data_dir), timestep)
    lbounds = {str(k): tuple(int(x) for x in v) for k, v in loaded["lbounds"].items()}
    r = vd.reblock(loaded, nproma)
    del loaded

    nlev = int(r["p_prog_vn"].shape[1])
    nlevp1 = int(r["p_prog_w"].shape[1])
    nblks_e = int(r["nblks_e"])
    arrays = {str(k): v for k, v in r.items() if isinstance(v, np.ndarray) and str(k) not in ("nflatlev", "nrdmax")}

    mask = "p_patch_cells_decomp_info_owner_mask"
    arrays[mask] = arrays[mask].astype(np.int8)
    for name in REFIN_NAMES:
        arrays[name] = embed_refin(arrays[name], lbounds[name][0])
        lbounds[name] = (REFIN_LBOUND, )
    # Never-associated in the artifact set (patched out by download_data.sh); zeros are
    # inert since ddt_vn_cor_associated=0 / l_vert_nested=0 guard every access.  Shapes
    # per velocity_tendencies_binding.json (cf. optarena velocity_tendencies.py initialize()).
    arrays["p_diag_ddt_vn_cor_pc"] = np.zeros((nproma, nlev, nblks_e, 3), dtype=np.float64, order="F")
    arrays["p_diag_vn_ie_ubc"] = np.zeros((nproma, 2, nblks_e), dtype=np.float64, order="F")
    arrays["p_diag_max_vcfl_dyn"] = np.asarray([float(r["p_diag_max_vcfl_dyn"])], dtype=np.float64)
    lbounds["p_diag_ddt_vn_cor_pc"] = (1, 1, 1, 1)
    lbounds["p_diag_vn_ie_ubc"] = (1, 1, 1)
    lbounds["p_diag_max_vcfl_dyn"] = (1, )

    meta: dict[str, object] = {
        name: int(r[name]) if name != "dtime" and name != "dt_linintp_ubc" else float(r[name])
        for name in SCALAR_META
    }
    meta.update({
        "nlev": nlev,
        "nlevp1": nlevp1,
        "l_vert_nested": 0,
        "ddt_vn_cor_associated": 0,
        "nflatlev_jg": int(r["nflatlev"][0]),
        "nrdmax_jg": int(r["nrdmax"][0]),
        "timestep": timestep,
        "lbounds": {
            k: list(v)
            for k, v in lbounds.items()
        },
    })
    return arrays, meta


def run_selftest() -> int:
    print(vd.selftest())
    window = embed_refin(np.asarray([5, 6, 7], dtype=np.int32), lbound=-1)
    assert window.shape == (REFIN_EXTENT, ) and window[14] == 5 and list(window[15:18]) == [5, 6, 7] \
        and window[18] == 7, "embed_refin window misplaced"
    print("convert_data selftest OK: refin window embed at lbound -16 verified")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert + re-block the R02B05 velocity serde data to one .npz")
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data_r02b05")
    ap.add_argument("--timestep", type=int, default=1)
    ap.add_argument("--nproma", type=int, default=32)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--selftest", action="store_true", help="synthetic parse+reblock check, no data needed")
    args = ap.parse_args()
    if args.selftest:
        return run_selftest()

    out = args.out or Path(f"velocity_r02b05_nproma{args.nproma}.npz")
    arrays, meta = convert(args.data_dir, args.timestep, args.nproma)
    np.savez_compressed(out, meta=json.dumps(meta), **arrays)
    print(f"wrote {out}: {len(arrays)} arrays, nproma={meta['nproma']} nblks_c/e/v="
          f"{meta['nblks_c']}/{meta['nblks_e']}/{meta['nblks_v']} nlev={meta['nlev']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
