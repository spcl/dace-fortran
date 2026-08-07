"""R02B05 velocity inputs -> raw stream-binary dump the standalone Fortran driver reads.

Same load + re-block path as convert_data.py (reused, not reimplemented); the only new part is
the on-disk shape: one little-endian, Fortran-order ``<name>.bin`` per array plus two text
manifests, because Fortran stream I/O reads that and cannot read an .npz.

  manifest.txt   name dtype rank d1..dr l1..lr      (dtype in f8/i4/i1, l = lower bound)
  scalars.txt    name dtype value                   (dtype in f8/i4)
  ref/<name>.bin --reference only: the DaCe kernel's outputs on these inputs, for driver verify

Scalar overrides match run_velocity_perf.FIXED_SCALARS so the Fortran lanes and the DaCe lanes
solve the same problem.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

from convert_data import convert
from run_velocity_perf import FIXED_SCALARS, OUTPUTS, VARIANT_TUS, cache_root, flat_name, git_describe

DTYPE_CODE = {np.dtype(np.float64): "f8", np.dtype(np.int32): "i4", np.dtype(np.int8): "i1"}

# The kernel reads these off config modules / p_patch scalars; the driver seeds them by name.
SCALAR_NAMES = ("nproma", "nblks_c", "nblks_e", "nblks_v", "nlev", "nlevp1", "istep", "lvn_only", "ntnd", "ldeepatmo",
                "lextra_diffu", "l_vert_nested", "ddt_vn_cor_associated", "nflatlev_jg", "nrdmax_jg", "dtime",
                "dt_linintp_ubc")


def write_bin(path: Path, arr: np.ndarray) -> None:
    """Raw Fortran-order stream.  ndarray.tofile ALWAYS writes C order, so transpose first: the
    C-order traversal of the transpose of an F-contiguous array is exactly the Fortran stream."""
    arr = np.asfortranarray(arr)
    arr.astype(arr.dtype.newbyteorder("<"), copy=False).T.tofile(path)


def write_arrays(out_dir: Path, arrays: dict) -> list[str]:
    lines = []
    for name in sorted(arrays):
        arr = np.asfortranarray(arrays[name])
        code = DTYPE_CODE.get(arr.dtype)
        if code is None:
            raise SystemExit(f"{name}: unsupported dtype {arr.dtype} (extend DTYPE_CODE and the driver)")
        write_bin(out_dir / f"{name}.bin", arr)
        lines.append(f"{name} {code} {arr.ndim} " + " ".join(str(int(s)) for s in arr.shape))
    return lines


def write_dump(out_dir: Path, arrays: dict, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lbounds = meta["lbounds"]
    lines = []
    for entry in write_arrays(out_dir, arrays):
        name = entry.split(" ", 1)[0]
        rank = arrays[name].ndim
        lb = lbounds.get(name, [1] * rank)
        lines.append(entry + " " + " ".join(str(int(x)) for x in lb))
    (out_dir / "manifest.txt").write_text("\n".join(lines) + "\n")

    scalars = []
    for name in SCALAR_NAMES:
        value = meta[name]
        if isinstance(value, float):
            scalars.append(f"{name} f8 {value!r}")
        else:
            scalars.append(f"{name} i4 {int(value)}")
    (out_dir / "scalars.txt").write_text("\n".join(scalars) + "\n")
    print(f"wrote {out_dir}: {len(lines)} arrays, {len(scalars)} scalars")


def read_dump(out_dir: Path) -> tuple[dict, dict]:
    """Read a dump back the way driver_velocity.f90 does, so the oracle sees the same bytes."""
    arrays, lbounds = {}, {}
    for line in (out_dir / "manifest.txt").read_text().splitlines():
        name, code, rank, *nums = line.split()
        rank = int(rank)
        shape, lbs = [int(x) for x in nums[:rank]], [int(x) for x in nums[rank:]]
        dtype = next(d for d, c in DTYPE_CODE.items() if c == code)
        flat = np.fromfile(out_dir / f"{name}.bin", dtype=dtype.newbyteorder("<"))
        # .copy: reshape(order="F") is a VIEW, and DaCe refuses views as kernel arguments
        arrays[name] = flat.reshape(shape, order="F").copy(order="F")
        lbounds[name] = lbs
    meta: dict = {"lbounds": lbounds}
    for line in (out_dir / "scalars.txt").read_text().splitlines():
        name, code, value = line.split()
        meta[name] = float(value) if code == "f8" else int(value)
    return arrays, meta


def write_reference(out_dir: Path, arrays: dict, meta: dict, variant: str) -> None:
    """Run the DaCe kernel on these inputs and dump its write set as the driver's oracle."""
    from run_velocity_perf import bind_call, load_or_build
    root = cache_root()
    tag = f"velocity_{variant}_gcc_{git_describe()}"
    sdfg = load_or_build(variant, root / f"{tag}.sdfgz", root / tag)
    compiled = sdfg.compile()
    call = bind_call(sdfg, arrays, meta)
    compiled(**call)
    ref_dir = out_dir / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    reverse = {flat_name(k): k for k in call}
    written = 0
    for name in OUTPUTS:
        key = reverse.get(name)
        if key is None:
            raise SystemExit(f"reference: kernel argument for output {name} not found in the SDFG arglist")
        write_bin(ref_dir / f"{name}.bin", call[key])
        written += 1
    print(f"wrote {ref_dir}: {written} reference outputs from the DaCe {variant} kernel")


def main() -> int:
    ap = argparse.ArgumentParser(description="Dump the re-blocked R02B05 velocity inputs as raw Fortran-order binary")
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data_r02b05")
    ap.add_argument("--out", type=Path, required=True, help="dump directory (created; must be on disk, not /tmp)")
    ap.add_argument("--timestep", type=int, default=1)
    ap.add_argument("--nproma", type=int, default=32)
    ap.add_argument("--reference", action="store_true", help="also dump the DaCe kernel's outputs (needs flang)")
    ap.add_argument("--variant", choices=sorted(VARIANT_TUS), default="loopexch", help="--reference TU variant")
    ap.add_argument("--force", action="store_true", help="re-dump even when the manifest is already there")
    args = ap.parse_args()

    if (args.out / "manifest.txt").is_file() and not args.force:
        print(f"{args.out}/manifest.txt exists; keeping it (pass --force to re-dump)")
    else:
        if args.out.exists():
            shutil.rmtree(args.out)
        arrays, meta = convert(args.data_dir, args.timestep, args.nproma)
        meta.update(FIXED_SCALARS)
        write_dump(args.out, arrays, meta)
    if args.reference:
        arrays, meta = read_dump(args.out)
        write_reference(args.out, arrays, meta, args.variant)
    return 0


if __name__ == "__main__":
    sys.exit(main())
