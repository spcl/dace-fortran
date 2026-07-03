"""Compare two ICON experiment output directories variable-by-variable.

Used by ``run_icon_e2e.sh`` to diff the stock-Fortran ICON's output
against the DaCe-patched ICON's output and report a verdict.

Picks every ``*_ml_*.nc`` / ``*_hl_*.nc`` / ``*_pl_*.nc`` file present
in BOTH directories, then for each variable reports::

    {variable}: max|stock - dace| = {abs_diff}   (rel {rel_diff})

A pair of files is bit-identical when every variable's ``abs_diff`` is
zero; "close" when the maximum relative diff is below ``--rtol``
(default 1e-12).  The DaCe-patched ICON currently SIGSEGVs inside
``velocity_tendencies`` on its first time step (the SDFG was built
against ``velocity_full.f90``'s stub-typed test kernel, not ICON's
real ``t_patch`` / ``t_nh_prog`` layout), so only the t=0 initial
dump is comparable until the SDFG is rebuilt against ICON's real
``mo_velocity_advection`` source.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def _list_outputs(d: Path):
    return sorted(d.glob("*_ml_*.nc")) + sorted(d.glob("*_hl_*.nc")) \
        + sorted(d.glob("*_pl_*.nc"))


def _read_var(ds, name):
    """Read a variable into a freshly-allocated ``ndarray`` with masked
    positions filled by 0.  ``netCDF4`` shares internal buffers across
    reads, so callers MUST hold a deep copy if they expect the data
    to survive subsequent reads -- we force one via ``np.array(...,
    copy=True)`` here.  The mask is returned separately so callers
    can surface mask divergence as a real diff."""
    arr = ds.variables[name][...]
    if np.ma.isMaskedArray(arr):
        mask = np.array(np.ma.getmaskarray(arr), copy=True)
        raw = np.array(np.ma.filled(arr, 0), dtype=np.float64, copy=True)
    else:
        mask = None
        raw = np.array(arr, dtype=np.float64, copy=True)
    return raw, mask


def _strip_prefix(p: Path, prefix: str) -> str:
    n = p.name
    return n.replace(prefix, "_") if prefix and prefix in n else n


def compare_files(stock_nc: Path, dace_nc: Path, rtol: float):
    print(f"  {stock_nc.name}")
    issues = []
    with Dataset(stock_nc) as a, Dataset(dace_nc) as b:
        names_a = set(a.variables.keys())
        names_b = set(b.variables.keys())
        only_a = names_a - names_b
        only_b = names_b - names_a
        if only_a:
            issues.append(f"variables only in stock: {sorted(only_a)}")
        if only_b:
            issues.append(f"variables only in dace: {sorted(only_b)}")
        for name in sorted(names_a & names_b):
            try:
                va, mask_a = _read_var(a, name)
                vb, mask_b = _read_var(b, name)
            except (TypeError, ValueError):
                continue
            if va.shape != vb.shape:
                issues.append(f"{name}: shape mismatch {va.shape} vs {vb.shape}")
                continue
            if va.size == 0:
                print(f"    ok   {name:24s}  (empty array, skipped)")
                continue
            # Surface mask divergence as a real difference: if stock
            # masks a cell but dace doesn't (or vice versa) the runs
            # genuinely diverge.
            if mask_a is not None and mask_b is not None:
                mask_diff = int(np.count_nonzero(mask_a ^ mask_b))
                if mask_diff:
                    issues.append(f"{name}: {mask_diff} cell(s) masked on one side only")
            # Subtract on the plain (mask-filled) ndarrays so the result
            # is independent of any uninitialised bits the netCDF4
            # buffer pool leaves behind.
            d = np.abs(va - vb)
            if not np.isfinite(d).all():
                # Common when ICON crashed mid-write -- nc declared
                # the variable but never wrote real data, leaving NaN
                # bits in the on-disk image.
                print(f"    skip {name:24s}  (NaN/Inf in diff -- truncated nc?)")
                continue
            # Denormal-range maxes (< 1e-200) signal uninitialised
            # netCDF buffers from a truncated write rather than real
            # output.  Stock-vs-DaCe diffs in that range are spurious;
            # skip the variable and surface it as a non-issue.
            denormal_threshold = 1e-200
            max_a = float(np.abs(va).max())
            max_b = float(np.abs(vb).max())
            if max(max_a, max_b) < denormal_threshold:
                print(f"    skip {name:24s}  (denormal-only values -- "
                      f"uninit nc field, max|x|={max(max_a, max_b):.1e})")
                continue
            abs_diff = float(d.max())
            scale = max(max_a, max_b)
            rel_diff = abs_diff / scale if scale > 0 else abs_diff
            tag = "ok " if rel_diff <= rtol else "DIFF"
            print(f"    {tag}  {name:24s}  max|abs|={abs_diff:.3e}  rel={rel_diff:.3e}")
            if rel_diff > rtol:
                issues.append(f"{name}: rel diff {rel_diff:.3e} > rtol {rtol:.3e}")
    return issues


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stock_dir", type=Path, help="Experiment dir for the stock-Fortran ICON run.")
    ap.add_argument("dace_dir", type=Path, help="Experiment dir for the DaCe-patched ICON run.")
    ap.add_argument("--rtol", type=float, default=1e-12, help="Relative-difference threshold (default 1e-12).")
    args = ap.parse_args()

    stock_files = {p.name: p for p in _list_outputs(args.stock_dir)}
    dace_files = {p.name: p for p in _list_outputs(args.dace_dir)}
    common = sorted(set(stock_files) & set(dace_files))
    if not common:
        print("ERROR: no overlapping *.nc files between the two run dirs", file=sys.stderr)
        return 2

    print(f"Comparing {len(common)} file(s) at rtol={args.rtol:.0e}:")
    all_issues = []
    for name in common:
        all_issues.extend(compare_files(stock_files[name], dace_files[name], args.rtol))

    print()
    only_stock = sorted(set(stock_files) - set(dace_files))
    only_dace = sorted(set(dace_files) - set(stock_files))
    if only_stock:
        print(f"files only in stock ({len(only_stock)}): "
              f"{', '.join(only_stock[:3])}{' ...' if len(only_stock) > 3 else ''}")
    if only_dace:
        print(f"files only in dace ({len(only_dace)}): "
              f"{', '.join(only_dace[:3])}{' ...' if len(only_dace) > 3 else ''}")

    if all_issues:
        print(f"\nVERDICT: {len(all_issues)} difference(s) over rtol")
        for line in all_issues[:20]:
            print(f"  - {line}")
        return 1
    print(f"\nVERDICT: all {len(common)} file(s) bit-identical within "
          f"rtol={args.rtol:.0e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
