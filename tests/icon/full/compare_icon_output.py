#!/usr/bin/env python3
"""Field-by-field comparison of two ICON output directories.

Differential verification for the ICON dycore integration test: run stock ICON
(original ``solve_nh``) and patched ICON (our SDFG binding swapped in for the
dycore) on the *same* experiment, then compare every output field. A bit-exact
match means the binding reproduces the original dycore inside a real ICON run;
any divergence is a real bug (never a tolerance to relax -- see the project's
never-use-norm_error rule).

The comparison matches output files by basename (both runs share the same
experiment name and output times), then compares each data variable. NaNs are
treated as equal to NaNs in the same slot so a deterministic all-fill field does
not read as a diff. Coordinate/dimension variables (time, height, lon/lat, ...)
are skipped -- only model state is compared.

CLI::

    python compare_icon_output.py REF_DIR TEST_DIR [--pattern '*_atm_*.nc']

Exit code 0 iff every compared field is bit-exact.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import netCDF4


def _load(var):
    """Return a variable's values as a plain ndarray, masked entries -> NaN."""
    data = var[:]
    if np.ma.isMaskedArray(data):
        data = data.filled(np.nan) if data.dtype.kind == "f" else data.filled(0)
    return np.asarray(data)


def _data_variables(ds):
    """Yield (name, var) for model-state variables, skipping coordinates/grid."""
    coords = set(ds.dimensions)
    for name, var in ds.variables.items():
        if name in coords:  # a coordinate variable (time, height, ...)
            continue
        if var.dtype.kind not in ("f", "i", "u"):  # skip char/string metadata
            continue
        yield name, var


def _bit_exact(a, b):
    """True iff a and b are identical, counting NaN==NaN (determinism)."""
    if a.dtype.kind == "f" and b.dtype.kind == "f":
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def compare_files(ref_path, test_path):
    """Compare one netcdf pair. Return list of per-variable result dicts."""
    rows = []
    with netCDF4.Dataset(ref_path) as ref, netCDF4.Dataset(test_path) as test:
        ref_vars = dict(_data_variables(ref))
        test_vars = dict(_data_variables(test))
        for name in sorted(ref_vars):
            if name not in test_vars:
                rows.append({"var": name, "status": "MISSING", "max_abs": None, "max_rel": None})
                continue
            a = _load(ref_vars[name])
            b = _load(test_vars[name])
            if a.shape != b.shape:
                rows.append({"var": name, "status": "SHAPE", "max_abs": None, "max_rel": None,
                             "detail": f"{a.shape} vs {b.shape}"})
                continue
            if _bit_exact(a, b):
                rows.append({"var": name, "status": "EXACT", "max_abs": 0.0, "max_rel": 0.0})
                continue
            fa = a.astype(np.float64)
            fb = b.astype(np.float64)
            diff = np.abs(fa - fb)
            max_abs = float(np.nanmax(diff)) if diff.size else 0.0
            denom = np.abs(fa)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.where(denom > 0.0, diff / denom, 0.0)
            max_rel = float(np.nanmax(rel)) if rel.size else 0.0
            rows.append({"var": name, "status": "DIFF", "max_abs": max_abs, "max_rel": max_rel})
    return rows


def compare_dirs(ref_dir, test_dir, pattern="*_atm_*.nc"):
    """Compare all matching output files present in both dirs.

    Returns (all_exact: bool, report: list[dict]) where each report entry is
    {file, rows} and rows is the per-variable result list from compare_files.
    Raises FileNotFoundError if the reference dir has no matching files.
    """
    ref_dir, test_dir = Path(ref_dir), Path(test_dir)
    ref_files = sorted(ref_dir.glob(pattern))
    if not ref_files:
        raise FileNotFoundError(f"no files matching {pattern!r} under {ref_dir}")
    all_exact = True
    report = []
    for ref_file in ref_files:
        test_file = test_dir / ref_file.name
        if not test_file.exists():
            report.append({"file": ref_file.name, "rows": [{"var": "*", "status": "MISSING_FILE"}]})
            all_exact = False
            continue
        rows = compare_files(ref_file, test_file)
        if any(r["status"] != "EXACT" for r in rows):
            all_exact = False
        report.append({"file": ref_file.name, "rows": rows})
    return all_exact, report


def format_report(report, verbose=False):
    """Human-readable summary; non-exact rows always shown, exact only if verbose."""
    lines = []
    for entry in report:
        bad = [r for r in entry["rows"] if r["status"] != "EXACT"]
        n_exact = sum(1 for r in entry["rows"] if r["status"] == "EXACT")
        if bad:
            lines.append(f"  {entry['file']}: {n_exact} exact, {len(bad)} DIVERGENT")
            for r in bad:
                detail = f"  ({r['detail']})" if r.get("detail") else ""
                mag = ""
                if r.get("max_abs") is not None:
                    mag = f"  max_abs={r['max_abs']:.3e} max_rel={r['max_rel']:.3e}"
                lines.append(f"      {r['var']}: {r['status']}{mag}{detail}")
        elif verbose:
            lines.append(f"  {entry['file']}: {n_exact} exact")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare two ICON output directories field-by-field.")
    ap.add_argument("ref_dir", help="reference output dir (stock ICON)")
    ap.add_argument("test_dir", help="test output dir (patched ICON / our binding)")
    ap.add_argument("--pattern", default="*_atm_*.nc", help="glob for output files to compare")
    ap.add_argument("-v", "--verbose", action="store_true", help="list exact files too")
    args = ap.parse_args(argv)

    all_exact, report = compare_dirs(args.ref_dir, args.test_dir, args.pattern)
    n_files = len(report)
    n_vars = sum(len(e["rows"]) for e in report)
    body = format_report(report, verbose=args.verbose)
    if body:
        print(body)
    verdict = "BIT-EXACT" if all_exact else "DIVERGENT"
    print(f"\n{verdict}: {n_files} files, {n_vars} fields compared "
          f"({args.ref_dir} vs {args.test_dir})")
    return 0 if all_exact else 1


if __name__ == "__main__":
    sys.exit(main())
