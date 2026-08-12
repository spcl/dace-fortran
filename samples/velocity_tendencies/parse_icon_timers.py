#!/usr/bin/env python3
"""Median-per-configuration report for ``VELO_TIMER`` lines captured from an ICON-integration run.

The instrumented call sites (``tests/icon/full/velocity_full_caller.f90`` for the in-process
binding e2e, ``scripts/build_icon_dace_libs.py``'s ``velocity_tendencies_dace_icon`` wrapper for a
real patched-ICON run) print one line per ``velocity_tendencies`` invocation::

    VELO_TIMER istep=1 lvn=0 ms=1.234567890E+000 path=run_velocity_flat_c

Both sites carry ``istep``/``lvn``, so the normal path groups directly into the four
``(istep, lvn)`` configurations.  When a log carries unlabelled ``VELO_TIMER ms=...`` lines (a
call site where neither scalar was in scope) the samples are split into two clusters instead --
1-D k-means seeded at the extremes, or a sorted largest-gap split -- and the four nominal
configurations are attributed to those clusters rather than measured.

Runs under the project's measurement venv interpreter (``$PARSE_PY``, see
``run_icon_velo_timing.sh``); numpy is the only non-stdlib import.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_LINE_RE = re.compile(r"VELO_TIMER\b(?P<fields>.*)")
_KV_RE = re.compile(r"(\w+)=([^\s]+)")

#: (istep, lvn) matrix ICON's ``solve_nh`` can reach; istep in {1,2} x lvn in {0,1}.
CONFIGS: tuple[tuple[int, int], ...] = ((1, 0), (1, 1), (2, 0), (2, 1))

#: Which cluster each configuration is attributed to when the log carries no labels.  ``lvn=1``
#: skips the whole w/kinetic-energy half of the kernel, so it is the cheap arm.
_ATTRIBUTION = {(1, 0): "slow", (1, 1): "fast", (2, 0): "slow", (2, 1): "fast"}

CSV_HEADER = ("config", "count", "median_ms", "p25", "p75")


@dataclass(frozen=True)
class Sample:
    ms: float
    istep: int | None
    lvn: int | None
    path: str | None


def parse_log(text: str) -> list[Sample]:
    out: list[Sample] = []
    for line in text.splitlines():
        m = _LINE_RE.search(line)
        if m is None:
            continue
        kv = dict(_KV_RE.findall(m.group("fields")))
        if "ms" not in kv:
            continue
        try:
            ms = float(kv["ms"])
        except ValueError:
            continue
        istep = _as_int(kv.get("istep"))
        lvn = _as_int(kv.get("lvn"))
        out.append(Sample(ms=ms, istep=istep, lvn=lvn, path=kv.get("path")))
    return out


def _as_int(val: str | None) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _stats(values: np.ndarray) -> tuple[float, float, float]:
    return (float(np.median(values)), float(np.percentile(values, 25)), float(np.percentile(values, 75)))


def kmeans_1d(values: np.ndarray, k: int = 2, iters: int = 100) -> np.ndarray:
    """Deterministic 1-D Lloyd's algorithm; centroids seeded at evenly spaced order statistics."""
    if values.size == 0:
        return np.zeros(0, dtype=np.int64)
    if values.size < k:
        return np.arange(values.size, dtype=np.int64)
    ordered = np.sort(values)
    centroids = ordered[np.linspace(0, ordered.size - 1, k).round().astype(np.int64)]
    labels = np.zeros(values.size, dtype=np.int64)
    for _ in range(iters):
        labels = np.abs(values[:, None] - centroids[None, :]).argmin(axis=1)
        moved = False
        for j in range(k):
            member = values[labels == j]
            if member.size == 0:
                continue
            new = float(member.mean())
            if new != centroids[j]:
                centroids[j] = new
                moved = True
        if not moved:
            break
    order = np.argsort(centroids)
    remap = np.empty(k, dtype=np.int64)
    remap[order] = np.arange(k, dtype=np.int64)
    return remap[labels]


def gap_split_1d(values: np.ndarray) -> np.ndarray:
    """Two clusters split at the largest consecutive gap of the sorted samples."""
    if values.size < 2:
        return np.zeros(values.size, dtype=np.int64)
    ordered = np.sort(values)
    cut = int(np.argmax(np.diff(ordered)))
    threshold = 0.5 * (ordered[cut] + ordered[cut + 1])
    return (values > threshold).astype(np.int64)


def labelled_rows(samples: list[Sample], split_path: bool) -> list[tuple[str, int, float, float, float]]:
    paths = sorted({s.path for s in samples if s.path}) if split_path else []
    groups = paths if len(paths) > 1 else [None]
    rows: list[tuple[str, int, float, float, float]] = []
    for path in groups:
        subset = [s for s in samples if path is None or s.path == path]
        for istep, lvn in CONFIGS:
            vals = np.asarray([s.ms for s in subset if s.istep == istep and s.lvn == lvn], dtype=np.float64)
            name = f"istep={istep},lvn={lvn}"
            if path is not None:
                name = f"{path}:{name}"
            if vals.size == 0:
                rows.append((name, 0, float("nan"), float("nan"), float("nan")))
            else:
                rows.append((name, int(vals.size), *_stats(vals)))
    return rows


def cluster_rows(samples: list[Sample], method: str) -> tuple[list[tuple[str, int, float, float, float]], np.ndarray]:
    vals = np.asarray([s.ms for s in samples], dtype=np.float64)
    labels = kmeans_1d(vals, 2) if method == "kmeans" else gap_split_1d(vals)
    rows: list[tuple[str, int, float, float, float]] = []
    for j, name in enumerate(("cluster0(fast)", "cluster1(slow)")):
        member = vals[labels == j]
        if member.size == 0:
            rows.append((name, 0, float("nan"), float("nan"), float("nan")))
        else:
            rows.append((name, int(member.size), *_stats(member)))
    return rows, labels


def _fmt(v: float) -> str:
    return "-" if np.isnan(v) else f"{v:.6f}"


def render_table(rows: list[tuple[str, int, float, float, float]]) -> str:
    width = max((len(r[0]) for r in rows), default=6)
    head = f"{'config'.ljust(width)}  {'count':>6}  {'median_ms':>12}  {'p25':>12}  {'p75':>12}"
    lines = [head, "-" * len(head)]
    for name, count, med, p25, p75 in rows:
        lines.append(f"{name.ljust(width)}  {count:>6d}  {_fmt(med):>12}  {_fmt(p25):>12}  {_fmt(p75):>12}")
    return "\n".join(lines)


def write_csv(rows: list[tuple[str, int, float, float, float]], dest: Path | None) -> None:
    stream = dest.open("w", newline="") if dest is not None else sys.stdout
    try:
        writer = csv.writer(stream)
        writer.writerow(CSV_HEADER)
        for name, count, med, p25, p75 in rows:
            writer.writerow([
                name, count, "" if np.isnan(med) else f"{med:.6f}", "" if np.isnan(p25) else f"{p25:.6f}",
                "" if np.isnan(p75) else f"{p75:.6f}"
            ])
    finally:
        if dest is not None:
            stream.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path, help="captured stdout log ('-' for stdin)")
    ap.add_argument("--csv", type=Path, default=None, help="write the CSV here (default: stdout after the table)")
    ap.add_argument("--path", default=None, help="keep only VELO_TIMER lines whose path= matches this")
    ap.add_argument("--no-split-path", action="store_true", help="pool every path= into one set of configurations")
    ap.add_argument("--method", choices=("kmeans", "gap"), default="kmeans", help="unlabelled clustering method")
    ap.add_argument("--min-samples", type=int, default=1, help="fail when fewer than this many lines were found")
    args = ap.parse_args(argv)

    text = sys.stdin.read() if str(args.log) == "-" else args.log.read_text(errors="replace")
    samples = parse_log(text)
    if args.path is not None:
        samples = [s for s in samples if s.path == args.path]

    print(f"VELO_SAMPLES={len(samples)}")
    if len(samples) < args.min_samples:
        print(f"VELO_PARSE_EXIT=1 (found {len(samples)} VELO_TIMER lines, need {args.min_samples})")
        return 1

    labelled = all(s.istep is not None and s.lvn is not None for s in samples)
    print(f"VELO_LABELS={'present' if labelled else 'absent'}")

    if labelled:
        rows = labelled_rows(samples, split_path=not args.no_split_path)
        print(render_table(rows))
    else:
        rows, labels = cluster_rows(samples, args.method)
        print(render_table(rows))
        print("\nlabels absent -- the four configurations are ATTRIBUTED to the clusters above,")
        print("not measured per configuration:")
        by_name = {r[0]: r for r in rows}
        for istep, lvn in CONFIGS:
            arm = _ATTRIBUTION[(istep, lvn)]
            target = "cluster0(fast)" if arm == "fast" else "cluster1(slow)"
            _, count, med, _, _ = by_name[target]
            print(f"  istep={istep},lvn={lvn} -> {target}  count={count}  median_ms={_fmt(med)}")
        print(f"cluster sizes: {int((labels == 0).sum())} / {int((labels == 1).sum())}")

    if args.csv is None:
        print()
    write_csv(rows, args.csv)
    print(f"VELO_PARSE_EXIT=0 rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
