"""QE us_exx CPU figure for this sample, in the shared f2dace-artifact style.

Reads the 10-column measurement CSVs this sample's run_*_cpu.sbatch writes and emits one
thread-sweep panel per material deck: bars (mean + 95% t CI) and violins (raw reps).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f2dace_style as fs  # noqa: E402

HERE = Path(__file__).resolve().parent
KERNEL = HERE.name
REPO = HERE.parents[1]
WORK_ROOT = Path(os.environ.get('WORK_ROOT', REPO / 'samples' / '_work'))
OUT = HERE / 'figures'

LANES = ['original-openmp', 'flang-openmp']


def sources() -> list[str]:
    override = os.environ.get('QE_CSVS', '')
    if override:
        return [q for q in re.split(r'[:,]', override) if q]
    found = sorted({
        str(q)
        for root in (HERE / 'output_data', WORK_ROOT)
        for q in root.rglob(f'{KERNEL}_cpu_*.csv') if q.is_file()
    })
    if not found:
        raise SystemExit(f'no {KERNEL} CSVs under {HERE / "output_data"} or {WORK_ROOT}; '
                         f'set QE_CSVS=<file>[,<file>...]')
    return found


def main() -> int:
    paths = sources()
    print(f'{KERNEL} sources:')
    for p in paths:
        print('  ', p)
    df = fs.load_runs(paths, KERNEL, alloc=os.environ.get('QE_ALLOC'))
    if df.height == 0:
        raise SystemExit(f'no rows for kernel {KERNEL} in {paths}')

    decks = sorted(df['mode'].unique().to_list())
    lanes = [l for l in LANES if l in set(df['lane'].unique().to_list())]
    OUT.mkdir(parents=True, exist_ok=True)

    for panel, name in ((fs.bar_panel, f'{KERNEL}_cpu'), (fs.violin_panel, f'{KERNEL}_cpu_violin')):
        fig, axes = plt.subplots(len(decks), 1, figsize=(10.5, 2.8 * len(decks)), squeeze=False)
        for ax, deck in zip(axes.flatten(), decks):
            sub = df.filter((pl.col('mode') == deck) & pl.col('lane').is_in(lanes))
            nnr = sub['size_a'][0] if sub.height else 0
            ngmt = sub['size_b'][0] if sub.height else 0
            panel(ax,
                  sub,
                  lanes,
                  'threads',
                  title=f'{deck} (nnr={nnr:,}, ngm_t={ngmt:,})',
                  xlabel_text='OpenMP threads',
                  legend=(deck == decks[0]),
                  ylabel_text='Run Time')
        fig.tight_layout()
        fs.save(fig, str(OUT), name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
