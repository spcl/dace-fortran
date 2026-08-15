"""QE us_exx CPU figure for this sample, in the shared f2dace-artifact style.

Reads the measurement CSVs this sample's run_*_cpu.sbatch writes and emits one thread-sweep
panel per material deck: bars (mean + 95% t CI) and violins (raw reps).

Only one deck replication factor is ever plotted -- the largest present -- because a 1x deck
and a 72x deck are different problems and averaging them together is meaningless.  That filter
is internal: it is reported on stdout, never drawn on the figure.  Lanes are whatever the CSVs
contain, coloured through f2dace_style's shared maps.
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

# Preferred hue order only: the lanes actually drawn come from the CSVs.  Pairing each DaCe
# lane with the Fortran baseline of its own compiler family keeps the families adjacent.
LANE_ORDER = ['dace-gcc', 'original-openmp', 'dace-llvm', 'flang-openmp']


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

    # run_lane_worker.sh encodes the replication factor in `inputs` (set<n>_rep<R>, bare set<n>
    # at R=1); that is the column load_runs keeps, so no reliance on the trailing deck_rep one.
    df = df.with_columns(pl.col('inputs').str.extract(r'_rep(\d+)$', 1).cast(pl.Int64).fill_null(1).alias('deck_rep'))
    # Largest R alone is the WRONG pick: an older run that reached a bigger R on only some lanes
    # silently deletes a newer complete one (newdxx job 4481020 has all four lanes at rep64, while
    # 4479255 has rep72 for gfortran/flang ONLY -- max() chose 72 and dropped both DaCe lanes with
    # no warning).  Rank candidates by lane COVERAGE first and R second, and name whatever the
    # winner still drops instead of letting a lane vanish silently.
    reps = sorted(df['deck_rep'].unique().to_list())
    cover = {r: set(df.filter(pl.col('deck_rep') == r)['lane'].unique().to_list()) for r in reps}
    deck_rep = max(reps, key=lambda r: (len(cover[r]), r))
    dropped = sorted(set(df['lane'].unique().to_list()) - cover[deck_rep])
    df = df.filter(pl.col('deck_rep') == deck_rep)
    summary = ', '.join(f'{r}x:{len(cover[r])}' for r in reps)
    print(f'  deck replication plotted: {deck_rep}x (R:lanes = {summary}; picked by lane coverage '
          f'then size; never pooled across R)')
    if dropped:
        print(f'WARNING: lane(s) {dropped} have no rows at deck_rep={deck_rep}x and are NOT drawn', file=sys.stderr)

    present = set(df['lane'].unique().to_list())
    unknown = sorted(l for l in present if l not in fs.LANE_COLOR or l not in fs.LANE_LABEL)
    if unknown:
        raise SystemExit(f'{KERNEL}: lane(s) {unknown} have no LANE_COLOR/LANE_LABEL entry in '
                         f'{Path(fs.__file__).resolve()}; add them there rather than here')
    lanes = [l for l in LANE_ORDER if l in present] + sorted(present - set(LANE_ORDER))

    decks = sorted(df['mode'].unique().to_list())
    OUT.mkdir(parents=True, exist_ok=True)

    for panel, name in ((fs.bar_panel, f'{KERNEL}_cpu'), (fs.violin_panel, f'{KERNEL}_cpu_violin')):
        fig, axes = plt.subplots(len(decks), 1, figsize=(10.5, 2.8 * len(decks)), squeeze=False)
        for ax, deck in zip(axes.flatten(), decks):
            sub = df.filter(pl.col('mode') == deck)
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
