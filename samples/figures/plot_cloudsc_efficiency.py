"""FIGURE C -- cloudsc CPU parallel efficiency, gcc toolchain only.

efficiency(N) = T(1) / (N * T(N)), from per-config median runtimes, in percent.
Lanes: dace-gcc (red), original-openmp (blue), gfortran-autopar (green) -- the
artifact's Set1 assignment.  One panel per measured data layout (mode/nproma),
the block-decomposed 'nblks' layout first because that is the one that scales.

Line/marker figure rather than bars, but it keeps the artifact's palette,
legend design (above the axes, frameon=False) and log-2 thread axis.
"""
import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import polars as pl

import f2dace_style as st

REPO = Path(__file__).resolve().parents[2]
WORK_ROOT = Path(os.environ.get("WORK_ROOT", REPO / "samples" / "_work"))
DEFAULT_RUNS = [
    str(WORK_ROOT / 'meas' / 'runs'),
    str(WORK_ROOT / 'dev' / 'runs'),
]
LANES = ['dace-gcc', 'original-openmp', 'gfortran-autopar']
MARKERS = {'dace-gcc': 'o', 'original-openmp': 's', 'gfortran-autopar': '^'}
MODE_TITLE = {'nblks': 'blocked layout (N_Pt=32 × N_Blk)', 'klon': 'flat layout (N_Pt=65536 × 1 block)'}


def efficiency(df, lane, mode):
    sub = df.filter((pl.col('lane') == lane) & (pl.col('mode') == mode))
    if sub.height == 0:
        return None
    med = sub.group_by('threads').agg(pl.median('ms').alias('t')).sort('threads')
    if 1 not in med['threads'].to_list():
        return None
    t1 = med.filter(pl.col('threads') == 1)['t'][0]
    return [(int(n), 100.0 * t1 / (n * t)) for n, t in zip(med['threads'], med['t'])]


def build(df, missing, out_dir, name, footer=None):
    modes = [m for m in ('nblks', 'klon') if df.height and m in set(df['mode'].unique().to_list())]
    for m in ('nblks', 'klon'):
        if m not in modes:
            missing.note(f'efficiency: layout "{m}" not measured')
    for lane in LANES:
        if df.height == 0 or lane not in set(df['lane'].unique().to_list()):
            missing.note(f'efficiency lane "{lane}": not measured')
    if not modes:
        missing.note('efficiency: nothing to plot, no figure written')
        return None

    fig, axes = plt.subplots(1, len(modes), squeeze=False, figsize=(5.2 * len(modes), 3.0), sharey=True)
    axes = axes.flatten()
    plotted = {}
    for ax, mode in zip(axes, modes):
        for lane in LANES:
            series = efficiency(df, lane, mode)
            if not series:
                missing.note(f'efficiency: {lane} has no 1-thread point for "{mode}"')
                continue
            xs = [n for n, _ in series]
            ys = [e for _, e in series]
            plotted.setdefault(mode, []).append(lane)
            ax.plot(xs,
                    ys,
                    marker=MARKERS.get(lane, 'o'),
                    markersize=4.5,
                    linewidth=1.6,
                    color=st.LANE_COLOR[lane],
                    label=lane)
        ax.axhline(100, color='0.6', linewidth=0.9, linestyle='--', zorder=0)
        ax.set_xscale('log', base=2)
        ax.set_xticks(sorted(df['threads'].unique().to_list()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_ylim(0, 115)
        ax.set_xlabel('OpenMP threads')
        ax.set_title(MODE_TITLE.get(mode, mode), fontsize=10, pad=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[0].set_ylabel('Parallel efficiency  T₁ / (N·T_N)  [%]')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,
               st.legend_for(labels),
               loc='lower center',
               ncol=min(len(labels), 3),
               bbox_to_anchor=(0.5, -0.16),
               frameon=False)
    fig.tight_layout()
    st.save(fig,
            out_dir,
            name,
            footer=footer,
            status={
                'figure': name,
                'panels': modes,
                'lanes': plotted,
                'missing': list(missing)
            })
    return plotted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', nargs='+', default=DEFAULT_RUNS)
    ap.add_argument('--out-name', default='fig_cloudsc_eff')
    ap.add_argument('--out-dir', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--alloc', default='mimalloc')
    args = ap.parse_args()

    missing = st.Missing()
    df = st.load_runs(args.runs_dir, 'cloudsc', alloc=args.alloc, missing=missing)
    print(f'cloudsc rows loaded: {df.height}')
    build(df, missing, args.out_dir, args.out_name, footer='data: ' + ', '.join(args.runs_dir))
    if missing:
        print('MISSING:')
        for m in missing:
            print(' -', m)


if __name__ == '__main__':
    main()
