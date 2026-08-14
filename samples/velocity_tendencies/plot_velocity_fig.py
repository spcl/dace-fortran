"""FIGURE B -- ICON velocity_tendencies, in the layout of the artifact's
4-panel ICON figure (GridSpec 1x4, log y, grid resolution on x, titles above
the panels and one shared legend under the row).

Artifact panels were CPU 16 Thread / CPU 32 Thread / GPU A100 / GPU V100; ours
are CPU (16 Threads) / CPU (32 Threads) / CPU (72 Threads) / GPU GH200.  Two
variants share the layout and palette:
  <prefix>_bar     -- artifact design (mean + 95% t CI + value labels)
  <prefix>_violin  -- violins over the raw reps, median labelled

Panels, lanes and grid resolutions absent from the CSVs are dropped and listed
under MISSING; nothing is invented.
"""
import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import polars as pl

import f2dace_style as st

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
WORK_ROOT = Path(os.environ.get("WORK_ROOT", REPO / "samples" / "_work"))
# output_data first, as in plot_velocity_integration.py: it is the promoted, canonical copy --
# lane names collapsed to the ones f2dace_style knows, and the deck relabelled to the grid its
# edge count actually implies.  _work/meas/runs holds the same job 4421055 pre-promotion, where
# the Fortran baseline is still split three ways (original-openmp-{flang,gcc,nvhpc}, none of
# them in CPU_LANES, so all three are silently dropped) and the 160 km deck is still mislabelled
# r02b05/80 km.  Reading it too would add an 80 km column that no baseline lane appears in.
DEFAULT_RUNS = [
    str(HERE / 'output_data'),
    str(WORK_ROOT / 'dev' / 'runs'),
]
XLABEL = 'Grid Resolution'
WANT_GRIDS = ['R02B04', 'R02B06']


def build(df, missing, out_dir, prefix, cpu_threads, footer=None):
    panels = []
    for th in cpu_threads:
        sub = df.filter((pl.col('lane').is_in(st.CPU_LANES)) & (pl.col('threads') == th))
        panels.append((f'CPU ({th} Thread{"s" if th != 1 else ""})', sub))
    panels.append(('GPU GH200', df.filter(pl.col('lane').is_in(st.GPU_LANES))))

    live = []
    for title, sub in panels:
        if sub.height == 0:
            missing.note(f'velocity panel "{title}": no rows in data')
            continue
        # sort so multiple grids (r02b05, r02b06, ...) keep a stable x order
        sub = st.best_variant(sub, keys=('lane', 'grid', 'threads')).sort('grid')
        lanes = [l for l in st.CPU_LANES + st.GPU_LANES if l in set(sub['lane'].unique().to_list())]
        live.append((title, sub, lanes))

    have_grids = sorted(df['grid'].unique().to_list()) if df.height else []
    for g in WANT_GRIDS:
        if g not in have_grids:
            missing.note(f'velocity grid "{g}" ({st.GRID_LABEL.get(g, g)}): not measured')
    for lane in st.CPU_LANES + st.GPU_LANES:
        if lane not in set(df['lane'].unique().to_list()):
            missing.note(f'velocity lane "{lane}": not measured')

    if not live:
        missing.note('velocity: nothing to plot, no figure written')
        return None

    widths = [0.75 if t.startswith('GPU') else 1.0 for t, _, _ in live]
    for kind, panel in (('bar', st.bar_panel), ('violin', st.violin_panel)):
        fig = plt.figure(figsize=(3.1 * len(live) + 0.3, 2.8))
        gs = GridSpec(1, len(live), height_ratios=[1], width_ratios=widths, wspace=0.3)
        axes = [fig.add_subplot(gs[0, i]) for i in range(len(live))]
        for ax, (title, sub, lanes) in zip(axes, live):
            panel(ax,
                  sub,
                  lanes,
                  'grid',
                  title=title,
                  xlabel_text=XLABEL,
                  xtick_map=lambda s: st.GRID_LABEL.get(s, s),
                  legend=False)

        all_lanes = [l for l in st.CPU_LANES + st.GPU_LANES if any(l in ls for _, _, ls in live)]
        handles = [plt.Rectangle((0, 0), 1, 1, color=st.LANE_COLOR.get(l, '0.5')) for l in all_lanes]
        fig.legend(handles,
                   st.legend_for(all_lanes),
                   loc='lower center',
                   ncol=min(len(all_lanes), 4),
                   bbox_to_anchor=(0.5, -0.28),
                   frameon=False)
        fig.tight_layout()
        st.save(fig,
                out_dir,
                f'{prefix}_{kind}',
                footer=footer,
                status={
                    'figure': f'{prefix}_{kind}',
                    'panels': [t for t, _, _ in live],
                    'lanes': all_lanes,
                    'grids': have_grids,
                    'missing': list(missing)
                })
    return live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', nargs='+', default=DEFAULT_RUNS)
    ap.add_argument('--out-prefix', default='fig_velocity')
    ap.add_argument('--out-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures'))
    ap.add_argument('--alloc', default='mimalloc')
    ap.add_argument('--kernel', default='velocity_tendencies')
    ap.add_argument('--cpu-threads', default='16,32,72')
    args = ap.parse_args()

    missing = st.Missing()
    df = st.load_runs(args.runs_dir, args.kernel, alloc=args.alloc, missing=missing)
    print(f'velocity rows loaded: {df.height}')
    build(df,
          missing,
          args.out_dir,
          args.out_prefix, [int(x) for x in args.cpu_threads.split(',')],
          footer='data: ' + ', '.join(args.runs_dir))
    if missing:
        print('MISSING:')
        for m in missing:
            print(' -', m)


if __name__ == '__main__':
    main()
