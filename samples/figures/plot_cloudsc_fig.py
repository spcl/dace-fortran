"""FIGURE A -- cloudsc, in the layout of f2dace-artifact's stacked 3-row figure.

Artifact rows were CPU 1 Thread / CPU 32 Thread / GPU A100 with grouped bars
over the problem-size sweep; ours are CPU (1 Thread) / CPU (72 Threads) / GPU
GH200.  Two variants are produced from the same layout and palette:
  <prefix>_bar     -- artifact design (mean + 95% t CI + value labels)
  <prefix>_violin  -- violins over the raw reps, median labelled

Everything is data driven: rows, lanes and x categories that are not in the
CSVs are dropped and listed under MISSING instead of being invented.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import polars as pl

import f2dace_style as st

DEFAULT_RUNS = [
    '/capstor/scratch/cscs/ybudanaz/aarch64/dace-fortran-samples-meas/runs',
    '/capstor/scratch/cscs/ybudanaz/aarch64/dace-fortran-samples/runs',
]
XLABEL = 'Problem size (N_Pt × N_Lvl × N_Blk)'


def build(df, missing, out_dir, prefix, cpu_threads, footer=None):
    rows = []
    for th in cpu_threads:
        sub = df.filter((pl.col('lane').is_in(st.CPU_LANES)) & (pl.col('threads') == th))
        rows.append((f'CPU ({th} Thread{"s" if th != 1 else ""})', sub))
    rows.append(('GPU GH200', df.filter(pl.col('lane').is_in(st.GPU_LANES))))

    live = []
    for title, sub in rows:
        if sub.height == 0:
            missing.note(f'cloudsc row "{title}": no rows in data')
            continue
        sub = st.best_variant(sub)
        lanes = [l for l in st.CPU_LANES + st.GPU_LANES
                 if l in set(sub['lane'].unique().to_list())]
        live.append((title, sub, lanes))

    for lane in st.CPU_LANES + st.GPU_LANES:
        if lane not in set(df['lane'].unique().to_list()):
            missing.note(f'cloudsc lane "{lane}": not measured')
    sizes = sorted(df['problem_size'].unique().to_list()) if df.height else []
    if len(sizes) < 2:
        missing.note(f'cloudsc problem-size sweep: only {sizes} measured '
                     f'(artifact figure sweeps 4096..262144)')

    if not live:
        missing.note('cloudsc: nothing to plot, no figure written')
        return None

    for kind, panel in (('bar', st.bar_panel), ('violin', st.violin_panel)):
        num_rows, num_cols = len(live), 1
        fig, axes = plt.subplots(num_rows, num_cols, squeeze=False,
                                 figsize=(1.5 * num_cols * 7, 2.4 * num_rows))
        axes = axes.flatten()
        for i, (title, sub, lanes) in enumerate(live):
            panel(axes[i], sub, lanes, 'problem_size', title=None,
                  xlabel_text=XLABEL if i == len(live) - 1 else '',
                  xtick_map=lambda s: f'{int(float(s)):,}',
                  ylabel_text=title)
        fig.tight_layout()
        st.save(fig, out_dir, f'{prefix}_{kind}', footer=footer,
                status={'figure': f'{prefix}_{kind}',
                        'rows': [t for t, _, _ in live],
                        'lanes': sorted({l for _, _, ls in live for l in ls}),
                        'problem_sizes': sizes,
                        'missing': list(missing)})
    return live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', nargs='+', default=DEFAULT_RUNS)
    ap.add_argument('--out-prefix', default='fig_cloudsc')
    ap.add_argument('--out-dir', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--alloc', default='mimalloc')
    ap.add_argument('--cpu-threads', default='1,72')
    args = ap.parse_args()

    missing = st.Missing()
    df = st.load_runs(args.runs_dir, 'cloudsc', alloc=args.alloc, missing=missing)
    print(f'cloudsc rows loaded: {df.height}')
    footer = 'data: ' + ', '.join(args.runs_dir)
    build(df, missing, args.out_dir, args.out_prefix,
          [int(x) for x in args.cpu_threads.split(',')], footer=footer)
    if missing:
        print('MISSING:')
        for m in missing:
            print(' -', m)


if __name__ == '__main__':
    main()
