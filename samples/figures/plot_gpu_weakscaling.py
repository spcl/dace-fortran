"""FIGURE D -- GPU scaling over the cloudsc problem-size sweep.

cuda-ref (green, the artifact's C-rewrite colour) vs dace-gpu (red), median
runtime on a log y axis over problem sizes 4096..262144, in the artifact's
palette and legend design.  Runs entirely off whatever GPU lanes are present:
with no GPU rows in the CSVs it writes no figure and reports MISSING instead of
inventing points.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import polars as pl

import f2dace_style as st

DEFAULT_RUNS = [
    '/capstor/scratch/cscs/ybudanaz/aarch64/dace-fortran-samples-meas/runs',
    '/capstor/scratch/cscs/ybudanaz/aarch64/dace-fortran-samples/runs',
]
LANES = ['dace-gpu', 'cuda-ref', 'openacc-gpu']
MARKERS = {'dace-gpu': 'o', 'cuda-ref': '^', 'openacc-gpu': 's'}
WANT_SIZES = [4096, 8192, 16384, 32768, 65536, 131072, 262144]


def build(df, missing, out_dir, name, footer=None):
    gpu = df.filter(pl.col('lane').is_in(LANES)) if df.height else df
    for lane in LANES:
        if gpu.height == 0 or lane not in set(gpu['lane'].unique().to_list()):
            missing.note(f'GPU scaling lane "{lane}": not measured')
    empty = gpu.height == 0
    if empty:
        missing.note('GPU scaling: no GPU rows in the data, empty scaffold written')
    gpu = gpu if empty else st.best_variant(gpu)
    have = sorted(gpu['problem_size'].unique().to_list()) if not empty else WANT_SIZES
    for s in WANT_SIZES:
        if s not in have:
            missing.note(f'GPU scaling: problem size {s} not measured')

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    lanes_plotted = []
    for lane in LANES:
        sub = gpu.filter(pl.col('lane') == lane)
        if sub.height == 0:
            continue
        med = sub.group_by('problem_size').agg(
            pl.median('ms').alias('t')).sort('problem_size')
        ax.plot(med['problem_size'], med['t'], marker=MARKERS.get(lane, 'o'),
                markersize=5, linewidth=1.6, color=st.LANE_COLOR[lane], label=lane)
        lanes_plotted.append(lane)

    if empty:
        ax.text(0.5, 0.5, 'no GPU runs measured yet', transform=ax.transAxes,
                ha='center', va='center', fontsize=11, color='0.55')
        ax.set_ylim(1, 1000)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(have)
    if empty:
        ax.set_xlim(have[0] / 1.3, have[-1] * 1.3)
    ax.tick_params(axis='x', labelrotation=30)
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.yaxis.set_major_formatter(FuncFormatter(st.human_readable_time))
    ax.set_xlabel('Problem size (N_Pt × N_Lvl × N_Blk)')
    ax.set_ylabel('GPU GH200')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    if empty:
        handles = [plt.Line2D([], [], color=st.LANE_COLOR[l], marker=MARKERS[l])
                   for l in LANES]
        labels = LANES
    ax.legend(handles, st.legend_for(labels), loc='upper center', frameon=False,
              ncol=min(len(labels), 3), bbox_to_anchor=(0.5, 1.26), fontsize=9)
    fig.tight_layout()
    st.save(fig, out_dir, name, footer=footer,
            status={'figure': name, 'lanes': lanes_plotted, 'problem_sizes': have,
                    'missing': list(missing)})
    return lanes_plotted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', nargs='+', default=DEFAULT_RUNS)
    ap.add_argument('--out-name', default='fig_gpu_weakscaling')
    ap.add_argument('--out-dir', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--alloc', default='mimalloc')
    ap.add_argument('--kernel', default='cloudsc')
    args = ap.parse_args()

    missing = st.Missing()
    df = st.load_runs(args.runs_dir, args.kernel, alloc=args.alloc, missing=missing)
    print(f'{args.kernel} rows loaded: {df.height}')
    build(df, missing, args.out_dir, args.out_name,
          footer='data: ' + ', '.join(args.runs_dir))
    if missing:
        print('MISSING:')
        for m in missing:
            print(' -', m)


if __name__ == '__main__':
    main()
