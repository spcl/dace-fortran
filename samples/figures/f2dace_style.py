"""Shared plotting style + data loading for the f2dace-artifact look-alike figures.

The styling helpers (human_readable_time, format_duration, the log-y bar panel
with per-bar value annotations, the t-distribution confidence intervals and the
Set1 palette) are copied from f2dace-artifact/analysis/f2dace_viz_cloudsc.ipynb
and f2dace_viz_icon.ipynb.  Nothing from the artifact's transformation or
optimization code is used here.

Data loading is schema-driven: any CSV with the columns
kernel,mode,<sizeA>,<sizeB>,threads,rep,ms,inputs,lane[,alloc]
is accepted, so old (9 column) and new (10 column, with alloc) runs both load.
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import t

# ------------------------------------------------------------------ lanes ---
_SET1 = sns.color_palette("Set1", 9)

# Red = DaCe, blue = the Fortran OpenMP/OpenACC baseline, green = the
# third-party reference (C rewrite / CUDA reference) -- as in the artifact.
LANE_COLOR = {
    'dace-gcc': _SET1[0],
    'dace-llvm': _SET1[3],
    'dace-gpu': _SET1[0],
    'original-openmp': _SET1[1],
    'openacc-gpu': _SET1[1],
    'c-openmp': _SET1[2],       # the artifact's green C-rewrite slot
    'cuda-ref': _SET1[2],
    'gfortran-autopar': _SET1[4],
    'openacc-cpu': _SET1[6],
    'flang-serial': _SET1[5],
    'gfortran-serial': _SET1[7],
    'flang-openmp': _SET1[8],
}

LANE_LABEL = {
    'dace-gcc': 'Original Code ▶ DaCe ▶ G++ w. OpenMP',
    'dace-llvm': 'Original Code ▶ DaCe ▶ Clang++ w. OpenMP',
    'dace-gpu': 'Original Code ▶ DaCe ▶ nvcc w. CUDA',
    'original-openmp': 'Original Code ▶ GFortran w. OpenMP',
    'openacc-gpu': 'Original Code ▶ NVHPC w. OpenACC',
    'gfortran-autopar': 'Original Code ▶ GFortran w. -ftree-parallelize-loops',
    'c-openmp': 'C Rewrite ▶ G++ w. OpenMP',
    'cuda-ref': 'C Rewrite ▶ nvcc w. CUDA',
    'openacc-cpu': 'Original Code ▶ NVHPC w. OpenACC (multicore)',
    'flang-serial': 'Original Code ▶ Flang (serial)',
    'gfortran-serial': 'Original Code ▶ GFortran (serial)',
    'flang-openmp': 'Original Code ▶ Flang w. OpenMP',
}

# Artifact hue order: DaCe first, the Fortran baseline second, the C rewrite third.
CPU_LANES = ['dace-gcc', 'dace-llvm', 'original-openmp', 'c-openmp',
             'gfortran-autopar', 'openacc-cpu', 'flang-serial', 'gfortran-serial']
GPU_LANES = ['dace-gpu', 'openacc-gpu', 'cuda-ref']

# ICON grid resolutions, from the artifact's RELABEL_XTICK.
GRID_LABEL = {
    'R02B03': '320 km', 'R02B04': '160 km', 'R02B05': '80 km',
    'R02B06': '40 km', 'R02B07': '20 km',
}


def palette_for(labels):
    return [LANE_COLOR.get(l, _SET1[7]) for l in labels]


def legend_for(labels):
    return [LANE_LABEL.get(l, l) for l in labels]


# ------------------------------------------------------- artifact helpers ---
def human_readable_time(ms, _=None):
    if ms == 0:
        return "0"
    micros = ms * 1000
    seconds = ms / 1000

    candidates = []
    if micros >= 1:
        candidates.append(f"{micros:.0f}µs")
    if ms >= 1:
        candidates.append(f"{ms:.0f}ms")
    if seconds >= 0.1:
        candidates.append(f"{seconds:.1f}s" if seconds < 10 else f"{seconds:.0f}s")

    return min(candidates, key=len) if candidates else "0"


def format_duration(mean_millis):
    if mean_millis == 0:
        return "0 ms"
    elif mean_millis < 1:
        return f"{mean_millis * 1000:.1f} µs"
    elif mean_millis < 1000:
        return f"{mean_millis:.0f} ms"
    seconds = mean_millis / 1000
    return f"{seconds:.2f} s" if seconds < 10 else f"{seconds:.1f} s"


def _annotation_text(height):
    """Artifact bar annotation: shortest of s / ms / us rendering."""
    s_txt = f"{height / 1000.0:.2f}s"
    ms_txt = f"{height:.1f}\nms" if height < 100 else f"{height:.0f}\nms"
    us_txt = f"{height * 1000.0:.1f}\nus"
    if height >= 1000 and len(s_txt) < len(ms_txt) - 3:
        return s_txt
    return us_txt if len(us_txt) < len(ms_txt) else ms_txt


# --------------------------------------------------------------- loading ---
class Missing(list):
    """Collects 'series X not in the data' notes for the manifest."""

    def note(self, what):
        if what not in self:
            self.append(what)
        return self


def load_runs(paths, kernel, alloc=None, missing=None):
    """Load every CSV under `paths` (dirs or globs) belonging to `kernel`.

    Returns a polars frame with normalized columns:
      lane, mode, size_a, size_b, problem_size, threads, rep, ms, inputs,
      grid, alloc
    """
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, '*.csv')))
        else:
            files += sorted(glob.glob(p))

    frames = []
    for f in files:
        try:
            df = pl.read_csv(f, infer_schema_length=10000)
        except Exception as exc:  # unreadable / empty file must not kill the figure
            print(f'  skip {os.path.basename(f)}: {exc}')
            continue
        cols = df.columns
        need = {'kernel', 'mode', 'threads', 'rep', 'ms', 'inputs', 'lane'}
        if not need.issubset(set(cols)):
            continue
        size_a, size_b = cols[2], cols[3]
        df = df.filter(pl.col('kernel') == kernel)
        if df.height == 0:
            continue
        if 'alloc' not in cols:
            df = df.with_columns(pl.lit('unspecified').alias('alloc'))
        df = df.with_columns([
            pl.col(size_a).cast(pl.Int64).alias('size_a'),
            pl.col(size_b).cast(pl.Int64).alias('size_b'),
            pl.col('ms').cast(pl.Float64),
            pl.col('threads').cast(pl.Int64),
            pl.col('rep').cast(pl.Int64),
            pl.col('inputs').cast(pl.Utf8),
            pl.col('lane').cast(pl.Utf8),
            pl.col('mode').cast(pl.Utf8),
            pl.col('alloc').cast(pl.Utf8),
        ]).select(['lane', 'mode', 'size_a', 'size_b', 'threads', 'rep', 'ms',
                   'inputs', 'alloc'])
        frames.append(df)
        print(f'  + {os.path.basename(f)}: {df.height} rows')

    if not frames:
        if missing is not None:
            missing.note(f'{kernel}: no CSV rows found under {paths}')
        return pl.DataFrame(schema={
            'lane': pl.Utf8, 'mode': pl.Utf8, 'size_a': pl.Int64, 'size_b': pl.Int64,
            'threads': pl.Int64, 'rep': pl.Int64, 'ms': pl.Float64,
            'inputs': pl.Utf8, 'alloc': pl.Utf8, 'problem_size': pl.Int64,
            'grid': pl.Utf8})

    df = pl.concat(frames).with_columns([
        (pl.col('size_a') * pl.col('size_b')).alias('problem_size'),
        pl.col('inputs').str.to_uppercase().alias('grid'),
    ])

    allocs = sorted(df['alloc'].unique().to_list())
    if alloc and alloc in allocs:
        df = df.filter(pl.col('alloc') == alloc)
    elif alloc and len(allocs) > 1:
        if missing is not None:
            missing.note(f'{kernel}: alloc={alloc} absent, kept {allocs}')
    print(f'  allocators present: {allocs}')
    return df


def best_variant(df, keys=('lane', 'problem_size', 'threads')):
    """Artifact's 'best nproma' pick, generalized to (mode, size_a) variants."""
    if df.height == 0:
        return df
    knobs = ['mode', 'size_a']
    means = df.group_by(list(keys) + knobs).agg(pl.mean('ms').alias('_mean'))
    best = means.sort('_mean').group_by(list(keys), maintain_order=True).first()
    return df.join(best.select(list(keys) + knobs), on=list(keys) + knobs, how='inner')


def ci_table(df, x_col):
    """Artifact's construct_performance_plot_table (mean + 95% t CI)."""
    g = df.group_by(['lane', x_col]).agg(
        pl.mean('ms').alias('mean_millis'),
        pl.std('ms').alias('std_millis').fill_null(0.0),
        pl.col('ms').count().alias('n'),
        pl.median('ms').alias('median_millis'),
    ).sort([x_col, 'lane'])
    return g.with_columns([
        pl.struct(['std_millis', 'n']).map_elements(
            lambda s: (t.ppf(0.975, df=s['n'] - 1) * s['std_millis'] / (s['n'] ** 0.5))
            if s['n'] > 1 else 0.0,
            return_dtype=pl.Float64,
        ).alias('margin_of_error')
    ])


def ylimits(values, headroom=2.5, floor_frac=0.9):
    v = [x for x in values if x and x > 0]
    if not v:
        return 1.0, 10.0
    lo = 10 ** math.floor(math.log10(min(v) * floor_frac))
    hi = 10 ** math.ceil(math.log10(max(v) * headroom))
    return lo, hi


# ---------------------------------------------------------------- panels ---
def _finish_axis(ax, lanes, ylo, yhi, title, xlabel_text, xtick_map=None,
                 legend=True, ylabel_text=''):
    ax.set_yscale('log')
    ax.set_ylabel(ylabel_text)
    ax.set_ylim(top=yhi, bottom=ylo)
    yticks = sorted(ax.get_yticks())
    if yticks:
        yticks[0] = ylo
        ax.set_yticks(yticks)
    ax.set_ylim(top=yhi, bottom=ylo)
    ax.yaxis.set_major_formatter(FuncFormatter(human_readable_time))
    ax.spines['top'].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    if legend and handles:
        ncol = min(len(labels), 3)
        nlines = math.ceil(len(labels) / ncol)
        ax.legend(handles=handles, labels=legend_for(labels), loc='upper center',
                  frameon=False, ncol=ncol,
                  bbox_to_anchor=(0.5, 1.19 + 0.07 * nlines), fontsize=9)
    elif ax.get_legend():
        ax.get_legend().set_visible(False)
    ax.set_xlabel(xlabel_text)
    if title:
        ax.set_title(title, pad=-6)
    if xtick_map:
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels([xtick_map(l.get_text()) for l in ax.get_xticklabels()])
    if len(ax.get_xticks()) > 1:
        ax.margins(x=0.01)  # Very small horizontal margin
    else:
        ax.set_xlim(-0.5, 0.5)


def bar_panel(ax, df, lanes, x_col, title, xlabel_text, xtick_map=None,
              legend=True, ylabel_text='', ylim=None, annotate=True):
    """Artifact bar panel: log y, Set1 hues, own t-CI error bars, value labels."""
    tab = ci_table(df, x_col)
    pdf = tab.to_pandas()
    ylo, yhi = ylim if ylim else ylimits(pdf['mean_millis'].tolist())
    single = pdf[x_col].nunique() == 1

    sns.barplot(data=pdf, x=x_col, y='mean_millis', hue='lane',
                palette=palette_for(lanes), errorbar=None, capsize=0.1,
                hue_order=lanes, width=0.4 if single else 0.8, ax=ax)

    n_x = max(pdf[x_col].nunique(), 1)
    for i, patch in enumerate(ax.patches):
        x = patch.get_x() + patch.get_width() / 2
        height = patch.get_height()
        if not height or np.isnan(height):
            continue
        row = pdf.iloc[(pdf['mean_millis'] - height).abs().argsort().iloc[0]]
        ax.errorbar(x=x, y=height, yerr=[[row['margin_of_error']], [row['margin_of_error']]],
                    fmt='none', ecolor='black', capsize=4, linewidth=1)
        if annotate:
            overflow = height > yhi * 0.8
            stagger = 5 + 11 * ((i // n_x) % 2)  # neighbouring hues alternate height
            ax.annotate(_annotation_text(height),
                        xy=(x, yhi / 10 if overflow else height), xytext=(0, stagger),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7, color="white" if overflow else "black",
                        bbox=dict(facecolor="none", edgecolor="none",
                                  boxstyle="round,pad=0.5"))

    _finish_axis(ax, lanes, ylo, yhi, title, xlabel_text, xtick_map, legend,
                 ylabel_text)
    return ylo, yhi


def violin_panel(ax, df, lanes, x_col, title, xlabel_text, xtick_map=None,
                 legend=True, ylabel_text='', ylim=None, annotate=True):
    """Same layout/palette as bar_panel, but the raw reps as violins."""
    pdf = df.to_pandas()
    med = df.group_by(['lane', x_col]).agg(pl.median('ms').alias('median_millis')).to_pandas()
    ylo, yhi = ylim if ylim else ylimits(pdf['ms'].tolist())
    single = pdf[x_col].nunique() == 1

    vwidth = 0.8
    sns.violinplot(data=pdf, x=x_col, y='ms', hue='lane',
                   palette=palette_for(lanes), hue_order=lanes, log_scale=(False, True),
                   density_norm='width', cut=0, inner='quartile', linewidth=0.6,
                   width=vwidth, ax=ax)

    if annotate:
        xcats = list(dict.fromkeys(sorted(pdf[x_col].unique())))
        n = len(lanes)
        for _, row in med.iterrows():
            if row[x_col] not in xcats or row['lane'] not in lanes:
                continue
            xi = xcats.index(row[x_col])
            li = lanes.index(row['lane'])
            off = (li - (n - 1) / 2) * vwidth / n
            ax.annotate(_annotation_text(row['median_millis']),
                        xy=(xi + off, row['median_millis']),
                        xytext=(0, 6 + 11 * (li % 2)),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=6.5, color="black")

    _finish_axis(ax, lanes, ylo, yhi, title, xlabel_text, xtick_map, legend,
                 ylabel_text)
    return ylo, yhi


# ---------------------------------------------------------------- output ---
def save(fig, out_dir, name, status=None, footer=None):
    if footer:
        if len(footer) > 110:
            footer = footer[:54] + ' … ' + footer[-54:]
        fig.text(0.5, -0.04, footer, ha='center', va='top', fontsize=6, color='0.45')
    pdf = os.path.join(out_dir, f'{name}.pdf')
    png = os.path.join(out_dir, f'{name}.png')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(png, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'wrote {pdf}\nwrote {png}')
    if status is not None:
        with open(os.path.join(out_dir, f'{name}.status.json'), 'w') as fh:
            json.dump(status, fh, indent=2)
    return pdf, png
