"""CLOUDSC CPU figure in the exact style of f2dace-artifact/analysis/f2dace_viz_cloudsc.ipynb.

Style code (helpers, construct_performance_plot_table, plot_performance, figure
assembly) is copied verbatim from that notebook; only the data loading and the
panel/lane selection are adapted to our measurements.  No code from the
artifact's transformation/optimization pipeline is used.
"""
import io
import os
import re
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
from scipy.stats import t

REPO = Path(__file__).resolve().parents[2]
WORK_ROOT = Path(os.environ.get("WORK_ROOT", REPO / "samples" / "_work"))
OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- our data ---
# Inputs, in priority order: an explicit CLOUDSC_CSVS (colon- or comma-separated), else every
# cloudsc CSV under $WORK_ROOT. Hardcoding job ids here is what made this script rot -- a run with
# a new id silently plotted the old numbers instead of failing.
def _discover_sources() -> list:
    override = os.environ.get('CLOUDSC_CSVS', '')
    if override:
        return [q for q in re.split(r'[:,]', override) if q]
    found = sorted(
        {str(q) for pat in ('cloudsc*_*.csv', 'cloudsc-*_*.csv')
         for q in WORK_ROOT.rglob(pat) if q.is_file()})
    if not found:
        raise SystemExit(
            f'no cloudsc CSVs under {WORK_ROOT}; set CLOUDSC_CSVS=<file>[,<file>...]')
    return found


CLOUDSC_SOURCES = _discover_sources()
print('cloudsc sources:')
for _s in CLOUDSC_SOURCES:
    print('  ', _s)

# klon = points per block == artifact's nproma; klon*nblocks == artifact's problem_size.
frames = [
    pl.read_csv(p).select(['lane', 'klon', 'nblocks', 'threads', 'ms']).with_columns(pl.col('ms').cast(pl.Float64))
    for p in CLOUDSC_SOURCES
]
daata = pl.concat(frames).with_columns([
    pl.col('lane').alias('label'),
    (pl.col('klon') * pl.col('nblocks')).alias('problem_size'),
    pl.col('klon').alias('nproma'),
    pl.col('ms').alias('millis'),
]).select(['label', 'threads', 'problem_size', 'nproma', 'millis'])

pl.Config.set_tbl_rows(20)
print(daata)

# Group by 'label', 'problem_size', 'threads', and 'nproma' and calculate the mean 'millis'
mean_millis_by_nproma = daata.group_by(['label', 'problem_size', 'threads',
                                        'nproma']).agg(pl.mean('millis').alias('mean_millis'))

best_configs_for_min_mean_millis = mean_millis_by_nproma.sort('mean_millis').group_by(
    ['label', 'problem_size', 'threads'], maintain_order=True).first()

# Keep rows from 'daata' only if the combination of ['label', 'problem_size', 'threads', 'nproma'] is in 'best_configs_for_min_mean_millis'
daata_best_nproma = daata.join(best_configs_for_min_mean_millis,
                               on=['label', 'problem_size', 'threads', 'nproma'],
                               how='inner')

print(daata_best_nproma)


# ------------------------------------------------- verbatim notebook cell 2 ---
def format_yaxis(value, tick_number):
    """Formats the y-axis ticks."""
    if abs(value) >= 1e9:
        return f'{value / 1e9:.0f}G'
    elif abs(value) >= 1e6:
        return f'{value / 1e6:.0f}M'
    elif abs(value) >= 1e3:
        return f'{value / 1e3:.0f}k'
    elif abs(value) >= 1:
        return f'{value:.0f}'
    else:
        return f'{value:.0f}'


def human_readable_time(mean_millis, _):
    if mean_millis == 0:
        return "0 ms"
    elif mean_millis < 1:
        micros = mean_millis * 1000
        return f"{micros:.0f} µs"
    elif mean_millis < 1000:
        return f"{mean_millis:.0f} ms"
    else:
        seconds = mean_millis / 1000
        if seconds < 10:
            return f"{seconds:.0f} s"
        else:
            return f"{seconds:.0f} s"


def format_duration(mean_millis):
    """
    Converts a duration in milliseconds to a short string representation with units.

    Args:
        mean_millis: The duration in milliseconds.

    Returns:
        A string representing the duration with an appropriate unit suffix (µs, ms, or s).
    """
    if mean_millis == 0:
        return "0 ms"
    elif mean_millis < 1:
        micros = mean_millis * 1000
        return f"{micros:.1f} µs"
    elif mean_millis < 1000:
        return f"{mean_millis:.0f} ms"
    else:
        seconds = mean_millis / 1000
        if seconds < 10:
            return f"{seconds:.2f} s"
        else:
            return f"{seconds:.1f} s"


# ------------------------------------------------- verbatim notebook cell 3 ---
# (x_col / xlabel_text are the only added parameters: they default to the
#  notebook's own 'problem_size' behaviour and are used by the thread-sweep figure.)
def construct_performance_plot_table(pl_df: pl.DataFrame,
                                     baseline_label: str,
                                     selected_labels: list,
                                     threads: int,
                                     x_col: str = 'problem_size'):

    # Filter the DataFrame
    filtered_data = pl_df.filter((pl.col('label').is_in(selected_labels))
                                 & (pl.col('threads') == threads if threads is not None else pl.lit(True)))

    # Group by 'label' and 'problem_size' and calculate mean of 'millis'
    grouped_data = filtered_data.group_by(['label', x_col]).agg(
        pl.mean('millis').alias('mean_millis'),
        pl.std('millis').alias('std_millis'),
        pl.col("millis").count().alias("n")).sort([x_col, 'label'])
    grouped_data = grouped_data.with_columns([
        pl.struct(["std_millis", "n"]).map_elements(
            lambda s: t.ppf(0.975, df=s["n"] - 1) * s["std_millis"] / (s["n"]**0.5),
            return_dtype=pl.Float64,
        ).alias("margin_of_error")
    ]).with_columns([(pl.col("mean_millis") - pl.col("margin_of_error")).alias("ci_lower"),
                     (pl.col("mean_millis") + pl.col("margin_of_error")).alias("ci_upper")])

    baseline_data = grouped_data.filter(pl.col('label') == baseline_label).rename({'mean_millis': 'baseline_millis'})

    merged_data = grouped_data.join(baseline_data.select([x_col, 'baseline_millis']), on=x_col, how='left')

    speedup_data = merged_data.with_columns(
        (pl.col('baseline_millis') / pl.col('mean_millis')).alias('speedup')).filter(
            pl.col('label').is_in(selected_labels))

    return speedup_data


def plot_performance(pl_df: pl.DataFrame,
                     baseline_label: str,
                     selected_labels: list,
                     legend_mapping: dict,
                     YLIM_LO: float,
                     YLIM_HI: float,
                     ax: plt.Axes = None,
                     xlabel: bool = False,
                     ylabel_text: str = 'Run Time (ms)',
                     annotation_unit='',
                     x_col: str = 'problem_size',
                     xlabel_text: str = 'Problem size (N_Pt × N_Lvl × N_Blk)'):

    assert ax

    # Pick color palette
    color = sns.color_palette("Set1", len(selected_labels))

    # Convert to Pandas DataFrame for Seaborn
    pandas_df = pl_df.to_pandas()

    # Only deviation from the notebook: with a single x category the default
    # group width (0.8) makes the bars span the whole axis, so keep the bar
    # aspect ratio of the reference figure.
    bar_width = 0.4 if pandas_df[x_col].nunique() == 1 else 0.8

    ax = sns.barplot(
        data=pandas_df,
        x=x_col,
        y="mean_millis",
        hue="label",
        palette=color,
        errorbar=None,  # we provide our own error bars
        capsize=0.1,
        hue_order=selected_labels,
        width=bar_width,
        ax=ax,
    )
    ax.set_yscale('log')
    ax.set_ylabel(ylabel_text)
    ax.set_ylim(top=YLIM_HI, bottom=YLIM_LO)
    yticks = sorted(ax.get_yticks())
    yticks[0] = YLIM_LO
    ax.set_yticks(yticks)
    ax.set_ylim(top=YLIM_HI, bottom=YLIM_LO)
    ax.yaxis.set_major_formatter(FuncFormatter(human_readable_time))
    handles, labels = ax.get_legend_handles_labels()
    ax.spines['top'].set_visible(False)
    ax.legend(handles=handles,
              labels=[legend_mapping.get(label, label) for label in labels],
              loc='upper center',
              frameon=False,
              ncol=len(labels),
              bbox_to_anchor=(0.5, 1.25),
              fontsize=9)
    ax.set_xlabel(xlabel_text if xlabel else '')
    xtick_labels = ax.get_xticklabels()
    ax.set_xticklabels([f"{int(label.get_text()):,}" for label in xtick_labels])
    if bar_width < 0.8:
        ax.set_xlim(-0.5, 0.5)  # keep the full category slot, don't shrink to the bars
    else:
        ax.margins(x=0.01)  # Very small horizontal margin

    # Instead of using ax.patches directly, loop with clarity:
    for patch in ax.patches:
        x = patch.get_x() + patch.get_width() / 2
        height = patch.get_height()

        # Get bar metadata from the axis tick positions
        # barplot arranges patches in hue-order inside each x tick
        # So we reconstruct the matching info:
        label_index = int(patch.get_facecolor() in color)  # not reliable

        # Better: loop over grouped data again
        for _, row in pandas_df.iterrows():
            if np.isclose(row["mean_millis"], height, atol=1e-3) and np.isclose(x,
                                                                                patch.get_x() + patch.get_width() / 2):
                ci = row["margin_of_error"]
                ax.errorbar(
                    x=x,
                    y=height,
                    yerr=[[ci], [ci]],
                    fmt='none',
                    ecolor='black',
                    capsize=4,
                    linewidth=1,
                )
                break

    # Loop through the bars and annotate them
    for p in ax.patches:
        # Get the height of the bar
        height = p.get_height()

        # Get the x-position of the bar (the center of the bar)
        overflow = height > YLIM_HI * 0.8
        x_pos = p.get_x() + p.get_width() / 2
        y_pos = YLIM_HI / 10 if overflow else height

        # Annotate with a line and text
        if annotation_unit:
            s_txt = f"{height/1000.0:.2f}s"
            ms_txt = f"{height:.1f}\nms"
            txt = s_txt if len(s_txt) < len(ms_txt) - 3 else ms_txt
            ax.annotate(
                txt,  # Annotation text (mean_millis)
                xy=(x_pos, y_pos),  # Position at the top of the bar
                xytext=(0, 5),  # Offset for the text (move a bit above the bar)
                textcoords="offset points",
                ha="center",  # Horizontal alignment of the text
                va="bottom",  # Vertical alignment (text goes above the top of the bar)
                fontsize=7,  # Font size of the annotation
                color="white" if overflow else "black",  # Text color
                bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.5")  # Box around text
            )


# --------------------------------------------------- figure (our lane set) ---
BASELINE = 'original-openmp'
# Six lanes, in three toolchain-paired groups, so each pair isolates the compiler on identical
# sources: our DaCe C++ codegen, the original Fortran, and the hand-written C rewrite.
LANES = [
    'dace-gcc', 'dace-llvm',
    'original-openmp', 'flang-openmp',
    'c-openmp', 'c-openmp-clang',
]
LEGEND = {
    'dace-gcc': 'Original Code ▶ DaCe ▶ G++ w. OpenMP',
    'dace-llvm': 'Original Code ▶ DaCe ▶ Clang++ w. OpenMP',
    'original-openmp': 'Original Code ▶ GFortran w. OpenMP',
    'flang-openmp': 'Original Code ▶ Flang w. OpenMP',
    'c-openmp': 'C Rewrite ▶ G++ w. OpenMP',
    'c-openmp-clang': 'C Rewrite ▶ Clang++ w. OpenMP',
}

num_rows, num_cols = 3, 1
fig, axes = plt.subplots(num_rows, num_cols, figsize=(1.5 * num_cols * 7, 2.4 * num_rows))
axes = axes.flatten()

for ax, threads, lo, hi, unit, last in [(axes[0], 1, 1e3, 1e4, 's', False), (axes[1], 32, 5e1, 5e3, 'ms', False),
                                        (axes[2], 72, 2e1, 5e3, 'ms', True)]:
    plot_df = construct_performance_plot_table(daata_best_nproma, BASELINE, LANES, threads)
    plot_performance(plot_df,
                     BASELINE,
                     LANES,
                     legend_mapping=LEGEND,
                     YLIM_LO=lo,
                     YLIM_HI=hi,
                     ax=ax,
                     xlabel=last,
                     ylabel_text=f"CPU ({threads} Thread)",
                     annotation_unit=unit)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'cloudsc_cpu.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'cloudsc_cpu.png'), bbox_inches='tight', dpi=200)
plt.close(fig)

# ------------------------------------------- thread sweep, identical style ---
# Our data has a single problem size, so the artifact's x sweep is degenerate;
# this companion figure puts the thread count on x with the same styling.
num_rows, num_cols = 2, 1
fig, axes = plt.subplots(num_rows, num_cols, figsize=(1.5 * num_cols * 9.5, 2.4 * num_rows))
axes = axes.flatten()

for ax, nproma, lo, hi, last in [(axes[0], 65536, 1e3, 3e4, False), (axes[1], 32, 2e1, 3e4, True)]:
    sub = daata.filter(pl.col('nproma') == nproma)
    plot_df = construct_performance_plot_table(sub, BASELINE, LANES, None, x_col='threads')
    plot_performance(plot_df,
                     BASELINE,
                     LANES,
                     legend_mapping=LEGEND,
                     YLIM_LO=lo,
                     YLIM_HI=hi,
                     ax=ax,
                     xlabel=last,
                     ylabel_text=f"CPU (N_Pt={nproma})",
                     annotation_unit='ms',
                     x_col='threads',
                     xlabel_text='OpenMP threads')

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'cloudsc_cpu_threads.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'cloudsc_cpu_threads.png'), bbox_inches='tight', dpi=200)
plt.close(fig)
print('wrote cloudsc_cpu.{pdf,png} and cloudsc_cpu_threads.{pdf,png}')
