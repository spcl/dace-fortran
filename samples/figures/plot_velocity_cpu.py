"""ICON velocity_tendencies CPU figure in the exact style of
f2dace-artifact/analysis/f2dace_viz_icon.ipynb.

Style code (helpers, construct_performance_plot_table, plot_performance, the
GridSpec figure assembly and the shared bottom legend) is copied verbatim from
that notebook; only the data loading and the panel/lane selection are adapted
to our measurements.  No code from the artifact's transformation/optimization
pipeline is used.
"""
import os

import polars as pl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from scipy.stats import t

ROOT = '/capstor/scratch/cscs/ybudanaz/aarch64'
OUT = os.path.join(ROOT, 'probe/f2dace_plots')

# ---------------------------------------------------------------- our data ---
VELOCITY_SOURCES = [
    f'{ROOT}/dace-fortran-samples-meas/runs/velocity-gcc_30c14cd_4391146.csv',
    f'{ROOT}/dace-fortran-samples-meas/runs/velocity-llvm_30c14cd_4391146.csv',
    f'{ROOT}/dace-fortran-samples/runs/velocity_baselines_4387049.csv',
    f'{ROOT}/dace-fortran-samples/runs/velocity_baselines_4387058.csv',
]

frames = [
    pl.read_csv(p).select(['lane', 'mode', 'threads', 'ms', 'inputs']).with_columns(
        pl.col('ms').cast(pl.Float64)) for p in VELOCITY_SOURCES
]
daata = pl.concat(frames).with_columns([
    pl.col('lane').alias('label'),
    pl.col('inputs').str.to_uppercase().alias('grid'),
    pl.col('ms').alias('millis'),
]).select(['label', 'grid', 'threads', 'mode', 'millis'])

# The loop-exchange variant plays the role of the artifact's 'nproma' knob: pick
# the best variant per (label, grid, threads) exactly as the cloudsc notebook
# picks the best nproma.
best_mode = daata.group_by(['label', 'grid', 'threads', 'mode']).agg(
    pl.mean('millis').alias('mean_millis')
).sort('mean_millis').group_by(['label', 'grid', 'threads'], maintain_order=True).first()
daata = daata.join(best_mode, on=['label', 'grid', 'threads', 'mode'], how='inner')

pl.Config.set_tbl_rows(20)
print(daata)


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

def human_readable_time(ms, _):
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
#  notebook's own 'grid' behaviour and are used by the thread-sweep figure.)
def construct_performance_plot_table(
    pl_df: pl.DataFrame, baseline_label: str, selected_labels: list, threads: int,
    x_col: str = 'grid'):

    # Filter the DataFrame
    filtered_data = pl_df.filter(
        (pl.col('label').is_in(selected_labels)) &
        (pl.col('threads') == threads if threads is not None else pl.lit(True))
    )

    # Group by 'label' and 'grid' and calculate mean of 'millis'
    grouped_data = filtered_data.group_by(['label', x_col]).agg(
        pl.mean('millis').alias('mean_millis'),
        pl.std('millis').alias('std_millis').fill_null(0.0),
        pl.col("millis").count().alias("n")
    ).sort([x_col, 'label'])
    grouped_data = grouped_data.with_columns([
        pl.struct(["std_millis", "n"]).map_elements(
            lambda s: t.ppf(0.975, df=s["n"] - 1) * s["std_millis"] / (s["n"] ** 0.5),
            return_dtype=pl.Float64,
        ).alias("margin_of_error")
    ]).with_columns([
        (pl.col("mean_millis") - pl.col("margin_of_error")).alias("ci_lower"),
        (pl.col("mean_millis") + pl.col("margin_of_error")).alias("ci_upper")
    ])

    baseline_data = grouped_data.filter(pl.col('label') == baseline_label).rename({'mean_millis': 'baseline_millis'})

    merged_data = grouped_data.join(
        baseline_data.select([x_col, 'baseline_millis']),
        on=x_col,
        how='left'
    )

    speedup_data = merged_data.with_columns(
        (pl.col('baseline_millis') / pl.col('mean_millis')).alias('speedup')
    ).filter(pl.col('label').is_in(selected_labels))

    return speedup_data

def plot_performance(
    pl_df: pl.DataFrame, baseline_label: str, selected_labels: list,
    legend_mapping: dict, YLIM_LO: float, YLIM_HI: float,
    ax: plt.Axes = None, xlabel: bool = False,
    ylabel_text: str = 'Run Time (ms)', annotation_unit='',
    x_col: str = 'grid', xlabel_text: str = 'Grid Resolution'):

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
    ax.set_ylabel('')
    ax.set_ylim(top=YLIM_HI, bottom=YLIM_LO)
    yticks = sorted(ax.get_yticks())
    yticks[0] = YLIM_LO
    ax.set_yticks(yticks)
    ax.set_ylim(top=YLIM_HI, bottom=YLIM_LO)
    ax.yaxis.set_major_formatter(FuncFormatter(human_readable_time))
    handles, labels = ax.get_legend_handles_labels()
    ax.spines['top'].set_visible(False)
    ax.legend(
        handles=handles,
        labels=[legend_mapping.get(label, label) for label in labels],
        loc='upper center', frameon=False, ncol=len(labels),
        bbox_to_anchor=(0.5, 1.25), fontsize=9)
    ax.set_xlabel(xlabel_text)
    ax.set_title(ylabel_text, pad=-6)
    RELABEL_XTICK = {
        'R02B03': '320 km',
        'R02B04': '160 km',
        'R02B05': '80 km',
        'R02B06': '40 km',
        'R02B07': '20 km',
    }
    xtick_labels = ax.get_xticklabels()
    ax.set_xticklabels([f"{RELABEL_XTICK.get(label.get_text(), label.get_text())}" for label in xtick_labels])
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
            if np.isclose(row["mean_millis"], height, atol=1e-3) and np.isclose(x, patch.get_x() + patch.get_width() / 2):
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
        y_pos = YLIM_HI/10 if overflow else height

        # Annotate with a line and text
        if annotation_unit:
          s_txt = f"{height/1000.0:.2f}s"
          ms_txt = f"{height:.1f}\nms" if height < 100 else f"{height:.0f}\nms"
          us_txt = f"{height*1000.0:.1f}\nus"
          txt = us_txt if len(us_txt) < len(ms_txt) else ms_txt
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
BASELINE = 'gfortran-autopar'
LANES = ['dace-gcc', 'dace-llvm', 'gfortran-autopar', 'openacc-cpu']
LEGEND = {
    'dace-gcc': 'Original Code ▶ DaCe ▶ G++ w. OpenMP',
    'dace-llvm': 'Original Code ▶ DaCe ▶ Clang++ w. OpenMP',
    'gfortran-autopar': 'Original Code ▶ GFortran w. Autopar',
    'openacc-cpu': 'Original Code ▶ NVHPC w. OpenACC (multicore)',
}
LEGEND_FIG = [
    'Original Code ▶ DaCe ▶ G++ w. OpenMP',
    'Original Code ▶ DaCe ▶ Clang++ w. OpenMP',
    'Original Code ▶ GFortran w. Autopar',
    'Original Code ▶ NVHPC w. OpenACC (multicore)',
]

fig = plt.figure(figsize=(12.5, 2.8))
gs = GridSpec(1, 4, height_ratios=[1], width_ratios=[1, 1, 0.75, 0.75], wspace=0.3)

axs = [fig.add_subplot(gs[0, i]) for i in range(4)]

for ax, threads, lo, hi in [(axs[0], 1, 1e1, 1e3), (axs[1], 16, 1e0, 1e3),
                            (axs[2], 32, 1e0, 1e3), (axs[3], 72, 1e0, 1e3)]:
    plot_df = construct_performance_plot_table(daata, BASELINE, LANES, threads)
    plot_performance(plot_df, BASELINE, LANES, legend_mapping=LEGEND,
                     YLIM_LO=lo, YLIM_HI=hi, ax=ax, xlabel=True,
                     ylabel_text=f"CPU ({threads} Thread)", annotation_unit='ms')

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, LEGEND_FIG,
           loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.2), frameon=False)
for ax in axs:
    ax.legend().set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'velocity_cpu.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'velocity_cpu.png'), bbox_inches='tight', dpi=200)
plt.close(fig)

# ------------------------------------------- thread sweep, identical style ---
# Our data has a single grid (R02B05), so the artifact's x sweep is degenerate;
# this companion figure puts the thread count on x with the same styling.
fig = plt.figure(figsize=(12.5, 2.8))
gs = GridSpec(1, 1, height_ratios=[1], width_ratios=[1])
ax = fig.add_subplot(gs[0, 0])

plot_df = construct_performance_plot_table(daata, BASELINE, LANES, None, x_col='threads')
plot_performance(plot_df, BASELINE, LANES, legend_mapping=LEGEND,
                 YLIM_LO=1e0, YLIM_HI=5e2, ax=ax, xlabel=True,
                 ylabel_text="CPU (R02B05 / 80 km)", annotation_unit='ms',
                 x_col='threads', xlabel_text='OpenMP threads')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, LEGEND_FIG,
           loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.2), frameon=False)
ax.legend().set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'velocity_cpu_threads.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'velocity_cpu_threads.png'), bbox_inches='tight', dpi=200)
plt.close(fig)
print('wrote velocity_cpu.{pdf,png} and velocity_cpu_threads.{pdf,png}')
