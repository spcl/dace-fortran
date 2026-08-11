"""Stack the artifact's reference figure above ours for a fidelity check."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

OUT = '/capstor/scratch/cscs/ybudanaz/aarch64/probe/f2dace_plots'

PAIRS = [
    ('reference/ref_cloudsc.png', 'cloudsc_cpu.png', 'cloudsc_cpu_threads.png',
     'compare_cloudsc.png',
     ['artifact reference (f2dace_viz_cloudsc.ipynb)',
      'ours, artifact axes (x = problem size, single size measured)',
      'ours, thread sweep in the same style']),
    ('reference/ref_icon_velocity.png', 'velocity_cpu.png', 'velocity_cpu_threads.png',
     'compare_velocity.png',
     ['artifact reference (f2dace_viz_icon.ipynb)',
      'ours, artifact axes (x = grid resolution, single grid measured)',
      'ours, thread sweep in the same style']),
]

for ref, mine, mine2, out, titles in PAIRS:
    imgs = [mpimg.imread(os.path.join(OUT, p)) for p in (ref, mine, mine2)]
    heights = [im.shape[0] / im.shape[1] for im in imgs]
    fig, axes = plt.subplots(3, 1, figsize=(11, 11 * sum(heights) + 1.2),
                             height_ratios=heights)
    for ax, im, title in zip(axes, imgs, titles):
        ax.imshow(im)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, out), dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)
