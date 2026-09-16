import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path
tools_parent = Path(__file__).parent.parent.resolve()
sys.path.append(str(tools_parent))
from tools.style import apply_style, LABELS, savefig

apply_style()

fig, ax = plt.subplots(1, 1, figsize=(4, 4))

p = 0.25  # event frequency

# Frequency-bias isolines. Every LABEL_EVERY-th line carries its FB value
# inline (rotated along the line, masking it like a contour label).
FB_VALUES = np.linspace(0, 4, 31)
LABEL_EVERY = 2
ISO_KW = dict(color='gray', lw=1)
LABEL_KW = dict(fontsize=8, color='gray')

# The axes use an equal aspect over equal ranges, so the on-screen angle of a
# line equals the angle of its data-space slope.
slope = -(1-p)/p
angle = np.degrees(np.arctan(slope))

for i, FB in enumerate(FB_VALUES):
    x_start, x_end = max(0, (FB-1)*p/(1-p)), min(1, FB*p/(1-p))
    x = np.linspace(x_start, x_end, 100)
    y = FB - (1-p)/p * x
    ax.plot(x, y, **ISO_KW)

    # Skip degenerate lines (FB = 0 collapses to the origin).
    if i % LABEL_EVERY or x_end <= x_start:
        continue

    x_mid = 0.5 * (x_start + x_end)
    y_mid = FB - (1-p)/p * x_mid
    ax.text(x_mid, y_mid, f'FB={FB:.1f}', rotation=angle, rotation_mode='anchor',
            ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', pad=1),
            **LABEL_KW)

ax.set_xlabel(LABELS['fpr'])
ax.set_ylabel(LABELS['tpr'])
ax.set_xlim(-.01, 1.01)
ax.set_ylim(-.01, 1.01)
ax.set_aspect('equal', adjustable='box')

fig.tight_layout()
savefig(fig, 'figures/constant-FB_ROC.png')