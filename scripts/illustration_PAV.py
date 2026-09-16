import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from sklearn.metrics import roc_curve
from sklearn.isotonic import IsotonicRegression

import sys
from pathlib import Path
tools_parent = Path(__file__).parent.parent.resolve()
sys.path.append(str(tools_parent))
from tools.style import apply_style, COLORS, LW as lw, LABELS, add_panel_labels, savefig

apply_style()

# Curve colours, taken from the central palette.
COLOR_RAW = COLORS[1]  # crimson
COLOR_PAV = COLORS[2]  # forestgreen
markersize = 20

# Interpolator for PR curve
# NB: this could be replaced by an element-wise linear interpolation of contingency matrices
class CustomInterpolator:
    def __init__(self, F, H, p, default_value):
        self.F = np.array(F)
        self.H = np.array(H)
        self.p = p
        self.default_value = default_value

    def __call__(self, Re):
        Re = np.array(Re)
        result = np.ones_like(Re, dtype=float) * self.default_value

        for i in range(len(self.H) - 1):
            mask = (Re > self.H[i]) & (Re < self.H[i+1])
            if self.H[i+1] == self.H[i]:
                continue  # avoid division by zero
            if self.F[i+1] == self.F[i]:
                result[mask] = self.p * Re[mask] / ((1-self.p)*self.F[i] + self.p*Re[mask])
            else:
                beta = (self.F[i+1] - self.F[i]) / (self.H[i+1] - self.H[i])
                result[mask] = self.p * Re[mask] / ((1-self.p)*self.F[i] + self.p*Re[mask] + (1-self.p)*beta*(Re[mask]-self.H[i]))

        return result
    
def PR_interpolation(p, fpr, tpr, recall, precision, min_value, add_concatenate=True):
    pr_interp = CustomInterpolator(fpr, tpr, p, default_value=min_value)
    Re = np.linspace(0, 1, 501)
    Pre_interp = pr_interp(Re)
    Re = np.array(Re[~np.isnan(Pre_interp)])
    Pre_interp = np.array(Pre_interp[~np.isnan(Pre_interp)])
    
    if add_concatenate:
        Re = np.concatenate((Re,np.array(recall)))
        arg = np.argsort(Re)
        Re = Re[arg]
        Pre_interp = np.concatenate((Pre_interp,np.array(precision)))[arg]
    return Re, Pre_interp

# ---------------------------------------------------------------------------- #

np.random.seed(3971)
p = 0.1
rho = 0.45
n = 30
cov_matrix = np.array([[1, rho], [rho, 1]])
t_Y = norm.ppf(p)
sample = mvn(mean=[0, 0], cov=cov_matrix).rvs(size=n)

x = sample[:,0]
y = (sample[:,1] > t_Y).astype(int)

# Raw ROC curve
fpr, tpr, thresholds = roc_curve(y, x)

# PAV-transformed forecast
iso_reg = IsotonicRegression().fit(x, y)
x_pav = iso_reg.predict(x)
fpr2, tpr2, thresholds2 = roc_curve(y, x_pav)

# Plot ROC and PR curves
fig, ax = plt.subplots(1,2,figsize=(2*4, 4))
ax[0].plot(fpr, tpr, color=COLOR_RAW, lw=lw, label='raw', ls='dashed')
ax[0].plot(fpr2, tpr2, color=COLOR_PAV, lw=lw, linestyle='solid', label='PAV-transformed')
ax[0].plot([0, 1], [0, 1], color='k', lw=1.5, linestyle='dotted')
ax[0].scatter(fpr, tpr, color=COLOR_RAW, s=markersize, zorder=10)  # empirical points
ax[0].scatter(fpr2, tpr2, color=COLOR_PAV, s=markersize, zorder=10)  # PAV points

ax[0].set_xlabel(LABELS['fpr'])
ax[0].set_ylabel(LABELS['tpr'])
ax[0].set_xlim(-.01,1.01)
ax[0].set_ylim(-.01,1.01)
ax[0].legend(loc='lower right',framealpha=1)
ax[0].set_aspect('equal', adjustable='box')

# Raw PR curve
precision, recall = p*tpr/(p*tpr+(1-p)*fpr), tpr

# PAV-transformed forecast
precision2, recall2 = p*tpr2/(p*tpr2+(1-p)*fpr2), tpr2

Re, Pre_interp = PR_interpolation(p, fpr, tpr, recall, precision, min_value=min(precision), add_concatenate=False)
Re2, Pre_interp2 = PR_interpolation(p, fpr2, tpr2, recall2, precision2, min_value=min(precision))


ax[1].plot(np.concatenate((Re,np.array([recall[-1]]))), np.concatenate((Pre_interp,np.array([precision[-1]]))), color=COLOR_RAW, lw=lw, label='raw', linestyle='dashed')
ax[1].plot(Re2, Pre_interp2, color=COLOR_PAV, lw=lw, linestyle='solid', label='PAV-transformed')
ax[1].hlines(y=p,xmin=0,xmax=1, colors = 'k', lw=1.5, linestyle='dotted')
ax[1].scatter(recall, precision, color=COLOR_RAW, s=markersize, zorder=10)  # empirical points
ax[1].scatter(recall2, precision2, color=COLOR_PAV, s=markersize, zorder=10)  # PAV points

ax[1].set_xlabel(LABELS['recall'])
ax[1].set_ylabel(LABELS['precision'])
ax[1].set_xlim(-.01,1.01)
ax[1].set_ylim(-.01,1.01)
ax[1].set_aspect('equal', adjustable='box')

fig.tight_layout()
add_panel_labels(ax)
savefig(fig, 'figures/illustration_PAV.png')