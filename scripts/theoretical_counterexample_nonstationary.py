import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal

import sys
from pathlib import Path
tools_parent = Path(__file__).parent.parent.resolve()
sys.path.append(str(tools_parent))
from tools.utils import compute_roc_pr
from tools.style import apply_style, LW as lw, LABELS, COLORS, savefig, add_panel_labels

apply_style()

mu_s = np.array([[0,0],[0,0]])
sigma_s = np.array([[1,1],[2,1]])

mu_Ys = np.array([0,.5])
sigma_Ys = np.array([1,.5])

P_Y = 0.95
t_Y = np.array([norm.ppf(P_Y), norm.ppf(P_Y)])*sigma_Ys + mu_Ys
rho_list = [.85,.8]


def solve_dcp_pl_thresholds(mvn, z_y, sigma, c_values, p_y=P_Y, z_min=-8.0, z_max=8.0, grid_size=4001):
    """Fast monotone inversion of c(z) on a dense z-grid."""
    z_grid = np.linspace(z_min, z_max, grid_size)
    z_y_vec = np.full_like(z_grid, z_y)
    joint_grid = mvn.cdf(np.column_stack((z_grid, z_y_vec)))
    c_grid = (norm.cdf(z_grid) - joint_grid) / (1 - p_y) + joint_grid / p_y

    # Numerical noise can break strict monotonicity; sort/unique for safe interpolation.
    order = np.argsort(c_grid)
    c_sorted = c_grid[order]
    z_sorted = z_grid[order]
    c_unique, idx_unique = np.unique(c_sorted, return_index=True)
    z_unique = z_sorted[idx_unique]

    c_clipped = np.clip(c_values, c_unique[0], c_unique[-1])
    z_x = np.interp(c_clipped, c_unique, z_unique)
    return sigma * z_x


def compute_confusion_from_zx(z_x, z_y, mvn):
    """Vectorized contingency entries for a full threshold vector."""
    z_y_vec = np.full_like(z_x, z_y)
    cdf_xy = mvn.cdf(np.column_stack((z_x, z_y_vec)))
    cdf_x = norm.cdf(z_x)
    cdf_y = norm.cdf(z_y)

    a = 1 - cdf_x - cdf_y + cdf_xy
    b = cdf_y - cdf_xy
    c = cdf_x - cdf_xy
    d = cdf_xy
    return a, b, c, d


n=1000
n_models = 2
n_locations = 2
confusion_dict = {
    "GraphCast's parameterization": [
        np.full((n_locations, n_models, n), np.nan, dtype=float),
        np.full((n_locations, n_models, n), np.nan, dtype=float),
        np.full((n_locations, n_models, n), np.nan, dtype=float),
        np.full((n_locations, n_models, n), np.nan, dtype=float)],
    "Location-scale parameterization": [
        np.full((n_locations, n_models, n), np.nan, dtype=float),
        np.full((n_locations, n_models, n), np.nan, dtype=float),
        np.full((n_locations, n_models, n), np.nan, dtype=float),
        np.full((n_locations, n_models, n), np.nan, dtype=float)],
    "DCP-FB": [
        np.full((n_locations, n_models, n), np.nan, dtype=float),
        np.full((n_locations, n_models, n), np.nan, dtype=float),
        np.full((n_locations, n_models, n), np.nan, dtype=float),
        np.full((n_locations, n_models, n), np.nan, dtype=float)],
    "DCP-PL": [
        np.full((n_locations, n_models, n), np.nan, dtype=float),
        np.full((n_locations, n_models, n), np.nan, dtype=float),
        np.full((n_locations, n_models, n), np.nan, dtype=float),
        np.full((n_locations, n_models, n), np.nan, dtype=float)],
}

forecastnames = "AB"
for m,rho in enumerate(rho_list):
    cov_matrix = np.array([[1, rho], [rho, 1]])
    mvn = multivariate_normal(mean=[0, 0], cov=cov_matrix)
    for s in [0,1]:
        z_y = (t_Y[s] - mu_Ys[s]) / sigma_Ys[s]
        for agg_strat in confusion_dict.keys():
            if agg_strat == "GraphCast's parameterization":
                gains = np.linspace(.01, 10, n)
                t_X = mu_s[s,m] + (t_Y[s]-mu_s[s,m]) / gains
                
            elif agg_strat == "Location-scale parameterization":
                gains = np.linspace(-3, 3, n)
                t_X = t_Y[s] - gains*sigma_s[s,m]
            elif agg_strat == "DCP-FB":
                u = np.linspace(0.0001, .9999, n)
                t_X = sigma_s[s,m] * norm.ppf(1-u)
            elif agg_strat == "DCP-PL":
                c_values = np.linspace(0.0, 2.0, n)
                t_X = solve_dcp_pl_thresholds(
                    mvn=mvn,
                    z_y=z_y,
                    sigma=sigma_s[s, m],
                    c_values=c_values,
                )
                
            Z_X = (t_X - mu_s[s,m])/sigma_s[s,m]

            a, b, c, d = compute_confusion_from_zx(Z_X, z_y, mvn)
            confusion_dict[agg_strat][0][s,m] = a
            confusion_dict[agg_strat][1][s,m] = b
            confusion_dict[agg_strat][2][s,m] = c
            confusion_dict[agg_strat][3][s,m] = d

# ---------------------------------------------------------------------------- #
#                                     PLOTS                                    #
# ---------------------------------------------------------------------------- #
nrows = 2
ncols = 3
fig_roc, ax_roc = plt.subplots(nrows,ncols, figsize=(4*ncols, nrows*4))
fig_pr, ax_pr = plt.subplots(nrows,ncols, figsize=(4*ncols, nrows*4))
fig_roc.tight_layout()
fig_pr.tight_layout()
colors = [COLORS[1], COLORS[2]]

for l in range(n_locations):
    agg_strat = 'DCP-FB'
    fpr1, tpr1, recall1, precision1 = compute_roc_pr(tps=confusion_dict[agg_strat][0][l,0,:], fps=confusion_dict[agg_strat][1][l,0,:], fns=confusion_dict[agg_strat][2][l,0,:], tns=confusion_dict[agg_strat][3][l,0,:])
    fpr2, tpr2, recall2, precision2 = compute_roc_pr(tps=confusion_dict[agg_strat][0][l,1,:], fps=confusion_dict[agg_strat][1][l,1,:], fns=confusion_dict[agg_strat][2][l,1,:], tns=confusion_dict[agg_strat][3][l,1,:])

    condition = np.concatenate((np.logical_and(fpr1 <= fpr2, tpr1 >= tpr2),[True]))

    for m in range(n_models):
        label = f"Forecast {forecastnames[m]}"
        fpr, tpr, recall, precision = compute_roc_pr(tps=confusion_dict[agg_strat][0][l,m,:], fps=confusion_dict[agg_strat][1][l,m,:], fns=confusion_dict[agg_strat][2][l,m,:], tns=confusion_dict[agg_strat][3][l,m,:])
        fpr, tpr = np.concatenate((fpr,[1])), np.concatenate((tpr,[1]))
        ax_roc[0,l].plot(fpr, tpr, label=f'{label}', color = colors[m], linewidth=lw)
        ax_roc[0,l].plot(fpr[~condition], tpr[~condition], color = colors[m], ls='None', marker='o', markersize=2)
        ax_pr[0,l].plot(recall, precision, label=f'{label}', color = colors[m], linewidth=lw)
    ax_roc[0,l].set_xlabel(LABELS['fpr'])
    ax_roc[0,l].set_ylabel(LABELS['tpr'])
    ax_pr[0,l].set_xlabel(LABELS['recall'])
    ax_pr[0,l].set_ylabel(LABELS['precision'])
    for i, ax in enumerate([ax_roc,ax_pr]):
        ax[0,l].set_xlim(-.01,1.01)
        ax[0,l].set_ylim(-.01,1.01)
        if l == 0:
            ax[0,l].legend(loc='lower right' if i==0 else 'lower left')
        ax[0,l].set_aspect('equal', adjustable='box')
        ax[0,l].set_title(f'Location {l+1}')

# Aggregated curves
for a, agg_strat in enumerate(confusion_dict.keys()):
    r,c = (a+2)//ncols, (a+2)%ncols
    a_agg = np.sum(confusion_dict[agg_strat][0], axis=0)
    b_agg = np.sum(confusion_dict[agg_strat][1], axis=0)
    c_agg = np.sum(confusion_dict[agg_strat][2], axis=0)
    d_agg = np.sum(confusion_dict[agg_strat][3], axis=0)
    
    # ---------------------------------------------------------------------------- #
    fpr_agg1, tpr_agg1, _, _ = compute_roc_pr(tps=a_agg[0,:], fps=b_agg[0,:], fns=c_agg[0,:], tns=d_agg[0,:], extend='none')
    fpr_agg1, tpr_agg1 = np.concatenate((fpr_agg1,[1])), np.concatenate((tpr_agg1,[1]))
    fpr_agg2, tpr_agg2, _, _ = compute_roc_pr(tps=a_agg[1,:], fps=b_agg[1,:], fns=c_agg[1,:], tns=d_agg[1,:], extend='none')
    fpr_agg2, tpr_agg2 = np.concatenate((fpr_agg2,[1])), np.concatenate((tpr_agg2,[1]))
    
    # ---------------------------------------------------------------------------- #
    
    for m in range(n_models):
        label = f"Forecast {forecastnames[m]}"
        fpr_agg, tpr_agg, recall_agg, precision_agg = compute_roc_pr(tps=a_agg[m,:], fps=b_agg[m,:], fns=c_agg[m,:], tns=d_agg[m,:], extend='none')
        fpr_agg, tpr_agg, recall_agg, precision_agg = np.concatenate((fpr_agg,[1])), np.concatenate((tpr_agg,[1])), np.concatenate((recall_agg,[1])), np.concatenate((precision_agg,[0.05]))
        ax_roc[r, c].plot(fpr_agg, tpr_agg, label=label, color = colors[m], linewidth=lw)
        ax_pr[r, c].plot(recall_agg, precision_agg, label=label, color = colors[m], linewidth=lw)
        
    ax_roc[r, c].set_xlabel(LABELS['fpr'])
    ax_roc[r, c].set_ylabel(LABELS['tpr'])
    ax_pr[r, c].set_xlabel(LABELS['recall'])
    ax_pr[r, c].set_ylabel(LABELS['precision'])
    for ax in [ax_roc,ax_pr]:
        ax[r, c].set_title(agg_strat)
        ax[r, c].set_xlim(-.01,1.01)
        ax[r, c].set_ylim(-.01,1.01)
        ax[r, c].set_aspect('equal', adjustable='box')

fig_roc.tight_layout(h_pad=3.5)
fig_pr.tight_layout(h_pad=3.5)
add_panel_labels(ax_roc)
add_panel_labels(ax_pr)
savefig(fig_roc, 'figures/CE_ROC_nonstationary.png')
savefig(fig_pr, 'figures/CE_PR_nonstationary.png')
