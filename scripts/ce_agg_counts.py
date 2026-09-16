import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm, multivariate_normal

import sys
from pathlib import Path
tools_parent = Path(__file__).parent.parent.resolve()
sys.path.append(str(tools_parent))
from tools.utils import compute_roc_pr, intersection_segment_ROC
from tools.style import apply_style, COLORS, LW as lw, LABELS, add_panel_labels, savefig

apply_style()

# ---------------------------------------------------------------------------- #
def FH_to_contingency(fpr, tpr, p, n):
    n_p = n*p
    tps = n_p * tpr
    fps = fpr * (n - n_p)
    fns = n_p - tps
    tns = n - tps - fps - fns
    return tps, fps, fns, tns


def confusion_preliminary(n=1000, rho_list=[[.65, .75], [.55, .65]], FB_lists=[[[.75, 1.25],[.4, 1, 1.6]],[[.75, 1.25],[.4, 1, 1.6]]]):
    t_Y = norm.ppf(0.5)
    z_y = t_Y

    confusion_prelim = [
        # location 1
        [[[],[],[],[]], # forecast 1
        [[],[],[],[]]], # forecast 2
        # location 2
        [[[],[],[],[]], # forecast 1
        [[],[],[],[]]] # forecast 2
    ]
    # ---------------------------------------------------------------------------- #
    for l in range(2):
        for m in range(2):
            FB_list = FB_lists[l][m]
            rho = rho_list[l][m]
            cov_matrix = np.array([[1, rho], [rho, 1]])
            mvn = multivariate_normal(mean=[0, 0], cov=cov_matrix)
            Z_X = norm.ppf([fb*.5 for fb in FB_list])
            for z_x in Z_X:
                cdf_xy = mvn.cdf([z_x, z_y])
                cdf_x = norm.cdf(z_x)
                cdf_y = norm.cdf(z_y)

                confusion_prelim[l][m][0].append(int(n*(1- cdf_x - cdf_y + cdf_xy)))
                confusion_prelim[l][m][1].append(int(n*(cdf_y - cdf_xy)))
                confusion_prelim[l][m][2].append(int(n*(cdf_x - cdf_xy)))
                confusion_prelim[l][m][3].append(int(n*(cdf_xy)))
    return confusion_prelim

def confusion_baserate(confusion_prelim, base_rate, n=1000):
    confusion_raw = [
        # location 1
        [[], # forecast 1
        []], # forecast 2
        # location 2
        [[], # forecast 1
        []] # forecast 2
    ]

    for l in range(2):
        for m in range(2):
            tps, fps, fns, tns = np.array(confusion_prelim[l][m][0]), np.array(confusion_prelim[l][m][1]), np.array(confusion_prelim[l][m][2]), np.array(confusion_prelim[l][m][3])
            fpr, tpr, _, _ = compute_roc_pr(tns=tns, fps=fps, fns=fns, tps=tps, extend='both')
            
            tps, fps, fns, tns = FH_to_contingency(fpr, tpr, p=base_rate[l], n=n)
            confusion_raw[l][m] = [tps, fps, fns, tns]
    return confusion_raw

P_PARAM = .6

def confusion_parameterization(confusion_raw, base_rate, n=1000, N=1000, param='FB'):
    confusion = np.empty((4+1, 2, 2, N), dtype=float)

    for l in range(2):
        for m in range(2):
            tps, fps, fns, tns = np.array(confusion_raw[l][m][0]), np.array(confusion_raw[l][m][1]), np.array(confusion_raw[l][m][2]), np.array(confusion_raw[l][m][3])
            p = base_rate[l]
            n_p = n*p
            u = np.linspace(0,1,N)
            if param == 'u' or param == 'FB':
                FB = (1+(1-p)/p) * u
            elif param == 'indep_u':
                FB = (1+(1-P_PARAM)/P_PARAM) * u
            fpr, tpr, _, _ = compute_roc_pr(tns=tns, fps=fps, fns=fns, tps=tps, extend='both')
            if param == 'indep_u':
                new_fpr, new_tpr = intersection_segment_ROC(fpr=fpr, tpr=tpr, FB=FB, p=P_PARAM)
            else:
                new_fpr, new_tpr = intersection_segment_ROC(fpr=fpr, tpr=tpr, FB=FB, p=p)
            new_tps = n_p * new_tpr
            new_fps = new_fpr * (n - n_p)
            new_fns = n_p - new_tps
            new_tns = n - new_tps - new_fps - new_fns
            
            confusion[0,l,m] = new_tps
            confusion[1,l,m] = new_fps
            confusion[2,l,m] = new_fns
            confusion[3,l,m] = new_tns
            if param == 'FB':
                confusion[4,l,m] = FB
            elif param == 'u' or param == 'indep_u':
                confusion[4,l,m] = u

    return confusion

def bounds(confusion):
    u2s, upper_bounds, lower_bounds = [], [], []
    for l in range(2):
        u1 = confusion[4,l,0]
        u2 = confusion[4,l,1]
        a,b,c,d = confusion[0,l,0], confusion[1,l,0], confusion[2,l,0], confusion[3,l,0]
        fpr1, tpr1, _, _ = compute_roc_pr(tns=d, fps=b, fns=c, tps=a, extend='none')
        a,b,c,d = confusion[0,l,1], confusion[1,l,1], confusion[2,l,1], confusion[3,l,1]
        fpr2, tpr2, _, _ = compute_roc_pr(tns=d, fps=b, fns=c, tps=a, extend='none')
        
        try:
            f1_inv, h1_inv = interp1d(fpr1, u1 ,kind='linear'), interp1d(tpr1, u1, kind='linear')
            upper_bounds.append(h1_inv(tpr2))
            lower_bounds.append(f1_inv(fpr2))
            u2s.append(u2)
        except ValueError:
            fpr1 = np.concatenate([fpr1[:-1], [1]])
            tpr1 = np.concatenate([tpr1[:-1], [1]])
            f1_inv, h1_inv = interp1d(fpr1, u1 ,kind='linear'), interp1d(tpr1, u1, kind='linear')
            upper_bounds.append(h1_inv(tpr2))
            lower_bounds.append(f1_inv(fpr2))
            u2s.append(u2)
    return u2s, lower_bounds, upper_bounds

def confusion_alphas(confusion_raw, alphas, lower_bounds, upper_bounds, base_rate, n=1000, N=1000, locationwise=False, param='u'):
    n_alpha = len(alphas)
    confusion_reparam = np.empty((4+1, 2, n_alpha, N), dtype=float)
    # temp_list = []
    for l in range(2):
        # Reparametrize forecast 1
        tps, fps, fns, tns = np.array(confusion_raw[l][0][0]), np.array(confusion_raw[l][0][1]), np.array(confusion_raw[l][0][2]), np.array(confusion_raw[l][0][3])
        p = base_rate[l]
        n_p = n*p
        fpr, tpr, _, _ = compute_roc_pr(tns=tns, fps=fps, fns=fns, tps=tps, extend='both')
        # Compute intersection with new FB values
        for i,alpha in enumerate(alphas):
            if locationwise:
                lower_bound = lower_bounds[l]
                upper_bound = upper_bounds[l]
            else:
                lower_bound = np.max(lower_bounds, axis=0)
                upper_bound = np.min(upper_bounds, axis=0)
            if param == 'u':
                FB = (lower_bound + alpha * (upper_bound - lower_bound)) * (1+(1-p)/p)
            elif param == 'FB':
                FB = lower_bound + alpha * (upper_bound - lower_bound)
            elif param == 'indep_u':
                FB = (1+(1-P_PARAM)/P_PARAM) * (lower_bound + alpha * (upper_bound - lower_bound))
            
            if param == 'indep_u':
                new_fpr, new_tpr = intersection_segment_ROC(fpr=fpr, tpr=tpr, FB=FB, p=P_PARAM)
            else:
                new_fpr, new_tpr = intersection_segment_ROC(fpr=fpr, tpr=tpr, FB=FB, p=p)
            # temp_list.append([new_fpr,new_tpr])

            new_tps = n_p * new_tpr
            new_fps = new_fpr * (n - n_p)
            new_fns = n_p - new_tps
            new_tns = n - new_tps - new_fps - new_fns
            
            confusion_reparam[0,l,i] = new_tps
            confusion_reparam[1,l,i] = new_fps
            confusion_reparam[2,l,i] = new_fns
            confusion_reparam[3,l,i] = new_tns
            confusion_reparam[4,l,i] = np.linspace(0,1,N)
    return confusion_reparam

def aggregate_counts(confusion, extend='none'):
    agg_fprs, agg_tprs, agg_recalls, agg_precisions = [], [], [], []
    agg_a, agg_b, agg_c, agg_d, _ = np.sum(confusion, axis=1)
    for m in range(agg_a.shape[0]):
        agg_fpr, agg_tpr, agg_recall, agg_precision = compute_roc_pr(tns=agg_d[m], fps=agg_b[m], fns=agg_c[m], tps=agg_a[m], extend=extend)
        agg_fprs.append(agg_fpr)
        agg_tprs.append(agg_tpr)
        agg_recalls.append(agg_recall)
        agg_precisions.append(agg_precision)

    return agg_fprs, agg_tprs, agg_recalls, agg_precisions

def aggregate_counts_combinations(confusion, extend='none'):
    agg_fprs, agg_tprs, agg_recalls, agg_precisions = [], [], [], []
    for m in range(confusion.shape[1]):
        agg_a, agg_b, agg_c, agg_d, indices = [], [], [], [], []
        for i in range(confusion.shape[-1]):
            for j in range(i,confusion.shape[-1]):
                agg_a.append(confusion[0,m,0,i] + confusion[1,m,0,j])
                agg_b.append(confusion[0,m,1,i] + confusion[1,m,1,j])
                agg_c.append(confusion[0,m,2,i] + confusion[1,m,2,j])
                agg_d.append(confusion[0,m,3,i] + confusion[1,m,3,j])
                indices.append((i,j))
        agg_a, agg_b, agg_c, agg_d = np.array(agg_a), np.array(agg_b), np.array(agg_c), np.array(agg_d)
        agg_fpr, agg_tpr, agg_recall, agg_precision = compute_roc_pr(tns=agg_d, fps=agg_b, fns=agg_c, tps=agg_a, extend=extend)
        agg_fprs.append(agg_fpr)
        agg_tprs.append(agg_tpr)
        agg_recalls.append(agg_recall)
        agg_precisions.append(agg_precision)

    return agg_fprs, agg_tprs, agg_recalls, agg_precisions, indices

def fprs_tprs_locationwise(confusion):
    n_locations = len(confusion[0])
    n_forecasts = len(confusion[0][0])
    fpr_list, tpr_list = [], []
    for l in range(n_locations):
        fpr_list.append([])
        tpr_list.append([])
        for m in range(n_forecasts):
            a,b,c,d = confusion[0,l,m], confusion[1,l,m], confusion[2,l,m], confusion[3,l,m]
            fpr, tpr, _, _ = compute_roc_pr(tns=d, fps=b, fns=c, tps=a, extend='none')
            fpr_list[l].append(fpr)
            tpr_list[l].append(tpr)
    return fpr_list, tpr_list


def recalls_precisions_locationwise(confusion):
    n_locations = len(confusion[0])
    n_forecasts = len(confusion[0][0])
    recall_list, precision_list = [], []
    for l in range(n_locations):
        recall_list.append([])
        precision_list.append([])
        for m in range(n_forecasts):
            a,b,c,d = confusion[0,l,m], confusion[1,l,m], confusion[2,l,m], confusion[3,l,m]
            _, _, recall, precision = compute_roc_pr(tns=d, fps=b, fns=c, tps=a, extend='none')
            recall_list[l].append(recall)
            precision_list[l].append(precision)
    return recall_list, precision_list

def monotone_sequences(m: int, n: int) -> list[list[tuple[int, int]]]:
    """
    All componentwise non-decreasing sequences of pairs (i, j) starting at
    (0, 0) and ending at (m-1, n-1), covering all values in each component.
    Steps: (+1,0), (0,+1), (+1,+1).
    """
    results = []

    def backtrack(i: int, j: int, path: list) -> None:
        if i == m - 1 and j == n - 1:
            results.append(path.copy())
            return
        for di, dj in ((1, 0), (0, 1), (1, 1)):
            ni, nj = i + di, j + dj
            if ni <= m - 1 and nj <= n - 1:
                path.append((ni, nj))
                backtrack(ni, nj, path)
                path.pop()

    backtrack(0, 0, [(0, 0)])
    return results

def is_concave_roc(fpr, tpr):
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)

    dx = np.diff(fpr)
    dy = np.diff(tpr)

    # cross product of consecutive edge vectors: positive <-> left turn <-> concave
    cross = dx[:-1] * dy[1:] - dy[:-1] * dx[1:]

    violations = np.where(cross > 0)[0] + 1  # +1: index into original arrays
    return violations.size == 0, violations

# ---------------------------------------------------------------------------- #

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-br', nargs=2, metavar=('p1', 'p2'), default=(.9, .1), type=float)
args = parser.parse_args()
base_rate = args.br
print(base_rate)


# Forecast 1 / Forecast 2, from the central palette.
colors = [COLORS[1], COLORS[2]]

n = 1000
confusion_prelim = confusion_preliminary(n=n, rho_list=[[.75, .65], [.65, .55]], FB_lists=[[ [1.],[.85]],[[.85],[1.],]])
confusion_raw = confusion_baserate(confusion_prelim=confusion_prelim, base_rate=base_rate, n=n)
confusion_raw = np.array(confusion_raw)
print(confusion_raw.shape)

# ---------------------------------------------------------------------------- #
#                              PLOT LOCATION-WISE                              #
# ---------------------------------------------------------------------------- #

fig, ax = plt.subplots(1, 3, figsize=(4*3, 4*1))
for l in range(2):
    fpr_list, tpr_list = [], []
    for m in range(2):
        a,b,c,d = confusion_raw[l,m,0], confusion_raw[l,m,1], confusion_raw[l,m,2], confusion_raw[l,m,3]
        fpr, tpr, _, _ = compute_roc_pr(tns=d, fps=b, fns=c, tps=a, extend='none')
    
        ax[l].plot(fpr, tpr, label=f'Forecast {"AB"[m]}', color = colors[m],zorder=2 if m==1 else 0, marker='o', markersize=4, lw=lw)
        if m == 1:
            ax[l].vlines(fpr[1], ymin=tpr[1], ymax=1, color='gray', lw=1, linestyle='dashed', zorder=1)
            ax[l].hlines(tpr[1], xmin=fpr[1], xmax=0, color='gray', lw=1, linestyle='dashed', zorder=1)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    ax[l].set_xlabel(LABELS['fpr'])
    ax[l].set_ylabel(LABELS['tpr'])
    ax[l].set_xlim(-.01,1.01)
    ax[l].set_ylim(-.01,1.01)
    ax[l].set_aspect('equal', adjustable='box')
    if l==0:
        ax[l].legend(loc='lower right')
    ax[l].set_title(f'Location {l+1}')

agg_fprs, agg_tprs, _, _, indices = aggregate_counts_combinations(confusion=confusion_raw, extend='none')
sequences = monotone_sequences(m=3, n=3)
seq = sequences[-1]
for m in range(2):
    seq_fprs = [agg_fprs[m][i] for i in range(len(agg_fprs[m])) if indices[i] in seq]
    seq_tprs = [agg_tprs[m][i] for i in range(len(agg_tprs[m])) if indices[i] in seq]
    ax[2].plot(seq_fprs, seq_tprs, color = colors[m], lw=lw, zorder=1, label=f'Forecast {"AB"[m]}', marker='o', markersize=4)

ax[2].set_xlabel(LABELS['fpr'])
ax[2].set_ylabel(LABELS['tpr'])
ax[2].set_xlim(-.01,1.01)
ax[2].set_ylim(-.01,1.01)
ax[2].set_aspect('equal', adjustable='box')
ax[2].set_title(f'Aggregated ROC curves')

fig.tight_layout(h_pad=3.5)
add_panel_labels(ax)
savefig(fig, f'figures/ce_agg_counts_p{base_rate[0]}-{base_rate[1]}.png')


# ---------------------------------------------------------------------------- #
#                          DOMINANCE / CONCAVITY CHECK                         #
# ---------------------------------------------------------------------------- #

print('### CHECK DOMINANCE ###')

agg_fprs, agg_tprs, _, _, indices = aggregate_counts_combinations(confusion=confusion_raw, extend='none')
concave_sequences = [[],[]]
all_sequences = [[],[]]
for m in range(2):
    for k,seq in enumerate(sequences):
        seq_fprs = [agg_fprs[m][i] for i in range(len(agg_fprs[m])) if indices[i] in seq or indices[i][::-1] in seq]
        seq_tprs = [agg_tprs[m][i] for i in range(len(agg_tprs[m])) if indices[i] in seq or indices[i][::-1] in seq]
        if is_concave_roc(seq_fprs, seq_tprs)[0]:
            concave_sequences[m].append(seq)
        all_sequences[m].append(seq)

s1 = len(all_sequences[0])
s2 = len(all_sequences[1])
for m in range(2):
    print(f'Aggregation strategies leading to concave ROC curves for Forecast {m+1}:')
    for i in range(len(concave_sequences[m])):
        print(all_sequences[m].index(concave_sequences[m][i]), concave_sequences[m][i])

fig, ax = plt.subplots(s1, s2, figsize=(4*s2, 4*s1))
for m in range(2):
    for k,seq in enumerate(concave_sequences[m]):
        seq_fprs = [agg_fprs[m][i] for i in range(len(agg_fprs[m])) if indices[i] in seq or indices[i][::-1] in seq]
        seq_tprs = [agg_tprs[m][i] for i in range(len(agg_tprs[m])) if indices[i] in seq or indices[i][::-1] in seq]
        for j in range(len(concave_sequences[1-m])):
            if m == 0:
                if s1 == 1 and s2 > 1:
                    ax[j].plot(seq_fprs, seq_tprs, color = colors[m], lw=2, zorder=1, marker='o', markersize=4)
                elif s1==1 and s2==1:
                    ax.plot(seq_fprs, seq_tprs, color = colors[m], lw=2, zorder=1, marker='o', markersize=4)
                else:
                    ax[k,j].plot(seq_fprs, seq_tprs, color = colors[m], lw=2, zorder=1, marker='o', markersize=4)
            else:
                if s2 == 1 and s2 > 1:
                    ax[k].plot(seq_fprs, seq_tprs, color = colors[m], lw=2, zorder=1, marker='o', markersize=4)
                elif s1==1 and s2==1:
                    ax.plot(seq_fprs, seq_tprs, color = colors[m], lw=2, zorder=1, marker='o', markersize=4)
                else:
                    ax[j,k].plot(seq_fprs, seq_tprs, color = colors[m], lw=2, zorder=1, marker='o', markersize=4)
if s1 == 1 and s2 == 1:
    ax.set_xlim(-.01,1.01)
    ax.set_ylim(-.01,1.01)
    ax.set_aspect('equal', adjustable='box')
else:
    for i in range(s1):
        for j in range(s2):
            ax[i,j].set_xlim(-.01,1.01)
            ax[i,j].set_ylim(-.01,1.01)
            ax[i,j].set_aspect('equal', adjustable='box')
fig.suptitle(f'Forecast comparison')
fig.tight_layout(h_pad=3.5)
savefig(fig, f'figures/grid_agg_roc_p{base_rate[0]}-{base_rate[1]}.png', dpi=150)

print('### CHECK DOMINANCE OF NON-TRIVIAL POINTS ###')
idx1 = indices.index((1,1))
idx2 = indices.index((1,1))
print('FAR (1,1)')
print(agg_fprs[0][idx2], agg_fprs[1][idx2], agg_fprs[0][idx2] < agg_fprs[1][idx2])
print('HR (1,1)')
print(agg_tprs[0][idx1], agg_tprs[1][idx1], agg_tprs[0][idx1] < agg_tprs[1][idx1])
