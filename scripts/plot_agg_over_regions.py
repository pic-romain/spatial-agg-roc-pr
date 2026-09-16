import argparse
import matplotlib.pyplot as plt
import numpy as np
from math import ceil

import sys
from pathlib import Path
tools_parent = Path(__file__).parent.parent.resolve()
sys.path.append(str(tools_parent))
from tools.utils import compute_roc_pr
from tools.style import (
    apply_style, COLORS as colors, LW as lw, LABELS,
    model_legend_handles, add_panel_labels, savefig,
)

from matplotlib.lines import Line2D

apply_style()

parser = argparse.ArgumentParser()
parser.add_argument("--thresholds", default="q98,max",
                    help="Comma-separated threshold values, e.g. 'q98,max'.")
parser.add_argument("--lead-times", default="48,120,240",
                    help="Comma-separated lead times in hours.")
parser.add_argument("--regions", default="world,northern_hemisphere_extratropics,southern_hemisphere_extratropics,tropics,north_america,europe_north_africa,asia,australia_new_zealand,northern_polar_region,southern_polar_region",
                    help="Comma-separated region names.")
parser.add_argument("--filter", default="none",
                    choices=["none", "land_noant", "land_summer", "land_noant_summer"],
                    help=(
                        "Spatial/temporal filter applied when running agg_over_regions: "
                        "'none' (no filter), "
                        "'land_noant' (land only, Antarctic excluded), "
                        "'land_summer' (land only, local summer months)."
                    ))
args = parser.parse_args()

WB_var = '2m_temperature'
region_name = 'world'
year = 2020
resolution = '0p25' # '0p25' or 'low-res'

thresholds_list = [t.strip() for t in args.thresholds.split(',')]
lead_times_list = [int(x) for x in args.lead_times.split(',')]
regions_list = [r.strip() for r in args.regions.split(',')]
file_suffix = {'none': '', 'land_noant': '_land_noant', 'land_summer': '_land_summer', 'land_noant_summer': '_land_noant_summer'}[args.filter]

agg_strategies = (
    'GraphCast',
    'location-scale',
    'FB',
    'parallel',
)

panel_indices = {
    'GraphCast': 0,
    'location-scale': 1,
    'FB': 2,
    'parallel': 3,
}
# ---------------------------------------------------------------------------- #

legend_elements = model_legend_handles()
labels_dict = {
    12: 'Lead time = 12h',
    48: 'Lead time = 2 days',
    120: 'Lead time = 5 days',
    240: 'Lead time = 10 days',
}
# ---------------------------------------------------------------------------- #

strategy_titles = {
    'GraphCast': "GraphCast's parameterization",
    'location-scale': "Location-scale parameterization",
    'FB': "DCP-FB",
    'parallel': "DCP-PL",
}
if lead_times_list == [48, 120, 240]:
    lead_time_styles = {
        48: {'ls': 'solid', 'alpha': 1.0},
        120: {'ls': 'dashed', 'alpha': 1.0},
        240: {'ls': 'dotted', 'alpha': 1.0},
    }
else:
    lead_time_styles = {
        12: {'ls': 'solid', 'alpha': 1.0},
        48: {'ls': 'dashed', 'alpha': 1.0},
        120: {'ls': 'dotted', 'alpha': 1.0},
    }
for lead_time, params in lead_time_styles.items():
    label = labels_dict[lead_time]
    legend_elements.append(Line2D([0], [0], color='gray', lw=lw, ls=params['ls'], label=label))

models_list = ['hres', 'graphcast', 'pangu']

n_panels = 4#len(agg_strategies)
ncols = 2
nrows = ceil(n_panels / ncols)

for region in regions_list:
    print(f"Processing region: {region}")
    for threshold in thresholds_list:
        fig_roc, ax_roc = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        fig_pr, ax_pr = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        ax_roc = np.atleast_1d(ax_roc).reshape(nrows, ncols)
        ax_pr = np.atleast_1d(ax_pr).reshape(nrows, ncols)

        for lead_time in lead_times_list:
            plot_params = lead_time_styles[lead_time]

            for a,agg_strat in enumerate(agg_strategies):
                try:
                    contingency_matrices = np.load(f'data/contingency_matrices_{region}_t{threshold}_{lead_time}h_{agg_strat}{file_suffix}.npy')
                except:
                    print(f"Warning: Contingency matrices not found for {agg_strat} at lead time {lead_time}h and threshold {threshold} in region {region}. Skipping.")
                    continue
                r, c = divmod(panel_indices[agg_strat], ncols)
                for m, model in enumerate(models_list):
                    label = model
                    
                    ls = plot_params['ls']
                    alpha = plot_params['alpha']
                    fpr_agg, tpr_agg, recall_agg, precision_agg = compute_roc_pr(tns=contingency_matrices[3,m,:], fps=contingency_matrices[1,m,:], fns=contingency_matrices[2,m,:], tps=contingency_matrices[0,m,:],extend='both')
                    ax_roc[r,c].plot(fpr_agg, tpr_agg, label=f'{label} ({lead_time}h)', color = colors[m%len(colors)], linewidth=lw, linestyle=ls, alpha=alpha)#, marker='o', markersize=4)
                    ax_pr[r,c].plot(recall_agg[1:-1], precision_agg[1:-1], label=f'{label} ({lead_time}h)', color = colors[m%len(colors)], linewidth=lw, linestyle=ls, alpha=alpha)#, marker='o', markersize=4)
            
                ax_roc[r,c].set_xlabel(LABELS['fpr'])
                ax_roc[r,c].set_ylabel(LABELS['tpr'])
                ax_pr[r,c].set_xlabel(LABELS['recall'])
                ax_pr[r,c].set_ylabel(LABELS['precision'])
                for ax in [ax_roc, ax_pr]:
                    title = strategy_titles[agg_strat]
                    ax[r,c].set_title(title)
                    if title == "DCP-FB" and (ax==ax_pr).all() and args.thresholds == "q98":
                        ax[r,c].set_ylim(-.01,.71)
                        ax[r,c].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
                    elif title == "DCP-PL" and (ax==ax_pr).all() and args.thresholds == "q98":
                        ax[r,c].set_ylim(-.01,.41)
                        ax[r,c].set_yticks([0, 0.1, 0.2, 0.3, 0.4])
                    elif title == "DCP-FB" and (ax==ax_pr).all() and args.thresholds == "max":
                        ax[r,c].set_ylim(-.01,.51)
                        ax[r,c].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
                    elif title == "DCP-PL" and (ax==ax_pr).all() and args.thresholds == "max":
                        ax[r,c].set_ylim(-.001,.051)
                        ax[r,c].set_yticks([0, .01, 0.02, 0.03, 0.04, 0.05])
                    else:
                        ax[r,c].set_ylim(-.01,1.01)
                        ax[r,c].set_aspect('equal', adjustable='box')
                    ax[r,c].set_xlim(-.01,1.01)
                    ax[r,c].set_box_aspect(1)

        if args.thresholds == "q98":
            ax_roc[0,0].legend(handles=legend_elements, loc='lower right')
            ax_pr[1,0].legend(handles=legend_elements, loc='upper right')
        elif args.thresholds == "max":
            ax_roc[0,0].legend(handles=legend_elements, loc='lower right')
            ax_pr[1,0].legend(handles=legend_elements, loc='upper right')

        fig_roc.tight_layout(h_pad=2.5)
        fig_pr.tight_layout(h_pad=2.5)
        add_panel_labels(ax_roc)
        add_panel_labels(ax_pr)
        fig_filter = f'_{args.filter}' if args.filter != 'none' else ''
        savefig(fig_roc, f'figures/aggROC_{region}_t{threshold}{fig_filter}.png')
        savefig(fig_pr, f'figures/aggPR_{region}_t{threshold}{fig_filter}.png')

    plt.close()