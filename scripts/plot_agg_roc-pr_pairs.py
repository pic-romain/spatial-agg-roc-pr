import xarray as xr
import matplotlib.pyplot as plt
import os
import numpy as np
import dask
dask.config.set(scheduler='threads')
import pickle
from math import ceil
from collections import OrderedDict

from sklearn.isotonic import IsotonicRegression

import sys
from pathlib import Path
tools_parent = Path(__file__).parent.parent.resolve()
sys.path.append(str(tools_parent))
from tools.utils import confusion_matrices_locations, get_climatological_quantile, get_climatological_std, load_models, compute_roc_pr
from tools.utils import contingency_for_strategy, sanitize_roc_curve, MODEL_DICT
from tools.style import apply_style, COLORS as colors, LW as lw, LABELS, savefig, add_panel_labels

import yaml
from pathlib import Path

with open("tools/config.yaml") as f:
    config = yaml.safe_load(f)

wdr = Path(config["wb2_dir"])

apply_style()

def sanitize_pr_curve(recall, precision):
    valid = ~(np.isnan(recall) | np.isnan(precision))
    recall, precision = recall[valid], precision[valid]
    order = np.argsort(recall)
    recall, precision = recall[order], precision[order]
    _, unique_idx = np.unique(np.column_stack([recall, precision]), axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return recall[unique_idx], precision[unique_idx]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--threshold",
    default='q75',
    help="Threshold for binary classification. Can be a float (e.g., 300.0) or a quantile string (e.g., 'q75')."
)
parser.add_argument(
    "--lead-time",
    type=int,
    default=48,
    help="Forecast lead time in hours.",
)
parser.add_argument(
    "--dominance-scenario",
    default="hres>graphcast>pangu",
    help="Dominance ordering, e.g. 'hres>graphcast>pangu'.",
)
parser.add_argument(
    "--locations",
    default="(60.25, -1.5) -- (70.25, -25.0)",
    help="Two locations as '(lat1, lon1) -- (lat2, lon2)'.",
)
parser.add_argument(
    "--gc-gain-range",
    type=float,
    nargs=2,
    default=[.8, 4.5],
    metavar=("MIN", "MAX"),
    help="Gain range for GraphCast's parameterization strategy (default: 0.8 4.5).",
)
parser.add_argument(
    "--location-scale-gain-range",
    type=float,
    nargs=2,
    default=[-1.5, 1.5],
    metavar=("MIN", "MAX"),
    help="Gain range for the location-scale parameterization (default: -1.5 1.5).",
)
parser.add_argument(
    "--gain-steps",
    type=int,
    default=1000,
    help="Number of gain steps (np.linspace num) for GraphCast's and the location-scale parameterization strategies. Defaults to 100000.",
)
args = parser.parse_args()

WB_var = '2m_temperature'
region_name = 'world'
year = 2020
resolution = '0p25' # '0p25' or 'low-res'

lead_time = args.lead_time

# ---------------------------------------------------------------------------- #

from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

legend_elements = [Line2D([0], [0], color='gray', lw=lw, ls='solid', label='PAV, interp'),
                   Line2D([0], [0], color='gray', lw=lw, ls='dashed', label='PAV, no interp'),
                   Line2D([0], [0], color='gray', lw=lw, ls='dotted', label='no PAV, interp'),
                   Line2D([0], [0], color='gray', lw=lw, ls='dashdot', label='no PAV, no interp'),
                   ]
extra = Line2D([0], [0], color='gray', lw=lw, ls='dashed', label='raw')
# ---------------------------------------------------------------------------- #


strategy_titles = {
    'GraphCast': "GraphCast's parameterization",
    'location-scale': "Location-scale parameterization",
    'FB': "DCP-FB",
    'parallel': "DCP-PL",
}

# Lower-left zoom insets on the aggregated PR panels, for the parameterized
# strategies only. Each value is (recall_min, recall_max, precision_min,
# precision_max) — edit to reframe the zoom. Remove a key to drop its inset.
PR_INSET_LIMITS = {
    'GraphCast': (.9, 1., .66, .76),
    'location-scale':     (.9, 1., .5, .6),
}
PR_INSET_BOX = [0.11, 0.08, 0.45, 0.45]  # [x0, y0, w, h] in axes fraction: lower-left

# Same convention for the aggregated ROC panels; each value is
# (fpr_min, fpr_max, tpr_min, tpr_max). ROC curves hug the top-left, so the
# inset sits in the (usually empty) lower-right corner.
ROC_INSET_LIMITS = {
    'GraphCast':       (.2, .3, .9, 1.0),
    'location-scale':  (.4, .5, .9, 1.0),
}
ROC_INSET_BOX = [0.5, 0.08, 0.45, 0.45]  # [x0, y0, w, h] in axes fraction: lower-right

# Connector lines from the zoomed rectangle to the inset. By default matplotlib
# picks two rectangle corners automatically and joins each to the *same* corner
# of the inset. To join a rectangle corner to a *different* inset corner, list
# (rectangle_corner, inset_corner) pairs for that strategy; corners are
# 'LL','UL','LR','UR'. Strategies mapped to None (or absent) keep the automatic
# connectors.
_CORNER_XY = {'LL': (0., 0.), 'UL': (0., 1.), 'LR': (1., 0.), 'UR': (1., 1.)}
ROC_INSET_CONNECT = {'location-scale': [('LL', 'UL'), ('LR', 'UR')], 'GraphCast': [('LL', 'UL'), ('LR', 'UR')]}  # rect bottom -> inset top
PR_INSET_CONNECT = {'location-scale': [('UL', 'UL'), ('LR', 'LR')], 'GraphCast': [('UL', 'UL'), ('LR', 'LR')]}


def _set_inset_connectors(indicator, parent_ax, axins, rect_limits, specs,
                          color='black', alpha=0.6, lw=0.8):
    """Replace the indicator's automatic connectors with explicit
    (rectangle_corner -> inset_corner) lines. No-op when specs is falsy."""
    if not specs:
        return
    for cp in indicator.connectors:  # drop matplotlib's automatic pair
        cp.set_visible(False)
    xmin, xmax, ymin, ymax = rect_limits
    rect_xy = {'LL': (xmin, ymin), 'UL': (xmin, ymax),
               'LR': (xmax, ymin), 'UR': (xmax, ymax)}
    for rect_corner, inset_corner in specs:
        parent_ax.add_patch(ConnectionPatch(
            xyA=_CORNER_XY[inset_corner], coordsA=axins.transAxes,
            xyB=rect_xy[rect_corner], coordsB=parent_ax.transData,
            color=color, alpha=alpha, lw=lw, zorder=5))

dominance_scenario = args.dominance_scenario
locations_str = args.locations

# Str to dict
locations = {}
for loc_str in locations_str.split('--'):
    lat_str, lon_str = loc_str.strip()[1:-1].split(',')
    lat, lon = float(lat_str), float(lon_str)
    locations[f'({lat}, {lon})'] = (lat, lon)

models = dominance_scenario.split('>')
m1 = MODEL_DICT[models[0]][0]
m2 = MODEL_DICT[models[1]][0]
m3 = MODEL_DICT[models[2]][0]
model1 = models[0]
model2 = models[1]
model3 = models[2]

# ---------------------------------------------------------------------------- #
#                        EXTRACT DATA OR LOAD FROM CACHE                       #
# ---------------------------------------------------------------------------- #
l_str = ''.join(locations.keys())
cache_dir = "./cache"
os.makedirs(cache_dir, exist_ok=True)
cache_file = os.path.join(
    cache_dir, 
    f"extracted_{l_str}_{WB_var}_{year}_{region_name}_lead{lead_time}_t{args.threshold}.pkl"
)

models = load_models(wdr=wdr, year=year, region_name=region_name, WB_var=WB_var, resolution=resolution, lead_time=lead_time)


# 2. Try to load from cache first
if os.path.exists(cache_file):
    print(f"Found cached data at {cache_file}. Loading...")
    with open(cache_file, 'rb') as f:
        models = pickle.load(f)
    print("Data loaded from cache.")

else:
    print("Cache not found. Starting raw data extraction...")
    loc_names = list(locations.keys())
    target_lats = xr.DataArray([locations[k][0] for k in loc_names], dims="location", coords={"location": loc_names})
    target_lons = xr.DataArray([locations[k][1] for k in loc_names], dims="location", coords={"location": loc_names})

    # Use the first forecast dataset to find the grid points corresponding to your cities
    # (Assumes all models share the same grid, which is standard for 0.25deg data)
    grid_lats = models['hres']['fcst']['lat'].sel(lat=target_lats, method='nearest')
    grid_lons = models['hres']['fcst']['lon'].sel(lon=target_lons, method='nearest')

    # Batch Lazy Loading
    lazy_objects = []
    model_keys_ordered = []

    for model_key in models:
        # Use exact selection (faster than nearest)
        fcst_lazy = models[model_key]['fcst'].sel(lat=grid_lats, lon=grid_lons)
        obs_lazy = models[model_key]['obs'].sel(lat=grid_lats, lon=grid_lons)
        
        # Collect lazy objects into a list
        lazy_objects.extend([fcst_lazy, obs_lazy])
        model_keys_ordered.append(model_key)

    print(f"Triggering parallel download for {len(lazy_objects)} items...")
    loaded_results = dask.compute(*lazy_objects)

    # --- Reassign loaded data back to dictionary ---
    for i, model_key in enumerate(model_keys_ordered):
        # Results are returned in the same order they were passed
        models[model_key]['fcst'] = loaded_results[i * 2]
        models[model_key]['obs'] = loaded_results[i * 2 + 1]

    print("Data loaded. Starting analysis loop.")
    
    # 3. Save the result to cache
    print(f"Saving extracted data to {cache_file}...")
    with open(cache_file, 'wb') as f:
        pickle.dump(models, f)
    print("Data saved.")

# ---------------------------------------------------------------------------- #
print("Data ready. Starting dominance violation analysis...")
if args.threshold.startswith('q'):
    t_Y = {
        model: get_climatological_quantile(level=float(args.threshold[1:])/100, months=models[model]['obs'].time.dt.month, hours=models[model]['obs'].time.dt.hour, region_name='europe', WB_var=WB_var, resolution=resolution)
          for model in models.keys()}
else:
    t_Y = {
        model: 273.15 + float(args.threshold)
        for model in models.keys()
    }

# Compute confusion matrices at each location and put in numpy array

n = 1000
gain_steps = args.gain_steps if args.gain_steps is not None else n
n_models = len(models.keys())

agg_strategies_all = (
    'GraphCast',
    'location-scale',
    'FB',
    'parallel',
)


def _n_for_strategy(strat):
    return gain_steps if strat.startswith(('GraphCast', 'location-scale', 'no_param')) else n

confusion_dict = {
    agg_strat: np.full((4, len(locations), n_models, _n_for_strategy(agg_strat)), np.nan, dtype=np.float64)
    for agg_strat in agg_strategies_all+('no_param','raw')
}

climatological_median_obs = {
    model: get_climatological_quantile(level=0.5, months=models[model]['obs'].time.dt.month, hours=models[model]['obs'].time.dt.hour, region_name='europe', WB_var=WB_var, resolution=resolution)
    for model in models.keys()
}

climatological_std = {
    model: get_climatological_std(
        model_key=model,
        lead_time=lead_time,
        months=models[model]['obs'].time.dt.month,
        region_name='world',
        WB_var=WB_var,
        resolution=resolution,
        year=year,
    )
    for model in models.keys()
}

for l, location in enumerate(locations.keys()):
    lat, lon = locations[location]
    for m, model in enumerate(models.keys()):
        # Select specific location from the in-memory array
        fc_loc = models[model]['fcst'].sel(location=location).values
        obs_loc = models[model]['obs'].sel(location=location).values
        label = models[model]['name']

        if isinstance(t_Y[model],xr.DataArray):
            lats, lons = t_Y[model].lat.values, t_Y[model].lon.values
            lat_idx, lon_idx = np.argmin(np.abs(lats - lat)), np.argmin(np.abs(lons - lon))
            obs_bin = (obs_loc > t_Y[model].isel(lat=lat_idx, lon=lon_idx).values).astype(int)
        else:
            lats, lons = climatological_median_obs[model].lat.values, climatological_median_obs[model].lon.values
            lat_idx, lon_idx = np.argmin(np.abs(lats - lat)), np.argmin(np.abs(lons - lon))
            obs_bin = (obs_loc > t_Y[model]).astype(int)
        fc_pav = np.round(
            IsotonicRegression().fit(X=fc_loc, y=obs_bin).predict(fc_loc),
            decimals=6)
        
        for agg_strat in agg_strategies_all:
            confusion_dict[agg_strat][:,l,m,:] = contingency_for_strategy(agg_strat, fc_loc, fc_pav, obs_bin, t_Y[model], n, climatological_median_obs[model], lat_idx, lon_idx, climatological_std=climatological_std[model], gc_gain_range=tuple(args.gc_gain_range), location_scale_gain_range=tuple(args.location_scale_gain_range), gain_steps=gain_steps)
confusion_dict["no_param"] = confusion_matrices_locations(models, locations, t_Y, raw=False, n=gain_steps)
confusion_dict["raw"] = confusion_matrices_locations(models, locations, t_Y, raw=True)

print("Confusion matrices computed. Starting plotting loop...")

agg_strategies = tuple(OrderedDict.fromkeys(['GraphCast', 'location-scale', 'FB', 'parallel']))

n_panels = len(locations) + len([a for a in agg_strategies if not ('_noPAV' in a or '_nointerp' in a)])
ncols = 3
nrows = ceil(n_panels / ncols)
fig_roc, ax_roc = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
fig_pr, ax_pr = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
ax_roc = np.atleast_1d(ax_roc).reshape(nrows, ncols)
ax_pr = np.atleast_1d(ax_pr).reshape(nrows, ncols)

# Individual location
locationwise_strat = ["raw", "no_param"]

for agg_strat in locationwise_strat:
    if agg_strat == "raw":
        ls = '--'
        alpha = 1.0
    else:
        ls = "-"
        alpha = 1.0
    for l, location in enumerate(locations.keys()):
        # ---------------------------------------------------------------------------- #
        
        fpr1, tpr1, recall1, precision1 = compute_roc_pr(tps=confusion_dict[agg_strat][0,l,m1,:], fps=confusion_dict[agg_strat][1,l,m1,:], fns=confusion_dict[agg_strat][2,l,m1,:], tns=confusion_dict[agg_strat][3,l,m1,:],extend='none')
        fpr2, tpr2, recall2, precision2 = compute_roc_pr(tps=confusion_dict[agg_strat][0,l,m2,:], fps=confusion_dict[agg_strat][1,l,m2,:], fns=confusion_dict[agg_strat][2,l,m2,:], tns=confusion_dict[agg_strat][3,l,m2,:],extend='none')
        fpr3, tpr3, recall3, precision3 = compute_roc_pr(tps=confusion_dict[agg_strat][0,l,m3,:], fps=confusion_dict[agg_strat][1,l,m3,:], fns=confusion_dict[agg_strat][2,l,m3,:], tns=confusion_dict[agg_strat][3,l,m3,:],extend='none')
        
        # Check if model 1 dominates model 2
        fpr1, tpr1 = sanitize_roc_curve(fpr1, tpr1)
        fpr2, tpr2 = sanitize_roc_curve(fpr2, tpr2)
        fpr3, tpr3 = sanitize_roc_curve(fpr3, tpr3)
        
        x_common = np.linspace(0, 1, num=n)
        
        y1_interp = np.interp(x_common, fpr1, tpr1)
        y2_interp = np.interp(x_common, fpr2, tpr2)
        y3_interp = np.interp(x_common, fpr3, tpr3)

        ## Diagnostic for dominance violations
        # print(f"Checking dominance for {agg_strat} at location {location}")
        # if np.any(y1_interp < y2_interp):
        #     print(f"\t\tDominance violation found for {agg_strat}: {model2}>{model1}")
        #     print(np.min(y1_interp-y2_interp))
        # if np.any(y1_interp < y3_interp):
        #     print(f"\t\tDominance violation found for {agg_strat}: {model3}>{model1}")
        #     print(np.min(y1_interp-y3_interp))
        # if np.any(y2_interp < y3_interp):
        #     print(f"\t\tDominance violation found for {agg_strat}: {model3}>{model2}")
        #     print(np.min(y2_interp-y3_interp))
        # ---------------------------------------------------------------------------- #

        row, col = divmod(l, ncols)
        for m, model in enumerate(models.keys()):
            fpr, tpr, recall, precision = compute_roc_pr(tps=confusion_dict[agg_strat][0,l,m,:], fps=confusion_dict[agg_strat][1,l,m,:], fns=confusion_dict[agg_strat][2,l,m,:], tns=confusion_dict[agg_strat][3,l,m,:],extend='both')
            recall, precision = sanitize_pr_curve(recall, precision)
            label = f"{models[model]['name']} ({lead_time}h)" if agg_strat=='no_param' else None
            ax_roc[row,col].plot(fpr, tpr, label=label, color = colors[m%len(colors)], linewidth=lw, linestyle=ls, alpha=alpha)
            ax_pr[row,col].plot(recall, precision, label=label, color = colors[m%len(colors)], linewidth=lw, linestyle=ls, alpha=alpha)
        
        if agg_strat == "no_param":
            ax_roc[row,col].set_xlabel(LABELS['fpr'])
            ax_roc[row,col].set_ylabel(LABELS['tpr'])
            ax_pr[row,col].set_xlabel(LABELS['recall'])
            ax_pr[row,col].set_ylabel(LABELS['precision'])
            for ax in [ax_roc,ax_pr]:
                ax[row,col].set_xlim(-.01,1.01)
                ax[row,col].set_ylim(-.01,1.01)
                if l == 0:
                    handles, labels = ax[row,col].get_legend_handles_labels()
                    if 'raw' in locationwise_strat:
                        ax[row,col].legend(handles=handles + [extra], loc='lower right',framealpha=1)
                    else:
                        ax[row,col].legend(handles=handles, loc='lower right',framealpha=1)
                ax[row,col].set_aspect('equal', adjustable='box')
                ax[row,col].set_title('°,'.join(location[:-1].split(','))+'°)')

# Aggregated curves
a_diff = 0
for a, agg_strat in enumerate(agg_strategies):
    if '_noPAV_nointerp' in agg_strat:
        a_diff += -1
        ls = 'dashdot'
        alpha = .7
    elif '_nointerp' in agg_strat:
        a_diff += -1
        ls = 'dashed'
        alpha = .7
    elif '_noPAV' in agg_strat:
        a_diff += -1
        ls = 'dotted'
        alpha = .7
    else:
        ls = 'solid'
        alpha = 1.0
    panel_idx = len(locations) + a + a_diff
    r, c = divmod(panel_idx, ncols)
    tps_agg, fps_agg, fns_agg, tns_agg = np.sum(confusion_dict[agg_strat], axis=1)

    def _sort_by(arr4, mi, key):
        """Return copies of tps/fps/fns/tns for model mi sorted by key ('far' or 'recall')."""
        tp, fp, fn, tn = arr4
        if key == 'far':
            denom = fp[mi] + tn[mi]
            vals = np.where(denom > 0, fp[mi] / denom, 0.0)
        else:
            denom = tp[mi] + fn[mi]
            vals = np.where(denom > 0, tp[mi] / denom, 0.0)
        order = np.argsort(vals)
        return tp[mi, order], fp[mi, order], fn[mi, order], tn[mi, order]

    # ---------------------------------------------------------------------------- #

    def _roc(mi, sort_key='far'):
        tp, fp, fn, tn = _sort_by((tps_agg, fps_agg, fns_agg, tns_agg), mi, sort_key) if agg_strat.startswith('GraphCast') else (tps_agg[mi], fps_agg[mi], fns_agg[mi], tns_agg[mi])
        fpr, tpr, _, _ = compute_roc_pr(tns=tn, fps=fp, fns=fn, tps=tp)
        return sanitize_roc_curve(fpr, tpr)

    def _pr(mi):
        tp, fp, fn, tn = _sort_by((tps_agg, fps_agg, fns_agg, tns_agg), mi, 'recall') if agg_strat.startswith('GraphCast') else (tps_agg[mi], fps_agg[mi], fns_agg[mi], tns_agg[mi])
        _, _, recall, precision = compute_roc_pr(tns=tn, fps=fp, fns=fn, tps=tp, extend='none')
        return sanitize_pr_curve(recall, precision)

    fpr_agg1, tpr_agg1 = _roc(m1)
    fpr_agg2, tpr_agg2 = _roc(m2)
    fpr_agg3, tpr_agg3 = _roc(m3)

    x_common = np.linspace(0, 1, num=1000)

    y1_interp = np.interp(x_common, fpr_agg1, tpr_agg1)
    y2_interp = np.interp(x_common, fpr_agg2, tpr_agg2)
    y3_interp = np.interp(x_common, fpr_agg3, tpr_agg3)

    ## Diagnostic for dominance violations
    # if np.any(y1_interp < y2_interp) or np.any(y1_interp < y3_interp) or np.any(y2_interp < y3_interp):
    #     print(f"\tDominance violation found for {agg_strat}")
    #     print(np.min(y1_interp - y2_interp), np.min(y1_interp - y3_interp), np.min(y2_interp - y3_interp))

    mask1 = y1_interp < y2_interp
    mask2 = y1_interp < y3_interp
    mask3 = y2_interp < y3_interp
    # ---------------------------------------------------------------------------- #
    roc_curves = []  # (fpr, tpr) per model, for the optional zoom inset
    pr_curves = []  # (recall, precision) per model, for the optional zoom inset
    for m, model in enumerate(models.keys()):
        label = models[model]['name']
        fpr_agg, tpr_agg = _roc(m)
        recall_agg, precision_agg = _pr(m)
        roc_curves.append((fpr_agg, tpr_agg))
        pr_curves.append((recall_agg, precision_agg))
        ax_roc[r,c].plot(fpr_agg, tpr_agg, label=f'{label} ({lead_time}h)', color = colors[m%len(colors)], linewidth=lw, linestyle=ls, alpha=alpha)
        ax_pr[r,c].plot(recall_agg, precision_agg, label=f'{label} ({lead_time}h)', color = colors[m%len(colors)], linewidth=lw, linestyle=ls, alpha=alpha)

    if agg_strat in ['GraphCast', 'location-scale', 'parallel', 'FB']:
        ax_roc[r,c].set_xlabel(LABELS['fpr'])
        ax_roc[r,c].set_ylabel(LABELS['tpr'])
        ax_pr[r,c].set_xlabel(LABELS['recall'])
        ax_pr[r,c].set_ylabel(LABELS['precision'])
        for ax in [ax_roc, ax_pr]:
            title = strategy_titles[agg_strat]
            ax[r,c].set_title(title)
            ax[r,c].set_xlim(-.01,1.01)
            ax[r,c].set_ylim(-.01,1.01)
            ax[r,c].set_aspect('equal', adjustable='box')

    # Zoom inset on the ROC panel for the parameterized strategies.
    if agg_strat in ROC_INSET_LIMITS:
        f_min, f_max, t_min, t_max = ROC_INSET_LIMITS[agg_strat]
        axins = ax_roc[r,c].inset_axes(ROC_INSET_BOX)
        for m, (fpr_agg, tpr_agg) in enumerate(roc_curves):
            axins.plot(fpr_agg, tpr_agg, color=colors[m%len(colors)], linewidth=lw, linestyle=ls, alpha=alpha)
        axins.set_xlim(f_min, f_max)
        axins.set_ylim(t_min, t_max)
        axins.tick_params(labelsize=6)
        # Box in the parent axes + connector lines to the inset.
        ind = ax_roc[r,c].indicate_inset_zoom(axins, edgecolor='black', alpha=0.6)
        _set_inset_connectors(ind, ax_roc[r,c], axins, ROC_INSET_LIMITS[agg_strat], ROC_INSET_CONNECT.get(agg_strat))

    # Zoom inset on the PR panel for the parameterized strategies.
    if agg_strat in PR_INSET_LIMITS:
        r_min, r_max, p_min, p_max = PR_INSET_LIMITS[agg_strat]
        axins = ax_pr[r,c].inset_axes(PR_INSET_BOX)
        for m, (recall_agg, precision_agg) in enumerate(pr_curves):
            axins.plot(recall_agg, precision_agg, color=colors[m%len(colors)], linewidth=lw, linestyle=ls, alpha=alpha)
        axins.set_xlim(r_min, r_max)
        axins.set_ylim(p_min, p_max)
        axins.tick_params(labelsize=6)
        # Box in the parent axes + connector lines to the inset.
        ind = ax_pr[r,c].indicate_inset_zoom(axins, edgecolor='black', alpha=0.6)
        _set_inset_connectors(ind, ax_pr[r,c], axins, PR_INSET_LIMITS[agg_strat], PR_INSET_CONNECT.get(agg_strat))


fig_roc.tight_layout(h_pad=3.5)
fig_pr.tight_layout(h_pad=3.5)
add_panel_labels(ax_roc)
add_panel_labels(ax_pr)
agg_loc = ''.join(locations.keys())
threshold = args.threshold
savefig(fig_roc, f'figures/aggROC_{agg_loc}_t{threshold}_{lead_time}h.png')
savefig(fig_pr, f'figures/aggPR_{agg_loc}_t{threshold}_{lead_time}h.png')

plt.close()