import os
import xarray as xr
import numpy as np

from datetime import timedelta

from scipy.interpolate import interp1d
from scipy.stats import norm, multivariate_normal

from sklearn.metrics import confusion_matrix_at_thresholds, confusion_matrix
from sklearn.isotonic import IsotonicRegression

import yaml
from pathlib import Path

with open("tools/config.yaml") as f:
    config = yaml.safe_load(f)

wb2_dir = Path(config["wb2_dir"])

REGIONS = {
    'world': {
        'lat_min': -90,
        'lat_max': 90,
        'lon_min': -180,
        'lon_max': 180,
    },
    'northern_hemisphere_extratropics': {
        'lat_min': 20,
        'lat_max': 90,
        'lon_min': -180,
        'lon_max': 180,
    },
    'southern_hemisphere_extratropics': {
        'lat_min': -90,
        'lat_max': -20,
        'lon_min': -180,
        'lon_max': 180,
    },
    'tropics': {
        'lat_min': -20,
        'lat_max': 20,
        'lon_min': -180,
        'lon_max': 180,
    },
    'north_america': {
        'lat_min': 25,
        'lat_max': 60,
        'lon_min': -145,
        'lon_max': -50,
    },
    'europe_north_africa': {
        'lat_min': 25,
        'lat_max': 70,
        'lon_min': -10,
        'lon_max': 28,
    },
    'asia': {
        'lat_min': 25,
        'lat_max': 65,
        'lon_min': 60,
        'lon_max': 145,
    },
    'australia_new_zealand': {
        'lat_min': -55,
        'lat_max': -10,
        'lon_min': 90,
        'lon_max': 180,
    },
    'northern_polar_region': {
        'lat_min': 60,
        'lat_max': 90,
        'lon_min': -180,
        'lon_max': 180,
    },
    'southern_polar_region': {
        'lat_min': -90,
        'lat_max': -60,
        'lon_min': -180,
        'lon_max': 180,
    },
    'europe': {
        'lat_min': 35,
        'lat_max': 72,
        'lon_min': -25,
        'lon_max': 45,
    }
}




#transform Kelvin to Celsius
def Kelvin2Celsius(data: xr.DataArray) -> xr.DataArray:
    return data-273.15

#unify dimension names (longitude and latitude)
def rename_latlon(data: xr.DataArray) -> xr.DataArray:
    return data.rename({'longitude': 'lon', 'latitude': 'lat'})

def change_lonlat(data: xr.DataArray) -> xr.DataArray:
    if 'lon' not in data.coords:
        data = rename_latlon(data)
    if data.lon.max() > 180:
        data = data.assign_coords(lon=(((data.lon + 180) % 360) - 180)).sortby('lon')
    if any(np.diff(data.lat) < 0):
        data = data.sortby('lat')
    return data

def load_models(wdr, year, region_name, WB_var, resolution, lead_time):
    ds_obs = xr.open_zarr(os.path.join(wdr, f'hres_t0/hres_t0_{year}_{region_name}_{WB_var}_{resolution}.zarr'))
    ds_fc1 = xr.open_zarr(os.path.join(wdr, f'hres/hres_{year}_{region_name}_{WB_var}_{resolution}.zarr')).sel(prediction_timedelta=timedelta(hours=lead_time))
    ds_fc2 = xr.open_zarr(os.path.join(wdr, f'graphcast/graphcast_{year}_{region_name}_{WB_var}_{resolution}.zarr')).sel(prediction_timedelta=timedelta(hours=lead_time))
    ds_fc3 = xr.open_zarr(os.path.join(wdr, f'pangu/pangu_{year}_{region_name}_{WB_var}_{resolution}.zarr')).sel(prediction_timedelta=timedelta(hours=lead_time))

    # ------------------------------ Matching times ------------------------------ #

    valid_times1 = ds_fc1['time'] + np.timedelta64(lead_time, 'h')
    valid_times2 = ds_fc2['time'] + np.timedelta64(lead_time, 'h')
    valid_times3 = ds_fc3['time'] + np.timedelta64(lead_time, 'h')

    obs_times = ds_obs['time']
    # Intersect across all three models as well as the obs, so every model is
    # scored on exactly the same set of valid times.
    common_times = np.intersect1d(valid_times1.values, obs_times.values)
    common_times = np.intersect1d(common_times, valid_times2.values)
    common_times = np.intersect1d(common_times, valid_times3.values)

    fc_sel1 = ds_fc1.sel(time=common_times - np.timedelta64(lead_time, 'h'))[WB_var]
    fc_sel2 = ds_fc2.sel(time=common_times - np.timedelta64(lead_time, 'h'))[WB_var]
    fc_sel3 = ds_fc3.sel(time=common_times - np.timedelta64(lead_time, 'h'))[WB_var]
    obs_sel1 = ds_obs.sel(time=common_times)[WB_var]
    obs_sel2 = ds_obs.sel(time=common_times)[WB_var]
    obs_sel3 = ds_obs.sel(time=common_times)[WB_var]

    models = {
        'hres': {'fcst': fc_sel1, 'obs': obs_sel1, 'name': 'HRES'},
        'graphcast': {'fcst': fc_sel2, 'obs': obs_sel2, 'name': 'GraphCast'},
        'pangu': {'fcst': fc_sel3, 'obs': obs_sel3, 'name': 'Pangu'}
        }
    
    return models

def select_region(models, region_name):
    for model_key in models:
        models[model_key]['fcst'] = models[model_key]['fcst'].sel(lat=slice(REGIONS[region_name]['lat_min'], REGIONS[region_name]['lat_max']), lon=slice(REGIONS[region_name]['lon_min'], REGIONS[region_name]['lon_max'])).load()
        models[model_key]['obs'] = models[model_key]['obs'].sel(lat=slice(REGIONS[region_name]['lat_min'], REGIONS[region_name]['lat_max']), lon=slice(REGIONS[region_name]['lon_min'], REGIONS[region_name]['lon_max'])).load()
    return models

# ---------------------------------------------------------------------------- #

# Strategies whose curve is traced by sweeping a gain, so their number of points
# is set by `gain_steps` rather than by `n`.
GAIN_STRATEGIES = ('GraphCast', 'location-scale')


def n_points_for_strategy(strategy, n, gain_steps=None):
    """Number of threshold points a strategy produces."""
    if gain_steps is not None and strategy is not None and strategy.startswith(GAIN_STRATEGIES):
        return gain_steps
    return n



def _confusion_matrices_one_latitude(lat_idx, d2, n, models_keys, model_data, t_Y, climatological_median_obs=None, climatological_std=None, agg_strategy=None, raw=False, gc_gain_range=(.8, 4.5), location_scale_gain_range=(-1.5, 1.5), gain_steps=None):
    n_models = len(models_keys)
    # GraphCast/location-scale sweep `gain_steps` gains; every other strategy yields n points.
    n_points = n_points_for_strategy(agg_strategy, n, gain_steps)
    row_conf = np.full((4, d2, n_models, n_points), np.nan, dtype=float)

    for lon_idx in range(d2):
        for m, model in enumerate(models_keys):
            fc_loc = model_data[model]['fcst'][:, lat_idx, lon_idx]
            if isinstance(t_Y[model], float):
                obs_bin = (model_data[model]['obs'][:, lat_idx, lon_idx] > t_Y[model]).astype(int)
            elif isinstance(t_Y[model], np.ndarray):
                obs_bin = (model_data[model]['obs'][:, lat_idx, lon_idx] > t_Y[model][:, lat_idx, lon_idx]).astype(int)
            # else:
            #     obs_bin = (model_data[model]['obs'][:, lat_idx, lon_idx] > t_Y[model].isel(lat=lat_idx, lon=lon_idx).values).astype(int)

            if raw:
                tns, fps, fns, tps, _ = confusion_matrix_at_thresholds(y_true=obs_bin, y_score=fc_loc)
            else:
                iso = IsotonicRegression().fit(X=fc_loc, y=obs_bin)
                fc_pav = np.round(iso.predict(fc_loc), decimals=6)
                if agg_strategy is None:
                    tns, fps, fns, tps, _ = confusion_matrix_at_thresholds(y_true=obs_bin, y_score=fc_pav)
                else:
                    tps, fps, fns, tns = contingency_for_strategy(
                        strategy=agg_strategy,
                        fc_loc=fc_loc,
                        fc_pav=fc_pav,
                        obs_bin=obs_bin,
                        t_Y=t_Y[model],
                        n=n,
                        climatological_median_obs=climatological_median_obs[model],
                        climatological_std=(climatological_std[model] if climatological_std is not None else None),
                        lat_idx=lat_idx,
                        lon_idx=lon_idx,
                        gc_gain_range=gc_gain_range,
                        location_scale_gain_range=location_scale_gain_range,
                        gain_steps=gain_steps,
                    )

            row_conf[1, lon_idx, m, :len(fps)] = fps
            row_conf[2, lon_idx, m, :len(fns)] = fns
            row_conf[3, lon_idx, m, :len(tns)] = tns
            row_conf[0, lon_idx, m, :len(tps)] = tps

    return lat_idx, row_conf


def confusion_matrices_whole_grid(models, t_Y, raw=False):
    n = max(len(models['hres']['fcst'].time), len(models['graphcast']['fcst'].time), len(models['pangu']['fcst'].time))
    models_keys = list(models.keys())
    n_models = len(models_keys)
    d1, d2 = models[models_keys[0]]['fcst'].lat.size, models[models_keys[0]]['fcst'].lon.size

    conf_matrices = np.full((4, d1, d2, n_models, n), np.nan, dtype=float)
    lats =  models[models_keys[0]]['fcst'].lat.values
    lons =  models[models_keys[0]]['fcst'].lon.values

    model_data = {}
    for model_key in models_keys:
        model_data[model_key] = {
            'fcst': models[model_key]['fcst'].values,
            'obs': models[model_key]['obs'].values,
            'name': models[model_key]['name']
        }

    print("Computing confusion matrices for each location...")
    for lat_idx, lon_idx in np.ndindex(d1, d2):
        if lon_idx == 0:
            print(f"Processing location {lats[lat_idx]}, {lons[lon_idx]}...")
        for i, model in enumerate(models_keys):
            fc_loc = model_data[model]['fcst'][:, lat_idx, lon_idx]
            if isinstance(t_Y[model],float):
                obs_bin = (model_data[model]['obs'][:, lat_idx, lon_idx] > t_Y[model]).astype(int)
            else:
                obs_bin = (model_data[model]['obs'][:, lat_idx, lon_idx] > t_Y[model].isel(lat=lat_idx, lon=lon_idx).values).astype(int)


            if raw:
                tns, fps, fns, tps, _ = confusion_matrix_at_thresholds(y_true = obs_bin, y_score = fc_loc)

            else:
                iso = IsotonicRegression().fit(X=fc_loc, y=obs_bin)
                fc_pav = np.round(iso.predict(fc_loc), decimals=6)
                tns, fps, fns, tps, _ = confusion_matrix_at_thresholds(y_true = obs_bin, y_score = fc_pav)
            
            conf_matrices[0, lat_idx, lon_idx, i, :len(tps)] = tps
            conf_matrices[1, lat_idx, lon_idx, i, :len(fps)] = fps
            conf_matrices[2, lat_idx, lon_idx, i, :len(fns)] = fns
            conf_matrices[3, lat_idx, lon_idx, i, :len(tns)] = tns

    return conf_matrices


def confusion_matrices_locations(models, locations, t_Y, raw=False, n=1000):
    models_keys = list(models.keys())
    n_models = len(models_keys)
    conf_matrices = np.full((4, len(locations), n_models, n), np.nan, dtype=float)

    for l, location in enumerate(locations.keys()):
        lat, lon = locations[location]
        for i, model in enumerate(models_keys):
            if isinstance(t_Y[model], float):
                obs_bin = (models[model]['obs'].sel(location=location).values > t_Y[model]).astype(int)
            else:
                lats, lons = t_Y[model].lat.values, t_Y[model].lon.values
                lat_idx, lon_idx = np.argmin(np.abs(lats - lat)), np.argmin(np.abs(lons - lon))
                obs_bin = (models[model]['obs'].sel(location=location).values > t_Y[model].isel(lat=lat_idx, lon=lon_idx).values).astype(int)

                
            fc_loc = models[model]['fcst'].sel(location=location).values

            if raw:
                tns, fps, fns, tps, _ = confusion_matrix_at_thresholds(y_true = obs_bin, y_score = fc_loc)

            else:
                iso = IsotonicRegression().fit(X=fc_loc, y=obs_bin)
                fc_pav = np.round(iso.predict(fc_loc), decimals=6)
                tns, fps, fns, tps, thresholds = confusion_matrix_at_thresholds(y_true = obs_bin, y_score = fc_pav)
                new_thresholds = np.linspace(thresholds.min()-.1, thresholds.max()+.1, n)
                tps, fps, fns, tns = interpolate_confusion_matrix(tns, fps, fns, tps, thresholds, new_thresholds)

            
            conf_matrices[0, l, i, :len(tps)] = tps
            conf_matrices[1, l, i, :len(fps)] = fps
            conf_matrices[2, l, i, :len(fns)] = fns
            conf_matrices[3, l, i, :len(tns)] = tns

    return conf_matrices

def interpolate_confusion_matrix(tns, fps, fns, tps, thresholds, new_thresholds):
    # Interpolate confusion matrix values at the given thresholds
    tns_interp = np.interp(new_thresholds, thresholds[::-1], tns[::-1])
    fps_interp = np.interp(new_thresholds, thresholds[::-1], fps[::-1])
    fns_interp = np.interp(new_thresholds, thresholds[::-1], fns[::-1])
    tps_interp = np.interp(new_thresholds, thresholds[::-1], tps[::-1])
    
    return np.array([tps_interp, fps_interp, fns_interp, tns_interp])

def new_thresholds_confusion_matrix(tns, fps, fns, tps, thresholds, new_thresholds):
    idx = np.array([np.where(thresholds == thresholds[thresholds <= t][0])[0][0] if len(thresholds[thresholds <= t])>0 else len(thresholds)-1 for t in new_thresholds])
    
    return np.array([tps[idx], fps[idx], fns[idx], tns[idx]])

def sanitize_roc_curve(fpr, tpr):
    valid = ~(np.isnan(fpr) | np.isnan(tpr))
    return fpr[valid], tpr[valid]
    # return np.concatenate((fpr[valid], [1])), np.concatenate((tpr[valid], [1]))


# ---------------------------------------------------------------------------- #
#                          PARAMETERIZATION STRATEGIES                         #
# ---------------------------------------------------------------------------- #

def parameterization_GC(fc_loc, fc_pav, obs_bin, t_Y, n, climatological_median_obs, lat_idx, lon_idx, PAV=False, interpolated_counts=False, gain_range=(.8, 4.5), gain_steps=None):
    gains = np.linspace(gain_range[0], gain_range[1], num=gain_steps if gain_steps is not None else n)
    gains = gains[gains != 0]

    if PAV:
        raise NotImplementedError("PAV-based thresholding for GC strategy is not implemented yet.")

    if isinstance(climatological_median_obs, np.ndarray):
        med_s = climatological_median_obs[:, lat_idx, lon_idx]
    else:
        med_s = climatological_median_obs.isel(lat=lat_idx, lon=lon_idx).values
    # med_s: (n_seasons,), gains: (n_gains,) → new_thresholds: (n_seasons, n_gains)
    if isinstance(t_Y, float):
        new_thresholds = med_s[:, None] + (t_Y - med_s[:, None]) / gains[None, :]
    elif isinstance(t_Y, np.ndarray):
        t_loc = t_Y[:, lat_idx, lon_idx]  # (n_seasons,)
        new_thresholds = med_s[:, None] + (t_loc[:, None] - med_s[:, None]) / gains[None, :]
    else:
        t_loc = t_Y.isel(lat=lat_idx, lon=lon_idx).values  # (n_seasons,)
        new_thresholds = med_s[:, None] + (t_loc[:, None] - med_s[:, None]) / gains[None, :]

    if interpolated_counts:
        raise NotImplementedError("Interpolation for GC strategy is not implemented yet.")
    else:
        fc_bin = fc_loc[:, None] > new_thresholds  # (n_samples, n_gains)
        obs_bool = obs_bin[:, None].astype(bool)
        tps = np.sum(fc_bin & obs_bool, axis=0)
        fps = np.sum(fc_bin & ~obs_bool, axis=0)
        fns = np.sum(~fc_bin & obs_bool, axis=0)
        tns = np.sum(~fc_bin & ~obs_bool, axis=0)
        contingency_matrix = np.array([tps, fps, fns, tns], dtype=float)  # (4, n_gains)

    return contingency_matrix

def parameterization_location_scale(fc_loc, fc_pav, obs_bin, t_Y, n, lat_idx, lon_idx, climatological_std=None, PAV=False, interpolated_counts=False, gain_range=(-1.5, 1.5), gain_steps=None):
    gains = np.linspace(gain_range[0], gain_range[1], num=gain_steps if gain_steps is not None else n)

    if PAV:
        raise NotImplementedError("PAV-based thresholding for location-scale strategy is not implemented yet.")

    if isinstance(climatological_std, np.ndarray):
        std_loc = climatological_std[:, lat_idx, lon_idx]  # (n_samples,)
    else:
        std_loc = climatological_std.isel(lat=lat_idx, lon=lon_idx).values  # (n_samples,)
    # std_loc: (n_seasons,), gains: (n_gains,) → new_thresholds: (n_seasons, n_gains)
    if isinstance(t_Y, float):
        new_thresholds = t_Y - gains[None, :] * std_loc[:, None]
    elif isinstance(t_Y, np.ndarray):
        t_loc = t_Y[:, lat_idx, lon_idx]  # (n_seasons,)
        new_thresholds = t_loc[:, None] - gains[None, :] * std_loc[:, None]
    else:
        t_loc = t_Y.isel(lat=lat_idx, lon=lon_idx).values  # (n_seasons,)
        new_thresholds = t_loc[:, None] - gains[None, :] * std_loc[:, None]

    fc_bin = fc_loc[:, None] > new_thresholds  # (n_samples, n_gains)
    obs_bool = obs_bin[:, None].astype(bool)       # (n_samples, 1)
    tps = np.sum(fc_bin & obs_bool, axis=0)
    fps = np.sum(fc_bin & ~obs_bool, axis=0)
    fns = np.sum(~fc_bin & obs_bool, axis=0)
    tns = np.sum(~fc_bin & ~obs_bool, axis=0)
    contingency_matrix = np.array([tps, fps, fns, tns], dtype=float)  # (4, n_gains)

    return contingency_matrix

def _degenerate_frequency_contingency(N, p, u):
    """Contingency counts at a location whose base rate is 0 or 1.

    The line swept by the FB parameterizations is p*tpr + (1-p)*fpr = u, and it
    degenerates when every sample shares the same outcome:

      p = 0 : no positives, so tpr is undefined and the line becomes the
              vertical fpr = u. The intersection fixes fps = u*(N - n_p) = u*N
              and tns = (1-u)*N, while tps = fns = 0 identically.
      p = 1 : no negatives, so fpr is undefined and the line becomes the
              horizontal tpr = u. The intersection fixes tps = u*n_p = u*N and
              fns = (1-u)*N, while fps = tns = 0 identically.

    In both cases the surviving coordinate alone determines every count, so no
    ROC hull is needed.
    """
    zeros = np.zeros_like(u)
    if p <= 0.0:
        return np.array([zeros, u * N, zeros, (1.0 - u) * N])
    return np.array([u * N, zeros, (1.0 - u) * N, zeros])


def parameterization_FB(fc_pav, obs_bin, n):
    tns, fps, fns, tps, _ = confusion_matrix_at_thresholds(y_true = obs_bin, y_score = fc_pav)

    N = len(obs_bin)
    p = np.mean(obs_bin)
    n_p = N*p
    u = np.linspace(0,1,n)
    if p <= 0.0 or p >= 1.0:
        return _degenerate_frequency_contingency(N, p, u)
    FB = (1+(1-p)/p) * u

    fpr, tpr, _, _ = compute_roc_pr(tns=tns, fps=fps, fns=fns, tps=tps, extend='both')
    new_fpr, new_tpr = intersection_segment_ROC(fpr=fpr, tpr=tpr, FB=FB, p=p)
    new_tps = n_p * new_tpr
    new_fps = new_fpr * (N - n_p)
    new_fns = n_p - new_tps
    new_tns = N - new_tps - new_fps - new_fns

    return np.array([new_tps, new_fps, new_fns, new_tns])

def parameterization_parallel(fc_pav, obs_bin, n):
    P_PARAM = .5
    tns, fps, fns, tps, _ = confusion_matrix_at_thresholds(y_true = obs_bin, y_score = fc_pav)

    N = len(obs_bin)
    p = np.mean(obs_bin)
    n_p = N*p
    u = np.linspace(0,1,n)
    # Here the swept line keeps slope -1 whatever p is, but the ROC curve itself
    # collapses when p is 0 or 1 (one of tpr/fpr is undefined), so the same
    # vertical/horizontal limit applies.
    if p <= 0.0 or p >= 1.0:
        return _degenerate_frequency_contingency(N, p, u)
    FB = (1+(1-P_PARAM)/P_PARAM) * u

    fpr, tpr, _, _ = compute_roc_pr(tns=tns, fps=fps, fns=fns, tps=tps, extend='both')
    new_fpr, new_tpr = intersection_segment_ROC(fpr=fpr, tpr=tpr, FB=FB, p=P_PARAM)
    new_tps = n_p * new_tpr
    new_fps = new_fpr * (N - n_p)
    new_fns = n_p - new_tps
    new_tns = N - new_tps - new_fps - new_fns

    return np.array([new_tps, new_fps, new_fns, new_tns])

# ---------------------------------------------------------------------------- #

def compute_roc_pr(tps, fps, fns, tns, extend='both'):
    # fpr, tpr = fps/(fps+tns), tps/(tps+fns)
    fpr, tpr = np.divide(fps, fps+tns, where=(fps+tns)!=0, out=np.ones_like(fps)*np.nan), np.divide(tps, tps+fns, where=(tps+fns)!=0, out=np.ones_like(tps)*np.nan)
    rev = False
    if fpr[0] > fpr[-1]:
        rev = True
        fpr, tpr = fpr[::-1], tpr[::-1]
    if (extend == 'left' or extend == 'both'):# and fpr[0] != 0:
        fpr, tpr = np.concatenate(([0], fpr)), np.concatenate(([0], tpr))
    precision, recall = np.divide(tps, tps+fps, where=(tps+fps)!=0, out=np.ones_like(tps)*np.nan), tpr
    if (extend == 'right' or extend == 'both'):# and fpr[-1] != 1:
        fpr, tpr = np.concatenate((fpr, [1])), np.concatenate((tpr, [1]))
    
    if rev:
        precision = precision[::-1]
    if extend == 'left' or extend == 'both':
        precision = np.concatenate(([precision[0]], precision))
    # if extend == 'right' or extend == 'both':
    #     p = np.nanmean(tps+fns) / np.nanmean(tps+fps+fns+tns)
    #     precision = np.concatenate((precision, [p]))
    return fpr, tpr, recall, precision

def intersection_segment_ROC(fpr, tpr, FB, p):
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    FB = np.atleast_1d(FB)

    # Degenerate or invalid base rates do not define meaningful FB lines.
    if not np.isfinite(p) or p <= 0.0 or p >= 1.0:
        return np.full(FB.shape, np.nan, dtype=float), np.full(FB.shape, np.nan, dtype=float)

    # ROC segments
    x1 = fpr[:-1]
    y1 = tpr[:-1]
    x2 = fpr[1:]
    y2 = tpr[1:]

    one_minus_p = 1.0 - p
    offset = 0.1 * p / one_minus_p
    ratio = p / one_minus_p
    inv_ratio = one_minus_p / p
    x_FB1 = np.maximum(0.0, (FB - 1.0) * ratio) - offset
    x_FB2 = np.minimum(1.0, FB * ratio) + offset
    y_FB1 = FB - inv_ratio * x_FB1
    y_FB2 = FB - inv_ratio * x_FB2

    # Broadcast over (n_fb, n_segments)
    x1_b, y1_b = x1[None, :], y1[None, :]
    x2_b, y2_b = x2[None, :], y2[None, :]
    x_FB1_b, y_FB1_b = x_FB1[:, None], y_FB1[:, None]
    x_FB2_b, y_FB2_b = x_FB2[:, None], y_FB2[:, None]

    denom = (x_FB1_b - x_FB2_b) * (y1_b - y2_b) - (y_FB1_b - y_FB2_b) * (x1_b - x2_b)
    valid = denom != 0
    denom_safe = np.where(valid, denom, 1.0)

    px = np.where(
        valid,
        ((x_FB1_b * y_FB2_b - y_FB1_b * x_FB2_b) * (x1_b - x2_b) -
         (x_FB1_b - x_FB2_b) * (x1_b * y2_b - y1_b * x2_b)) / denom_safe,
        np.nan,
    )
    py = np.where(
        valid,
        ((x_FB1_b * y_FB2_b - y_FB1_b * x_FB2_b) * (y1_b - y2_b) -
         (y_FB1_b - y_FB2_b) * (x1_b * y2_b - y1_b * x2_b)) / denom_safe,
        np.nan,
    )

    px = np.clip(px, 0.0, 1.0)
    py = np.clip(py, 0.0, 1.0)

    cond = (
        valid &
        (px >= np.minimum(x_FB1_b, x_FB2_b)) &
        (px <= np.maximum(x_FB1_b, x_FB2_b)) &
        (py >= np.minimum(y_FB1_b, y_FB2_b)) &
        (py <= np.maximum(y_FB1_b, y_FB2_b)) &
        (px >= np.minimum(x1_b, x2_b)) &
        (px <= np.maximum(x1_b, x2_b)) &
        (py >= np.minimum(y1_b, y2_b)) &
        (py <= np.maximum(y1_b, y2_b))
    )

    has_hit = np.any(cond, axis=1)
    first_idx = np.argmax(cond, axis=1)

    new_fpr = np.full(FB.shape, np.nan, dtype=float)
    new_tpr = np.full(FB.shape, np.nan, dtype=float)
    hit_rows = np.where(has_hit)[0]
    if hit_rows.size > 0:
        new_fpr[hit_rows] = px[hit_rows, first_idx[hit_rows]]
        new_tpr[hit_rows] = py[hit_rows, first_idx[hit_rows]]

    return new_fpr, new_tpr


def prepare_interp_curve(contingency_matrices_raw, lat_idx, lon_idx, model_idx, x_common):
    tps, fps, fns, tns = contingency_matrices_raw[:, lat_idx, lon_idx, model_idx, :]
    condition = np.logical_or(np.isnan(tps),
                              np.logical_or(np.isnan(fps), 
                                            np.logical_or(np.isnan(fns), np.isnan(tns))))
    tps = tps[~condition]
    fps = fps[~condition]
    fns = fns[~condition]
    tns = tns[~condition]

    if tps.size == 0 or fns.size == 0 or (np.sum(tps) + np.sum(fns) == 0):
        return None

    fpr, tpr, _, _ = compute_roc_pr(tps=tps, fps=fps, fns=fns, tns=tns, extend='both')
    fpr, tpr = sanitize_roc_curve(fpr, tpr)
    return np.interp(x_common, fpr, tpr)



# ---------------------------------------------------------------------------- #
#                                CHECK DOMINANCE                               #
# ---------------------------------------------------------------------------- #


def check_dominance_latitude(
    lat_idx,
    d2,
    contingency_matrices_raw,
    x_common,
):
    row_hg = np.zeros(d2, dtype=float)
    row_hp = np.zeros(d2, dtype=float)
    row_gp = np.zeros(d2, dtype=float)
    pair_positions = {(0, 1): 0, (0, 2): 1, (1, 2): 2}

    for lon_idx in range(d2):
        y_interp = [
            prepare_interp_curve(contingency_matrices_raw, lat_idx, lon_idx, m, x_common)
            for m in range(3)
        ]

        for m1, m2 in ((0, 1), (0, 2), (1, 2)):
            y1_interp = y_interp[m1]
            y2_interp = y_interp[m2]
            if y1_interp is None or y2_interp is None:
                continue

            test_1_dom_2 = np.all(y1_interp >= y2_interp) and np.any(y1_interp > y2_interp)
            test_2_dom_1 = np.all(y2_interp >= y1_interp) and np.any(y2_interp > y1_interp)

            if test_1_dom_2:
                dom_value = 1
            elif test_2_dom_1:
                dom_value = -1
            else:
                continue

            pair_pos = pair_positions[(m1, m2)]
            if pair_pos == 0:
                row_hg[lon_idx] = dom_value
            elif pair_pos == 1:
                row_hp[lon_idx] = dom_value
            else:
                row_gp[lon_idx] = dom_value

    return lat_idx, row_hg, row_hp, row_gp

# ---------------------------------------------------------------------------- #
#                         CHECK DOMINANCE PRESERVATION                         #
# ---------------------------------------------------------------------------- #

def precompute_point_series(fcst_values, obs_values, points, t_Y):
    point_series = []
    for lat_idx, lon_idx in points:
        fc_loc = fcst_values[:, lat_idx, lon_idx]
        if isinstance(t_Y, np.ndarray):
            obs_bin = (obs_values[:, lat_idx, lon_idx] > t_Y[:, lat_idx, lon_idx]).astype(np.int8)
        elif isinstance(t_Y, xr.DataArray):
            obs_bin = (obs_values[:, lat_idx, lon_idx] > t_Y.isel(lat=lat_idx, lon=lon_idx).values).astype(np.int8)
        else:
            obs_bin = (obs_values[:, lat_idx, lon_idx] > t_Y).astype(np.int8)
        iso = IsotonicRegression().fit(X=fc_loc, y=obs_bin)
        fc_pav = np.round(iso.predict(fc_loc), decimals=6)
        point_series.append((fc_loc, fc_pav, obs_bin))
    return point_series


def precompute_strategy_data(point_series, agg_strategies, n, t_Y, points, climatological_median_obs, climatological_std=None):
    n_locations = len(point_series)
    n_strats = len(agg_strategies)
    contingency_data = np.full((n_locations, n_strats, 4, n), np.nan, dtype=np.float32)
    for point_idx, (fc_loc, fc_pav, obs_bin) in enumerate(point_series):
        lat_idx, lon_idx = points[point_idx]
        for strat_idx, agg_strat in enumerate(agg_strategies):
            contingency = contingency_for_strategy(strategy=agg_strat, fc_loc=fc_loc, fc_pav=fc_pav, obs_bin=obs_bin, t_Y=t_Y, n=n, climatological_median_obs=climatological_median_obs, lat_idx=lat_idx, lon_idx=lon_idx, climatological_std=climatological_std)
            contingency_data[point_idx, strat_idx, :, :] = contingency.astype(np.float32)
    return contingency_data



def contingency_for_strategy(strategy, fc_loc, fc_pav, obs_bin, t_Y, n, climatological_median_obs, lat_idx, lon_idx, climatological_std=None, gc_gain_range=(.8, 4.5), location_scale_gain_range=(-1.5, 1.5), gain_steps=None):
    if strategy == 'GraphCast':
        return parameterization_GC(fc_loc, fc_pav, obs_bin, t_Y, n, climatological_median_obs, lat_idx, lon_idx, PAV=False, interpolated_counts=False, gain_range=gc_gain_range, gain_steps=gain_steps)
    if strategy == 'location-scale':
        return parameterization_location_scale(fc_loc, fc_pav, obs_bin, t_Y, n, lat_idx, lon_idx, climatological_std=climatological_std, PAV=False, interpolated_counts=False, gain_range=location_scale_gain_range, gain_steps=gain_steps)
    if strategy == 'FB':
        return parameterization_FB(fc_pav, obs_bin, n)
    if strategy == 'parallel':
        return parameterization_parallel(fc_pav, obs_bin, n)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

MODEL_DICT = {
    'hres': [0, 'HRES'],
    'graphcast': [1, 'GC'],
    'pangu': [2, 'Pangu']
}
def check_dominance_pairs(
    l1,
    points,
    precomputed_by_model,
    lats,
    lons,
    x_common,
    agg_strategies,
    dominance_scenario,
    crossing_tol=1e-6,
    min_violation_area=1e-7,
):  
    
    models = dominance_scenario.split('>')
    m1 = MODEL_DICT[models[0]][0]
    m2 = MODEL_DICT[models[1]][0]
    m3 = MODEL_DICT[models[2]][0]
    m1_name = MODEL_DICT[models[0]][1]
    m2_name = MODEL_DICT[models[1]][1]
    m3_name = MODEL_DICT[models[2]][1]
    
    pair_violations = []
    lat_idx1, lon_idx1 = points[l1]

    for l2 in range(l1 + 1, len(points)):
        lat_idx2, lon_idx2 = points[l2]
        dict_key = f'({lats[lat_idx1]}, {lons[lon_idx1]}) -- ({lats[lat_idx2]}, {lons[lon_idx2]})'

        violation_entries = []
        for strat_idx, agg_strat in enumerate(agg_strategies):
            counts1 = precomputed_by_model[m1][l1, strat_idx, :, :] + precomputed_by_model[m1][l2, strat_idx, :, :]
            counts2 = precomputed_by_model[m2][l1, strat_idx, :, :] + precomputed_by_model[m2][l2, strat_idx, :, :]
            counts3 = precomputed_by_model[m3][l1, strat_idx, :, :] + precomputed_by_model[m3][l2, strat_idx, :, :]

            tps1, fps1, fns1, tns1 = counts1
            tps2, fps2, fns2, tns2 = counts2
            tps3, fps3, fns3, tns3 = counts3

            fpr_agg1, tpr_agg1, _, _ = compute_roc_pr(
                tns=tns1, fps=fps1, fns=fns1, tps=tps1
            )
            fpr_agg2, tpr_agg2, _, _ = compute_roc_pr(
                tns=tns2, fps=fps2, fns=fns2, tps=tps2
            )
            fpr_agg3, tpr_agg3, _, _ = compute_roc_pr(
                tns=tns3, fps=fps3, fns=fns3, tps=tps3
            )

            y1_interp = np.interp(x_common, fpr_agg1, tpr_agg1)
            y2_interp = np.interp(x_common, fpr_agg2, tpr_agg2)
            y3_interp = np.interp(x_common, fpr_agg3, tpr_agg3)

            crosses_12 = np.any(y1_interp + crossing_tol < y2_interp)
            crosses_13 = np.any(y1_interp + crossing_tol < y3_interp)
            crosses_23 = np.any(y2_interp + crossing_tol < y3_interp)
            

            combined_test = crosses_12 or crosses_13 or crosses_23

            if combined_test:
                auc_13 = np.trapezoid(np.where(y1_interp + crossing_tol < y3_interp, y1_interp - y3_interp, 0), x_common)
                auc_23 = np.trapezoid(np.where(y2_interp + crossing_tol < y3_interp, y2_interp - y3_interp, 0), x_common)
                auc_12 = np.trapezoid(np.where(y1_interp + crossing_tol < y2_interp, y1_interp - y2_interp, 0), x_common)
                n1_total = np.nansum(tps1[0] + fps1[0] + fns1[0] + tns1[0])
                base_rate = float((tps1[0] + fns1[0]) / n1_total) if n1_total > 0 else float('nan')
                violation_entries.append(
                    {
                        agg_strat: {
                            f'AUC {m1_name} - {m2_name}': float(auc_12),
                            f'AUC {m1_name} - {m3_name}': float(auc_13),
                            f'AUC {m2_name} - {m3_name}': float(auc_23),
                            f'{m1_name} >= {m2_name}': int(not crosses_12),
                            f'{m1_name} >= {m3_name}': int(not crosses_13),
                            f'{m2_name} >= {m3_name}': int(not crosses_23),
                            f'AUC {m1_name}': float(np.trapezoid(y1_interp, x_common)),
                            f'AUC {m2_name}': float(np.trapezoid(y2_interp, x_common)),
                            f'AUC {m3_name}': float(np.trapezoid(y3_interp, x_common)),
                            'base_rate': base_rate,
                        }
                    }
                )

        if violation_entries:
            pair_violations.append(
                {
                    dominance_scenario: {
                        dict_key: violation_entries,
                    }
                }
            )

    return pair_violations


def get_climatological_quantile(level, months, hours, region_name, WB_var, resolution):
    ds = xr.open_zarr(
        f"{wb2_dir}/era5_quantiles/era5-quantiles_1979-2019_world_{WB_var}_{resolution}.zarr"
    )[WB_var].sel(
        lat=slice(REGIONS[region_name]['lat_min'], REGIONS[region_name]['lat_max']), lon=slice(REGIONS[region_name]['lon_min'], REGIONS[region_name]['lon_max']),
        quantile=level,
        month=months,
        hour=hours,
        )
    return ds


def get_climatological_max(months, region_name, WB_var, resolution):
    ds = xr.open_zarr(
        f"{wb2_dir}/era5_records/era5-record_1979-2019_world_{WB_var}_{resolution}.zarr"
    )[WB_var].sel(
        lat=slice(REGIONS[region_name]['lat_min'], REGIONS[region_name]['lat_max']),
        lon=slice(REGIONS[region_name]['lon_min'], REGIONS[region_name]['lon_max']),
    ).max(dim='hour').sel(month=months)
    return ds


def get_climatological_std(model_key, lead_time, months, region_name, WB_var, resolution, year=2020):
    ds = xr.open_zarr(
        f"{wb2_dir}/forecast_std/{model_key}-std_{year}_world_{WB_var}_{resolution}.zarr"
    )[WB_var].sel(
        lat=slice(REGIONS[region_name]['lat_min'], REGIONS[region_name]['lat_max']),
        lon=slice(REGIONS[region_name]['lon_min'], REGIONS[region_name]['lon_max']),
        prediction_timedelta=timedelta(hours=lead_time),
        month=months,
    )
    return ds

# ---------------------------------------------------------------------------- #

