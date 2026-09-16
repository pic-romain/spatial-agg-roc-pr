import numpy as np

import sys
from pathlib import Path
tools_parent = Path(__file__).parent.parent.resolve()
sys.path.append(str(tools_parent))
from tools.utils import REGIONS, _confusion_matrices_one_latitude, get_climatological_max, get_climatological_quantile, get_climatological_std, load_models, n_points_for_strategy
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import dask
    HAS_DASK = True
except ImportError:
    HAS_DASK = False

import yaml
from pathlib import Path

with open("tools/config.yaml") as f:
    config = yaml.safe_load(f)

import argparse
parser = argparse.ArgumentParser()

parser.add_argument(
    "--max-workers",
    type=int,
    default=8,
    help="Number of parallel workers for pair evaluation.",
)
parser.add_argument(
    "--parallel-backend",
    choices=("threading", "loky"),
    default="threading",
    help="joblib backend. Use threading to avoid process-level data duplication.",
)
parser.add_argument(
    "--lat-chunk-size",
    type=int,
    default=1,
    help="Number of latitudes to compute per dispatch batch. Lower values reduce peak memory.",
)
parser.add_argument(
    "--lat-block-size",
    type=int,
    default=8,
    help="Number of latitude rows to load in memory at once.",
)
parser.add_argument(
    "--xarray-chunk-lat",
    type=int,
    default=0,
    help="Optional xarray chunk size along latitude (0 disables explicit rechunk).",
)
parser.add_argument(
    "--xarray-chunk-lon",
    type=int,
    default=0,
    help="Optional xarray chunk size along longitude (0 disables explicit rechunk).",
)
parser.add_argument(
    "--threshold",
    default="q90",
    help="Threshold for binary classification. Can be a float (e.g., 300.0) or a quantile string (e.g., 'q75')."
)
parser.add_argument(
    "--agg-strategy",
    choices=(
        'GraphCast',
        'location-scale',
        'FB',
        'parallel',
    ),
    default=None,
    help="Run a single aggregation strategy instead of looping over all of them."
)
parser.add_argument(
    "--lead-time",
    type=int,
    default=240,
    help="Forecast lead time in hours.",
)
parser.add_argument(
    "--land",
    action="store_true",
    help="Restrict accumulation to land grid points (requires regionmask).",
)
parser.add_argument(
    "--summer",
    action="store_true",
    help="Restrict to local summer months: JAS (Jul–Sep) for NH, DJF (Dec–Feb) for SH.",
)
parser.add_argument(
    "--no-antarctic",
    action="store_true",
    help="Exclude Antarctic land points (lat < -60) when --land is active.",
)
parser.add_argument(
    "--gc-gain-range",
    type=float,
    nargs=2,
    default=[0.8, 4.5],
    metavar=("MIN", "MAX"),
    help="Gain range for GraphCast parameterization (default: 0.8 4.5).",
)
parser.add_argument(
    "--location-scale-gain-range",
    type=float,
    nargs=2,
    default=[-1.5, 1.5],
    metavar=("MIN", "MAX"),
    help="Gain range for location-scale parameterization (default: -1.5 1.5).",
)
parser.add_argument(
    "--gain-steps",
    type=int,
    default=1000,
    help="Number of gain steps (np.linspace num) for GraphCast and location-scale parameterizations. Defaults to n (1000).",
)
args = parser.parse_args()

WB_var = '2m_temperature'
region_name = 'world'
year = 2020
resolution = '0p25' # '0p25' or 'low-res'
wdr = Path(config["wb2_dir"])

lead_time = args.lead_time
max_workers = args.max_workers
parallel_backend = args.parallel_backend
lat_chunk_size = max(1, args.lat_chunk_size)
lat_block_size = max(1, args.lat_block_size)
xarray_chunk_lat = max(0, args.xarray_chunk_lat)
xarray_chunk_lon = max(0, args.xarray_chunk_lon)

if args.agg_strategy is not None:
    agg_strategies = (args.agg_strategy,)
else:
    agg_strategies = (
        'GraphCast',
        'location-scale',
        'FB',
        'parallel',
    )
regions_list = ['world']#, 'northern_hemisphere_extratropics', 'southern_hemisphere_extratropics', 'tropics', 'north_america', 'europe_north_africa', 'asia', 'australia_new_zealand', 'northern_polar_region', 'southern_polar_region']


print('Loading data...')
models = load_models(wdr=wdr, year=year, region_name=region_name, WB_var=WB_var, resolution=resolution, lead_time=lead_time)

if (xarray_chunk_lat > 0 or xarray_chunk_lon > 0) and not HAS_DASK:
    print("Warning: xarray chunk options requested but dask is unavailable; skipping rechunk.")

if HAS_DASK and (xarray_chunk_lat > 0 or xarray_chunk_lon > 0):
    chunk_kwargs = {}
    if xarray_chunk_lat > 0:
        chunk_kwargs['lat'] = xarray_chunk_lat
    if xarray_chunk_lon > 0:
        chunk_kwargs['lon'] = xarray_chunk_lon
    for model in models.keys():
        models[model]['fcst'] = models[model]['fcst'].chunk(chunk_kwargs)
        models[model]['obs'] = models[model]['obs'].chunk(chunk_kwargs)

if args.threshold.startswith('q'):
    t_Y = {}
    for model in models.keys():
        q_da = get_climatological_quantile(
            level=float(args.threshold[1:]) / 100,
            months=models[model]['obs'].time.dt.month,
            hours=models[model]['obs'].time.dt.hour,
            region_name='world',
            WB_var=WB_var,
            resolution=resolution,
        )
        if HAS_DASK and (xarray_chunk_lat > 0 or xarray_chunk_lon > 0):
            chunk_kwargs = {}
            if xarray_chunk_lat > 0:
                chunk_kwargs['lat'] = xarray_chunk_lat
            if xarray_chunk_lon > 0:
                chunk_kwargs['lon'] = xarray_chunk_lon
            q_da = q_da.chunk(chunk_kwargs)
        t_Y[model] = q_da
elif args.threshold == 'max':
    t_Y = {}
    for model in models.keys():
        m_da = get_climatological_max(
            months=models[model]['obs'].time.dt.month,
            region_name='world',
            WB_var=WB_var,
            resolution=resolution,
        )
        if HAS_DASK and (xarray_chunk_lat > 0 or xarray_chunk_lon > 0):
            chunk_kwargs = {}
            if xarray_chunk_lat > 0:
                chunk_kwargs['lat'] = xarray_chunk_lat
            if xarray_chunk_lon > 0:
                chunk_kwargs['lon'] = xarray_chunk_lon
            m_da = m_da.chunk(chunk_kwargs)
        t_Y[model] = m_da
else:
    t_Y = {
        model: 273.15 + float(args.threshold)
        for model in models.keys()
    }

lats =  models['hres']['fcst'].lat.values
lons =  models['hres']['fcst'].lon.values


print("Data loaded. Starting analysis loop.")

models_keys = list(models.keys())
lats =  models[models_keys[0]]['fcst'].lat.values
lons =  models[models_keys[0]]['fcst'].lon.values
d1, d2 = models[models_keys[0]]['fcst'].lat.size, models[models_keys[0]]['fcst'].lon.size

climatological_median_obs = {
    model: get_climatological_quantile(level=0.5, months=models[model]['obs'].time.dt.month, hours=models[model]['obs'].time.dt.hour, region_name='world', WB_var=WB_var, resolution=resolution)
    for model in models.keys()
}
if HAS_DASK and (xarray_chunk_lat > 0 or xarray_chunk_lon > 0):
    chunk_kwargs = {}
    if xarray_chunk_lat > 0:
        chunk_kwargs['lat'] = xarray_chunk_lat
    if xarray_chunk_lon > 0:
        chunk_kwargs['lon'] = xarray_chunk_lon
    for model in climatological_median_obs:
        climatological_median_obs[model] = climatological_median_obs[model].chunk(chunk_kwargs)

# Per-sample forecast std (precomputed per lead time and valid-time month), used by the
# Location-scale strategy. Indexed by valid-time month → dims (time, lat, lon).
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
if HAS_DASK and (xarray_chunk_lat > 0 or xarray_chunk_lon > 0):
    chunk_kwargs = {}
    if xarray_chunk_lat > 0:
        chunk_kwargs['lat'] = xarray_chunk_lat
    if xarray_chunk_lon > 0:
        chunk_kwargs['lon'] = xarray_chunk_lon
    for model in climatological_std:
        climatological_std[model] = climatological_std[model].chunk(chunk_kwargs)

n = 1000
n_models = len(models_keys)

gc_gain_range = tuple(args.gc_gain_range)
location_scale_gain_range = tuple(args.location_scale_gain_range)
gain_steps = args.gain_steps

land_mask_2d = None
if args.land:
    import regionmask
    _land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    land_mask_2d = _land.mask(lons, lats).notnull().values  # (d1, d2), True = land
    if args.no_antarctic:
        land_mask_2d[lats < -60, :] = False
    print(f"Land mask: {land_mask_2d.sum()} / {land_mask_2d.size} grid points kept.")

file_suffix = (
    ('_land' if args.land else '')
    + ('_noant' if (args.land and args.no_antarctic) else '')
    + ('_summer' if args.summer else '')
)

# Hemisphere-aware summer: precompute time masks and latitude masks per pass.
# hem_lat_mask is read by _reduce_row_into_regions; updated each pass below.
hem_lat_mask = None
if args.summer:
    summer_passes = [
        {'name': 'NH', 'months': [7, 8, 9],  'lat_mask': lats >= 0},
        {'name': 'SH', 'months': [12, 1, 2], 'lat_mask': lats < 0},
    ]
else:
    summer_passes = [{'name': 'all', 'months': None, 'lat_mask': None}]

# Pre-compute region masks once to avoid repeated indexing work.
region_masks = {}
for region in regions_list:
    if region == 'world':
        region_masks[region] = {
            'lat_mask': np.ones(d1, dtype=bool),
            'lon_indices': np.arange(d2, dtype=int),
        }
        continue

    lat_indices = np.where((REGIONS[region]['lat_min'] <= lats) & (lats <= REGIONS[region]['lat_max']))[0]
    lon_indices = np.where((REGIONS[region]['lon_min'] <= lons) & (lons <= REGIONS[region]['lon_max']))[0]
    lat_mask = np.zeros(d1, dtype=bool)
    lat_mask[lat_indices] = True
    region_masks[region] = {
        'lat_mask': lat_mask,
        'lon_indices': lon_indices,
    }


def _reduce_row_into_regions(lat_idx, row_conf, region_accumulators, region_valid_counts):
    """Accumulate one latitude slice into region totals and valid-sample counts."""
    if hem_lat_mask is not None and not hem_lat_mask[lat_idx]:
        return
    for region in regions_list:
        mask = region_masks[region]
        if not mask['lat_mask'][lat_idx]:
            continue

        lon_indices = mask['lon_indices']
        if land_mask_2d is not None:
            lon_indices = lon_indices[land_mask_2d[lat_idx, lon_indices]]
        if lon_indices.size == 0:
            continue

        selected = row_conf[:, lon_indices, :, :]
        region_accumulators[region] += np.nansum(selected, axis=1)
        region_valid_counts[region] += np.sum(~np.isnan(selected), axis=1)


def _iter_latitude_rows(agg_strategy, block_d1, model_data_block, t_Y_block, climatological_median_obs_block, climatological_std_block):
    """Yield (lat_idx, row_conf) while keeping memory bounded."""
    if HAS_JOBLIB and max_workers > 1:

        print(
            f"Computing confusion matrices with max_workers={max_workers}, "
            f"backend={parallel_backend}, lat_chunk_size={lat_chunk_size}..."
        )

        try:
            row_generator = Parallel(
                n_jobs=max_workers,
                backend=parallel_backend,
                return_as='generator',
                pre_dispatch=lat_chunk_size * max_workers,
                batch_size=lat_chunk_size,
            )(
                delayed(_confusion_matrices_one_latitude)(
                    lat_idx=lat_idx,
                    d2=d2,
                    n=n,
                    models_keys=models_keys,
                    model_data=model_data_block,
                    t_Y=t_Y_block,
                    climatological_median_obs=climatological_median_obs_block,
                    climatological_std=climatological_std_block,
                    agg_strategy=agg_strategy,
                    gc_gain_range=gc_gain_range,
                    location_scale_gain_range=location_scale_gain_range,
                    gain_steps=gain_steps,
                )
                for lat_idx in range(block_d1)
            )
            for lat_idx, row_conf in row_generator:
                yield lat_idx, row_conf
            return
        except TypeError:
            # Older joblib versions do not support return_as.
            print("joblib version without streaming support detected; using chunked fallback.")
            for start in range(0, block_d1, lat_chunk_size):
                stop = min(start + lat_chunk_size, block_d1)
                chunk_rows = Parallel(n_jobs=max_workers, backend=parallel_backend)(
                    delayed(_confusion_matrices_one_latitude)(
                        lat_idx=lat_idx,
                        d2=d2,
                        n=n,
                        models_keys=models_keys,
                        model_data=model_data_block,
                        t_Y=t_Y_block,
                        climatological_median_obs=climatological_median_obs_block,
                        climatological_std=climatological_std_block,
                        agg_strategy=agg_strategy,
                    )
                    for lat_idx in range(start, stop)
                )
                for lat_idx, row_conf in chunk_rows:
                    yield lat_idx, row_conf
            return

    if not HAS_JOBLIB and max_workers > 1:
        print("joblib is unavailable; falling back to sequential execution.")
    print("Computing confusion matrices sequentially...")
    for lat_idx in range(block_d1):
        yield _confusion_matrices_one_latitude(
            lat_idx=lat_idx,
            d2=d2,
            n=n,
            models_keys=models_keys,
            model_data=model_data_block,
            t_Y=t_Y_block,
            climatological_median_obs=climatological_median_obs_block,
            climatological_std=climatological_std_block,
            agg_strategy=agg_strategy,
            gc_gain_range=gc_gain_range,
            location_scale_gain_range=location_scale_gain_range,
            gain_steps=gain_steps,
        )


for s,agg_strategy in enumerate(agg_strategies):
    print(f"Starting strategy: {agg_strategy}")
    # GraphCast/location-scale produce one point per gain step; the others produce n.
    n_points = n_points_for_strategy(agg_strategy, n, gain_steps)
    region_accumulators = {
        region: np.zeros((4, n_models, n_points), dtype=np.float64)
        for region in regions_list
    }
    region_valid_counts = {
        region: np.zeros((4, n_models, n_points), dtype=np.int64)
        for region in regions_list
    }

    for spass in summer_passes:
        hem_lat_mask = spass['lat_mask']
        if spass['name'] != 'all':
            _n_ex = np.isin(models[models_keys[0]]['obs'].time.dt.month.values, spass['months']).sum()
            print(f"  Summer pass: {spass['name']} (~{_n_ex} timesteps for first model)")

        for lat_start in range(0, d1, lat_block_size):
            lat_stop = min(lat_start + lat_block_size, d1)
            # Skip block if no latitude belongs to this hemisphere pass.
            if hem_lat_mask is not None and not np.any(hem_lat_mask[lat_start:lat_stop]):
                continue
            block_d1 = lat_stop - lat_start
            print(f"Processing latitude block [{lat_start}, {lat_stop}) for strategy {agg_strategy}")

            model_data_block = {}
            t_Y_block = {}
            climatological_median_obs_block = {}
            climatological_std_block = {}
            for model_key in models_keys:
                # Per-model time mask: each model may have a different number of timesteps.
                if spass['months'] is not None:
                    tm = np.isin(models[model_key]['obs'].time.dt.month.values, spass['months'])
                else:
                    tm = slice(None)

                model_data_block[model_key] = {
                    'fcst': models[model_key]['fcst'].isel(lat=slice(lat_start, lat_stop)).values[tm],
                    'obs':  models[model_key]['obs'].isel(lat=slice(lat_start, lat_stop)).values[tm],
                    'name': models[model_key]['name'],
                }

                if isinstance(t_Y[model_key], float):
                    t_Y_block[model_key] = t_Y[model_key]
                else:
                    t_Y_block[model_key] = t_Y[model_key].isel(lat=slice(lat_start, lat_stop)).values[tm]

                climatological_median_obs_block[model_key] = climatological_median_obs[model_key].isel(lat=slice(lat_start, lat_stop)).values[tm]
                climatological_std_block[model_key] = climatological_std[model_key].isel(lat=slice(lat_start, lat_stop)).values[tm]

            for local_lat_idx, row_conf in _iter_latitude_rows(agg_strategy, block_d1, model_data_block, t_Y_block, climatological_median_obs_block, climatological_std_block):
                global_lat_idx = lat_start + local_lat_idx
                _reduce_row_into_regions(global_lat_idx, row_conf, region_accumulators, region_valid_counts)

            del model_data_block
            del t_Y_block
            del climatological_median_obs_block
            del climatological_std_block

    entry_names = ['TP', 'FP', 'FN', 'TN']
    csv_cols = ','.join(f"{m}_{e}" for m in models_keys for e in entry_names)
    for region in regions_list:
        print(f"Saving region: {region} with strategy: {agg_strategy}")
        counts = region_valid_counts[region]
        masked_contingency = region_accumulators[region].copy()
        # A zero count means no grid point contributed a finite value there, which
        # is not the same as a genuine zero contingency count.
        masked_contingency[counts == 0] = np.nan

        # The aggregated curve is only a ROC of a fixed sample if every threshold
        # index was built from the same set of grid points.
        per_k = counts.reshape(-1, counts.shape[-1])
        if per_k.size and not np.all(per_k == per_k[:, [0]]):
            n_bad = int(np.sum(np.any(per_k != per_k[:, [0]], axis=0)))
            print(
                f"  WARNING: {region}/{agg_strategy}: contributing grid points vary "
                f"across threshold index at {n_bad}/{counts.shape[-1]} indices "
                f"(range {per_k.min()}..{per_k.max()})."
            )

        base_path = f'data/contingency_matrices_{region}_t{args.threshold}_{lead_time}h_{agg_strategy}{file_suffix}'
        np.save(f'{base_path}.npy', masked_contingency)
        np.save(f'{base_path}_validcounts.npy', counts)