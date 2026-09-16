import os
import sys
import gcsfs
import xarray as xr
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import yaml
from pathlib import Path

with open("tools/config.yaml") as f:
    config = yaml.safe_load(f)
wdr = Path(config["wb2_dir"])

from tools.utils import change_lonlat

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--download",
    type=str,
    default="observations",
    choices=["observations", "models", "climatology"],
    help="Specify whether to download 'observations', 'models', or 'climatology' data."
)
parser.add_argument("--wb-var", type=str, default="2m_temperature", help="WeatherBench variable name.")
parser.add_argument("--region-name", type=str, default="world", choices=["europe", "world"], help="Region key.")
parser.add_argument("--year", type=int, default=2020, help="Forecast year for model and observation downloads.")
parser.add_argument("--resolution", type=str, default="0p25", choices=["0p25", "low-res"], help="Grid resolution.")
parser.add_argument("--wdr", type=str, default=str(wdr), help="Root output data directory.")
parser.add_argument("--models", nargs="+", default=["HRES", "GraphCast", "Pangu"], help="Models to download when --download models.")
args = parser.parse_args()
print(args)

download = args.download

WB_var = args.wb_var
region_name = args.region_name
year = args.year
resolution = args.resolution
wdr = Path(args.wdr)
models = args.models

regions = {}
regions['europe'] = [22, 72, -22, 45] # lat min, lat max, lon min, lon max
regions['world'] = [-90, 90, -180, 180] # lat min, lat max, lon min, lon max


roots = {
    'hres_t0': 'gs://weatherbench2/datasets/hres_t0/',
    'ERA5': 'gs://weatherbench2/datasets/era5/',
    'GraphCast': 'gs://weatherbench2/datasets/graphcast_hres_init/',
    'Pangu': 'gs://weatherbench2/datasets/pangu_hres_init/',
    'HRES': 'gs://weatherbench2/datasets/hres/',   
}

files = {
    '0p25': {
        'hres_t0': '2016-2022-6h-1440x721.zarr',
        'ERA5': '1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr',
        'GraphCast': f'{year}/date_range_{year-1}-11-16_{year+1}-02-01_12_hours_derived.zarr',
        'Pangu': '2020_0012_0p25.zarr',
        'HRES': '2016-2022-0012-1440x721.zarr',
    },
    'low-res': {
        'hres_t0': '2016-2022-6h-64x32_equiangular_conservative.zarr',    
        'ERA5': '1959-2023_01_10-6h-64x32_equiangular_conservative.zarr',
        'GraphCast': f'{year}/date_range_{year-1}-11-16_{year+1}-02-01_12_hours-64x32_equiangular_conservative.zarr',
        'Pangu': '2020_0012_64x32_equiangular_conservative.zarr',
        'HRES': '2016-2022-0012-64x32_equiangular_conservative.zarr',
    }
}

# ---------------------------------------------------------------------------- #

fs = gcsfs.GCSFileSystem(token='anon', session_kwargs={'trust_env': True})

if download == 'observations':
    # Observations
    path = os.path.join(roots['hres_t0'], files[resolution]['hres_t0'])
    mapper = fs.get_mapper(path)
    hres_t0 = xr.open_zarr(mapper, consolidated=True, chunks={})

    # Preprocessing
    hres_t0 = change_lonlat(hres_t0)[WB_var].sel(
        lat=slice(regions[region_name][0], regions[region_name][1]), lon=slice(regions[region_name][2], regions[region_name][3])).sel(
            time=slice(datetime(year-1, 12, 31), datetime(year+1, 1, 1)))
    hres_t0 = hres_t0.compute()
    hres_t0.to_zarr(os.path.join(wdr, f'hres_t0/hres_t0_{year}_{region_name}_{WB_var}_{resolution}.zarr'))
    hres_t0.close()

elif download == 'models':
    for model in models:
        if model not in roots:
            raise ValueError(f"Unknown model '{model}'. Expected one of: HRES, GraphCast, Pangu")
        print(f'Downloading {model} data...')
        path = os.path.join(roots[model], files[resolution][model])
        mapper = fs.get_mapper(path)
        
        ds_all = xr.open_zarr(mapper, consolidated=True)
        ds_all = change_lonlat(ds_all)[WB_var].sel(
                lat=slice(regions[region_name][0], regions[region_name][1]), lon=slice(regions[region_name][2], regions[region_name][3])).sel(
                    time=slice(datetime(year-1, 12, 31), datetime(year+1, 1, 1)))
        
        for lt_idx in range(len(ds_all.prediction_timedelta.values)):
            print(f'  Lead time index: {lt_idx}')
            ds = ds_all.isel(prediction_timedelta=slice(lt_idx, lt_idx + 1))
            # Preprocessing
            if lt_idx == 0:
                ds.to_zarr(os.path.join(wdr, f'{model.lower()}/{model.lower()}_{year}_{region_name}_{WB_var}_{resolution}.zarr'),
                           mode='w')
            else:
                ds.to_zarr(os.path.join(wdr, f'{model.lower()}/{model.lower()}_{year}_{region_name}_{WB_var}_{resolution}.zarr'),
                           mode='a',
                           append_dim='prediction_timedelta')
            ds.close()

elif download == 'climatology':
    # Climatology
    path = os.path.join(roots['ERA5'], files[resolution]['ERA5'])
    mapper = fs.get_mapper(path)
    era5 = xr.open_zarr(mapper, consolidated=True, chunks='auto')

    # Preprocessing (lazy): subset once, then stream yearly chunks to zarr.
    era5 = change_lonlat(era5)[WB_var].sel(
        lat=slice(regions[region_name][0], regions[region_name][1]), lon=slice(regions[region_name][2], regions[region_name][3])
    ).sel(time=slice(datetime(1979, 1, 1), datetime(2019, 12, 31)))

    # Force safe write chunks: avoid creating any compressed chunk >2GB.
    # For world 0p25, this keeps per-chunk buffers in the tens of MB range.
    time_chunk = 31 * 4  # ~1 month at 6-hourly frequency
    lat_chunk = min(181, era5.sizes['lat'])
    lon_chunk = min(360, era5.sizes['lon'])
    era5 = era5.chunk({"time": time_chunk, "lat": lat_chunk, "lon": lon_chunk})

    var_name = era5.name or WB_var
    encoding = {var_name: {"chunks": (time_chunk, lat_chunk, lon_chunk)}}

    out_path = os.path.join(wdr, f'era5/era5_climatology_{region_name}_{WB_var}_{resolution}.zarr')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for idx, year_chunk in enumerate(range(1979, 2020)):
        print(f"Writing climatology year chunk: {year_chunk}")
        yearly = era5.sel(time=slice(datetime(year_chunk, 1, 1), datetime(year_chunk, 12, 31, 23, 59, 59)))
        if idx == 0:
            yearly.to_zarr(out_path, mode='w', encoding=encoding)
        else:
            yearly.to_zarr(out_path, mode='a', append_dim='time')

    era5.close()