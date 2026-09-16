import argparse
import os
import shutil
from pathlib import Path

import dask
import dask.array as dsa
import numpy as np
import xarray as xr

import yaml
with open("tools/config.yaml") as f:
    config = yaml.safe_load(f)

wdr = Path(config["wb2_dir"])


MODEL_FILE = {
	"hres": "hres",
	"graphcast": "graphcast",
	"pangu": "pangu",
}


def infer_input_path(wdr, model, region_name, wb_var, resolution, year):
	return os.path.join(
		wdr,
		MODEL_FILE[model],
		f"{MODEL_FILE[model]}_{year}_{region_name}_{wb_var}_{resolution}.zarr",
	)


def infer_output_path(wdr, model, region_name, wb_var, resolution, year):
	return os.path.join(
		wdr,
		"forecast_std",
		f"{MODEL_FILE[model]}-std_{year}_{region_name}_{wb_var}_{resolution}.zarr",
	)


def _get_dataarray(ds, wb_var):
	if wb_var in ds:
		return ds[wb_var]
	raise ValueError(
		f"Variable '{wb_var}' not found; dataset has: {list(ds.data_vars)}"
	)


def _init_output_store(output_path, da, ptds, months, var_name, source):
	"""Initialize output Zarr (prediction_timedelta, month, lat, lon) with empty dask arrays."""
	coords = {
		"prediction_timedelta": ptds,
		"month": months,
		"lat": da.lat,
		"lon": da.lon,
	}
	lat_chunk = min(da.sizes["lat"], 16)
	lon_chunk = min(da.sizes["lon"], 16)
	shape = (len(ptds), len(months), da.sizes["lat"], da.sizes["lon"])
	chunks = (1, len(months), lat_chunk, lon_chunk)
	dims = ("prediction_timedelta", "month", "lat", "lon")

	template = xr.DataArray(
		dsa.empty(shape, chunks=chunks, dtype=np.float32), coords=coords, dims=dims
	).to_dataset(name=var_name)
	template.attrs["source"] = source
	template.attrs["stat"] = "std"
	template.attrs["note"] = "Per-gridpoint forecast std grouped by valid-time month, per lead time."
	template.to_zarr(output_path, mode="w", compute=False)


def _compute_lead_time(da, ptd, months, var_name):
	"""Std over init time of one lead time, grouped by valid-time month -> (month, lat, lon)."""
	sub = da.sel(prediction_timedelta=ptd).chunk({"time": -1, "lat": 120, "lon": 120})
	valid_month = (sub["time"] + ptd).dt.month
	sub = sub.assign_coords(vmonth=("time", valid_month.values))
	std_m = sub.groupby("vmonth").std("time", ddof=0)
	std_m = std_m.rename({"vmonth": "month"}).reindex(month=months)
	# Re-attach the (dropped) lead-time dimension as size-1 so the region write aligns
	# with the (prediction_timedelta, month, lat, lon) store layout.
	std_m = std_m.expand_dims(prediction_timedelta=[ptd])
	# Materialize the reduced (month, lat, lon) result (~tens of MB) so the region
	# write is numpy-backed and avoids Dask/Zarr chunk-alignment issues. The compute
	# uses the configured Dask threaded scheduler, which parallelizes the (GIL-releasing)
	# Zarr decompression and numpy reductions across cores for this one lead time.
	return std_m.astype(np.float32).compute().to_dataset(name=var_name)


def process_model(model, args):
	input_path = infer_input_path(
		args.wdr, model, args.region_name, args.wb_var, args.resolution, args.year
	)
	output_path = infer_output_path(
		args.wdr, model, args.region_name, args.wb_var, args.resolution, args.year
	)

	if not os.path.exists(input_path):
		raise FileNotFoundError(f"Input forecast zarr does not exist: {input_path}")

	if os.path.exists(output_path):
		if not args.overwrite:
			raise FileExistsError(
				f"Output already exists: {output_path}. Use --overwrite to replace it."
			)
		shutil.rmtree(output_path)

	Path(output_path).parent.mkdir(parents=True, exist_ok=True)

	print(f"[{model}] Opening forecast: {input_path}")
	da = _get_dataarray(xr.open_zarr(input_path, chunks="auto"), args.wb_var)

	ptds = da.prediction_timedelta.values
	months = np.arange(1, 13, dtype=int)
	var_name = da.name or args.wb_var

	print(f"[{model}] Lead times: {[str(p) for p in ptds]}")
	print(f"[{model}] Initializing output store: {output_path}")
	_init_output_store(output_path, da, ptds, months, var_name, input_path)

	# Lead times are processed one at a time; parallelism comes from the Dask threaded
	# scheduler within each lead-time compute (set in main()). This keeps memory bounded
	# to a single lead-time buffer while still using all worker cores.
	for pi, ptd in enumerate(ptds):
		print(f"[{model}] Computing std for lead time {ptd} ({pi + 1}/{len(ptds)})", flush=True)
		tile_ds = _compute_lead_time(da, ptd, months, var_name)
		tile_ds.to_zarr(
			output_path,
			mode="r+",
			region={
				"prediction_timedelta": slice(pi, pi + 1),
				"month": slice(None),
				"lat": slice(None),
				"lon": slice(None),
			},
		)
	print(f"[{model}] Done -> {output_path}", flush=True)


def main():
	parser = argparse.ArgumentParser(
		description="Compute per-model forecast std grouped by lead time and valid-time month."
	)
	parser.add_argument(
		"--models",
		type=str,
		nargs="+",
		default=["hres", "graphcast", "pangu"],
		choices=list(MODEL_FILE.keys()),
		help="Models to process.",
	)
	parser.add_argument("--year", type=int, default=2020)
	parser.add_argument(
		"--region-name", type=str, default="world", choices=["europe", "world"]
	)
	parser.add_argument("--wb-var", type=str, default="2m_temperature")
	parser.add_argument(
		"--resolution", type=str, default="0p25", choices=["0p25", "low-res"]
	)
	parser.add_argument(
		"--wdr", type=str, default=str(wdr)
	)
	parser.add_argument(
		"--workers",
		type=int,
		default=8,
		help="Number of Dask threads used to parallelize each lead-time compute.",
	)
	parser.add_argument("--overwrite", action="store_true")
	args = parser.parse_args()

	# Dask threaded scheduler: blosc decompression and numpy reductions release the GIL,
	# so threads use multiple cores. (A process pool would be wrong here — it would
	# serialize via pickling and lose the shared chunk cache.)
	dask.config.set(scheduler="threads", num_workers=args.workers)

	for model in args.models:
		process_model(model, args)


if __name__ == "__main__":
	main()
