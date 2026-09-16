import argparse
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import dask.array as dsa
import numpy as np
import xarray as xr

import yaml
with open("tools/config.yaml") as f:
    config = yaml.safe_load(f)

wdr = Path(config["wb2_dir"])


def parse_quantile_levels(quantiles_str):
	levels = [float(q.strip()) for q in quantiles_str.split(",") if q.strip()]
	if not levels:
		raise ValueError("At least one quantile level must be provided.")
	for q in levels:
		if q <= 0 or q >= 1:
			raise ValueError(f"Invalid quantile level {q}. Levels must be in (0, 1).")
	return np.array(levels, dtype=float)


def infer_input_path(wdr, region_name, wb_var, resolution):
	return os.path.join(
		wdr,
		"era5",
		f"era5_climatology_{region_name}_{wb_var}_{resolution}.zarr",
	)


def infer_output_path(wdr, region_name, wb_var, resolution, stat):
	if stat == "quantiles":
		return os.path.join(
			wdr,
			"era5_quantiles",
			f"era5-quantiles_1979-2019_{region_name}_{wb_var}_{resolution}.zarr",
		)
	if stat == "max":
		return os.path.join(
			wdr,
			"era5_records",
			f"era5-record_1979-2019_{region_name}_{wb_var}_{resolution}.zarr",
		)
	raise ValueError(f"Unsupported stat: {stat}")


def _get_dataarray(ds, wb_var):
	# if isinstance(ds, xr.DataArray):
	# 	return ds
	if wb_var in ds:
		return ds[wb_var]
	# if len(ds.data_vars) == 1:
	# 	return ds[list(ds.data_vars)[0]]
	raise ValueError(
		f"Variable '{wb_var}' not found and dataset has multiple variables: {list(ds.data_vars)}"
	)


def _init_output_store(output_path, da, months, hours, stat, levels, var_name, source):
	"""Initialize output Zarr with dask-backed empty arrays (metadata + chunk schema)."""
	coords = {
		"month": months,
		"hour": hours,
		"lat": da.lat,
		"lon": da.lon,
	}
	lat_chunk = min(da.sizes["lat"], 16)
	lon_chunk = min(da.sizes["lon"], 16)

	if stat == "quantiles":
		coords["quantile"] = levels
		shape = (len(months), len(hours), len(levels), da.sizes["lat"], da.sizes["lon"])
		chunks = (len(months), len(hours), len(levels), lat_chunk, lon_chunk)
		dims = ("month", "hour", "quantile", "lat", "lon")
	else:
		shape = (len(months), len(hours), da.sizes["lat"], da.sizes["lon"])
		chunks = (len(months), len(hours), lat_chunk, lon_chunk)
		dims = ("month", "hour", "lat", "lon")

	template_data = dsa.empty(shape, chunks=chunks, dtype=np.float32)
	template_da = xr.DataArray(template_data, coords=coords, dims=dims)
	template_ds = template_da.to_dataset(name=var_name)
	template_ds.attrs["source"] = source
	template_ds.attrs["stat"] = stat
	if stat == "quantiles":
		template_ds.attrs["quantiles"] = ",".join(str(v) for v in levels.tolist())
	template_ds.to_zarr(output_path, mode="w", compute=False)


def _compute_tile_quantiles(tile, levels, months, hours):
	arr = tile.values  # (time, lat, lon)
	months_t = tile.time.dt.month.values
	hours_t = tile.time.dt.hour.values
	out = np.full((len(months), len(hours), len(levels), tile.sizes["lat"], tile.sizes["lon"]), np.nan, dtype=np.float32)

	for mi, month in enumerate(months):
		for hi, hour in enumerate(hours):
			mask = (months_t == month) & (hours_t == hour)
			if np.any(mask):
				q = np.nanquantile(arr[mask, :, :], levels, axis=0)
				out[mi, hi, :, :, :] = q.astype(np.float32)

	return xr.DataArray(
		out,
		coords={"month": months, "hour": hours, "quantile": levels, "lat": tile.lat, "lon": tile.lon},
		dims=("month", "hour", "quantile", "lat", "lon"),
	)


def _compute_tile_max(tile, months, hours):
	arr = tile.values  # (time, lat, lon)
	months_t = tile.time.dt.month.values
	hours_t = tile.time.dt.hour.values
	out = np.full((len(months), len(hours), tile.sizes["lat"], tile.sizes["lon"]), np.nan, dtype=np.float32)

	for mi, month in enumerate(months):
		for hi, hour in enumerate(hours):
			mask = (months_t == month) & (hours_t == hour)
			if np.any(mask):
				m = np.nanmax(arr[mask, :, :], axis=0)
				out[mi, hi, :, :] = m.astype(np.float32)

	return xr.DataArray(
		out,
		coords={"month": months, "hour": hours, "lat": tile.lat, "lon": tile.lon},
		dims=("month", "hour", "lat", "lon"),
	)


def _compute_tile_dataset(da, i0, i1, j0, j1, stat, levels, months, hours, var_name):
	"""Compute one spatial tile and return a Dataset ready for region write."""
	tile = da.isel(lat=slice(i0, i1), lon=slice(j0, j1)).transpose("time", "lat", "lon")
	if stat == "quantiles":
		tile_da = _compute_tile_quantiles(tile, levels, months, hours)
	else:
		tile_da = _compute_tile_max(tile, months, hours)
	return i0, i1, j0, j1, tile_da.to_dataset(name=var_name)


def main():
	parser = argparse.ArgumentParser(
		description="Compute hourly climatological statistics and save them to a Zarr file."
	)
	parser.add_argument(
		"--stat",
		type=str,
		default="quantiles",
		choices=["quantiles", "max"],
		help="Statistic to compute: quantiles or climatological record maximum.",
	)
	parser.add_argument(
		"--quantiles",
		type=str,
		default="0.5,0.75,0.98",
		help="Comma-separated quantile levels, e.g. '0.9,0.95' (used when --stat quantiles).",
	)
	parser.add_argument(
		"--region-name",
		type=str,
		default="world",
		choices=["europe", "world"],
		help="Spatial region key used in downloaded file naming.",
	)
	parser.add_argument(
		"--wb-var",
		type=str,
		default="2m_temperature",
		help="WeatherBench variable name.",
	)
	parser.add_argument(
		"--resolution",
		type=str,
		default="0p25",
		choices=["0p25", "low-res"],
		help="Grid resolution suffix used in downloaded file naming.",
	)
	parser.add_argument(
		"--wdr",
		type=str,
		default=str(wdr),
		help="Root data directory.",
	)
	parser.add_argument(
		"--time-chunk",
		type=int,
		default=366 * 4,
		help="Chunk size along time for parallel/lazy processing.",
	)
	parser.add_argument(
		"--lat-chunk",
		type=int,
		default=181,
		help="Chunk size along latitude for parallel/lazy processing.",
	)
	parser.add_argument(
		"--lon-chunk",
		type=int,
		default=360,
		help="Chunk size along longitude for parallel/lazy processing.",
	)
	parser.add_argument(
		"--lat-tile-size",
		type=int,
		default=16,
		help="Latitude tile size for bounded-memory computation and region writes.",
	)
	parser.add_argument(
		"--lon-tile-size",
		type=int,
		default=16,
		help="Longitude tile size for bounded-memory computation and region writes.",
	)
	parser.add_argument(
		"--workers",
		type=int,
		default=4,
		help="Number of worker threads for parallel tile computation.",
	)
	parser.add_argument(
		"--input",
		type=str,
		default=None,
		help="Optional explicit input Zarr path.",
	)
	parser.add_argument(
		"--output",
		type=str,
		default=None,
		help="Optional explicit output Zarr path.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite output if it already exists.",
	)

	args = parser.parse_args()

	levels = parse_quantile_levels(args.quantiles) if args.stat == "quantiles" else None

	input_path = args.input or infer_input_path(
		args.wdr, args.region_name, args.wb_var, args.resolution
	)
	output_path = args.output or infer_output_path(
		args.wdr, args.region_name, args.wb_var, args.resolution, args.stat
	)

	if not os.path.exists(input_path):
		raise FileNotFoundError(
			f"Input file does not exist: {input_path}\n"
			"If needed, run tools/download.py with --download climatology first."
		)

	output_dir = Path(output_path).parent
	output_dir.mkdir(parents=True, exist_ok=True)

	if os.path.exists(output_path) and not args.overwrite:
		raise FileExistsError(
			f"Output already exists: {output_path}. Use --overwrite to replace it."
		)

	print(f"Opening input climatology: {input_path}")
	ds = xr.open_zarr(input_path, chunks="auto")
	da = _get_dataarray(ds, args.wb_var)
	chunk_spec = {
		"time": min(args.time_chunk, da.sizes["time"]),
		"lat": min(args.lat_chunk, da.sizes["lat"]),
		"lon": min(args.lon_chunk, da.sizes["lon"]),
	}
	da = da.chunk(chunk_spec)
	print(f"Using chunks: {chunk_spec}")

	months = np.arange(1, 13, dtype=int)
	hours = np.unique(da.time.dt.hour.values).astype(int)
	var_name = da.name or args.wb_var

	if os.path.exists(output_path) and args.overwrite:
		import shutil

		shutil.rmtree(output_path)

	print(f"Initializing output store for {args.stat}: {output_path}")
	_init_output_store(output_path, da, months, hours, args.stat, levels, var_name, input_path)

	tiles = []
	for i0 in range(0, da.sizes["lat"], args.lat_tile_size):
		i1 = min(i0 + args.lat_tile_size, da.sizes["lat"])
		for j0 in range(0, da.sizes["lon"], args.lon_tile_size):
			j1 = min(j0 + args.lon_tile_size, da.sizes["lon"])
			tiles.append((i0, i1, j0, j1))

	if args.workers <= 1:
		for i0, i1, j0, j1 in tiles:
			print(f"Processing tile lat[{i0}:{i1}] lon[{j0}:{j1}]")
			_, _, _, _, tile_ds = _compute_tile_dataset(
				da, i0, i1, j0, j1, args.stat, levels, months, hours, var_name
			)
			region = {
				"month": slice(None),
				"hour": slice(None),
				"lat": slice(i0, i1),
				"lon": slice(j0, j1),
			}
			if args.stat == "quantiles":
				region["quantile"] = slice(None)
			tile_ds.to_zarr(
				output_path,
				mode="r+",
				region=region,
			)
	else:
		print(f"Processing tiles with {args.workers} workers")
		max_in_flight = max(2, args.workers * 2)
		in_flight = {}
		with ThreadPoolExecutor(max_workers=args.workers) as executor:
			tile_iter = iter(tiles)
			while True:
				while len(in_flight) < max_in_flight:
					try:
						i0, i1, j0, j1 = next(tile_iter)
					except StopIteration:
						break
					print(f"Queue tile lat[{i0}:{i1}] lon[{j0}:{j1}]")
					fut = executor.submit(
						_compute_tile_dataset,
						da,
						i0,
						i1,
						j0,
						j1,
						args.stat,
						levels,
						months,
						hours,
						var_name,
					)
					in_flight[fut] = (i0, i1, j0, j1)

				if not in_flight:
					break

				done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
				for fut in done:
					i0, i1, j0, j1, tile_ds = fut.result()
					region = {
						"month": slice(None),
						"hour": slice(None),
						"lat": slice(i0, i1),
						"lon": slice(j0, j1),
					}
					if args.stat == "quantiles":
						region["quantile"] = slice(None)
					tile_ds.to_zarr(
						output_path,
						mode="r+",
						region=region,
					)
					del in_flight[fut]

	print("Done.")


if __name__ == "__main__":
	main()
