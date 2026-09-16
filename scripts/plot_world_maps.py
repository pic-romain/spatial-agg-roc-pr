import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm

import sys
from pathlib import Path
tools_parent = Path(__file__).parent.parent.resolve()
sys.path.append(str(tools_parent))
from tools.utils import REGIONS
from tools.style import apply_style, savefig

apply_style()

WB_var = '2m_temperature'
year = 2020
resolution = '0p25'
import yaml
from pathlib import Path

with open("tools/config.yaml") as f:
    config = yaml.safe_load(f)

wdr = Path(config["wb2_dir"])
cache_dir = 'cache/'
output_dir = 'figures/'
os.makedirs(output_dir, exist_ok=True)

PROJ = ccrs.PlateCarree()

cache_q98 = os.path.join(cache_dir, 'count_q98.npy')
cache_max  = os.path.join(cache_dir, 'count_max.npy')

# ---------------------------------------------------------------------------- #
#                          Exceedance counts (cached)                          #
# ---------------------------------------------------------------------------- #

import regionmask

# Opened lazily in both paths: the cache only holds the counts, but the grid and
# the land mask are needed at plot time to keep non-land points transparent.
obs = xr.open_zarr(
    os.path.join(wdr, f'hres_t0/hres_t0_{year}_world_{WB_var}_{resolution}.zarr')
)[WB_var]
lats = obs.lat.values
lons = obs.lon.values
nlat, nlon = len(lats), len(lons)

land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(
    lons, lats
).notnull().values  # (nlat, nlon)

if os.path.exists(cache_q98) and os.path.exists(cache_max):
    print("Loading counts from cache...")
    count_q98 = np.load(cache_q98)
    count_max  = np.load(cache_max)
else:
    print("Loading observations...")
    obs_months = obs.time.dt.month.values
    obs_hours  = obs.time.dt.hour.values

    q98_zarr = xr.open_zarr(
        f"{wdr}/era5_quantiles/era5-quantiles_1979-2019_world_{WB_var}_{resolution}.zarr"
    )[WB_var]
    max_zarr = xr.open_zarr(
        f"{wdr}/era5_records/era5-record_1979-2019_world_{WB_var}_{resolution}.zarr"
    )[WB_var].max(dim='hour')  # (month, lat, lon)

    NH_SUMMER = {7, 8, 9}
    SH_SUMMER = {12, 1, 2}
    nh_rows = lats >= 0
    sh_rows = lats < 0

    spatial_max = land_mask & (lats >= -60)[:, np.newaxis]

    count_q98 = np.zeros((nlat, nlon), dtype=np.int32)
    count_max  = np.zeros((nlat, nlon), dtype=np.int32)

    print("Computing exceedances month by month...")
    for m in range(1, 13):
        print(f"  Month {m:02d}...")
        thresh_max = max_zarr.sel(month=m).values

        lat_summer = np.zeros(nlat, dtype=bool)
        if m in NH_SUMMER:
            lat_summer |= nh_rows
        if m in SH_SUMMER:
            lat_summer |= sh_rows
        spatial_q98 = land_mask & lat_summer[:, np.newaxis]

        for h in [0, 6, 12, 18]:
            time_idx = np.where((obs_months == m) & (obs_hours == h))[0]
            if time_idx.size == 0:
                continue

            obs_slice = obs.isel(time=time_idx).values

            thresh_q98 = q98_zarr.sel(month=m, hour=h, quantile=0.98).values

            count_q98 += np.where(spatial_q98, (obs_slice > thresh_q98).sum(axis=0), 0)
            count_max  += np.where(spatial_max, (obs_slice > thresh_max).sum(axis=0), 0)

        del obs_slice

    np.save(cache_q98, count_q98)
    np.save(cache_max,  count_max)
    print(f"Counts saved to {cache_dir}")

# ---------------------------------------------------------------------------- #
#                          Plot — Exceedance maps                              #
# ---------------------------------------------------------------------------- #

# Points excluded from the accumulation stay transparent; included points with
# zero exceedances are kept as 0 so they render in the colormap's "under" colour
# (grey) rather than vanishing. The masks mirror those used above: q98 is
# land-only, max is land-only *without* Antarctica (lat < -60).
mask_q98 = land_mask
mask_max = land_mask & (lats >= -60)[:, np.newaxis]

plot_q98 = np.where(mask_q98, count_q98.astype(float), np.nan)
plot_max  = np.where(mask_max, count_max.astype(float),  np.nan)

# Continuous YlOrRd for both maps: grey for zeros (under), clipped at the 99th
# percentile of the non-zero counts.
ZERO_GREY = '#bdbdbd'

def _ylorrd():
    cmap = plt.get_cmap('YlOrRd').copy()
    cmap.set_bad(color='none')     # excluded points -> transparent
    cmap.set_under(ZERO_GREY)      # zero exceedances -> grey
    return cmap

def _vmax(arr):
    """99th percentile of the strictly positive counts."""
    return np.nanpercentile(np.where(arr > 0, arr, np.nan), 99)

def _add_vmin_tick(cb, vmax):
    """Tick the bottom of the scale, so the grey (zero) end is readable."""
    ticks = [t for t in cb.get_ticks() if VMIN < t <= vmax]
    cb.set_ticks([VMIN] + ticks)

cmap_q98 = _ylorrd()
vmax_q98 = _vmax(plot_q98)

# max keeps its discrete scale: 1-7, 8-15, 16-25, >25 -> gold.
cmap_max = ListedColormap(['#460000', '#e06100', '#feff99'])
cmap_max.set_bad(color='none')     # excluded points -> transparent
cmap_max.set_under(ZERO_GREY)      # zero exceedances -> grey
cmap_max.set_over('gold')
bounds_max = [1, 7, 15, 25]
norm_max = BoundaryNorm(bounds_max, cmap_max.N)

# vmin=1 puts every zero below the scale, so it picks up the grey "under" colour
# on the map and the grey extension triangle on the colorbar.
VMIN = 1

# Vertical gap between the map and its colorbar (axes fraction).
CB_PAD = 0.10

print("Plotting exceedance maps...")

# q98 figure
fig, ax = plt.subplots(1, 1, figsize=(12, 5), subplot_kw={'projection': PROJ})
im = ax.pcolormesh(
    lons, lats, plot_q98,
    cmap=cmap_q98, shading='auto', transform=PROJ,
    vmin=VMIN, vmax=vmax_q98,
)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
cb = plt.colorbar(im, ax=ax, orientation='horizontal', pad=CB_PAD,
                  label='# exceedances', extend='both', shrink=0.4)
_add_vmin_tick(cb, vmax_q98)
fig.tight_layout()
out = os.path.join(output_dir, 'exceedances_q98.png')
savefig(fig, out)
plt.close(fig)
print(f"Saved {out}")

# max figure
fig, ax = plt.subplots(1, 1, figsize=(12, 5), subplot_kw={'projection': PROJ})
im = ax.pcolormesh(
    lons, lats, plot_max,
    cmap=cmap_max, norm=norm_max, shading='auto', transform=PROJ,
)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
cb = plt.colorbar(im, ax=ax, orientation='horizontal', pad=CB_PAD,
                  label='# exceedances', extend='both', shrink=0.4)
cb.set_ticks([4, 11.5, 20.5])
cb.set_ticklabels(['1–7', '8–15', '16–25'])
fig.tight_layout()
out = os.path.join(output_dir, 'exceedances_max.png')
savefig(fig, out)
plt.close(fig)
print(f"Saved {out}")