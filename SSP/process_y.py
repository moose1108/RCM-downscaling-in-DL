import os
import glob
import numpy as np
import xarray as xr
from tqdm import tqdm
import pandas as pd

# === Taiwan domain ===
MIN_LAT, MAX_LAT = 21.88, 25.32
MIN_LON, MAX_LON = 120.0, 122.0

# === Input paths ===
coord_file = "/work/moose1108/corrdiff-like/data/output/TAIESM_CORDEXEA_coord2d.nc"
wrf_dir = "/work/moose1108/corrdiff-like/data/output/CORDEXEA_historical/"
output_file = "./Taiwan_WRF_daily_rain_1950_2014.nc"

# === Load coordinate file ===
coord_ds = xr.open_dataset(coord_file)
XLAT = coord_ds["XLAT"].isel(Time=0)
XLONG = coord_ds["XLONG"].isel(Time=0)

# === Find Taiwan domain indices ===
mask = (XLAT >= MIN_LAT) & (XLAT <= MAX_LAT) & (XLONG >= MIN_LON) & (XLONG <= MAX_LON)
j_idx, i_idx = np.where(mask)
j_min, j_max = j_idx.min(), j_idx.max()
i_min, i_max = i_idx.min(), i_idx.max()

print(f"✅ Taiwan domain index range:")
print(f"   south_north: {j_min}–{j_max}")
print(f"   west_east  : {i_min}–{i_max}")

# === List all WRF files ===
wrf_files = sorted(glob.glob(os.path.join(wrf_dir, "wrfday_d01_*.nc")))
print(f"Found {len(wrf_files)} WRF files.")

# === Variables to extract ===
rain_vars = ["RAINC", "RAINNC"]

# === Prepare for merging ===
subset_list = []

for f in tqdm(wrf_files, desc="Processing WRF files"):
    try:
        ds = xr.open_dataset(f)

        # --- 讀出真實時間 (WRF "Times" 是 char array) ---
        if "Times" in ds:
            wrf_times = ["".join(t.astype(str)) for t in ds["Times"].values]
            wrf_times = pd.to_datetime(wrf_times, format="%Y-%m-%d_%H:%M:%S")
            ds = ds.assign_coords(Time=("Time", wrf_times))

        # --- 剪出台灣區域 ---
        ds_tw = ds.isel(south_north=slice(j_min, j_max + 1),
                        west_east=slice(i_min, i_max + 1))[rain_vars]

        # --- 計算每日降雨量（非累積）---
        # 注意：最後一筆通常是模擬結束時刻，多一天，diff後自動少一筆
        daily_rain = ds_tw["RAINC"].diff("Time") + ds_tw["RAINNC"].diff("Time")
        daily_rain.name = "precip_daily"  # 統一名稱

        # --- 時間與空間對齊 ---
        daily_rain = daily_rain.assign_coords(Time=ds_tw["Time"].isel(Time=slice(1, None)))

        subset_list.append(daily_rain)
        ds.close()

    except Exception as e:
        print(f"⚠️ Failed to process {f}: {e}")

# === Merge all monthly files ===
if len(subset_list) == 0:
    raise RuntimeError("No valid WRF files found!")

print("🔄 Concatenating along Time dimension...")
merged = xr.concat(subset_list, dim="Time")

# === Sort by time just in case ===
merged = merged.sortby("Time")

# === 移除閏年2月29日 ===
is_leap_day = (merged["Time.month"] == 2) & (merged["Time.day"] == 29)
merged = merged.sel(Time=~is_leap_day)

# === Save merged file ===
merged.to_netcdf(output_file)
print(f"🎯 Taiwan *daily* rain data saved to: {output_file}")

# === Optional check ===
print("\n✅ Final time range:")
print(pd.to_datetime(merged["Time"].values[0]), "→", pd.to_datetime(merged["Time"].values[-1]))
print(f"Total days: {len(merged['Time'])}")