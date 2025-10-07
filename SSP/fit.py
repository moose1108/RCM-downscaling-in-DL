import xarray as xr
import numpy as np
import pandas as pd

in_file = "/home/moose1108/corrdiff-like-project/new_project/Taiwan_WRF_daily_rain_1950_2014.nc"
out_file = "/work/moose1108/corrdiff-like/data/y_WRF_1950_2014_adjusted.nc"

ds = xr.open_dataset(in_file)

# 改維度名稱
ds = ds.rename({
    "Time": "time",
    "south_north": "lat",
    "west_east": "lon",
    "precip_daily": "pr"
})

# 更新 time 型別與範圍
ds["time"] = pd.to_datetime(ds["time"].values)

# 移除閏日
ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))

# 變數屬性
ds["pr"] = ds["pr"].fillna(0.0)
ds["pr"].attrs.update({
    "units": "mm",
    "description": "Daily total precipitation",
    "comment": "Converted from Taiwan_WRF_daily_rain_1950_2014.nc"
})

# 用 encoding 而非 attrs 來設定時間資訊
encoding = {
    "pr": {"_FillValue": 1e36, "dtype": "float32"},
    "time": {"units": "days since 1950-01-01 09:00:00", "calendar": "proleptic_gregorian"}
}

ds.to_netcdf(out_file, encoding=encoding)
print("✅ Saved:", out_file)
print("🕒 time range:", ds["time"].values[0], "→", ds["time"].values[-1])
