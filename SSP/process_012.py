import tqdm
import xarray as xr
import numpy as np

min_lat = 21.88
max_lat = 25.32
min_lon = 120
max_lon = 122

# TaiESM1 檔案中變數名稱對應表
var_map = {
    "u200": "ua",
    "u850": "ua",
    "v200": "va",
    "v850": "va",
    "w850": "wap",
    "pr": "pr",
    "ts": "ts",
}

variables = ["pr", "ts", "u200", "u850", "v200", "v850", "w850"]
data_values = {var: [] for var in variables}
coordinates = {"time": None, "latitude": None, "longitude": None}
start_year = 1950
end_year = 2014

def is_leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

for var in variables:
    print(f"處理變數 {var} ...")
    main_var = var_map[var]

    for year in tqdm.tqdm(range(start_year, end_year + 1)):
        for month in range(1, 13):
            # 🟡 跳過壞掉的檔案 (1979_12)
            if year == 1979 and month == 12:
                print(f"⚠️ Skip corrupted file for {year}-{month:02}")
                continue

            if var in ["u200", "u850", "v200", "v850", "w850"]:
                file_path = f"/work/moose1108/corrdiff-like/data/012-predictor_TaiESM1_ssp/historical_daily/{var}/{year}/TaiESM1_PRS_{var}_{year}{month:02}_r1440x721_day.nc"
            else:
                file_path = f"/work/moose1108/corrdiff-like/data/012-predictor_TaiESM1_ssp/historical_daily/{var}/{year}/TaiESM1_SFC_{var}_{year}{month:02}_r1440x721_day.nc"

            try:
                ds = xr.open_dataset(file_path)
            except Exception as e:
                print(f"⚠️ 無法開啟 {file_path}: {e}")
                continue

            # 移除閏日
            if is_leap_year(year) and month == 2:
                ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))

            # 選取台灣範圍（注意 lat 順序）
            lat_ascending = ds.lat.values[0] < ds.lat.values[-1]
            if lat_ascending:
                ds = ds.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
            else:
                ds = ds.sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))

            # 若有壓力層選擇
            if var.endswith("850") and "plev" in ds.dims:
                ds = ds.sel(plev=85000)
            elif var.endswith("200") and "plev" in ds.dims:
                ds = ds.sel(plev=20000)

            # 初始化座標（只設定一次）
            if coordinates["latitude"] is None:
                coordinates["latitude"] = ds["lat"].values.tolist()
                coordinates["longitude"] = ds["lon"].values.tolist()
            if coordinates["time"] is None and var == "pr":
                coordinates["time"] = []

            # 累積時間與資料
            if var == "pr":
                coordinates["time"].extend(ds["time"].to_index().tolist())

            # 加入資料值
            data_values[var].append(ds[main_var].values)

            ds.close()

    # 將每月資料拼接起來
    data_values[var] = np.concatenate(data_values[var], axis=0)

# ✅ 對齊時間長度（避免長度不符）
min_time_len = min([data_values[v].shape[0] for v in variables])
coordinates["time"] = coordinates["time"][:min_time_len]

# 建立 Dataset
new_data_arrays = {
    var: xr.DataArray(
        data=data_values[var][:min_time_len, :, :],
        dims=["time", "latitude", "longitude"],
        coords={
            "time": coordinates["time"],
            "latitude": coordinates["latitude"],
            "longitude": coordinates["longitude"],
        },
    )
    for var in variables
}

ds_combined = xr.Dataset(new_data_arrays)

# 儲存檔案
ds_combined.to_netcdf("TaiESM1_Taiwan_1948_2014.nc")
print("✅ 合併完成：TaiESM1_Taiwan_1948_2014.nc")
