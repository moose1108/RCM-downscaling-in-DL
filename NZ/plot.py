import cartopy.crs as ccrs
import matplotlib.cm as cm
import xarray as xr
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.prepare_data import format_features, prepare_training_dataset, create_test_train_split

config = dict(y = "/work/moose1108/corrdiff-like/data/y_adjust_1981_2022.nc",
              X = "/work/moose1108/corrdiff-like/data/1981_2022.nc",
             train_start = "1982-01-01",
             train_end = "2015-12-31",
             val_start = "2016-01-01",
             val_end = "2021-12-31",
             test_start = "2022-01-01",
             test_end = "2022-12-31",
             downscale_variables = ['w850', 'u200', 'u850', 'v200', 'v850', 'tp'])

x_train, x_val, x_test, y_train, y_val, y_test = create_test_train_split(config)
x_train, x_test, x_val, y_train, y_test, y_val = prepare_training_dataset(x_train, x_val, x_test, y_train, y_val, y_test)
pred = xr.open_dataset('test9.nc')
print(pred)
gt = y_test.unstack()
mask = xr.open_dataset("/work/moose1108/corrdiff-like/data/02-predictand_TReAD/TReAD_Regrid_2km_landmask.nc")
sea = mask.landmask.values == 0
gt['pr'].values[:,sea] = np.nan
pred['RAINNC'].values[:,sea] = np.nan

selected_days = ['2022-10-15', '2022-10-16', '2022-10-17', '2022-10-13', '2022-10-30', '2022-10-31', '2022-07-10', '2022-08-10', '2022-09-01', '2022-10-10', '2022-11-10', '2022-12-10']
num_days = len(selected_days)
standards = [0, 1, 2, 6, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300, 600]#[0, 0.05, 0.1, 0.
color_map = ['#ffffff','#98ffff','#00ceff','#009aff','#006af7','#2e9c00','#2bff00','#fefe08','#ffcb00','#ff9c00','#fe0005','#c90200','#9d0000','#9a009d','#cf00d7','#ff00f7','#fdcafe']
fig, axes = plt.subplots(num_days, 3, figsize=(20, 10 * num_days), subplot_kw={'projection': ccrs.PlateCarree()})
if num_days == 1:
    axes = [axes]
gt = y_test.unstack()
gt = gt.reindex(lon = sorted(gt.lon.values))
gt_lat = gt.variables['lat'][:]
gt_lon = gt.variables['lon'][:]

for i in range(len(selected_days)):
    ax = axes[i][0]
    daily_pred = pred['RAINNC'].sel(time = selected_days[i])
    daily_pred.values[sea] = np.nan
    pred_plot = ax.contourf(gt_lon, gt_lat, daily_pred, levels=standards, colors=color_map)
    ax.set_title(f'Model Prediction (RAINNC) - {selected_days[i]}', pad=20)
    ax.coastlines()

    ax = axes[i][1]
    daily_gt = gt['pr'].sel(time = selected_days[i])
    daily_gt.values[sea] = np.nan
    gt_plot = ax.contourf(gt_lon, gt_lat, daily_gt, levels=standards, colors=color_map)
    ax.set_title(f'Groundtruth (RAINNC) - {selected_days[i]}', pad=20)
    ax.coastlines()
    plt.colorbar(pred_plot, ax=ax, orientation='vertical')
    
    ax = axes[i][2]
    bias = daily_pred - daily_gt
    cmap = plt.get_cmap('RdBu')
    levels = np.linspace(-150, 150, 21)
    norm = plt.Normalize(min(levels), max(levels))
    colormap = [cmap(norm(level)) for level in levels]
    bias_plot = ax.contourf(gt_lon, gt_lat, bias, levels=levels, colors=colormap)
    ax.coastlines()
    plt.colorbar(bias_plot, ax=ax, orientation='vertical')
plt.savefig('temp.png')

fig, axes = plt.subplots(12, 3, figsize=(20, 60), subplot_kw={'projection': ccrs.PlateCarree()})
pred['RAINNC'].values[:,sea] = np.nan
gt['pr'].values[:,sea] = np.nan
gt = gt.reindex(lon = sorted(gt.lon.values))
gt_lat = gt.variables['lat'][:]
gt_lon = gt.variables['lon'][:]


monthly_avg_gt = gt['pr'].groupby('time.month').mean('time')
monthly_avg_pred = pred['RAINNC'].groupby('time.month').mean('time')

for i in range(12):
    ax1 = axes[i, 0]
    ax2 = axes[i, 1]
    ax3 = axes[i, 2]

    # Predicted Rainfall
    pred_plot = ax1.contourf(monthly_avg_pred.lon, monthly_avg_pred.lat, monthly_avg_pred.isel(month=i), levels=standards, colors=color_map)
    ax1.coastlines()
    ax1.set_title('Predicted Rainfall')
    plt.colorbar(pred_plot, ax=ax1, orientation='vertical')

    # Actual Rainfall
    gt_plot = ax2.contourf(monthly_avg_gt.lon, monthly_avg_gt.lat, monthly_avg_gt.isel(month=i), levels=standards, colors=color_map)
    ax2.coastlines()
    ax2.set_title('Actual Rainfall')
    plt.colorbar(gt_plot, ax=ax2, orientation='vertical')

    # Bias (Prediction - Ground Truth)
    bias = monthly_avg_pred.isel(month=i) - monthly_avg_gt.isel(month=i)
    cmap = plt.get_cmap('RdBu')
    levels = np.linspace(-10, 10, 21)
    norm = plt.Normalize(min(levels), max(levels))
    colormap = [cmap(norm(level)) for level in levels]
    bias_plot = ax3.contourf(monthly_avg_gt.lon, monthly_avg_gt.lat, bias, levels=levels, colors=colormap)
    ax3.coastlines()
    ax3.set_title('Bias (Pred - GT)')
    plt.colorbar(bias_plot, ax=ax3, orientation='vertical')

plt.tight_layout()
plt.savefig('current_Monthly_Average_Comparison_2013-2022.png')
plt.close()


dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
daily_correlations = []

# Loop through each day in the date range
for date in dates:
    try:
        # Select the data for the current day
        daily_pred = pred['RAINNC'].sel(time=date)
        daily_gt = gt['pr'].sel(time=date)

        # Flatten the data arrays to 1D
        daily_pred_flat = daily_pred.values.flatten()
        daily_gt_flat = daily_gt.values.flatten()

        # Remove NaN values to ensure correlation can be computed
        valid_mask = ~np.isnan(daily_pred_flat) & ~np.isnan(daily_gt_flat)
        daily_pred_valid = daily_pred_flat[valid_mask]
        daily_gt_valid = daily_gt_flat[valid_mask]

        # Compute correlation coefficient if valid data exists
        if daily_pred_valid.size > 0:
            daily_correlation = np.corrcoef(daily_pred_valid, daily_gt_valid)[0, 1]
            daily_correlations.append(daily_correlation)
        else:
            daily_correlations.append(np.nan)  # Append NaN if no valid data for correlation
        print(date, daily_correlation)
    except:
        print('No date:', date)

print(np.mean(daily_correlations))
