import netCDF4 as nc
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def calculate_mae(predictions, ground_truth):
    valid_mask = ~np.isnan(predictions) & ~np.isnan(ground_truth)
    pred_flat = predictions[valid_mask]
    gt_flat = ground_truth[valid_mask]
    print(pred_flat.shape)
    mae = np.mean(np.abs(pred_flat - gt_flat))
    return mae

def calculate_rmse(predictions, ground_truth):
    """Calculate the RMSE between predictions and ground truth arrays, considering only non-NaN values."""
    valid_mask = ~np.isnan(predictions) & ~np.isnan(ground_truth)
    pred_flat = predictions[valid_mask]
    gt_flat = ground_truth[valid_mask]
    squared_errors = np.square(pred_flat - gt_flat)
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)
    return rmse

monthly_sums = {i: None for i in range(1, 13)}
monthly_counts = {i: 0 for i in range(1, 13)}

monthly_rmse_years = {year: [] for year in range(2013, 2023)}
for year in range(2013, 2023):
    # year = '2022'

    pred_dataset = nc.Dataset('current_max.nc')
    gt_dataset = nc.Dataset(f'/work/moose1108/corrdiff-like/data/02-predictand_TReAD/RAINNC/TReAD_daily_{year}_RAINNC.nc')
    landmask_dataset = nc.Dataset('/work/moose1108/corrdiff-like/data/02-predictand_TReAD/TReAD_Regrid_2km_landmask.nc')

    time = pred_dataset.variables['time']
    dates = nc.num2date(time[:], units=time.units)  # Ensure this matches the actual units
    dates = np.array(dates).astype('datetime64[D]')
    start_date = np.datetime64(f'{year}-01-01')
    end_date = np.datetime64(f'{year}-12-31')
    indices = (dates >= start_date) & (dates <= end_date)



    predictand = 'RAINNC'
    rain = pred_dataset.variables[predictand][indices]
    gt_rain = gt_dataset.variables[predictand][:] * 24
    lat = pred_dataset.variables['lat'][:]
    lon = pred_dataset.variables['lon'][:]
    gt_lat = gt_dataset.variables['Lat'][:]
    gt_lon = gt_dataset.variables['Lon'][:]
    land_sea_mask = landmask_dataset.variables['landmask'][:]
    gt_rain_masked = np.where(land_sea_mask == 1, gt_rain, np.nan)

    rain = np.where(np.isnan(gt_rain_masked), np.nan, rain)

    mae1 = calculate_mae(rain, gt_rain_masked)

    # Print the MAE for each model
    print(f'MAE for CNN-Gamma (wrong version): {mae1}')

    # Apply land/sea mask
    rain_masked = np.where(land_sea_mask == 1, rain, np.nan)
    gt_rain_masked = np.where(land_sea_mask == 1, gt_rain, np.nan)

    #
    squared_differences = np.square(rain_masked - gt_rain_masked)
    # months = np.array([date.month for date in dates])

    
    # Assuming the dataset provides time in a compatible format or using a range
    dates = pd.date_range(start='2020-01-01', periods=rain.shape[0], freq='D')

    #
    for month in range(1, 13):
        month_indices = dates.month == month
        if monthly_sums[month] is None:
            monthly_sums[month] = np.nansum(squared_differences[month_indices], axis=0)
        else:
            monthly_sums[month] += np.nansum(squared_differences[month_indices], axis=0)
        monthly_counts[month] += np.sum(month_indices)

    # Calculate RMSE for each month
    monthly_rmse = {}
    for month in range(1, 13):
        month_mask = dates.month == month
        monthly_rmse[month] = calculate_rmse(rain_masked[month_mask], gt_rain_masked[month_mask])
        monthly_rmse_years[year].append(monthly_rmse[month])

    # Convert results to DataFrame for easier viewing
    monthly_rmse_df = pd.DataFrame(list(monthly_rmse.items()), columns=['Month', 'RMSE'])
    print(monthly_rmse_df)

    overall_rmse = calculate_rmse(rain_masked, gt_rain_masked)
    print(f'Overall average RMSE for {year}: {overall_rmse}')


# plt.figure(figsize=(10, 8))
# for year, rmses in monthly_rmse_years.items():
#     plt.plot(range(1, 13), rmses, label=str(year))

# plt.title('Monthly RMSE from 2013 to 2022')
# plt.xlabel('Month')
# plt.ylabel('RMSE')
# plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
# plt.legend(title='Year')
# plt.grid(True)
# plt.savefig('fig.png')


# import matplotlib.colors as mcolors
# levels = np.linspace(0, 10, 21)
# norm = plt.Normalize(min(levels), max(levels))
# cmap = plt.get_cmap('Blues')
# colormap = [cmap(norm(level)) for level in levels]
# norm = mcolors.BoundaryNorm(levels, cmap.N)
# for i in range(1, 13):
#     mean_squared_errors = monthly_sums[i] / monthly_counts[i]
#     rmse_grid = np.sqrt(mean_squared_errors)

#     max_rmse = np.nanmax(rmse_grid)
#     levels = np.linspace(0, max_rmse, 21)  # 從0到最大RMSE，分為40個階段
#     fig, ax = plt.subplots(figsize=(8, 10))
#     im = ax.contourf(gt_lon, gt_lat, rmse_grid, levels=levels, colors=colormap)
#     # ax.coastlines()
#     cbar = fig.colorbar(im, ax=ax, ticks=levels, extend='max')
#     cbar.set_label('RMSE (mm)')
#     plt.title(f'Spatial Distribution of RMSE for Precipitation Predictions - Month {i}')
#     plt.xlabel('Longitude Index')
#     plt.ylabel('Latitude Index')
#     plt.savefig(f'grid_RMSE_{i}.png')
#     plt.close()