# import netCDF4 as nc
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import cartopy.io.shapereader as shpreader
# import matplotlib.colors as mcolors
# import numpy as np
# import random
# import datetime
# import argparse

# parser = argparse.ArgumentParser(description="Plot a deep learning model for climate data.")
# parser.add_argument('--model', type=str, default='', help='model')
# parser.add_argument('--predict_path', type=str, default='', help='path to predict result')
# parser.add_argument('--gt_path', type=str, default='', help='ground truth')
# parser.add_argument('--predict_year', type=int, default='', help='year being predicted')
# parser.add_argument('--plot_path', type=str, default='', help='Path to put the visualization')
# parser.add_argument('--landmask_data', type=str, default='', help='Path to landmask')
# args = parser.parse_args()

# print('===== plotting settings =====')
# model = args.model
# predict_path = args.predict_path
# gt_path = args.gt_path
# predict_year = args.predict_year
# plot_path = args.plot_path
# landmask_data = args.landmask_data
# print(f'year: {predict_year}')
# print(f'{model}')

# pred_dataset = nc.Dataset(predict_path)
# gt_dataset = nc.Dataset(gt_path)
# landmask_dataset = nc.Dataset(landmask_data)

# rain = pred_dataset.variables['RAINNC'][:]
# gt_rain = gt_dataset.variables['RAINNC'][:]

# lat = pred_dataset.variables['latitude'][:]
# lon = pred_dataset.variables['longitude'][:]
# gt_lat = gt_dataset.variables['Lat'][:]
# gt_lon = gt_dataset.variables['Lon'][:]

# land_sea_mask = landmask_dataset.variables['landmask'][:]
# gt_rain_masked = np.where(land_sea_mask == 1, gt_rain, np.nan)

# # Convert days to dates
# start_date = datetime.date(predict_year, 1, 1)  # Start date of the dataset
# selected_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
# # selected_days = [130, 131, 132, 133, 134]
# num_days = len(selected_days)
# dates = [start_date + datetime.timedelta(days=day) for day in selected_days]

# # Setup the plot
# fig, axes = plt.subplots(num_days, 2, figsize=(20, 10 * num_days), subplot_kw={'projection': ccrs.PlateCarree()})
# if num_days == 1:
#     axes = [axes]  # Ensure axes is iterable if there's only one row

# # color_list = ["white", "gray", "lightskyblue", "deepskyblue", "dodgerblue", "blue", "green", "lime", "yellow", "orange", "darkorange", "red", "firebrick", "darkred", "purple", "mediumorchid", "magenta", "violet"]
# # cmap = mcolors.ListedColormap(color_list)
# cm_pw = ['#ffffff', '#808080', '#a0fffa','#00cdff','#0096ff','#0069ff',
#  '#329600','#32ff00','#ffff00','#ffc800',
#  '#ff9600','#ff0000','#c80000','#a00000',
#  '#96009b','#c800d2','#ff00f5','#ff64ff', '#ffc8ff', '#f284ba'] #20
# cMap = mcolors.ListedColormap(cm_pw)
# stnd = [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30] #17
# # stnd = [0, 0.1, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 35, 40]
# norm = mcolors.BoundaryNorm(stnd, (len(stnd) - 1))

# # Loop over selected days and plot
# for i, (day, date) in enumerate(zip(selected_days, dates)):
#     # Plot prediction
#     ax = axes[i][0]  # First column for predictions
#     rain_plot = ax.contourf(lon, lat, rain[day, :, :]*24, levels=stnd, cmap=cMap, norm=norm)
#     ax.set_title(f'Model Prediction - {date.strftime("%B %d")}', pad=20) 
#     ax.coastlines()

#     # Plot ground truth
#     ax = axes[i][1]  # Second column for ground truth
#     gt_plot = ax.contourf(gt_lon, gt_lat, gt_rain_masked[day, :, :]*24, levels=stnd, cmap=cMap, norm=norm)
#     ax.set_title(f'Ground Truth - {date.strftime("%B %d")}', pad=20)
#     ax.coastlines()

# plt.suptitle(f'Predict {args.predict_year} with model: {model}', fontsize=16, y=0.92)
# # Add Taiwan boundary using a shapefile
# shapefile_path = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
# reader = shpreader.Reader(shapefile_path)
# countries = list(reader.records())  # Store in list to avoid multiple iterations exhausting the generator

# for i in range(num_days):
#     for axi in axes[i]:
#         for country in countries:
#             if country.attributes['NAME'] == 'Taiwan':
#                 axi.add_geometries(country.geometry, ccrs.PlateCarree(),
#                                    edgecolor='black', facecolor='none', linewidth=2)

# # Add gridlines and labels
# for i in range(num_days):
#     for axi in axes[i]:
#         axi.gridlines(draw_labels=True)

# # Add a unified colorbar for each row
# for i in range(num_days):
#     fig.colorbar(rain_plot, ax=axes[i], orientation='horizontal', pad=0.05, fraction=0.05, label='Rainfall (mm)')

# plt.savefig(f'{plot_path}')

import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.colors as mcolors
import numpy as np
import random
import datetime
import argparse
import xarray as xr
import pandas as pd

parser = argparse.ArgumentParser(description="Plot a deep learning model for climate data.")
parser.add_argument('--model', type=str, default='', help='model')
parser.add_argument('--predict_path', type=str, default='', help='path to predict result')
parser.add_argument('--gt_path', type=str, default='', help='ground truth')
parser.add_argument('--predict_year', type=int, default='', help='year being predicted')
parser.add_argument('--plot_path', type=str, default='', help='Path to put the visualization')
parser.add_argument('--landmask_data', type=str, default='', help='Path to landmask')
parser.add_argument('--predictand', type=str, default='pr')
args = parser.parse_args()

def is_leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

print('===== plotting settings =====')
model = args.model
predict_path = args.predict_path
gt_path = args.gt_path
predict_year = args.predict_year
plot_path = args.plot_path
landmask_data = args.landmask_data
predictand = args.predictand
print(f'year: {predict_year}')
pred_dataset = nc.Dataset(predict_path)
gt_dataset = nc.Dataset(gt_path)
landmask_dataset = nc.Dataset(landmask_data)

rain = pred_dataset.variables[predictand][:]
gt_rain = gt_dataset.variables[predictand][:]

lat = pred_dataset.variables['latitude'][:]
lon = pred_dataset.variables['longitude'][:]
gt_lat = gt_dataset.variables['Lat'][:]
gt_lon = gt_dataset.variables['Lon'][:]

land_sea_mask = landmask_dataset.variables['landmask'][:]
gt_rain_masked = np.where(land_sea_mask == 1, gt_rain, np.nan)


start_date = datetime.date(predict_year, 1, 1)  # Start date of the dataset
selected_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334] # [i for i in range(180, 210)]# [135] # 

num_days = len(selected_days)
dates = [start_date + datetime.timedelta(days=day) for day in selected_days]
print(f'{dates[0]}~{dates[-1]}')

# Setup the plot
fig, axes = plt.subplots(num_days, 2, figsize=(20, 10 * num_days), subplot_kw={'projection': ccrs.PlateCarree()})
if num_days == 1:
    axes = [axes]  # Ensure axes is iterable if there's only one row

if predictand == 'RAINNC':
    standards = [0, 1, 2, 6, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300, 600]#[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1]# 
    color_map = ['#ffffff','#98ffff','#00ceff','#009aff','#006af7','#2e9c00','#2bff00','#fefe08','#ffcb00','#ff9c00','#fe0005','#c90200','#9d0000','#9a009d','#cf00d7','#ff00f7','#fdcafe']
elif predictand == 'T2':
    color_map = ['#117288','#207F95' ,'#2E899C' ,'#3C93A7' ,'#4D9EB1' ,'#5CA9BD' ,'#68B4C4' ,'#77BFCD' ,'#87CBD8' ,'#93D5E3' ,'#A4DFEB' ,'#B4E9F7' ,'#0D924E' ,'#1C9A53' ,'#2FA257' ,'#3FA95E' ,'#51B265' ,'#5EB96B' ,'#74C16F' ,'#83C976' ,'#95D07C' ,'#A6D984' ,'#BBDF88' ,'#CAE68F' ,'#D9F191' ,'#F2F4C3' ,'#F6E78B' ,'#F2D577' ,'#F1C363' ,'#EFB14C' ,'#E99E39' ,'#E58D29' ,'#DD7D05' ,'#EE5233' ,'#EF165B' ,'#AC0539' ,'#750204' ,'#9C68AA' ,'#864F99' ,'#7F279A']
    standards = np.arange(272.15, 313.15)


start_date = datetime.date(args.predict_year, 1, 1)
end_date = datetime.date(args.predict_year + 1, 1, 1)
date_range = [start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days)]

correlations = []
maxi = 1
index = 0
for i, date in enumerate(date_range):
    if is_leap_year(int(predict_year)) and i == 365:
        break
    daily_pred = rain[i, :, :]

    daily_gt = gt_rain_masked[i, :, :]
    valid_mask = ~np.isnan(daily_pred) & ~np.isnan(daily_gt)
    if np.any(valid_mask):
        daily_pred_valid = daily_pred[valid_mask]
        daily_gt_valid = daily_gt[valid_mask]
        correlation = np.corrcoef(daily_pred_valid, daily_gt_valid)[0, 1]
        # print(i, correlation)
        if np.isnan(correlation):
            correlations.append(0)
        else:
            correlations.append(correlation)
            if correlation < maxi:
                maxi = correlation
                index = date
    else:
        correlations.append(np.nan)
print(maxi, index)
# a = input()

if predictand == 'RAINNC':
    factor = 24
else:
    factor = 1

for i, (day, date) in enumerate(zip(selected_days, dates)):
    
    if is_leap_year(int(predict_year)):
        date += datetime.timedelta(days=1)
    daily_pred = rain[day, :, :] * factor
    daily_gt = gt_rain_masked[day, :, :]
    valid_mask = ~np.isnan(daily_pred) & ~np.isnan(daily_gt)
    if np.any(valid_mask):  # Check if there's any valid data to correlate
        daily_pred_valid = daily_pred[valid_mask]
        daily_gt_valid = daily_gt[valid_mask]

        # Compute correlation coefficient
        correlation_matrix = np.corrcoef(daily_pred_valid, daily_gt_valid)
        correlation_coefficient = correlation_matrix[0, 1]  # Extract the off-diagonal value which is the correlation coefficient
        print(f"Correlation on {date.strftime('%Y-%m-%d')} between model predictions and ground truth: {correlation_coefficient:.3f}")
    else:
        print(f"No valid data on {date.strftime('%Y-%m-%d')} to compute correlation.")

    # Plot prediction
    ax = axes[i][0]
    rain_plot = ax.contourf(lon, lat, rain[day, :, :] * factor, levels=standards, colors=color_map)
    ax.set_title(f'Model Prediction ({predictand}) - {date.strftime("%B %d")}', pad=20)
    ax.coastlines()

    # Plot ground truth
    ax = axes[i][1]
    gt_plot = ax.contourf(gt_lon, gt_lat, gt_rain_masked[day, :, :] * factor, levels=standards, colors=color_map)
    ax.set_title(f'Ground Truth ({predictand}) - {date.strftime("%B %d")}', pad=20)
    ax.coastlines()

# plt.suptitle(f'bgfdasgbsrfwergPredict {args.predict_year} with model: {model}', fontsize=16, y=0.92)
# Add Taiwan boundary using a shapefile
shapefile_path = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
reader = shpreader.Reader(shapefile_path)
countries = list(reader.records())  # Store in list to avoid multiple iterations exhausting the generator

for i in range(num_days):
    for axi in axes[i]:
        for country in countries:
            if country.attributes['NAME'] == 'Taiwan':
                axi.add_geometries(country.geometry, ccrs.PlateCarree(),
                                   edgecolor='black', facecolor='none', linewidth=2)

# Add gridlines and labels
for i in range(num_days):
    for axi in axes[i]:
        axi.gridlines(draw_labels=True)

label = 'Precipitation (mm)' if predictand == 'RAINNC' else 'Temperature (°C)'
for i in range(num_days):
    fig.colorbar(rain_plot, ax=axes[i], orientation='horizontal', pad=0.05, fraction=0.05, label=label)

plt.savefig(f'{plot_path}')