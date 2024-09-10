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

parser = argparse.ArgumentParser(description="Plot a deep learning model for climate data.")
parser.add_argument('--model', type=str, default='', help='model')
parser.add_argument('--predict_path', type=str, default='', help='path to predict result')
parser.add_argument('--gt_path', type=str, default='', help='ground truth')
parser.add_argument('--predict_year', type=int, default='', help='year being predicted')
parser.add_argument('--plot_path', type=str, default='', help='Path to put the visualization')
parser.add_argument('--landmask_data', type=str, default='', help='Path to landmask')
args = parser.parse_args()

print('===== plotting settings =====')
model = args.model
predict_path = args.predict_path
gt_path = args.gt_path
predict_year = args.predict_year
plot_path = args.plot_path
landmask_data = args.landmask_data
print(f'year: {predict_year}')

pred_dataset = nc.Dataset(predict_path)
gt_dataset = nc.Dataset(gt_path)
landmask_dataset = nc.Dataset(landmask_data)

rain = pred_dataset.variables['RAINNC'][:]
gt_rain = gt_dataset.variables['RAINNC'][:]

lat = pred_dataset.variables['latitude'][:]
lon = pred_dataset.variables['longitude'][:]
gt_lat = gt_dataset.variables['Lat'][:]
gt_lon = gt_dataset.variables['Lon'][:]

land_sea_mask = landmask_dataset.variables['landmask'][:]
gt_rain_masked = np.where(land_sea_mask == 1, gt_rain, np.nan)

# Convert days to dates
start_date = datetime.date(predict_year, 1, 1)  # Start date of the dataset
selected_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
# selected_days = [130, 131, 132, 133, 134]
num_days = len(selected_days)
dates = [start_date + datetime.timedelta(days=day) for day in selected_days]

# Setup the plot
fig, axes = plt.subplots(num_days, 2, figsize=(20, 10 * num_days), subplot_kw={'projection': ccrs.PlateCarree()})
if num_days == 1:
    axes = [axes]  # Ensure axes is iterable if there's only one row

# color_list = ["white", "gray", "lightskyblue", "deepskyblue", "dodgerblue", "blue", "green", "lime", "yellow", "orange", "darkorange", "red", "firebrick", "darkred", "purple", "mediumorchid", "magenta", "violet"]
# cmap = mcolors.ListedColormap(color_list)
cm_pw = ['#ffffff', '#808080', '#a0fffa','#00cdff','#0096ff','#0069ff',
 '#329600','#32ff00','#ffff00','#ffc800',
 '#ff9600','#ff0000','#c80000','#a00000',
 '#96009b','#c800d2','#ff00f5','#ff64ff', '#ffc8ff', '#f284ba'] #20
cMap = mcolors.ListedColormap(cm_pw)
stnd = [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30] #17
# stnd = [0, 0.1, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 35, 40]
norm = mcolors.BoundaryNorm(stnd, (len(stnd) - 1))

# Loop over selected days and plot
for i, (day, date) in enumerate(zip(selected_days, dates)):
    # Plot prediction
    ax = axes[i][0]  # First column for predictions
    rain_plot = ax.contourf(lon, lat, rain[day, :, :]*24, levels=stnd, cmap=cMap, norm=norm)
    ax.set_title(f'Model Prediction - {date.strftime("%B %d")}', pad=20) 
    ax.coastlines()

    # Plot ground truth
    ax = axes[i][1]  # Second column for ground truth
    gt_plot = ax.contourf(gt_lon, gt_lat, gt_rain_masked[day, :, :]*24, levels=stnd, cmap=cMap, norm=norm)
    ax.set_title(f'Ground Truth - {date.strftime("%B %d")}', pad=20)
    ax.coastlines()

plt.suptitle(f'Predict {args.predict_year} with model: {model}', fontsize=16, y=0.92)
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

# Add a unified colorbar for each row
for i in range(num_days):
    fig.colorbar(rain_plot, ax=axes[i], orientation='horizontal', pad=0.05, fraction=0.05, label='Rainfall (mm)')

plt.savefig(f'{plot_path}')