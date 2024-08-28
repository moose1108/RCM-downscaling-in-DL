import os
import xarray as xr
import tensorflow as tf
from tensorflow.keras import backend as K
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
# import cartopy.crs as ccrs
from utils.deepmodel import *
from utils.emulate import *
from utils.auxiliaryFunctions import *

parser = argparse.ArgumentParser(description="Train a deep learning model for climate data.")
parser.add_argument('--gcm', type=str, default='noresm', help='General Circulation Model')
parser.add_argument('--rcm', type=str, default='ald63', help='Regional Climate Model')
parser.add_argument('--variables', nargs='+', default=['q700'], help='List of variables')
parser.add_argument('--predictand', type=str, default='RAINNC', help='Predictand')
parser.add_argument('--topology', type=str, default='deepesd', help='Topology')
parser.add_argument('--approach', type=str, default='MOS-E', help='Approach')
parser.add_argument('--start_year', type=str, default='', help='start year')
parser.add_argument('--end_year', type=str, default='', help='end year')
parser.add_argument('--modelPath', type=str, default='', help='model path')
args = parser.parse_args()

def is_leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

gcm = args.gcm
rcm = args.rcm
variables = args.variables
predictand = args.predictand
topology = args.topology
approach = args.approach
start_year = args.start_year
end_year = args.end_year
modelPath = args.modelPath

print('===== concat predictand =====')
y_base_path = '/work/moose1108/corrdiff-like/data/02-predictand_TReAD/RAINNC/'
yearly_ys = []
for year in tqdm.tqdm(range(int(start_year), int(end_year)+1)):
    file_path = f'{y_base_path}TReAD_daily_{year}_RAINNC.nc'
    dataset = xr.open_dataset(file_path)
    yearly_ys.append(dataset)
y = xr.concat(yearly_ys, dim='time')

min_lat = y['Lat'].min().item()
max_lat = y['Lat'].max().item()
min_lon = y['Lon'].min().item()
max_lon = y['Lon'].max().item()
# print(min_lat, max_lat, min_lon, max_lon)
# a = input()

print('===== concat predictor =====')
base_path = '/work/moose1108/corrdiff-like/data/01-predictor_ERA5/'
x_datasets = {}
for var in variables:
    monthly_datasets = []
    for year in tqdm.tqdm(range(int(start_year), int(end_year)+1)):
        for month in range(1, 13):
            file_path = f'{base_path}{var}/{str(year)}/ERA5_PRS_{var}_{str(year)}{month:02}_r1440x721_day.nc'
            dataset = xr.open_dataset(file_path)
            if is_leap_year(year) and month == 2:
                dataset = dataset.sel(time=~((dataset.time.dt.month == 2) & (dataset.time.dt.day == 29)))
            dataset = dataset.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
            monthly_datasets.append(dataset)
    concatenated_dataset = xr.concat(monthly_datasets, dim='time')
    x_datasets[var] = concatenated_dataset
x = xr.merge([x_datasets[var] for var in variables])
filtered_x = x.drop_vars('time_bnds')

# for var in variables:
#     monthly_datasets = []
    # for year in range(start_year, end_year+1):
    # for month in range(1, 13):
    #     file_path = f'{base_path}{var}/1981/ERA5_PRS_{var}_1981{month:02}_r1440x721_day.nc'
    #     dataset = xr.open_dataset(file_path)
    #     monthly_datasets.append(dataset)
    # concatenated_dataset = xr.concat(monthly_datasets, dim='time')
    # print(concatenated_dataset)
    # x_datasets[var] = concatenated_dataset
    # print(x_datasets)
    # a = input()
# x = xr.merge([x_datasets[var] for var in variables])

# x_list = [xr.open_dataset(f'/home/moose1108/corrdiff-like-project/01-predictor_ERA5/q700/1981/ERA5_PRS_q700_1981{i:02}_r1440x721_day.nc') for i in range(1, 13)]
# x = xr.concat(x_list, dim='time')

variables_str = ''
for i in variables:
    variables_str += i

# y = xr.open_dataset('/work/moose1108/corrdiff-like/data/02-predictand_TReAD/RAINNC/TReAD_daily_1981_RAINNC.nc')
# modelPath = './models/' + predictand + '/' + topology + '-' + gcm + '-' + rcm + '-' + approach + '-' + variables_str + '-' + str(int(end_year) - int(start_year) + 1) + '.h5'
print(f'===== ModelName: {modelPath} =====')
# a = input()
# filtered_x = x.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
# filtered_x = filtered_x.drop_vars('time_bnds')

scale = True
if scale is True:
	filtered_x = scaleGrid(filtered_x, base = filtered_x, type = 'standardize', spatialFrame = 'gridbox')

y = binaryGrid(y, condition = 'GE', threshold = 1, partial = True)
x_array = filtered_x.to_stacked_array("var", sample_dims = ["longitude", "latitude", "time"]).values
print(x_array.shape)
print(x_array[0])
a = input()
outputShape = None
mask = xr.open_dataset('/work/moose1108/corrdiff-like/data/02-predictand_TReAD/TReAD_Regrid_2km_landmask.nc')

if topology == 'deepesd':
    mask.landmask.values[mask.landmask.values == 0] = np.nan
    mask_Onedim = mask.landmask.values.reshape((np.prod(mask.landmask.shape)))

    ind = [i for i in range(len(mask_Onedim)) if mask_Onedim[i] == 1]
    yTrain = y[predictand].values.reshape((x.dims['time'],np.prod(mask.landmask.shape)))[:,ind]
    if predictand == 'RAINNC':
        yTrain = yTrain - 0.99
        yTrain[yTrain < 0] = 0
    outputShape = yTrain.shape[1]

print('===== model structure =====')
model = deepmodel(topology = topology, predictand = predictand, inputShape = x_array.shape[1::], outputShape = outputShape)
model.summary()

loss = bernoulliGamma
model.compile(loss = loss, optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001))

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience = 30),
    tf.keras.callbacks.ModelCheckpoint(filepath = modelPath, monitor = 'val_loss', save_best_only = True)
]

history = model.fit(x = x_array, y = yTrain, batch_size = 100, epochs = 10000, validation_split = 0.1, callbacks = my_callbacks)

train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(f'./plots/loss_{variables_str}_{str(int(end_year) - int(start_year)+1)}y_bd.png')