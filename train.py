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
parser.add_argument('--variables', nargs='+', default=['q700'], help='List of variables')
parser.add_argument('--predictand', type=str, default='RAINNC', help='Predictand')
parser.add_argument('--topology', type=str, default='deepesd', help='Topology')
parser.add_argument('--approach', type=str, default='MOS-E', help='Approach')
parser.add_argument('--start_year', type=str, default='', help='start year')
parser.add_argument('--end_year', type=str, default='', help='end year')
parser.add_argument('--modelPath', type=str, default='', help='model path')
parser.add_argument('--variables_str', type=str, default='', help='variables string')
parser.add_argument('--scale', type=bool, help='scale')
parser.add_argument('--predictor_data', type=str, default='', help='Path to predictors')
parser.add_argument('--predictand_data', type=str, default='', help='Path to predictands')
parser.add_argument('--landmask_data', type=str, default='', help='Path to landmask')
parser.add_argument('--loss_path', type=str, default='', help='loss_path')
args = parser.parse_args()

def is_leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

variables = args.variables
predictand = args.predictand
topology = args.topology
approach = args.approach
start_year = args.start_year
end_year = args.end_year
modelPath = args.modelPath
variables_str = args.variables_str
scale = args.scale
predictor_data = args.predictor_data
predictand_data = args.predictand_data
landmask_data = args.landmask_data
loss_path = args.loss_path

print('===== training settings =====')
print(f'variables: {variables}')
print(f'y label: {predictand}')
print(f'years for training: {start_year}~{end_year}')
print(f'ModelName: {modelPath}')
print(f'scale: {scale}')
print(f'{loss_path}')
a = input()

print('===== concat predictand =====')
yearly_ys = []
for year in tqdm.tqdm(range(int(start_year), int(end_year)+1)):
    file_path = f'{predictand_data}TReAD_daily_{year}_{predictand}.nc'
    dataset = xr.open_dataset(file_path)
    yearly_ys.append(dataset)
y = xr.concat(yearly_ys, dim='Day')
min_lat = y['Lat'].min().item()
max_lat = y['Lat'].max().item()
min_lon = y['Lon'].min().item()
max_lon = y['Lon'].max().item()
print(f'latitude range: {min_lat}~{max_lat}')
print(f'longitude range: {min_lon}~{max_lon}')

print('===== concat predictor =====')
x_datasets = {}
for var in variables:
    monthly_datasets = []
    for year in tqdm.tqdm(range(int(start_year), int(end_year)+1)):
        for month in range(1, 13):
            file_path = f'{predictor_data}{var}/{str(year)}/ERA5_PRS_{var}_{str(year)}{month:02}_r1440x721_day.nc'
            dataset = xr.open_dataset(file_path)
            if is_leap_year(year) and month == 2:
                dataset = dataset.sel(time=~((dataset.time.dt.month == 2) & (dataset.time.dt.day == 29)))
            dataset = dataset.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
            monthly_datasets.append(dataset)
    concatenated_dataset = xr.concat(monthly_datasets, dim='time')
    x_datasets[var] = concatenated_dataset
x = xr.merge([x_datasets[var] for var in variables])
filtered_x = x.drop_vars('time_bnds')

if scale is True:
	filtered_x = scaleGrid(filtered_x, base = filtered_x, type = 'standardize', spatialFrame = 'gridbox')

if predictand == 'pr':
    y = binaryGrid(y, condition = 'GE', threshold = 1, partial = True)

x_array = filtered_x.to_stacked_array("var", sample_dims = ["longitude", "latitude", "time"]).values

outputShape = None
mask = xr.open_dataset(landmask_data)

if topology == 'deepesd':
    mask.landmask.values[mask.landmask.values == 0] = np.nan
    mask_Onedim = mask.landmask.values.reshape((np.prod(mask.landmask.shape)))

    ind = [i for i in range(len(mask_Onedim)) if mask_Onedim[i] == 1]
    yTrain = y[predictand].values.reshape((x.dims['time'],np.prod(mask.landmask.shape)))[:,ind]
    if predictand == 'RAINNC':
        yTrain = yTrain - 0.99
        yTrain[yTrain < 0] = 0
    outputShape = yTrain.shape[1]
if topology == 'unet':
		sea = mask.landmask.values == 0
		y[predictand].values[:,sea] = 0
		yTrain = y[predictand].values
		outputShape = None

print('===== model structure =====')
model = deepmodel(topology = topology, predictand = predictand, inputShape = x_array.shape[1::], outputShape = outputShape)
model.summary()

if predictand == 'T2':
	loss = 'mse'
elif predictand == 'RAINNC':
	loss = bernoulliGamma
model.compile(loss = loss, optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001))

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience = 30),
    tf.keras.callbacks.ModelCheckpoint(filepath = modelPath, monitor = 'val_loss', save_best_only = True)
]

history = model.fit(x = x_array, y = yTrain, batch_size = 40, epochs = 10000, validation_split = 0.1, callbacks = my_callbacks)

train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel(f'{loss}')
plt.title(f'{predictand} Model Loss Over Epochs: {start_year}~{end_year}\nparameter: {variables}')
plt.legend()
plt.grid(True)
plt.savefig(f'{loss_path}')