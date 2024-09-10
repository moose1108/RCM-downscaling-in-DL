import os
import xarray as xr
import argparse
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
from utils.emulate import *
from utils.auxiliaryFunctions import *
import tqdm

parser = argparse.ArgumentParser(description='Run climate model emulation.')
parser.add_argument('--variables', nargs='+', default=['q700'], help='List of variables')
parser.add_argument('--predictand', type=str, default='pr')
parser.add_argument('--topology', type=str, default='deepesd')
parser.add_argument('--approach', type=str, default='MOS-E')
parser.add_argument('--outputFileName', type=str, default='./predictions/a.nc')
parser.add_argument('--variables_str', type=str, default='', help='variables string')
parser.add_argument('--years', type=str, default='')
parser.add_argument('--scale', type=bool, default='True')
parser.add_argument('--bias_correction', type=str, default='False')
parser.add_argument('--modelPath', type=str, default='')
args = parser.parse_args()

variables = args.variables
predictand = args.predictand
topology = args.topology
approach = args.approach
outputFileName = args.outputFileName
scale = args.scale
bias_correction = args.bias_correction
modelPath = args.modelPath
variables_str = args.variables_str
years = args.years

print('===== training settings =====')
print(f'variables: {variables}')
print(f'y label: {predictand}')
print(f'years to predict: {years}')
print(f'ModelName: {modelPath}')
print(f'scale: {scale}')
a = input()

def is_leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

print('===== concat predictand =====')
predictand_base_path = '/work/moose1108/corrdiff-like/data/02-predictand_TReAD/RAINNC/'
y_datasets = []
for year in tqdm.tqdm(range(2001, 2010, 2)):
    file_path = f'{predictand_base_path}/TReAD_daily_{year}_RAINNC.nc'
    dataset = xr.open_dataset(file_path)
    y_datasets.append(dataset)
y = xr.concat(y_datasets, dim='Day')
min_lat = y['Lat'].min().item()
max_lat = y['Lat'].max().item()
min_lon = y['Lon'].min().item()
max_lon = y['Lon'].max().item()
print(f'latitude range: {min_lat}~{max_lat}')
print(f'longitude range: {min_lon}~{max_lon}')

print('===== concat predictor base data =====')
base_path = '/work/moose1108/corrdiff-like/data/01-predictor_ERA5/'
x_datasets = {}

for var in variables:
    monthly_datasets = []
    for month in range(1, 13):
        file_path = f'{base_path}{var}/{years}/ERA5_PRS_{var}_{years}{month:02}_r1440x721_day.nc'
        dataset = xr.open_dataset(file_path)
        monthly_datasets.append(dataset)
    concatenated_dataset = xr.concat(monthly_datasets, dim='time')
    x_datasets[var] = concatenated_dataset
x = xr.merge([x_datasets[var] for var in variables])

base_datasets = {}
for var in variables:
    monthly_datasets = []
    for year in tqdm.tqdm(range(2001, 2010, 2)):
        for month in range(1, 13):
            file_path = f'{base_path}{var}/{str(year)}/ERA5_PRS_{var}_{str(year)}{month:02}_r1440x721_day.nc'
            dataset = xr.open_dataset(file_path)
            if is_leap_year(year) and month == 2:
                dataset = dataset.sel(time=~((dataset.time.dt.month == 2) & (dataset.time.dt.day == 29)))
            dataset = dataset.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
            monthly_datasets.append(dataset)
    concatenated_dataset = xr.concat(monthly_datasets, dim='time')
    base_datasets[var] = concatenated_dataset
base = xr.merge([base_datasets[var] for var in variables])

base = base.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
base = base.drop_vars('time_bnds')
x = x.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
x = x.drop_vars('time_bnds')

if scale is True:
    print('scaling...')
    x = scaleGrid(x, base = base, type = 'standardize', spatialFrame = 'gridbox')

if predictand == 'T2':
    model = tf.keras.models.load_model(modelPath)
    description = {'description': 'air surface temperature (ºC)'}
elif predictand == 'RAINNC':
    model = tf.keras.models.load_model(modelPath, custom_objects = {'bernoulliGamma': bernoulliGamma})
    description = {'description': 'total daily precipitation (mm/day)'}
print('===== model structure =====')
model.summary()

x_array = x.to_stacked_array("var", sample_dims = ["latitude", "longitude", "time"]).values
pred = model.predict(x_array)

## Reshaping the prediction to a latitude-longitude grid
mask = xr.open_dataset('/work/moose1108/corrdiff-like/data/02-predictand_TReAD/TReAD_Regrid_2km_landmask.nc')
if topology == 'deepesd':
    mask.landmask.values[mask.landmask.values == 0] = np.nan
    mask_Onedim = mask.landmask.values.reshape((np.prod(mask.landmask.shape)))
    ind = [i for i in range(len(mask_Onedim)) if mask_Onedim[i] == 1]
    pred = reshapeToMap(grid = pred, ntime = x.dims['time'], nlat = mask.dims['Lat'], nlon = mask.dims['Lon'], indLand = ind)
if topology == 'unet':
    sea = mask.sftlf.values == 0
    pred = np.squeeze(pred)
    pred[:,sea] = np.nan

if predictand == 'RAINNC':
    y_bin = binaryGrid(y.RAINNC, condition = 'GE', threshold = 1)
    ## Prediction on the train set -----------
    base2 = scaleGrid(base, base = base, type = 'standardize', spatialFrame = 'gridbox')
    base_array = base2.to_stacked_array("var", sample_dims = ["longitude", "latitude", "time"]).values
    pred_ocu_train = model.predict(base_array)[:,:,0]
    pred_ocu_train = reshapeToMap(grid = pred_ocu_train, ntime = base2.dims['time'], nlat = mask.dims['Lat'], nlon = mask.dims['Lon'], indLand = ind)
    pred_ocu_train = xr.DataArray(pred_ocu_train, dims = ['time','latitude','longitude'], coords = {'longitude': mask.Lon.values, 'latitude': mask.Lat.values, 'time': y.Day.values})
    ## ---------------------------------------
    ## Recovering the complete serie -----------
    pred = xr.Dataset(data_vars = {'p': (['time','latitude','longitude'], pred[:,:,:,0]),
                                    'log_alpha': (['time','latitude','longitude'], pred[:,:,:,1]),
                                    'log_beta': (['time','latitude','longitude'], pred[:,:,:,2])},
                                    coords = {'longitude': mask.Lon.values, 'latitude': mask.Lat.values, 'time': x.time.values})
    pred_bin = adjustRainfall(grid = pred.p, refPred = pred_ocu_train, refObs = y_bin)
    pred_amo = computeRainfall(log_alpha = pred.log_alpha, log_beta = pred.log_beta, bias = 1, simulate = True)
    pred = pred_bin * pred_amo
    pred = pred.values
    ## -----------------------------------------

template_predictand = xr.open_dataset('/work/moose1108/corrdiff-like/data/02-predictand_TReAD/RAINNC/TReAD_daily_2009_RAINNC.nc')
pred = xr.Dataset(
    data_vars = {predictand: (['time','latitude','longitude'], pred)},
    coords = {'longitude': template_predictand.Lon.values, 'latitude': template_predictand.Lat.values, 'time': x.time.values},
    attrs = description
)
print(pred)
pred.to_netcdf(outputFileName)
