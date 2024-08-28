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

parser = argparse.ArgumentParser(description='Run climate model emulation.')
parser.add_argument('--gcm', type=str, default='noresm')
parser.add_argument('--rcm', type=str, default='ald63')
parser.add_argument('--variables', nargs='+', default=['q700'], help='List of variables')
parser.add_argument('--predictand', type=str, default='pr')
parser.add_argument('--topology', type=str, default='deepesd')
parser.add_argument('--approach', type=str, default='MOS-E')
parser.add_argument('--outputFileName', type=str, default='./predictions/a.nc')
parser.add_argument('--years', type=str, default='')
parser.add_argument('--scale', type=str, default='True')
parser.add_argument('--bias_correction', type=str, default='False')
parser.add_argument('--modelPath', type=str, default='')
args = parser.parse_args()

gcm = args.gcm
rcm = args.rcm
variables = args.variables
predictand = args.predictand
topology = args.topology
approach = args.approach
outputFileName = args.outputFileName
scale = args.scale
bias_correction = args.bias_correction
modelPath = args.modelPath
years = args.years
emulator = gcm + '-ald63'

def reshapeToMa(grid, ntime, nlat, nlon, indLand):
	if len(grid.shape) == 2:
		grid = np.expand_dims(grid, axis = 2)
		vars = 1
	InnerList=[]
	for i in range(grid.shape[2]):
		p = np.full((ntime,nlat*nlon), np.nan)
		p[:,indLand] = grid[:,:,i]
		print(p[:,indLand])
		# a = input()
		p = p.reshape((ntime,nlat,nlon,1))
		InnerList.append(p)
	grid = np.concatenate(InnerList, axis = 3)
	return grid.squeeze()

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
    print(var)
x = xr.merge([x_datasets[var] for var in variables])

# x_list = [xr.open_dataset(f'/home/moose1108/corrdiff-like-project/01-predictor_ERA5/q700/2009/ERA5_PRS_q700_2009{i:02}_r1440x721_day.nc') for i in range(1, 13)]
# x = xr.concat(x_list, dim='time')
if approach == 'MOS-E':
    modelEmulator = emulator
    # modelEmulator = emulator + '-' + rcm

y = xr.open_dataset('/work/moose1108/corrdiff-like/data/02-predictand_TReAD/RAINNC/TReAD_daily_2011_RAINNC.nc')
# y = yh.isel(Day=slice(0, 365))
min_lat = y['Lat'].min().item()
max_lat = y['Lat'].max().item()
min_lon = y['Lon'].min().item()
max_lon = y['Lon'].max().item()

base = x.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
base = base.drop_vars('time_bnds')
x = x.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
x = x.drop_vars('time_bnds')

# modelPath = './models/RAINNC/' + topology + '-' + modelEmulator + '-' + approach + '.h5'
# modelPath = './models/RAINNC/' + topology + '-' + predictand + '-' + modelEmulator + '-' + approach + '.h5'

# if years is not None:
#     base = xr.merge([base.sel(time = base['time.year'] == int(year)) for year in years])
#     modelPath = '../models/' + topology + '-' + predictand + '-' + modelEmulator + '-' + approach + '-year' + str(len(years)) + '.h5'

if scale is True:
    print('scaling...')
    x = scaleGrid(x, base = base, type = 'standardize', spatialFrame = 'gridbox')

## Loading the cnn model...
if predictand == 'tas':
    model = tf.keras.models.load_model(modelPath)
    description = {'description': 'air surface temperature (ºC)'}
elif predictand == 'pr':
    model = tf.keras.models.load_model(modelPath, custom_objects = {'bernoulliGamma': bernoulliGamma})
    description = {'description': 'total daily precipitation (mm/day)'}

model.summary()
# a = input()
## Converting xarray to a numpy array and predict on the test set
x_array = x.to_stacked_array("var", sample_dims = ["latitude", "longitude", "time"]).values
pred = model.predict(x_array)

## Reshaping the prediction to a latitude-longitude grid
mask = xr.open_dataset('/work/moose1108/corrdiff-like/data/02-predictand_TReAD/TReAD_Regrid_2km_landmask.nc')
if topology == 'deepesd':
    mask.landmask.values[mask.landmask.values == 0] = np.nan
    mask_Onedim = mask.landmask.values.reshape((np.prod(mask.landmask.shape)))
    ind = [i for i in range(len(mask_Onedim)) if mask_Onedim[i] == 1]
    print(pred.shape)
    # a = input()
    pred = reshapeToMa(grid = pred, ntime = x.dims['time'], nlat = mask.dims['Lat'], nlon = mask.dims['Lon'], indLand = ind)
if topology == 'unet':
    sea = mask.sftlf.values == 0
    pred = np.squeeze(pred)
    pred[:,sea] = np.nan

if predictand == 'pr':
    ## Loading the reference observation for the occurrence of precipitation ---------------------------
    # gcm = newdata.split("_")[1].split("-")[0]
    # yh = xr.open_dataset('../data/pr/pr_' + gcm + '-ald63_historical_1996-2005.nc') #, decode_times = False)
    # y85 = xr.open_dataset('../data/pr/pr_' + gcm + '-ald63_rcp85_2090-2099.nc') #, decode_times = False)
    # y = xr.concat([yh,y85], dim = 'time')
    y_bin = binaryGrid(y.RAINNC, condition = 'GE', threshold = 1)
    ## -------------------------------------------------------------------------------------------------
    ## Prediction on the train set -----------
    base2 = scaleGrid(base, base = base, type = 'standardize', spatialFrame = 'gridbox')
    # ind_time = np.intersect1d(y.Day.values, base2.time.values)
    # base2 = base2.sel(time = ind_time)
    # y = y.sel(time = ind_time)
    # print(base2)
    # a = input()
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
