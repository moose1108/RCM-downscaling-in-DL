# %load_ext autoreload
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import get_file
import cartopy.crs as ccrs
import matplotlib.cm as cm
from tensorflow.keras.models import Model

import tensorflow as tf
from dask.diagnostics import ProgressBar
import cmocean
from src.models import train_model, complex_conv, simple_conv, predict, simple_dense, linear_complex_model
from src.losses import gamma_loss_1d, gamma_mse_metric
from src.prepare_data import format_features, prepare_training_dataset, create_test_train_split

config = dict(y = "/work/moose1108/corrdiff-like/data/y_adjust_1981_2022.nc",
              X = "/work/moose1108/corrdiff-like/data/1981_2022.nc",
             train_start = "1981-01-01",
             train_end = "2016-12-31",
             val_start = "2017-01-01",
             val_end = "2021-12-31",
             test_start = "2022-01-01",
             test_end = "2022-12-31",
             downscale_variables = ['w850', 'u200', 'u850', 'v200', 'v850', 'tp'])

x_train, x_val, x_test, y_train, y_val, y_test = create_test_train_split(config)
x_train, x_test, x_val, y_train, y_test, y_val = prepare_training_dataset(x_train, x_val, x_test, y_train, y_val, y_test)

terrain_data = xr.open_dataset("/work/moose1108/corrdiff-like/data/TReAD_wrf_d02_info.nc")
y_data = xr.open_dataset("/work/moose1108/corrdiff-like/data/y_adjust_1981_2022.nc")
y_lat = y_data.lat
y_lon = y_data.lon
longitude = terrain_data['XLONG']
latitude = terrain_data['XLAT']
print(longitude)
max_longitude = longitude.max().values
max_latitude = latitude.max().values

interpolated_terrain = terrain_data.interp(south_north=[i for i in range(54, 227)], west_east=[i for i in range(63, 164)], method="linear")
terrain_array = interpolated_terrain['TER'].values
terrain_tensor = tf.convert_to_tensor(terrain_array, dtype=tf.float32)


gt = y_test.unstack()
gt = gt.reindex(lon = sorted(gt.lon.values))
gt_lat = gt.variables['lat'][:]
gt_lon = gt.variables['lon'][:]


initial_learning_rate =1e-3
dropout = 0.6
input_shape = x_train.shape[1:]
output_shape = y_train.z.size
hidden_layer_dense = 256
batch_size = 64
kernel_size = 5
layer_filters =[64, 256, 512]
epochs = 500
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=initial_learning_rate)

cnn_gamma = complex_conv(layer_filters=layer_filters, bn=True, padding='same', kernel_size=(kernel_size,kernel_size),
                pooling=True, dense_layers=[hidden_layer_dense], dense_activation='selu', input_shape=input_shape, terrain_features=None,
                dropout=dropout, activation='selu', output_shape = output_shape)

history, cnn_gamma = train_model(
cnn_gamma, [x_train.values, y_train['pr'].values], x_val = x_val.values, y_val = y_val['pr'].values,
                             loss = gamma_loss_1d, epochs = 150, batch_size=64,
                             optimizer = optimizer, model_weights_name = 'current_max.h5',
                            metrics =gamma_mse_metric)