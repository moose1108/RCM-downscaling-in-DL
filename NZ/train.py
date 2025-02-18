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
import argparse
import tensorflow as tf
from dask.diagnostics import ProgressBar
import cmocean
from src.models import train_model, complex_conv, simple_conv, predict, simple_dense, linear_complex_model
from src.losses import gamma_loss_1d, gamma_mse_metric
from src.prepare_data import format_features, prepare_training_dataset, create_test_train_split

parser = argparse.ArgumentParser(description="Train a deep learning model for climate data.")
parser.add_argument('--variables', nargs='+', default=['q700'], help='List of variables')
parser.add_argument('--train_start', type=str, default='1981-01-01', help='starting day of training')
parser.add_argument('--train_end', type=str, default='2016-12-31', help='ending day of training')
parser.add_argument('--val_start', type=str, default='2017-01-01', help='starting day of validation')
parser.add_argument('--val_end', type=str, default='2021-12-31', help='ending day of validation')
parser.add_argument('--test_start', type=str, default='2022-01-01', help='This slice of data won\'nt be used.')
parser.add_argument('--test_end', type=str, default='2022-12-31', help='This slice of data won\'nt be used.')
parser.add_argument('--x_data', type=str, default='/work/moose1108/corrdiff-like/data/1981_2022.nc', help='ending day of validation')
parser.add_argument('--y_data', type=str, default='/work/moose1108/corrdiff-like/data/y_adjust_1981_2022.nc', help='ending day of validation')
parser.add_argument('--model_output', type=str, default='model.h5', help='path to model directory')
parser.add_argument('--terrain_data', type=str, default='/work/moose1108/corrdiff-like/data/TReAD_wrf_d02_info.nc', help='ending day of validation')
parser.add_argument('--kernel_size', type=int, default=5, help='CNN kernel_size')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--terrain_enable', type=str, help='terrain')
parser.add_argument('--initial_learning_rate', type=float, default=1e-4, help='initial_learning_rate')
args = parser.parse_args()

variables = args.variables
train_start = args.train_start
train_end = args.train_end
val_start = args.val_start
val_end = args.val_end
test_start = args.test_start
test_end = args.test_end
x_data = args.x_data
y_data = args.y_data
model_output = args.model_output
terrain_data = args.terrain_data
kernel_size = args.kernel_size
batch_size = args.batch_size
initial_learning_rate = args.initial_learning_rate
terrain_enable = args.terrain_enable

config = dict(y = y_data,
              X = x_data,
             train_start = train_start,
             train_end = train_end,
             val_start = val_start,
             val_end = val_end,
             test_start = test_start,
             test_end = test_end,
             downscale_variables = variables
)

x_train, x_val, x_test, y_train, y_val, y_test = create_test_train_split(config)
x_train, x_test, x_val, y_train, y_test, y_val = prepare_training_dataset(x_train, x_val, x_test, y_train, y_val, y_test)

terrain_data = xr.open_dataset(terrain_data)
y = xr.open_dataset(y_data)
y_lat = y.lat
y_lon = y.lon
longitude = terrain_data['XLONG']
latitude = terrain_data['XLAT']

max_longitude = longitude.max().values
max_latitude = latitude.max().values

if terrain_enable == 'T':
    interpolated_terrain = terrain_data.interp(south_north=[i for i in range(54, 227)], west_east=[i for i in range(63, 164)], method="linear")
    terrain_array = interpolated_terrain['TER'].values
    terrain_tensor = tf.convert_to_tensor(terrain_array, dtype=tf.float32)
else:
    terrain_tensor = None

gt = y_test.unstack()
gt = gt.reindex(lon = sorted(gt.lon.values))
gt_lat = gt.variables['lat'][:]
gt_lon = gt.variables['lon'][:]

dropout = 0.6
input_shape = x_train.shape[1:]
output_shape = y_train.z.size
hidden_layer_dense = 256
layer_filters =[64, 256, 512]
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=initial_learning_rate)
terrain_features=terrain_tensor if terrain_enable else None

cnn_gamma = complex_conv(layer_filters=layer_filters, bn=True, padding='same', kernel_size=(kernel_size, kernel_size),
                pooling=True, dense_layers=[hidden_layer_dense], dense_activation='selu', input_shape=input_shape, terrain_features=terrain_tensor,
                dropout=dropout, activation='selu', output_shape = output_shape)

history, cnn_gamma = train_model(
cnn_gamma, [x_train.values, y_train['pr'].values], x_val = x_val.values, y_val = y_val['pr'].values,
                             loss = gamma_loss_1d, epochs = 150, batch_size=batch_size,
                             optimizer = optimizer, model_weights_name = model_output,
                            metrics =gamma_mse_metric)