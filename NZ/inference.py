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

parser = argparse.ArgumentParser(description="Plot downsclaing results.")
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
parser.add_argument('--prediction_output', type=str, default='model.nc', help='path to prediction directory')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
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
batch_size = args.batch_size
prediction_output = args.prediction_output

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

cnn_gamma = tf.keras.models.load_model(model_output, custom_objects = {'gamma_loss_1d': gamma_loss_1d, 'gamma_mse_metric': gamma_mse_metric})

gamma_prediction = predict(cnn_gamma, x_test, y_test, batch_size=batch_size, key ='pr', pred_name = "RAINNC", loss = 'gamma', thres = 0.5)
gamma_prediction = gamma_prediction.unstack()
gamma_prediction = gamma_prediction.reindex(lon = sorted(gamma_prediction.lon.values))
print(gamma_prediction)
gamma_prediction.to_netcdf(prediction_output)