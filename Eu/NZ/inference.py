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
             test_start = "2013-01-01",
             test_end = "2022-12-31",
             downscale_variables = ['w850', 'u200', 'u850', 'v200', 'v850', 'tp'])

x_train, x_val, x_test, y_train, y_val, y_test = create_test_train_split(config)
x_train, x_test, x_val, y_train, y_test, y_val = prepare_training_dataset(x_train, x_val, x_test, y_train, y_val, y_test)

cnn_gamma = tf.keras.models.load_model('current_max.h5', custom_objects = {'gamma_loss_1d': gamma_loss_1d, 'gamma_mse_metric': gamma_mse_metric})

gamma_prediction = predict(cnn_gamma, x_test, y_test, batch_size=32, key ='pr', pred_name = "RAINNC", loss = 'gamma', thres = 0.5)
gamma_prediction = gamma_prediction.unstack()
gamma_prediction = gamma_prediction.reindex(lon = sorted(gamma_prediction.lon.values))
print(gamma_prediction)
gamma_prediction.to_netcdf('current_max.nc')