def emulate(newdata, emulator, rcm, topology, vars, outputFileName, approach, predictand = "pr", years = None, scale = True, bias_correction = False, bias_correction_base = None):

    ### Load predictor data (.nc)
    # x = xr.open_dataset(newdata)
    x_list = [xr.open_dataset(f'/home/moose1108/corrdiff-like-project/new_data/q700/2009/ERA5_PRS_q700_2009{i:02}_r1440x721_day.nc') for i in range(1, 13)]
    x = xr.concat(x_list, dim='time')
    ### Load predictor base data (.nc)
    if approach == 'PP-E':
        base_h = xr.open_dataset('/home/moose1108/corrdiff-like-project/new_data/q700/2009/ERA5_PRS_q700_200901_r1440x721_day.nc')
        base_85 = xr.open_dataset('/home/moose1108/corrdiff-like-project/new_data/q700/2009/ERA5_PRS_q700_200901_r1440x721_day.nc')
        modelEmulator = emulator
    elif approach == 'MOS-E':
        # x_list = [xr.open_dataset(f'/home/moose1108/corrdiff-like-project/new_data/q700/2009/ERA5_PRS_q700_2009{i:02}_r1440x721_day.nc') for i in range(1, 13)]
        base_h = xr.concat(x_list, dim='time')
        modelEmulator = emulator
        # modelEmulator = emulator + '-' + rcm
    
    yh = xr.open_dataset('/home/moose1108/corrdiff-like-project/TReAD_Data/RAINNC/TReAD_daily_2009_RAINNC.nc')
    # print(yh.Day)
    # a = input()
    y = yh.isel(Day=slice(0, 365))
    base = base_h.interp(latitude=y['Lat'], longitude=y['Lon'], method='nearest')
    x = x.interp(latitude=y['Lat'], longitude=y['Lon'], method='nearest')
    # base = xr.concat([base_h,base_85], dim = 'time')
    # if vars is not None:
    #     x = x[vars]
    #     base = base[vars]

    modelPath = './models/RAINNC/' + topology + '-' + predictand + '-' + modelEmulator + '-' + approach + '.h5'
    
    if years is not None:
        base = xr.merge([base.sel(time = base['time.year'] == int(year)) for year in years])
        modelPath = '../models/' + topology + '-' + predictand + '-' + modelEmulator + '-' + approach + '-year' + str(len(years)) + '.h5'

    ## Bias correction?..
    if bias_correction is True:
        print('bias correction...')
        base_h   = xr.open_dataset('../data/predictors/gcm/x_' + bias_correction_base + '_historical_1996-2005.nc')
        base_85  = xr.open_dataset('../data/predictors/gcm/x_' + bias_correction_base + '_rcp85_2090-2099.nc')
        base_gcm = xr.concat([base_h,base_85], dim = 'time')
        if vars is not None:
            base_gcm = base_gcm[vars]
        if years is not None:
            base_gcm  = xr.merge([base_gcm.sel(time = base_gcm['time.year'] == int(year)) for year in years])
        x = scaleGrid(x, base = base_gcm, ref = base, type = 'center', timeFrame = 'monthly', spatialFrame = 'gridbox')

    ## Scaling..
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


    ## Converting xarray to a numpy array and predict on the test set
    x_array = x.to_stacked_array("a", sample_dims = ["Lat", "Lon", "time"]).values
    pred = model.predict(x_array)
    # print(pred)
    # print(pred.shape)
    # a = input()
    ## Reshaping the prediction to a latitude-longitude grid
    mask = xr.open_dataset('/home/moose1108/corrdiff-like-project/TReAD_Data/TReAD_Regrid_2km_landmask.nc')
    if topology == 'deepesd':
        mask.landmask.values[mask.landmask.values == 0] = np.nan
        mask_Onedim = mask.landmask.values.reshape((np.prod(mask.landmask.shape)))
        ind = [i for i in range(len(mask_Onedim)) if mask_Onedim[i] == 1]
        pred = reshapeToMap(grid = pred, ntime = x.dims['time'], nlat = mask.dims['Lat'], nlon = mask.dims['Lon'], indLand = ind)
    if topology == 'unet':
        sea = mask.sftlf.values == 0
        pred = np.squeeze(pred)
        pred[:,sea] = np.nan

    if predictand == 'pr':
        ## Loading the reference observation for the occurrence of precipitation ---------------------------
        gcm = newdata.split("_")[1].split("-")[0]
        # yh = xr.open_dataset('../data/pr/pr_' + gcm + '-ald63_historical_1996-2005.nc') #, decode_times = False)
        # y85 = xr.open_dataset('../data/pr/pr_' + gcm + '-ald63_rcp85_2090-2099.nc') #, decode_times = False)
        # y = xr.concat([yh,y85], dim = 'time')
        y_bin = binaryGrid(y.RAINNC, condition = 'GE', threshold = 1)
        ## -------------------------------------------------------------------------------------------------
        ## Prediction on the train set -----------
        base2 = scaleGrid(base, base = base,  type = 'standardize', spatialFrame = 'gridbox')
        # ind_time = np.intersect1d(y.Day.values, base2.time.values)
        # base2 = base2.sel(time = ind_time)
        # y = y.sel(time = ind_time)
        base_array = base2.to_stacked_array("var", sample_dims = ["Lon", "Lat", "time"]).values
        pred_ocu_train = model.predict(base_array)[:,:,0]
        pred_ocu_train = reshapeToMap(grid = pred_ocu_train, ntime = base2.dims['time'], nlat = mask.dims['Lat'], nlon = mask.dims['Lon'], indLand = ind)
        pred_ocu_train = xr.DataArray(pred_ocu_train, dims = ['time','Lat','Lon'], coords = {'Lon': mask.Lon.values, 'Lat': mask.Lat.values, 'time': y.Day.values})
        ## ---------------------------------------
        ## Recovering the complete serie -----------
        pred = xr.Dataset(data_vars = {'p': (['time','lat','lon'], pred[:,:,:,0]),
                                       'log_alpha': (['time','lat','lon'], pred[:,:,:,1]),
                                       'log_beta': (['time','lat','lon'], pred[:,:,:,2])},
                                       coords = {'lon': mask.Lon.values, 'lat': mask.Lat.values, 'time': x.time.values})
        pred_bin = adjustRainfall(grid = pred.p, refPred = pred_ocu_train, refObs = y_bin)
        pred_amo = computeRainfall(log_alpha = pred.log_alpha, log_beta = pred.log_beta, bias = 1, simulate = True)
        print(pred)
        pred = pred_bin * pred_amo
        pred = pred.values
        ## -----------------------------------------

    template_predictand = xr.open_dataset('/home/moose1108/corrdiff-like-project/TReAD_Data/RAINNC/TReAD_daily_2009_RAINNC.nc')
    pred = xr.Dataset(
		data_vars = {predictand: (['time', 'Lat', 'Lon'], pred)},
	    coords = {'lon': template_predictand.Lon.values, 'lat': template_predictand.Lat.values, 'time': x.time.values},
	    attrs = description
	)
    print(outputFileName)
    pred.to_netcdf(outputFileName)
