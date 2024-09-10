#!/bin/bash

predict_year=2011
model="q700t500u200v850_1991_2010"
predict_path="/home/moose1108/corrdiff-like-project/mycode/predictions/${model}.nc"
gt_path="/work/moose1108/corrdiff-like/data/02-predictand_TReAD/RAINNC/TReAD_daily_${predict_year}_RAINNC.nc"
plot_path="./plots/visualization/${model}_pred_${predict_year}.png"
landmask_data="/work/moose1108/corrdiff-like/data/02-predictand_TReAD/TReAD_Regrid_2km_landmask.nc"

python plot.py --model $model \
               --predict_year $predict_year \
               --predict_path $predict_path \
               --gt_path $gt_path \
               --plot_path $plot_path \
               --landmask_data $landmask_data