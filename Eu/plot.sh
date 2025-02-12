#!/bin/bash

predict_year=2022
model="deepesd_q700q850t500t850u200u850v200v850tp_1981_2020_${predict_year}_3_b"
predictand="RAINNC"
predict_path="/home/moose1108/corrdiff-like-project/RCM-downscaling-in-DL/Eu/predictions/${predictand}/${model}.nc"
gt_path="/work/moose1108/corrdiff-like/data/02-predictand_TReAD/${predictand}/TReAD_daily_${predict_year}_${predictand}.nc"
plot_path="./plots/visualization/${model}_${predictand}_${predict_year}.png"
landmask_data="/work/moose1108/corrdiff-like/data/02-predictand_TReAD/TReAD_Regrid_2km_landmask.nc"

python plot.py --model $model \
               --predict_year $predict_year \
               --predict_path $predict_path \
               --gt_path $gt_path \
               --plot_path $plot_path \
               --landmask_data $landmask_data \
               --predictand $predictand