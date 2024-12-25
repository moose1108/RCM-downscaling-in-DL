#!/bin/bash

variables="q700 q850 t500 t850 u200 u850 v200 v850"
predictand="RAINNC"
topology="deepesd"
approach="MOS-E"
start_year="1981"
end_year="2020"
scale=true
year_range=$((end_year - start_year + 1))
predictor_data="/work/moose1108/corrdiff-like/data/01-predictor_ERA5/"
predictand_data="/work/moose1108/corrdiff-like/data/02-predictand_TReAD/${predictand}/"
variables_str=$(echo $variables | tr -d ' ')
modelPath="./models/${predictand}/${topology}_${variables_str}_${start_year}_${end_year}_3_m.h5"
landmask_data="/work/moose1108/corrdiff-like/data/02-predictand_TReAD/TReAD_Regrid_2km_landmask.nc"
loss_path="./plots/loss/${predictand}_${topology}_${variables_str}_${start_year}_${end_year}_3_m.png"

python train.py --variables $variables \
                --predictand $predictand \
                --start_year $start_year \
                --end_year $end_year \
                --topology $topology \
                --approach $approach \
                --modelPath $modelPath \
                --variables_str $variables_str \
                --scale $scale \
                --predictor_data $predictor_data \
                --predictand_data $predictand_data \
                --landmask_data $landmask_data \
                --loss_path $loss_path
