#!/bin/bash

gcm='noresm'
rcm='ald63'
variables="q700 t500 u850 v850"
predictand="T2"
topology="deepesd"
approach="MOS-E"
start_year="1991"
end_year="2010"
scale=true
year_range=$((end_year - start_year + 1))
predictor_data="/work/moose1108/corrdiff-like/data/01-predictor_ERA5/"
predictand_data="/work/moose1108/corrdiff-like/data/02-predictand_TReAD/${predictand}/"
variables_str=$(echo $variables | tr -d ' ')
modelPath="./models/${predictand}/${variables_str}_${start_year}_${end_year}.h5"
landmask_data="/work/moose1108/corrdiff-like/data/02-predictand_TReAD/TReAD_Regrid_2km_landmask.nc"
loss_path="./plots/loss/${predictand}_${variables_str}_${start_year}_${end_year}.png"

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