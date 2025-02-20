#!/bin/bash

variables="q700 q850 t500 t850 u200 u850 v200 v850 tp"
predictand='T2'
topology='deepesd'
approach='MOS-E'
start_year="1981"
end_year="2020"
variables_str=$(echo $variables | tr -d ' ')
years="2022"
outputFileName="./predictions/${predictand}/${topology}_${variables_str}_${start_year}_${end_year}_${years}_3_b.nc"
scale=True
bias_correction='False'
modelPath="./models/${predictand}/${topology}_${variables_str}_${start_year}_${end_year}_3_b.h5"
predictand_base="/work/moose1108/corrdiff-like/data/02-predictand_TReAD/"
predictor_base="/work/moose1108/corrdiff-like/data/01-predictor_ERA5/"
x_data="/work/moose1108/corrdiff-like/data/1981_2022.nc"
landmask_data="/work/moose1108/corrdiff-like/data/02-predictand_TReAD/TReAD_Regrid_2km_landmask.nc"
template_predictand="/work/moose1108/corrdiff-like/data/02-predictand_TReAD/RAINNC/TReAD_daily_2009_RAINNC.nc"

python inference.py --variables $variables \
                        --predictand $predictand \
                        --topology $topology \
                        --approach $approach \
                        --start_year $start_year \
                        --end_year $end_year \
                        --outputFileName $outputFileName \
                        --years $years \
                        --scale $scale \
                        --bias_correction $bias_correction \
                        --variables_str $variables_str \
                        --modelPath $modelPath \
                        --predictand_base $predictand_base \
                        --predictor_base $predictor_base \
                        --x_data $x_data \
                        --landmask_data $landmask_data \
                        --template_predictand $template_predictand