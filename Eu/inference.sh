#!/bin/bash

variables="q700 q850 t500 t850 u200 u850 v200 v850"
predictand='RAINNC'
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
                        --modelPath $modelPath
