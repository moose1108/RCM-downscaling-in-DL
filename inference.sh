#!/bin/bash

gcm='noresm'
rcm='ald63'
variables="q700 t500 u200 v850"
predictand='RAINNC'
topology='deepesd'
approach='MOS-E'
start_year="1991"
end_year="2010"
variables_str=$(echo $variables | tr -d ' ')
outputFileName="./predictions/${variables_str}_${start_year}_${end_year}.nc"
years="2011"
scale=True
bias_correction='False'
modelPath="./models/${predictand}/${variables_str}_${start_year}_${end_year}.h5"

python inference.py --variables $variables \
                        --predictand $predictand \
                        --topology $topology \
                        --approach $approach \
                        --outputFileName $outputFileName \
                        --years $years \
                        --scale $scale \
                        --bias_correction $bias_correction \
                        --variables_str $variables_str \
                        --modelPath $modelPath