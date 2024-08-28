#!/bin/bash

gcm='noresm'
rcm='ald63'
variables='q700 t500'
predictand='RAINNC'
topology='deepesd'
approach='MOS-E'
start_year='2007'
end_year='2010'
variables_str='q700t500'
year_range=$((end_year - start_year + 1))
modelPath="./models/${predictand}/${variables_str}_${year_range}y.h5"

python train.py --gcm $gcm --rcm $rcm --variables $variables --predictand $predictand --start_year $start_year --end_year $end_year --topology $topology --approach $approach --modelPath $modelPath
