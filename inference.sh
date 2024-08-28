#!/bin/bash

gcm='noresm'
rcm='ald63'
variables="q700 t500 u850 v850"
predictand='pr'
topology='deepesd'
approach='MOS-E'
outputFileName='./predictions/q700t500u850v850_20y_one.nc'
years='2011'  # This can be set as needed, e.g., '1999 2000 2001'
scale='True'
bias_correction='False'
variables_str='q700t500u850v850'

modelPath="./models/RAINNC/${variables_str}_20y.h5"

python inference.py --gcm $gcm --rcm $rcm --variables $variables --predictand $predictand --topology $topology --approach $approach --outputFileName $outputFileName --years $years --scale $scale --bias_correction $bias_correction --modelPath $modelPath