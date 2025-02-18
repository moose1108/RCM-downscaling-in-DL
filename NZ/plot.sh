train_start="1981-01-01"
train_end="2016-12-31"
prediction_output="./pred/model.nc"
avg_output="./figures/current_Monthly_Average_Comparison_2013-2022.png"
days_output="./figures/selected.png"
val_start="2017-01-01"
val_end="2021-12-31"
test_start="2013-01-01"
test_end="2022-12-31"
variables="w850 u200 u850 v200 v850 tp"
x_data="/work/moose1108/corrdiff-like/data/1981_2022.nc"
y_data="/work/moose1108/corrdiff-like/data/y_adjust_1981_2022.nc"
mask_data="/work/moose1108/corrdiff-like/data/02-predictand_TReAD/TReAD_Regrid_2km_landmask.nc"
selected_days="'2022-10-15 2022-10-16 2022-10-17 2022-10-13 2022-10-30 2022-10-31 2022-07-10 2022-08-10 2022-09-01 2022-10-10 2022-11-10 2022-12-10"

python plot.py --prediction_output $prediction_output \
               --train_start $train_start \
               --train_end $train_end \
               --val_start $val_start \
               --val_end $val_end \
               --test_start $test_start \
               --test_end $test_end \
               --x_data $x_data \
               --y_data $y_data \
               --mask_data $mask_data \
               --variables $variables \
               --selected_days $selected_days \
               --avg_output $avg_output \
               --days_output $days_output