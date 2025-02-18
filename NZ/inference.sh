train_start="1981-01-01"
train_end="2016-12-31"
val_start="2017-01-01"
val_end="2021-12-31"
test_start="2013-01-01"
test_end="2022-12-31"
variables="w850 u200 u850 v200 v850 tp"
x_data="/work/moose1108/corrdiff-like/data/1981_2022.nc"
y_data="/work/moose1108/corrdiff-like/data/y_adjust_1981_2022.nc"
model_output="./models/model.h5"
prediction_output="./pred/model.nc"
batch_size=64

python inference.py --variables $variables \
                    --train_start $train_start \
                    --train_end $train_end \
                    --val_start $val_start \
                    --val_end $val_end \
                    --test_start $test_start \
                    --test_end $test_end \
                    --x_data $x_data \
                    --y_data $y_data \
                    --model_output $model_output \
                    --prediction_output $prediction_output \
                    --batch_size $batch_size