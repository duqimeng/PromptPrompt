rm data/batch/batch*
#shuf data/xy_train  > data/xy_train_shuffled;
a=$1;
split -l $a data/xy_train data/batch/batch;
