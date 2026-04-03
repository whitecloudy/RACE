#!/bin/bash

if [ $# -ge 2 ];then
  CONFIG_FILE=$2
else
  CONFIG_FILE='config/testing_config.ini'
fi

if [ $# -ge 1 ];then
  RANDOM_SEED=$1
else
  RANDOM_SEED=0
fi

if [ $# -ge 3 ];then
  W=$3
else
  W=6
fi

#for heu in {9,15,21,27}
model=$(awk '/^MODEL/{print $3}' ${CONFIG_FILE})
# aug_ratio=$(awk '/^AUG_RATIO/{print $3}' ${CONFIG_FILE})
aug_ratio1=$(awk '/^AUG_RATIO1/{print $3}' ${CONFIG_FILE})
aug_ratio2=$(awk '/^AUG_RATIO2/{print $3}' ${CONFIG_FILE})
lr=$(awk '/^LEARNING_RATE/{print $3}' ${CONFIG_FILE})
log_prefix=$(awk '/^LOG_PREFIX/{print $3}' ${CONFIG_FILE})
noise=$(awk '/^NOISE_ADDED/{print $3}' ${CONFIG_FILE})

testing_file=cache/$log_prefix\_$noise\_$model\_$lr\_$aug_ratio1\_$aug_ratio2\_$W\_$RANDOM_SEED
echo $log_filename
echo $testing_file
python3 testing.py --gpunum 1 --model $model --W $W --test $testing_file #--save-model
