#!/bin/bash

if [ $# -ge 3 ];then
  CONFIG_FILE=$3
else
  CONFIG_FILE='config/training_config.ini'
fi

#for heu in {9,15,21,27}

epochs=$(awk '/^EPOCHS/{print $3}' ${CONFIG_FILE})
patience=$(awk '/^PATIENCE/{print $3}' ${CONFIG_FILE})
model=$(awk '/^MODEL/{print $3}' ${CONFIG_FILE})
aug_ratio1=$(awk '/^AUG_RATIO1/{print $3}' ${CONFIG_FILE})
aug_ratio2=$(awk '/^AUG_RATIO2/{print $3}' ${CONFIG_FILE})
batch=$(awk '/^BATCH/{print $3}' ${CONFIG_FILE})
lr=$(awk '/^LEARNING_RATE/{print $3}' ${CONFIG_FILE})
noise=$(awk '/^NOISE_ADDED/{print $3}' ${CONFIG_FILE})
log_prefix=$(awk '/^LOG_PREFIX/{print $3}' ${CONFIG_FILE})
dataset_postfix=$(awk '/^DATA_POSTFIX/{print $3}' ${CONFIG_FILE})

gpunum=$1


for W in {6,8,10,12,14,16,18}
#for W in {8,10,12,14,16,18}
#for W in {14,16,18}
#for W in {6,18}
do
  for i in {0..4}
  do
#      if [ $# -ge 6 ];then
#        W_start=$5
#        V_start=$6
#        if [ $W -lt $W_start ];then
#          echo $W\_$i pass
#          continue
#        elif [ $W -eq $W_start -a $i -lt $V_start ];then
#          echo $W\_$i pass
#          continue
#        fi
#      fi
#
    if [ $# -ge 2 ];then
      V_specific=$2
      if [ $i -ne $V_specific ];then
        echo $W\_$i pass
        continue
      fi
    fi
    log_filename=$log_prefix\_$noise\_$model\_$lr\_$aug_ratio1\_$aug_ratio2\_$W\_$i
    echo $log_filename
    export CUDA_CACHE_DISABLE=1
    export LD_PRELOAD=/usr/local/lib/libjemalloc.so
    export LRU_CACHE_CAPACITY=1
    python3 training.py --gpunum $gpunum --seed $i --batch-size $batch --lr $lr --dataset-postfix $dataset_postfix --epochs $epochs --patience $patience --W $W --log $log_filename --model $model --aug-ratio1 $aug_ratio1 --aug-ratio2 $aug_ratio2 --test-batch-size 1024 --save-model > play_log/$log_filename.txt 2>&1 #--dry-run
  done
done

