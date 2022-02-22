#!/bin/bash

gender='male'

config='../../config/hifi/config_v1.json'
modeldir='../../checkpoints/hifi/'$gender
logdir='../../logs/hifi/'$gender


####################################################



python ../src/hifi_gan/train.py \
    --config $config \
    --input_training_file '../../data/hifi/'$gender'/train.txt' \
    --input_validation_file '../../data/hifi/'$gender'/valid.txt' \
    --checkpoint_path $modeldir \
    --logs_path $logdir \
    --checkpoint_interval 10000 \
    --stdout_interval 50
