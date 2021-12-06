#!/bin/bash

config=''
modeldir=''
logdir=''
init=1 # 1, start from scratch - 0, start from last checkpoint

if [[ $init -eq 1 ]]
then
  python ./src/glow_tts/init.py -c $config -m $modeldir -l $logdir
fi
python ./src/glow_tts/train.py -c $config -m $modeldir -l $logdir
