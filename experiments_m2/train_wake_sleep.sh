#!/bin/bash

outfolder='./fits/'
encoder_name='starnet'

python train_sleep.py \
  --outfolder $outfolder \
  --outfilename $encoder_name

python train_wake_sleep.py \
  --n_iter 2 \
  --init_encoder ${outfolder}${encoder_name} \
  --outfolder ${outfolder}  \
  --outfilename ${encoder_name}

