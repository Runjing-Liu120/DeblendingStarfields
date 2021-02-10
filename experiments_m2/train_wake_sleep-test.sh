#!/bin/bash

outfolder='./fits/'
encoder_name='starnet_m2test'

python train_wake_sleep.py \
  --n_iter 2 \
  --init_encoder ${outfolder}starnet \
  --outfolder ${outfolder}  \
  --outfilename ${encoder_name} \
  --x0 630 \
  --x1 210

