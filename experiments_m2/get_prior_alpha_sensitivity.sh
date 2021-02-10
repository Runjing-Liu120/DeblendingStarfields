#!/bin/bash
for alpha in 0.25 0.75
do
python train_sleep.py \
  --outfolder fits/prior_sensitivity/ \
  --outfilename starnet_alpha$alpha \
  --prior_alpha $alpha \

python train_wake_sleep.py \
  --n_iter 2 \
  --init_encoder ./fits/prior_sensitivity/starnet_alpha$alpha \
  --outfolder ./fits/prior_sensitivity/ \
  --outfilename starnet_alpha$alpha \
  --prior_alpha $alpha
done
