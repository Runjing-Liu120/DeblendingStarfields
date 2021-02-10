#!/bin/bash
for mu in 1000 2000
do
python train_sleep.py \
  --outfolder fits/prior_sensitivity/ \
  --outfilename starnet_mu$mu \
  --prior_mu $mu \

python train_wake_sleep.py \
  --n_iter 2 \
  --init_encoder ./fits/prior_sensitivity/starnet_mu$mu \
  --outfolder ./fits/prior_sensitivity/ \
  --outfilename starnet_mu$mu \
  --prior_mu $mu
done
