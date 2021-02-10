#!/bin/bash

# run 6 random restarts optimizing 
# the ELBO with REINFORCE
# for i in {1..6}
# do
# python train_elbo.py --seed $i
# done

# run 6 random restarts optimizing 
# the ELBO with the reparameterization trick
# for i in {1..6}
# do
# python train_elbo.py --seed $i --grad_estimator reparam
# done

# run 6 random restarts of the sleep phase
for i in {1..6}
do
python train_sleep.py --seed $i
done
