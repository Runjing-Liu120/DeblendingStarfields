import numpy as np

import torch
import torch.optim as optim

import deblending_runjingdev.simulated_datasets_lib as simulated_datasets_lib
import deblending_runjingdev.sdss_dataset_lib as sdss_dataset_lib
import deblending_runjingdev.starnet_lib as starnet_lib
import deblending_runjingdev.sleep_lib as sleep_lib
import deblending_runjingdev.psf_transform_lib as psf_transform_lib
import deblending_runjingdev.wake_lib as wake_lib

import json
import time

from deblending_runjingdev.which_device import device
print('device: ', device)

print('torch version: ', torch.__version__)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--outfolder', type=str, default='./fits/results_2020-05-15/')
parser.add_argument('--outfilename', type=str, default='starnet_ri')
parser.add_argument('--prior_mu', type=int, default=1500)
parser.add_argument('--prior_alpha', type=float, default=0.5)

args = parser.parse_args()

import os
assert os.path.isdir(args.outfolder)

###############
# set seed
###############
np.random.seed(65765)
_ = torch.manual_seed(3453453)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###############
# data parameters
###############
with open('../model_params/default_star_parameters.json', 'r') as fp:
    data_params = json.load(fp)

data_params['mean_stars'] = args.prior_mu
data_params['alpha'] = args.prior_alpha

print(data_params)

###############
# load psf and background
###############
psfield_file = '../sdss_stage_dir/2583/2/136/psField-002583-2-0136.fit'
init_psf_params = psf_transform_lib.get_psf_params(
                                    psfield_file,
                                    bands = [2, 3])
# init_psf_params = torch.Tensor(np.load('./data/fitted_powerlaw_psf_params.npy'))
power_law_psf = psf_transform_lib.PowerLawPSF(init_psf_params.to(device))
psf_og = power_law_psf.forward().detach()

# load background
sdss_background = \
    sdss_dataset_lib.load_m2_data(sdss_dir = './../sdss_stage_dir/',
                                    hubble_dir = './hubble_data/')[1]

sdss_background = sdss_background.to(device)


###############
# draw data
###############
print('generating data: ')
n_images = 200
t0 = time.time()
star_dataset = \
    simulated_datasets_lib.load_dataset_from_params(psf_og,
                            data_params,
                            background = sdss_background,
                            n_images = n_images,
                            transpose_psf = False,
                            add_noise = True)

print('data generation time: {:.3f}secs'.format(time.time() - t0))
# get loader
batchsize = 20


loader = torch.utils.data.DataLoader(
                 dataset=star_dataset,
                 batch_size=batchsize,
                 shuffle=True)

###############
# define VAE
###############
star_encoder = starnet_lib.StarEncoder(slen = data_params['slen'],
                                           ptile_slen = 8,
                                           step = 2,
                                           edge_padding = 3,
                                           n_bands = psf_og.shape[0],
                                           max_detections = 2)

star_encoder.to(device)

###############
# define optimizer
###############
learning_rate = 1e-3
weight_decay = 1e-5
optimizer = optim.Adam([
                    {'params': star_encoder.parameters(),
                    'lr': learning_rate}],
                    weight_decay = weight_decay)



###############
# Train!
###############
n_epochs = 201
print_every = 5
print('training')

t0 = time.time()
out_filename = args.outfolder + args.outfilename # './fits/results_2020-05-15/starnet_ri'
sleep_lib.run_sleep(star_encoder, loader, optimizer, n_epochs,
                        out_filename = out_filename,
                        print_every = print_every,
                        full_image = None)

print('DONE. Elapsed: {}secs'.format(time.time() - t0))
