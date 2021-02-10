import numpy as np
import torch

import json

import matplotlib.pyplot as plt

import deblending_runjingdev.simulated_datasets_lib as simulated_datasets_lib
import deblending_runjingdev.psf_transform_lib as psf_transform_lib

from deblending_runjingdev.which_device import device

np.random.seed(65765)
_ = torch.manual_seed(3453453)

# get the SDSS point spread function
bands = [2, 3]
psfield_file = './../sdss_stage_dir/2583/2/136/psField-002583-2-0136.fit'
init_psf_params = psf_transform_lib.get_psf_params(
                                    psfield_file,
                                    bands = bands)

power_law_psf = psf_transform_lib.PowerLawPSF(init_psf_params.to(device))
psf_og = power_law_psf.forward().detach().to(device)

##############################
# Simulate the 20 x 20 image
##############################
slen = 20

# get background
background = torch.zeros(len(bands), slen, slen)
background[0] = 686.
background[1] = 1123.
background = background.to(device)

# the simulator 
simulator = simulated_datasets_lib.StarSimulator(psf_og, slen, background, transpose_psf = False)

# set locations and fluxes 
true_locs = torch.Tensor([[2, 3], 
                         [5.5, 7.5], 
                         [12.5, 6.5], 
                         [8.5, 14.5]]).unsqueeze(0) / slen
true_locs = true_locs.to(device)

true_fluxes = torch.zeros(true_locs.shape[0], true_locs.shape[1], len(bands), 
                          device = device) + 4000.

# simulate image
full_image = simulator.draw_image_from_params(locs = true_locs, 
                                 fluxes = true_fluxes, 
                                 n_stars= torch.Tensor([4]).to(device).long(),
                                add_noise = True)

# save 
fname = './test_image_20x20.npz'
print('saving 20 x 20 test image into: ', fname)
np.savez(fname, 
        image = full_image.cpu().squeeze(0), 
        locs = true_locs.cpu().squeeze(0), 
        fluxes = true_fluxes.cpu().squeeze(0))

##############################
# Simulate 100 x 100 image 
##############################
np.random.seed(652)
_ = torch.manual_seed(3143)

# data parameters
with open('./../model_params/default_star_parameters.json', 'r') as fp:
    data_params = json.load(fp)
    
data_params['min_stars'] = 50
data_params['max_stars'] = 50
data_params['mean_stars'] = 50
data_params['slen'] = 110

# background
background = torch.zeros(len(bands), data_params['slen'], data_params['slen'])
background[0] = 686.
background[1] = 1123.
background = background.to(device)

# simulate image
n_images = 1
star_dataset = \
    simulated_datasets_lib.load_dataset_from_params(psf_og,
                            data_params,
                            background = background,
                            n_images = n_images,
                            transpose_psf = False,
                            add_noise = True)

fname = './test_image_100x100.npz'
print('saving 100 x 100 test image into: ', fname)

full_image = star_dataset[0]['image'].unsqueeze(0)
true_locs = star_dataset[0]['locs']
true_fluxes = star_dataset[0]['fluxes']

np.savez(fname, 
        image = full_image.cpu().squeeze(0), 
        locs = true_locs.cpu().squeeze(0), 
        fluxes = true_fluxes.cpu().squeeze(0))

