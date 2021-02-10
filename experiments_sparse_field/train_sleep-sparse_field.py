import numpy as np

import torch
import torch.optim as optim

from deblending_runjingdev import simulated_datasets_lib 
from deblending_runjingdev import starnet_lib
from deblending_runjingdev import sleep_lib
from deblending_runjingdev import psf_transform_lib
from deblending_runjingdev import wake_lib

import json
import time

from deblending_runjingdev.which_device import device
print('device: ', device)

print('torch version: ', torch.__version__)

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

data_params['mean_stars'] = 50
data_params['slen'] = 500
print(data_params)

###############
# load psf
###############
bands = [2]
psfield_file = '../sdss_stage_dir/94/1/12/psField-000094-1-0012.fit'
init_psf_params = psf_transform_lib.get_psf_params(
                                    psfield_file,
                                    bands = bands)
power_law_psf = psf_transform_lib.PowerLawPSF(init_psf_params.to(device))
psf_og = power_law_psf.forward().detach()

###############
# sky intensity: for the r band
###############
init_background_params = torch.zeros(len(bands), 3).to(device)
init_background_params[0, 0] = 862.
planar_background = wake_lib.PlanarBackground(image_slen = data_params['slen'],
                            init_background_params = init_background_params.to(device))
background = planar_background.forward().detach()

###############
# draw data
###############
print('generating data: ')
n_images = 200
t0 = time.time()
star_dataset = \
    simulated_datasets_lib.load_dataset_from_params(psf_og,
                            data_params,
                            background = background,
                            n_images = n_images,
                            transpose_psf = False,
                            add_noise = True)

print('data generation time: {:.3f}secs'.format(time.time() - t0))
# get loader
batchsize = 1

loader = torch.utils.data.DataLoader(
                 dataset=star_dataset,
                 batch_size=batchsize,
                 shuffle=True)

###############
# define VAE
###############
star_encoder = starnet_lib.StarEncoder(slen = data_params['slen'],
                                       ptile_slen = 50,
                                       step = 50,
                                       edge_padding = 0,
                                       n_bands = psf_og.shape[0],
                                       max_detections = 3,
                                       track_running_stats = False)

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
n_epochs = 141
print_every = 10
print('training')

out_filename = './starnet_sparsefield'
sleep_lib.run_sleep(star_encoder,
                    loader,
                    optimizer,
                    n_epochs,
                    out_filename = out_filename,
                    print_every = print_every)

# star_dataset2 = \
#     simulated_datasets_lib.load_dataset_from_params(psf_og,
#                             data_params,
#                             background = background,
#                             n_images = 1,
#                             transpose_psf = False,
#                             add_noise = True)
# sim_image = star_dataset2[0]['image'].unsqueeze(0)
# true_locs = star_dataset2[0]['locs'][0:star_dataset[0]['n_stars']].unsqueeze(0)
# true_fluxes = star_dataset2[0]['fluxes'][0:star_dataset[0]['n_stars']].unsqueeze(0)

# np.savez('./fits/results_2020-05-10/starnet_ri_sparse_field',
#         sim_image = sim_image.cpu().numpy(),
#         true_locs = true_locs.cpu().numpy(),
#         true_fluxes = true_fluxes.cpu().numpy())

# # check loss
# loss, counter_loss, locs_loss, fluxes_loss, perm_indx = \
#     sleep_lib.get_inv_kl_loss(star_encoder, sim_image,
#                                 true_locs, true_fluxes)[0:5]

# print(loss)
# print(counter_loss.mean())
# print(locs_loss.mean())
# print(fluxes_loss.mean())
