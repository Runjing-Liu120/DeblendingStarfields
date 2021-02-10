import numpy as np

import torch
import torch.optim as optim

import deblending_runjingdev.simulated_datasets_lib as simulated_datasets_lib
import deblending_runjingdev.starnet_lib as starnet_lib
import deblending_runjingdev.sleep_lib as sleep_lib
import deblending_runjingdev.psf_transform_lib as psf_transform_lib

import json
import time

from deblending_runjingdev.which_device import device
print('device: ', device)

print('torch version: ', torch.__version__)

###############
# set seed
###############
np.random.seed(5751)
_ = torch.manual_seed(1151)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###############
# data parameters
###############
with open('../model_params/default_star_parameters.json', 'r') as fp:
    data_params = json.load(fp)

data_params['min_stars'] = 1
# mean set so that P(n_stars <= 1) \approx 0.5
data_params['mean_stars'] = 1.65 
data_params['max_stars'] = 2
data_params['slen'] = 7
data_params['f_max'] = 10000

print(data_params)

###############
# load psf
###############
bands = [2, 3]
psfield_file = '../sdss_stage_dir/2583/2/136/psField-002583-2-0136.fit'
init_psf_params = psf_transform_lib.get_psf_params(
                                    psfield_file,
                                    bands = bands)
power_law_psf = psf_transform_lib.PowerLawPSF(init_psf_params.to(device))
psf = power_law_psf.forward().detach()

###############
# set background
###############
background = torch.zeros(len(bands), data_params['slen'], data_params['slen']).to(device)
background[0] = 686.
background[1] = 1123.

###############
# draw data
###############
print('generating data: ')
n_images = 60000
t0 = time.time()
star_dataset = \
    simulated_datasets_lib.load_dataset_from_params(psf,
                            data_params,
                            background = background,
                            n_images = n_images,
                            transpose_psf = False,
                            add_noise = True)

print('data generation time: {:.3f}secs'.format(time.time() - t0))

# get data loader
batchsize = 2000

loader = torch.utils.data.DataLoader(
                 dataset=star_dataset,
                 batch_size=batchsize,
                 shuffle=True)


###############
# define VAE
###############
star_encoder = starnet_lib.StarEncoder(slen = data_params['slen'],
                                        ptile_slen = data_params['slen'],
                                        step = data_params['slen'],
                                        edge_padding = 0,
                                        n_bands = len(bands),
                                        max_detections = 2)

star_encoder.to(device)

###############
# define optimizer
###############
learning_rate = 1e-3
weight_decay = 1e-3
optimizer = optim.Adam([
                    {'params': star_encoder.parameters(),
                    'lr': learning_rate}],
                    weight_decay = weight_decay)



###############
# Train!
###############
n_epochs = 30
print_every = 10
print('training')

out_filename = './starnet'

sleep_lib.run_sleep(star_encoder, 
                    loader,
                    optimizer,
                    n_epochs,
                    out_filename = out_filename,
                    print_every = print_every)
