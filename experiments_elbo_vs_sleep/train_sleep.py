import numpy as np

import torch
import torch.optim as optim

import deblending_runjingdev.simulated_datasets_lib as simulated_datasets_lib
import deblending_runjingdev.starnet_lib as starnet_lib
import deblending_runjingdev.sleep_lib as sleep_lib
import deblending_runjingdev.psf_transform_lib as psf_transform_lib
import deblending_runjingdev.elbo_lib as elbo_lib

import json
import time

from deblending_runjingdev.which_device import device
print('device: ', device)

print('torch version: ', torch.__version__)

###############
# set seed
###############
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type = int, default = 0)
parser.add_argument("--test_image", 
                    type = str,
                    default = 'small')
args = parser.parse_args()

print(args.seed)

np.random.seed(5751 + args.seed * 17)
_ = torch.manual_seed(11512 + args.seed * 13)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

####################
# Get test image
#####################
if args.test_image == 'small': 
    # test image file
    test_image_file = './test_image_20x20.npz'
    
    # parameters for encoder
    ptile_slen = 10
    step = 10
    edge_padding = 0
    
    # prior parameters 
    mean_stars = 4
    max_stars = 6

elif args.test_image == 'large': 
    # test image file
    test_image_file = './test_image_100x100.npz'
    
    # parameters for encoder
    ptile_slen = 20
    step = 10
    edge_padding = 5
    
    # prior parameters 
    mean_stars = 50
    max_stars = 100
    
else: 
    print('Specify whether to use the large (100 x 100) test image', 
          'or the small (30 x 30) test image')
    raise NotImplementedError()

full_image_np = np.load(test_image_file)['image']
full_image = torch.Tensor(full_image_np).unsqueeze(0).to(device)

###############
# data parameters
###############
with open('../model_params/default_star_parameters.json', 'r') as fp:
    data_params = json.load(fp)

data_params['mean_stars'] = mean_stars
data_params['min_stars'] = 0
data_params['max_stars'] = max_stars
data_params['slen'] = full_image.shape[-1]

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
psf_og = power_law_psf.forward().detach()

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
n_images = 20000
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
batchsize = 64

loader = torch.utils.data.DataLoader(
                 dataset=star_dataset,
                 batch_size=batchsize,
                 shuffle=True)


###############
# define VAE
###############
star_encoder = starnet_lib.StarEncoder(slen = data_params['slen'],
                                        ptile_slen = ptile_slen,
                                        step = step,
                                        edge_padding = edge_padding,
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
out_filename = './fits/starnet_klpq-restart' + str(args.seed)
if args.test_image == 'small': 
    n_epochs = 31
    print_every = 1
    out_filename = out_filename + '_20x20'
else: 
    n_epochs = 500
    print_every = 10
    out_filename = out_filename + '_100x100'
    
print('training')

sleep_lib.run_sleep(star_encoder, loader, optimizer, n_epochs,
                        out_filename = out_filename,
                        print_every = print_every,
                        full_image = full_image,
                        mean_stars = data_params['mean_stars'])
