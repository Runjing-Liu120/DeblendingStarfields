import numpy as np

import torch
import torch.optim as optim

import deblending_runjingdev.sdss_dataset_lib as sdss_dataset_lib
import deblending_runjingdev.simulated_datasets_lib as simulated_datasets_lib
import deblending_runjingdev.starnet_lib as starnet_lib
import deblending_runjingdev.sleep_lib as sleep_lib
from deblending_runjingdev.sleep_lib import run_sleep
import deblending_runjingdev.wake_lib as wake_lib
import deblending_runjingdev.psf_transform_lib as psf_transform_lib

import time

import json

import os

from deblending_runjingdev.which_device import device
print('device: ', device)

print('torch version: ', torch.__version__)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--x0', type=int, default=630)
parser.add_argument('--x1', type=int, default=310)
parser.add_argument('--init_encoder', type=str, default='./fits/results_2020-05-15/starnet_ri')
parser.add_argument('--outfolder', type=str, default='./fits/results_2020-05-15/')
parser.add_argument('--outfilename', type=str, default='starnet_ri_wake-sleep')
parser.add_argument('--n_iter', type=int, default=2)

parser.add_argument('--prior_mu', type=int, default=1500)
parser.add_argument('--prior_alpha', type=float, default=0.5)

args = parser.parse_args()

assert os.path.isfile(args.init_encoder)
assert os.path.isdir(args.outfolder)

#######################
# set seed
########################
np.random.seed(32090275)
_ = torch.manual_seed(120457)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#######################
# get sdss data
#######################
sdss_image = sdss_dataset_lib.load_m2_data(sdss_dir = './../sdss_stage_dir/',
                                    hubble_dir = './hubble_data/',
                                    x0 = args.x0,
                                    x1 = args.x1)[0]

sdss_image = sdss_image.unsqueeze(0).to(device)

#######################
# simulated data parameters
#######################
with open('../model_params/default_star_parameters.json', 'r') as fp:
    data_params = json.load(fp)

data_params['alpha'] = args.prior_alpha
data_params['mean_stars'] = args.prior_mu
print(data_params)


###############
# load model parameters
###############
#### the psf
psfield_file = '../sdss_stage_dir/2583/2/136/psField-002583-2-0136.fit'
init_psf_params = psf_transform_lib.get_psf_params(
                                    psfield_file,
                                    bands = [2, 3]).to(device)

model_params = wake_lib.ModelParams(sdss_image,
                                init_psf_params = init_psf_params,
                                init_background_params = None)
psf_og = model_params.get_psf().detach()
background_og = model_params.get_background().detach().squeeze(0)

###############
# draw data
###############
print('generating data: ')
n_images = 200
t0 = time.time()
star_dataset = \
    simulated_datasets_lib.load_dataset_from_params(psf_og,
                            data_params,
                            n_images = n_images,
                            background = background_og,
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
                                           n_bands = 2,
                                           max_detections = 2)

init_encoder = args.init_encoder
star_encoder.load_state_dict(torch.load(init_encoder,
                                   map_location=lambda storage, loc: storage))
star_encoder.to(device)
star_encoder.eval();

####################
# optimzer
#####################
encoder_lr = 1e-5
sleep_optimizer = optim.Adam([
                    {'params': star_encoder.parameters(),
                    'lr': encoder_lr}],
                    weight_decay = 1e-5)

# initial loss:
sleep_loss, sleep_counter_loss, sleep_locs_loss, sleep_fluxes_loss = \
    sleep_lib.eval_sleep(star_encoder, loader, train = False)

print('**** INIT SLEEP LOSS: {:.3f}; counter loss: {:.3f}; locs loss: {:.3f}; fluxes loss: {:.3f} ****'.format(\
    sleep_loss, sleep_counter_loss, sleep_locs_loss, sleep_fluxes_loss))

wake_loss = wake_lib.get_wake_loss(sdss_image, star_encoder, model_params,
                        n_samples = 1, run_map = True).detach()
print('**** INIT WAKE LOSS: {:.3f}'.format(wake_loss))

# file header to save results
outfolder = args.outfolder # './fits/results_2020-03-04/'
outfile_base = outfolder + args.outfilename
print(outfile_base)

############################
# Run wake-sleep
############################
t0 = time.time()
n_iter = args.n_iter
map_losses = torch.zeros(n_iter)
for iteration in range(0, n_iter):
    #######################
    # wake phase training
    #######################
    print('RUNNING WAKE PHASE. ITER = ' + str(iteration))

    if iteration == 0:
        powerlaw_psf_params = init_psf_params
        planar_background_params = None
        encoder_file = init_encoder
    else:
        powerlaw_psf_params = \
            torch.Tensor(np.load(outfile_base + '-iter' + str(iteration -1) + \
                                    '-powerlaw_psf_params.npy')).to(device)
        planar_background_params = \
            torch.Tensor(np.load(outfile_base + '-iter' + str(iteration -1) + \
                                    '-planarback_params.npy')).to(device)

        encoder_file = outfile_base + '-encoder-iter' + str(iteration)

    print('loading encoder from: ', encoder_file)
    star_encoder.load_state_dict(torch.load(encoder_file,
                                   map_location=lambda storage, loc: storage))
    star_encoder.to(device);
    star_encoder.eval();

    model_params, map_losses[iteration] = \
        wake_lib.run_wake(sdss_image, star_encoder, powerlaw_psf_params,
                        planar_background_params,
                        n_samples = 25,
                        out_filename = outfile_base + '-iter' + str(iteration),
                        lr = 1e-3,
                        n_epochs = 100,
                        run_map = False,
                        print_every = 10)

    print(list(model_params.planar_background.parameters())[0])
    print(list(model_params.power_law_psf.parameters())[0])
    print(map_losses[iteration])
    np.save(outfolder + 'map_losses', map_losses.cpu().detach())

    ########################
    # sleep phase training
    ########################
    print('RUNNING SLEEP PHASE. ITER = ' + str(iteration + 1))

    # update psf and background
    loader.dataset.simulator.psf = model_params.get_psf().detach()
    loader.dataset.simulator.background = model_params.get_background().squeeze(0).detach()

    run_sleep(star_encoder,
                loader,
                sleep_optimizer,
                n_epochs = 11,
                out_filename = outfile_base + '-encoder-iter' + str(iteration + 1))


print('DONE. Elapsed: {}secs'.format(time.time() - t0))
