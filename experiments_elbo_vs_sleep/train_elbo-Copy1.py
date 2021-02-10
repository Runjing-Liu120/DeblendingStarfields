import numpy as np

import torch
import torch.optim as optim

import deblending_runjingdev.elbo_lib as elbo_lib
import deblending_runjingdev.simulated_datasets_lib as simulated_datasets_lib
import deblending_runjingdev.starnet_lib as starnet_lib
import deblending_runjingdev.psf_transform_lib as psf_transform_lib

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
args = parser.parse_args()

print(args.seed)

np.random.seed(575 + args.seed * 17)
_ = torch.manual_seed(1512 + args.seed * 13)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###############
# load psf
###############
bands = [2, 3]
psfield_file = '../sdss_stage_dir/2583/2/136/psField-002583-2-0136.fit'
init_psf_params = psf_transform_lib.get_psf_params(
                                    psfield_file,
                                    bands = bands)
# init_psf_params = torch.Tensor(np.load('./data/fitted_powerlaw_psf_params.npy'))
power_law_psf = psf_transform_lib.PowerLawPSF(init_psf_params.to(device))
psf_og = power_law_psf.forward().detach()

###############
# Get image
###############
test_image_file = './test_image_20x20.npz'
full_image_np = np.load(test_image_file)['image']
full_image = torch.Tensor(full_image_np).unsqueeze(0).to(device)
slen = full_image.shape[-1]
fmin = 1000.
mean_stars = 4

# load true locations and fluxes
true_locs = torch.Tensor(np.load(test_image_file)['locs']).to(device)
true_fluxes = torch.Tensor(np.load(test_image_file)['fluxes']).to(device)

###############
# background
###############
background = torch.zeros(len(bands), slen, slen).to(device)
background[0] = 686.
background[1] = 1123.

###############
# Get simulator
###############
simulator = simulated_datasets_lib.StarSimulator(psf_og,
                                                slen,
                                                background,
                                                transpose_psf = False)

###############
# define VAE
###############
star_encoder = starnet_lib.StarEncoder(slen = slen,
                                       ptile_slen = 10,
                                       step = 10,
                                       edge_padding = 0,
                                       n_bands = psf_og.shape[0],
                                       max_detections = 2,
                                       fmin = fmin,
                                       constrain_logflux_mean = True,
                                       track_running_stats = False)
# star_encoder = elbo_lib.MFVBEncoder(slen = slen,
#                                     patch_slen = 10,
#                                     step = 10,
#                                     edge_padding = 0,
#                                     n_bands = psf_og.shape[0],
#                                     max_detections = 2,
#                                     fmin = 1000.)
#
# star_encoder.load_state_dict(torch.load('./fits/results_2020-04-29/starnet_klpq',
#                                map_location=lambda storage, loc: storage))

star_encoder.eval();
star_encoder.to(device);

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
out_filename = './foo' # './fits/results_2020-05-06/starnet_encoder_allsum-restart' + str(args.seed)

n_epochs = 2500
print_every = 100
n_samples = 2000
print('training')

elbo_results_vec = elbo_lib.save_elbo_results(full_image, star_encoder,
                                                simulator, mean_stars, n_samples)

t0 = time.time()
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad()

    # get pseudo loss
    # ps_loss = elbo_lib.get_pseudo_loss(full_image, star_encoder,
    #                                     simulator,mean_stars, n_samples)
    ps_loss = elbo_lib.get_pseudo_loss_all_sum(full_image, star_encoder,
                                        simulator, mean_stars, n_samples)
    # ps_loss = elbo_lib.loss_on_true_nstars(full_image, star_encoder, simulator,
    #                                 mean_stars, n_samples,
    #                                 true_locs, true_fluxes)
    ps_loss.backward()
    optimizer.step()

    if ((epoch % print_every) == 0) or (epoch == n_epochs):
        print('epoch = {}; elapsed = {:.1f}sec'.format(epoch, time.time() - t0))

        elbo_results = elbo_lib.save_elbo_results(full_image, star_encoder,
                                                    simulator, mean_stars, n_samples)

        elbo_results_vec = np.vstack((elbo_results_vec, elbo_results))

        np.savetxt(out_filename + '-elbo_results', elbo_results_vec)

        print("writing the encoder parameters to " + out_filename)
        torch.save(star_encoder.state_dict(), out_filename)
        # torch.save(star_encoder.params, out_filename)

        t0 = time.time()
