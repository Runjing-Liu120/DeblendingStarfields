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
parser.add_argument("--test_image", 
                    type = str,
                    default = 'small')
parser.add_argument("--grad_estimator", 
                    type = str,
                    default = 'reinforce')

args = parser.parse_args()

print(args.seed)

np.random.seed(8910 + args.seed * 17)
_ = torch.manual_seed(8910 + args.seed * 13)
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

power_law_psf = psf_transform_lib.PowerLawPSF(init_psf_params.to(device))
psf_og = power_law_psf.forward().detach()

###############
# Get image
###############
if args.test_image == 'small': 
    # test image file
    test_image_file = './test_image_20x20.npz'
    
    # parameters for encoder
    ptile_slen = 10
    step = 10
    edge_padding = 0
    
    # prior parameters 
    mean_stars = 4

elif args.test_image == 'large': 
    # test image file
    test_image_file = './test_image_100x100.npz'
    
    # parameters for encoder
    ptile_slen = 20
    step = 10
    edge_padding = 5
    
    # prior parameters 
    mean_stars = 50
    
else: 
    print('Specify whether to use the large (100 x 100) test image', 
          'or the small (20 x 20) test image')
    raise NotImplementedError()
    
full_image_np = np.load(test_image_file)['image']
full_image = torch.Tensor(full_image_np).unsqueeze(0).to(device)
slen = full_image.shape[-1]
fmin = 1000.

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
                                       ptile_slen = ptile_slen,
                                       step = step,
                                       edge_padding = edge_padding,
                                       n_bands = psf_og.shape[0],
                                       max_detections = 2,
                                       fmin = fmin,
                                       constrain_logflux_mean = True,
                                       track_running_stats = False)

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
out_filename = './fits/starnet_elbo_' + args.grad_estimator + '-restart' + str(args.seed)
if args.test_image == 'small': 
    n_epochs = 2500
    print_every = 100
    n_samples = 2000
    out_filename = out_filename + '_20x20'
else: 
    raise NotImplementedError()
    out_filename = out_filename + '_100x100'

print('training')

elbo_results_vec = elbo_lib.save_elbo_results(full_image, star_encoder,
                                                simulator, mean_stars,
                                                n_samples = n_samples,
                                                pad = star_encoder.edge_padding)

t0 = time.time()
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad()

    # get pseudo loss
    if args.grad_estimator == 'reinforce': 
        ps_loss = elbo_lib.get_pseudo_loss(full_image, star_encoder,
                                            simulator, mean_stars,
                                            n_samples = n_samples,
                                            pad = star_encoder.edge_padding)
    elif args.grad_estimator == 'reparam': 
        ps_loss = elbo_lib.get_pseudo_loss_all_sum(full_image, star_encoder,
                                            simulator, mean_stars, n_samples)
    else: 
        print(args.grad_estimator, 'not implemented. Specify either reinforce or reparam')
        raise NotImplementedError()

    ps_loss.backward()
    optimizer.step()

    if ((epoch % print_every) == 0) or (epoch == n_epochs):
        print('epoch = {}; elapsed = {:.1f}sec'.format(epoch, time.time() - t0))

        elbo_results = elbo_lib.save_elbo_results(full_image, star_encoder,
                                                    simulator, mean_stars,
                                                    n_samples = n_samples,
                                                    pad = star_encoder.edge_padding)

        elbo_results_vec = np.vstack((elbo_results_vec, elbo_results))

        np.savetxt(out_filename + '-elbo_results', elbo_results_vec)

        print("writing the encoder parameters to " + out_filename)
        torch.save(star_encoder.state_dict(), out_filename)

        t0 = time.time()
