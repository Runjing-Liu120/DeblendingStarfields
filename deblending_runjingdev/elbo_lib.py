import torch
import numpy as np
import time

from torch import nn

import deblending_runjingdev.starnet_lib as starnet_lib
from deblending_runjingdev.which_device import device

def get_neg_elbo(simulator, full_image, locs, fluxes, n_stars, \
                    log_q_locs, log_q_fluxes, log_q_n_stars,
                    mean_stars,
                    pad = 0,
                    clamp = None,
                    uniform_nstars = False):

    # get reconstruction
    recon = \
        simulator.draw_image_from_params(locs, fluxes, n_stars, add_noise = False)

    # option to mask outliers
    if clamp is not None:
        mask_bool = (((full_image - recon) / recon).abs() < clamp).detach().float()
    else:
        mask_bool = 1

    # get log likelihood-
    loglik = (- 0.5 * (full_image - recon)**2 / recon - 0.5 * torch.log(recon)) * mask_bool
    padm = full_image.shape[-1] - pad
    loglik = loglik[:, :, pad:padm, pad:padm]
    loglik = loglik.sum(-1).sum(-1).sum(-1)

    # get entropy terms
    entropy = - log_q_locs - log_q_fluxes - log_q_n_stars

    # TODO: need to pass in prior parameters
    alpha = 0.5

    if uniform_nstars:
        log_prior_nstars = 0.0
    else:
        log_prior_nstars = n_stars * np.log(mean_stars) - torch.lgamma(n_stars.float() + 1)

    is_on_fluxes = (fluxes[:, :, 0] > 0.).detach().float()
    log_prior_fluxes = (- (alpha + 1) * torch.log(fluxes[:, :, 0] + 1e-16) * \
                                        is_on_fluxes).sum(-1)

    if fluxes.shape[-1] > 1:
        # TODO assumes two bands
        color = -2.5 * (torch.log10(fluxes[:, :, 0] + 1e-16) - \
                            torch.log10(fluxes[:, :, 1] + 1e-16)) * is_on_fluxes

        log_prior_color = (- 0.5 * color**2).sum(-1)
    else:
        log_prior_color = 0.0

    log_prior = log_prior_nstars + log_prior_fluxes + log_prior_color

    return -(loglik + entropy + log_prior), -loglik, -log_prior, recon

def eval_star_encoder_on_elbo(full_image, star_encoder, simulator,
                                n_samples,
                                mean_stars,
                                return_map = False,
                                training = True,
                                clamp = None,
                                pad = 0):

    # sample
    locs_sampled, fluxes_sampled, n_stars_sampled, \
        log_q_locs, log_q_fluxes, log_q_n_stars = \
            star_encoder.sample_star_encoder(full_image,
                                    n_samples = n_samples,
                                    training = training,
                                    return_map_n_stars = return_map,
                                    return_map_star_params = return_map,
                                    return_log_q = True)

    # get elbo
    neg_elbo, neg_loglik, neg_logprior,  recon = \
        get_neg_elbo(simulator, full_image,
                            locs_sampled, fluxes_sampled, n_stars_sampled.detach(),
                            log_q_locs, log_q_fluxes, log_q_n_stars, mean_stars,
                            clamp = clamp,
                            pad = pad)

    return neg_elbo, neg_loglik, recon, log_q_n_stars

def save_elbo_results(full_image, star_encoder, simulator, mean_stars,
                            n_samples = 100, pad = 0):

    neg_elbo, neg_loglik, _, _ = \
        eval_star_encoder_on_elbo(full_image, star_encoder,
                                    simulator,
                                    n_samples = n_samples,
                                    mean_stars = mean_stars,
                                    training = False,
                                    pad = pad)

    map_neg_elbo, map_neg_loglik, _, _ = \
        eval_star_encoder_on_elbo(full_image, star_encoder,
                                    simulator,
                                    n_samples = 1,
                                    mean_stars = mean_stars,
                                    return_map = True,
                                    training = False,
                                    pad = pad)


    print('neg elbo: {:.3e}; neg log-likelihood: {:.3e}'.format(neg_elbo.mean(), neg_loglik.mean()))
    print('neg elbo (map): {:.3e}; neg log-likelihood (map): {:.3e}'.format(map_neg_elbo.mean(),
                                                                                    map_neg_loglik.mean()))

    return np.array([neg_elbo.detach().mean().cpu().numpy(),
                    neg_loglik.detach().mean().cpu().numpy(),
                    map_neg_elbo.detach().mean().cpu().numpy(),
                    map_neg_loglik.detach().mean().cpu().numpy(),
                    time.time()])

def get_pseudo_loss(full_image, star_encoder, simulator, mean_stars, n_samples,
                    pad = 0):
    # get elbo
    neg_elbo, loglik, _, log_q_n_stars = \
        eval_star_encoder_on_elbo(full_image, star_encoder, simulator,
                                            n_samples,
                                             mean_stars,
                                            training = True,
                                            pad = pad)

    # get control variate
    cv, loglik, _, _ = \
        eval_star_encoder_on_elbo(full_image, star_encoder, simulator,
                                            n_samples,
                                             mean_stars,
                                            training = False,
                                            pad = pad)

    # get pseudo-loss
    ps_loss = ((neg_elbo.detach() - cv.detach()) * log_q_n_stars + \
                    neg_elbo).mean()

    return ps_loss

def get_pseudo_loss_all_sum(full_image, star_encoder, simulator,
                                mean_stars, n_samples,
                                clamp = None,
                                pad = 0):

    locs_sampled, fluxes_sampled, n_stars_sampled, \
        log_q_locs, log_q_fluxes, log_q_n_stars = \
            star_encoder.sample_star_encoder(full_image,
                                    return_map_n_stars = False,
                                    return_map_star_params = False,
                                    n_samples = n_samples,
                                    return_log_q = True,
                                    training = True,
                                    enumerate_all_n_stars = True)

    # get elbo
    neg_elbo, neg_loglik, neg_logprior, recon = \
        get_neg_elbo(simulator, full_image,
                            locs_sampled, fluxes_sampled, n_stars_sampled.detach(),
                            log_q_locs, log_q_fluxes, log_q_n_stars, mean_stars,
                            clamp = clamp,
                            pad = pad)

    return (neg_elbo * log_q_n_stars.exp()).sum() / n_samples

def loss_on_true_nstars(full_image, star_encoder, simulator,
                            mean_stars, n_samples,
                            true_locs, true_fluxes,
                            clamp = None,
                            pad = 0):

    image_ptiles, tile_locs, tile_fluxes, \
        tile_n_stars, tile_is_on_array = \
            star_encoder.get_image_ptiles(full_image,
                                            true_locs.unsqueeze(0),
                                            true_fluxes.unsqueeze(0))

    locs_sampled, fluxes_sampled, n_stars_sampled, \
        log_q_locs, log_q_fluxes, log_q_n_stars = \
            star_encoder.sample_star_encoder(full_image,
                                    return_map_star_params = False,
                                    n_samples = n_samples,
                                    return_log_q = True,
                                    training = True,
                                    tile_n_stars = tile_n_stars);

    # get elbo
    neg_elbo, neg_loglik, neg_logprior, recon = \
        get_neg_elbo(simulator, full_image,
                            locs_sampled, fluxes_sampled, n_stars_sampled.detach(),
                            log_q_locs, log_q_fluxes, log_q_n_stars, mean_stars,
                            clamp = clamp,
                            pad = pad)

    return neg_elbo.mean()
