import torch
from torch import nn

import deblending_runjingdev.utils as utils
from deblending_runjingdev.which_device import device



# This function copied from
# https://gist.github.com/dem123456789/23f18fd78ac8da9615c347905e64fc78
def _extract_patches_2d(img,patch_shape,step=[1.0,1.0],batch_first=False):
    patch_H, patch_W = patch_shape[0], patch_shape[1]
    if(img.size(2)<patch_H):
        num_padded_H_Top = (patch_H - img.size(2))//2
        num_padded_H_Bottom = patch_H - img.size(2) - num_padded_H_Top
        padding_H = nn.ConstantPad2d((0,0,num_padded_H_Top,num_padded_H_Bottom),0)
        img = padding_H(img)
    if(img.size(3)<patch_W):
        num_padded_W_Left = (patch_W - img.size(3))//2
        num_padded_W_Right = patch_W - img.size(3) - num_padded_W_Left
        padding_W = nn.ConstantPad2d((num_padded_W_Left,num_padded_W_Right,0,0),0)
        img = padding_W(img)
    step_int = [0,0]
    step_int[0] = int(patch_H*step[0]) if(isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W*step[1]) if(isinstance(step[1], float)) else step[1]
    patches_fold_H = img.unfold(2, patch_H, step_int[0])
    if((img.size(2) - patch_H) % step_int[0] != 0):
        patches_fold_H = torch.cat((patches_fold_H,img[:,:,-patch_H:,].permute(0,1,3,2).unsqueeze(2)),dim=2)
    patches_fold_HW = patches_fold_H.unfold(3, patch_W, step_int[1])
    if((img.size(3) - patch_W) % step_int[1] != 0):
        patches_fold_HW = torch.cat((patches_fold_HW,patches_fold_H[:,:,:,-patch_W:,:].permute(0,1,2,4,3).unsqueeze(3)),dim=3)
    patches = patches_fold_HW.permute(2,3,0,1,4,5)
    patches = patches.reshape(-1,img.size(0),img.size(1),patch_H,patch_W)
    if(batch_first):
        patches = patches.permute(1,0,2,3,4)
    return patches


def tile_images(images, subimage_slen, step):
    # images should be batchsize x n_bands x slen x slen
    # breaks up a large image into smaller tiles
    # of size subimage_slen x subimage_slen

    # the output tensor is (batchsize * tiles per image) x n_bands x subimage_slen x subimage_slen
    # where tiles per image  is (slen - subimage_sel / step)**2

    # NOTE: input and output are torch tensors, not numpy arrays
    #       (need the unfold command from torch)

    assert len(images.shape) == 4

    image_xlen = images.shape[2]
    image_ylen = images.shape[3]

    # my tile coords doens't work otherwise ...
    assert (image_xlen - subimage_slen) % step == 0
    assert (image_ylen - subimage_slen) % step == 0

    n_bands = images.shape[1]
    for b in range(n_bands):
        image_tiles_b = _extract_patches_2d(images[:, b:(b+1), :, :],
                                            patch_shape = [subimage_slen, subimage_slen],
                                            step = [step, step],
                                            batch_first = True).reshape(-1, 1, subimage_slen, subimage_slen)

        if b == 0:
            image_tiles = image_tiles_b
        else:
            image_tiles = torch.cat((image_tiles, image_tiles_b), dim = 1)

    return image_tiles

def get_tile_coords(image_xlen, image_ylen, subimage_slen, step):
    # this function is used in conjuction with tile_images above.
    # this records (x0, x1) indices each image image tile comes from

    nx_tiles = ((image_xlen - subimage_slen) // step) + 1
    ny_tiles = ((image_ylen - subimage_slen) // step) + 1
    n_tiles = nx_tiles * ny_tiles

    return_coords = lambda i : [(i // ny_tiles) * step,
                                (i % ny_tiles) * step]

    tile_coords = torch.LongTensor([return_coords(i) \
                                    for i in range(n_tiles)]).to(device)

    return tile_coords


def get_params_in_tiles(tile_coords, locs, fluxes, slen, subimage_slen,
                            edge_padding = 0):

    # locs are the coordinates in the full image, in coordinates between 0-1
    assert torch.all(locs <= 1.)
    assert torch.all(locs >= 0.)

    n_tiles = tile_coords.shape[0] # number of tiles in a full image
    fullimage_batchsize = locs.shape[0] # number of full images

    subimage_batchsize = n_tiles * fullimage_batchsize # total number of tiles

    max_stars = locs.shape[1]

    tile_coords = tile_coords.unsqueeze(0).unsqueeze(2).float()
    locs = locs * (slen - 1)
    which_locs_array = (locs.unsqueeze(1) > tile_coords + edge_padding - 0.5) & \
                        (locs.unsqueeze(1) < tile_coords - 0.5 + subimage_slen - edge_padding) & \
                        (locs.unsqueeze(1) != 0)
    which_locs_array = (which_locs_array[:, :, :, 0] * which_locs_array[:, :, :, 1]).float()

    tile_locs = \
        (which_locs_array.unsqueeze(3) * locs.unsqueeze(1) - \
            (tile_coords + edge_padding - 0.5)).view(subimage_batchsize, max_stars, 2) / \
                (subimage_slen - 2 * edge_padding)
    tile_locs = torch.relu(tile_locs) # by subtracting off, some are negative now; just set these to 0
    if fluxes is not None:
        assert fullimage_batchsize == fluxes.shape[0]
        assert max_stars == fluxes.shape[1]
        n_bands = fluxes.shape[2]
        tile_fluxes = \
            (which_locs_array.unsqueeze(3) * fluxes.unsqueeze(1)).view(subimage_batchsize, max_stars, n_bands)
    else:
        tile_fluxes = torch.zeros(tile_locs.shape[0], tile_locs.shape[1], 1)
        n_bands = 1

    # sort locs so all the zeros are at the end
    is_on_array = which_locs_array.view(subimage_batchsize, max_stars).type(torch.bool).to(device)
    n_stars_per_tile = is_on_array.float().sum(dim = 1).type(torch.LongTensor).to(device)

    is_on_array_sorted = utils.get_is_on_from_n_stars(n_stars_per_tile, n_stars_per_tile.max())

    indx = is_on_array_sorted.clone()
    indx[indx == 1] = torch.nonzero(is_on_array, as_tuple=False)[:, 1]

    tile_fluxes = torch.gather(tile_fluxes, dim = 1, index = indx.unsqueeze(2).repeat(1, 1, n_bands)) * \
                        is_on_array_sorted.float().unsqueeze(2)
    tile_locs = torch.gather(tile_locs, dim = 1, index = indx.unsqueeze(2).repeat(1, 1, 2)) * \
                        is_on_array_sorted.float().unsqueeze(2)

    tile_is_on_array = is_on_array_sorted

    return tile_locs, tile_fluxes, n_stars_per_tile, tile_is_on_array

def get_full_params_from_tile_params(tile_locs, tile_fluxes,
                                        tile_coords,
                                        full_slen,
                                        stamp_slen,
                                        edge_padding,
                                        # TODO: default is to assume the full image is square
                                        # make this a systematic change.
                                        full_slen2 = None):

    # off stars should have tile_locs == 0 and tile_fluxes == 0

    assert (tile_fluxes.shape[0] % tile_coords.shape[0]) == 0
    batchsize = int(tile_fluxes.shape[0] / tile_coords.shape[0])

    assert (tile_fluxes.shape[0] % batchsize) == 0
    n_stars_in_batch = int(tile_fluxes.shape[0] * tile_fluxes.shape[1] / batchsize)

    n_bands = tile_fluxes.shape[2]
    fluxes = tile_fluxes.view(batchsize, n_stars_in_batch, n_bands)

    scale = (stamp_slen - 2 * edge_padding)
    bias = tile_coords.repeat(batchsize, 1).unsqueeze(1).float() + edge_padding - 0.5
    if full_slen2 is None:
        locs = (tile_locs * scale + bias) / (full_slen - 1)
    else:
        locs = (tile_locs * scale + bias) / torch.Tensor([[[full_slen - 1, full_slen2 - 1]]]).to(device)

    locs = locs.view(batchsize, n_stars_in_batch, 2)

    tile_is_on_bool = (fluxes > 0).any(2).float() # if flux in any band is nonzero
    n_stars = torch.sum(tile_is_on_bool > 0, dim = 1)

    # puts all the on stars in front
    is_on_array_full = utils.get_is_on_from_n_stars(n_stars, n_stars.max())
    indx = is_on_array_full.clone()
    indx[indx == 1] = torch.nonzero(tile_is_on_bool)[:, 1]

    fluxes = torch.gather(fluxes, dim = 1, index = indx.unsqueeze(2).repeat(1, 1, n_bands)) * \
                        is_on_array_full.float().unsqueeze(2)
    locs = torch.gather(locs, dim = 1, index = indx.unsqueeze(2).repeat(1, 1, 2)) * \
                        is_on_array_full.float().unsqueeze(2)

    return locs, fluxes, n_stars


def trim_images(images, edge_padding):
    slen = images.shape[-1] - edge_padding

    return images[:, :, edge_padding:slen, edge_padding:slen]
