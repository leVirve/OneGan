# Copyright (c) 2018- Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch


def img_normalize(img, val_range=None):
    ''' Normalize the tensor into (0, 1)

    Args:
        tensor: torch.Tensor
        val_range: tuple of (min_val, max_val)
    Returns:
        img: normalized tensor
    '''
    t = img.clone()
    if val_range:
        mm, mx = val_range[0], val_range[1]
    else:
        mm, mx = t.min(), t.max()
    try:
        return t.add_(-mm).div_(mx - mm)
    except RuntimeError:
        return img


def stack_visuals(*args):
    ''' Merge results in one image (R, G, B) naively
    Args:
        args: each should shape in (N, H, W) or (N, C, H, W)
    '''
    def make_valid_batched_dim(x):
        return x.unsqueeze(1) if x.dim() == 3 else x

    dtype = args[0].type()
    channels = [img_normalize(make_valid_batched_dim(ch.type(dtype))) for ch in args]
    padding_channel = torch.zeros_like(channels[0])

    IMAGE_CHANNEL = 3
    for _ in range(IMAGE_CHANNEL - len(channels)):
        channels.append(padding_channel)

    return torch.cat(channels, dim=1)


def as_rgb_visual(tensor, vallina=False):
    ''' Make tensor into colorful image
    Args:
        tensor: shape in (C, H, W) or (N, C, H, W)
        vallina: (bool) if True, then use the `stack_visuals`
    '''

    def batched_colorize(batched_x):
        n, c, h, w = batched_x.size()
        channels = [batched_x[:, i] for i in range(c)]  # list of N x h x w

        if vallina:
            return stack_visuals(*channels)
        else:
            dtype = channels[0].type()
            colors = torch.tensor([[.2, .8, .4], [.6, .4, .8], [.8, .4, .2]]).type(dtype)

            canvas = torch.zeros(n, h, w, 3).to(batched_x)
            for i in range(c):
                canvas += channels[i].unsqueeze(-1) * colors[i]
            return canvas.permute(0, 3, 1, 2)

    if tensor.dim() == 3:
        return batched_colorize(tensor.unsqueeze(0)).squeeze(-1)
    return batched_colorize(tensor)
