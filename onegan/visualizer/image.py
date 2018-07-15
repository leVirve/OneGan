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
        args: each should shape in (N, h, w) or (N, c, h, w)
    '''
    def make_valid_batched_dim(x):
        return x.unsqueeze(1) if x.dim() == 3 else x

    channels = [img_normalize(make_valid_batched_dim(ch)) for ch in args]
    padding_channel = torch.zeros_like(channels[0])

    IMAGE_CHANNEL = 3
    for _ in range(IMAGE_CHANNEL - len(channels)):
        channels.append(padding_channel)

    return torch.cat(channels, dim=1)


def as_rgb_visual(batched_x, vallina=True):
    # TODO: make it for 3 channels
    assert batched_x.size(1) <= 3
    ch1, ch2 = batched_x[:, :1], batched_x[:, 1:]

    if vallina:
        return stack_visuals(ch1, ch2)
    else:
        color1, color2 = [.2, .8, .4], [.6, .4, .8]
        n, _, h, w = batched_x.size()
        canvas = torch.zeros(n, h, w, 3).to(batched_x)
        canvas += ch1.squeeze(1).unsqueeze(-1) * torch.tensor([color1]).to(ch1)
        canvas += ch2.squeeze(1).unsqueeze(-1) * torch.tensor([color2]).to(ch2)
        return canvas.permute(0, 3, 1, 2)
