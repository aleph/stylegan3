# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
from matplotlib.pyplot import axis
import dnnlib
import numpy as np
from numpy import linalg as LA
import PIL.Image
import torch

import legacy

import csv

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=False)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    translate: Tuple[float,float],
    rotate: float,
    class_idx: Optional[int]
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    \b
    # Generate uncurated images with truncation using the MetFaces-U dataset
    python gen_images.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    input_seed_list = [[2, 4, 8], [16, 32, 64], [128, 256, 512]]
    seed_list = []

    print(f"Generating {len(input_seed_list)} paths from seed_list")
    for list_idx, list in enumerate(input_seed_list):
        pos_array = []
        pos_list = []
        idx_list = []

        out_list = []
        for seed_idx, seed in enumerate(list):
            # z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            z_m = np.random.RandomState(seed).randn(1, G.z_dim)
            if seed_idx == 0:
                pos_array = z_m
            else:
                pos_array = np.append(pos_array, z_m, axis=0)

            z = np.random.RandomState(seed).randn(G.z_dim)
            pos_list.append(z)
            idx_list.append(seed_idx)

        print(f"before mean: {pos_array}, dimension: {pos_array.ndim}")
        pos_array = np.mean(pos_array, axis=0)
        print(f"after mean: {pos_array}, dimension: {pos_array.ndim}")

        current_idx = 0
        current_item = 0
        idx_counter = 0

        while current_idx >= 0 or idx_counter > 10000:
            min_dist = 10000000.0
            min_idx = -1
            current_item = idx_list[current_idx]
            out_list.append(list[current_item])
            print(f"out_list: {out_list}")

            current_pos = pos_list[current_idx]
            print(f"current item: {list[current_item]} | popping {current_idx} of {len(pos_list)}")
            pos_list.pop(current_idx)
            idx_list.pop(current_idx)
            print(f"length: {len(pos_list)}")

            for t_idx, t in enumerate(pos_list):
                distance = np.linalg.norm(t - current_pos)

                if distance < min_dist:
                    min_dist = distance
                    min_idx = t_idx
                    # item_idx = idx_list[t_idx]
                    print(f"min_idx: {t_idx} | distance: {distance}")

            current_idx = min_idx
            idx_counter += 1
            print(f"-------> idx: {current_idx} | item: {current_item}") 

        shift_dist = int(np.floor(len(out_list) * .5))
        out_list = out_list[shift_dist:] + out_list[:shift_dist]
        seed_list.append(out_list)
    
    print(f"seed_list: {seed_list}")


    # seed_list = input_seed_list.copy()
    for list_idx, list in enumerate(seed_list):
        for seed_idx, seed in enumerate(list):
            print('Generating image for seed %d (%d/%d) path (%d/%d) ...' % (seed, seed_idx, len(list) - 1, list_idx, len(seed_list) - 1))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

            # Construct an inverse rotation/translation matrix and pass to the generator.  The
            # generator expects this matrix as an inverse to avoid potentially failing numerical
            # operations in the network.
            if hasattr(G.synthesis, 'input'):
                m = make_transform(translate, rotate)
                m = np.linalg.inv(m)
                G.synthesis.input.transform.copy_(torch.from_numpy(m))

            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}\path_{list_idx:02d}_{seed_idx:03d}_seed_{seed:04d}_t_{truncation_psi:0.2f}.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
