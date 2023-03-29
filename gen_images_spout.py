# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""
from Library.Spout import Spout

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import datetime
import asyncio
from scipy.interpolate import CubicSpline as CubSpline
import os.path as osp

import legacy
import argparse

from util.utilgan import img_read, img_list      # using ada

# ---- OSC -----
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.osc_server import AsyncIOOSCUDPServer

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

# def get_pinned_buf(self, ref):
#     key = (tuple(ref.shape), ref.dtype)
#     buf = self._pinned_bufs.get(key, None)
#     if buf is None:
#         buf = torch.empty(ref.shape, dtype=ref.dtype).pin_memory()
#         self._pinned_bufs[key] = buf
#     return buf

# def to_cpu(self, buf):
#     return self.get_pinned_buf(buf).copy_(buf).clone()


#----------------------------------------------------------------------------

def lerp_single(z1, z2, x): 
    vectors = []
    interpol = z1 + (z2 - z1) * x
    vectors.append(interpol)
    return np.array(vectors)

def slerp_single(z1, z2, x):
    z1_norm = np.linalg.norm(z1)
    z2_norm = np.linalg.norm(z2)
    z2_normal = z2 * (z1_norm / z2_norm)
    vectors = []

    interplain = z1 + (z2 - z1) * x
    interp = z1 + (z2_normal - z1) * x
    interp_norm = np.linalg.norm(interp)
    interpol_normal = interplain * (z1_norm / interp_norm)
    vectors.append(interpol_normal)

    return np.array(vectors)

def cublerp_spline(points):
    steps = len(points)
    keys = np.array([i for i in range(steps)] + [steps])
    points = np.concatenate((points, np.expand_dims(points[0], 0)))
    cspline = CubSpline(keys, points)
    return cspline

def cublerp(points, steps, fstep):
    keys = np.array([i*fstep for i in range(steps)] + [steps*fstep])
    points = np.concatenate((points, np.expand_dims(points[0], 0)))
    cspline = CubSpline(keys, points)
    return cspline(range(steps*fstep+1))

#----------------------------------------------------------------------------

def get_speed(address, *args):
    global speed
    speed = args[0]

def get_psi(address, *args):
    global psi,counter
    psi = args[0]
    # print(f"psi: {psi} | counter: {counter}")

def get_y(address, *args):
    global lat_y
    lat_y = args[0]

def get_seeds(address, *args):
    global seeds
    seeds_buffer = args[:]
    seeds.append(seeds_buffer[0])

def get_rotate(address, *args):
    global rotate
    rotate = args[0]

def get_translate(address, *args):
    global translate
    if len(args) > 1:
        translate = (args[0], args[1])

def get_interpolation(address, *args):
    global interpolation
    interpolation = args[0]
    print(f"interpolation: {interpolation}")

def default_handler(address, *args):
    print(f"DEFAULT {address}: {args}")

#----------------------------------------------------------------------------

parser = argparse.ArgumentParser()

# folders
parser.add_argument('--network', help='Network pickle filename', required=True)
parser.add_argument('--outdir', help='Where to save the output images', type=str, metavar='DIR')
parser.add_argument('--seeds_csv', help='use a csv list of seeds', default='')
parser.add_argument('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
# general
parser.add_argument('--psi', type=float, help='Truncation psi', default=.8,)
parser.add_argument('--speed', type=float, help='Truncation psi', default=.1,)
parser.add_argument('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const')
parser.add_argument('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0', metavar='VEC2')
parser.add_argument('--rotate', help='Rotation angle in degrees', type=float, default=0, metavar='ANGLE')
parser.add_argument('--send_texture', help='send texture over spout', action='store_true', default=True)
parser.add_argument('--save_imgs', help='save images', action='store_true')

# ada
parser.add_argument('--extended_gen',  action='store_true', default=False)
parser.add_argument('--size', default='1024-1024', help='output resolution, set in X-Y format')
parser.add_argument('--scale_type', default='symm', help="main types: pad, padside, symm, symmside")
parser.add_argument('--latmask', default=None, help='external mask file (or directory) for multi latent blending')
parser.add_argument('--nXY', '-n', default='1-1', help='multi latent frame split count by X (width) and Y (height)')

args = parser.parse_args()


#---Parameters
# control
seeds = args.seeds
psi = args.psi
speed = args.speed
lat_x = 0.
lat_y = 0.
translate = args.translate
rotate = args.rotate
interpolation = 'slerp'

counter = 0

if args.size is not None: args.size = [int(s) for s in args.size.split('-')][::-1]



#---OSC
dispatcher = Dispatcher()
dispatcher.set_default_handler(default_handler)
dispatcher.map("/speed", get_speed)
dispatcher.map("/psi", get_psi)
dispatcher.map("/y", get_y)
dispatcher.map("/seeds", get_seeds)
dispatcher.map("/rotate", get_rotate)
dispatcher.map("/translate", get_translate)
dispatcher.map("/interpolation", get_interpolation)

ip = "127.0.0.1"
port = 7000



async def loop():
    global args, counter, lat_x, lat_y
    run = True
    device = torch.device('cuda')

    #---Spout
    # create spout object
    spout = Spout(silent = False, width = 1024, height = 1024)
    # create sender
    spout.createSender('input_gan')


    # setup generator   #ADA

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.verbose = True
    Gs_kwargs.size = args.size
    # print(a.size)
    Gs_kwargs.scale_type = args.scale_type

    # setup aux 
    dconst = np.zeros([2, 1, 1, 1, 1])
    dconst = torch.from_numpy(dconst).to(device)

    # mask/blend latents with external latmask or by splitting the frame
    if args.latmask is None:
        nHW = [int(s) for s in args.nXY.split('-')][::-1]
        assert len(nHW)==2, ' Wrong count nXY: %d (must be 2)' % len(nHW)
        n_mult = nHW[0] * nHW[1]
        if n_mult > 1: print(' Latent blending w/split frame %d x %d' % (nHW[1], nHW[0]))
        lmask = np.tile(np.asarray([[[[1]]]]), (1,n_mult,1,1))
        Gs_kwargs.countHW = nHW
        Gs_kwargs.splitfine = 0.
    else:
        print(' Latent blending with mask', args.latmask)
        n_mult = 2
        if osp.isfile(args.latmask): # single file
            lmask = np.asarray([[img_read(args.latmask)[:,:,0] / 255.]]) # [1,1,h,w]
        elif osp.isdir(args.latmask): # directory with frame sequence
            lmask = np.expand_dims(np.asarray([img_read(f)[:,:,0] / 255. for f in img_list(args.latmask)]), 1) # [n,1,h,w]
        else:
            print(' !! Blending mask not found:', args.latmask); exit(1)
        print(' latmask shape', lmask.shape)
        lmask = np.concatenate((lmask, 1 - lmask), 1) # [frm,2,h,w]
    lmask = torch.from_numpy(lmask).to(device)

    #ADA


    #Load Network
    print('Loading networks from "%s"...' % args.network)
    with dnnlib.util.open_url(args.network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        
    if (args.extended_gen):
        with dnnlib.util.open_url(r"models\network-snapshot-010990-Gs.pkl") as fs:
            custom = True
            G = legacy.load_network_pkl(fs, **Gs_kwargs)['G_ema'].to(device) # type: ignore
            print(' out shape', G.output_shape[1:])


    if args.save_imgs:
        os.makedirs(parser.add_argumentoutdir, exist_ok=True)

    # setup seeds
    seed = 0
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    label = torch.zeros([1, G.c_dim], device=device)
    cubic_key_latents = np.array([(np.random.RandomState(seeds[i]).randn(1, G.z_dim)) for i in range(len(seeds))])
    cs = cublerp_spline(cubic_key_latents)
    # cubic_latents = cublerp(cubic_key_latents, len(seeds), transit)


    # setup time
    current_time = datetime.datetime.now()
    last_time = datetime.datetime.now()
    fps = 0


    #---UPDATE
    while run: # Ctrl+C to stop

        # check on close window
        spout.check()

        for i in range (16):
            await asyncio.sleep(0.0)

        # update time
        current_time = datetime.datetime.now()
        time_diff = current_time - last_time
        time_diff = time_diff.total_seconds()
        last_time = datetime.datetime.now()
        if (time_diff > 0.):
            fps = fps * .98 + (1. / time_diff) * .02


        #---UPDATE SEEDS    
        lat_x += time_diff * speed
        last_val = np.floor(lat_x)
        frac_val = abs(lat_x - np.trunc(lat_x))
        if lat_x < 0.:
            frac_val = 1. - frac_val
        next_val = np.ceil(lat_x)

        last_item = int(last_val)%len(seeds)
        next_item = int(next_val)%len(seeds)

        z0 = np.random.RandomState(last_item).randn(1, G.z_dim)
        z1 = np.random.RandomState(next_item).randn(1, G.z_dim)

        z = slerp_single(z0, z1, frac_val)[0]
        if interpolation == 'cubic':
            z = cs(last_item + frac_val) 

        z = torch.from_numpy(z).to(device)


        #---UPDATE AUX  #ADA
        dc      = dconst[0]
        latmask = lmask[last_item % len(lmask)] if lmask is not None else [None]


        #---GENERATION
        # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))


        if not args.extended_gen:
            img = G(z, label, truncation_psi=psi, noise_mode=args.noise_mode)
        else:
            img = G(z, label, latmask, dc, truncation_psi=psi, noise_mode=args.noise_mode)

        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        data = img[0].cpu().numpy()


        #---SEND
        if (args.send_texture):
            spout.send(data)
            # image = PIL.Image.fromarray(data, 'RGB')
            # spout.send(image)


        if (counter % 600 == 0 and counter > 0):
            print(f"avg_fps: {fps:0.4f}")

        counter += 1
        if (args.save_imgs):
            PIL.Image.fromarray(data, 'RGB').save(f'{args.outdir}/seed_{seed:04d}_t_{psi:0.2f}.png')
            # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed_{seed:04d}_t_{psi:0.2f}.png')



async def main():

    #---SETUP
    # server = BlockingOSCUDPServer(("localhost", 8000), dispatcher)
    server = AsyncIOOSCUDPServer((ip, port), dispatcher, asyncio.get_event_loop())
    transport, protocol = await server.create_serve_endpoint()  # Create datagram endpoint and start serving

    await loop()    # enter main loop

    transport.close() # clean up serve endpoint
            


#----------------------------------------------------------------------------

if __name__ == "__main__":
    # main() # pylint: disable=no-value-for-parameter
    asyncio.run(main())

#----------------------------------------------------------------------------