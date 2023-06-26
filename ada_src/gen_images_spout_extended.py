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

import torch.nn.functional as F

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

    if not np.isnan(args[0]):
        speed[0] = args[0]
    if len(args) > 1:
        if not np.isnan(args[1]):
            speed[1] = args[1]

def get_psi(address, *args):
    global psi

    if not np.isnan(args[0]):
        psi[0] = args[0]
    if len(args) > 1:
        if not np.isnan(args[1]):
            psi[1] = args[1]

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
    
def get_projection(address, *args):
    global projection
    projection = bool(args[0])
    print(f"projection: {projection}")

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
parser.add_argument('--speed', type=float, help='speed', default=.1,)
parser.add_argument('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const')
parser.add_argument('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0', metavar='VEC2')
parser.add_argument('--rotate', help='Rotation angle in degrees', type=float, default=0, metavar='ANGLE')
# parser.add_argument('--send_texture', help='send texture over spout', action='store_true', default=True)
parser.add_argument('--save_imgs', help='save images', action='store_true')

# ada
parser.add_argument('--extended_gen',  action='store_true', default=False)
parser.add_argument('--size', default='1024-1024', help='output resolution, set in X-Y format')
parser.add_argument('--scale_type', default='symm', help="main types: pad, padside, symm, symmside")
parser.add_argument('--latmask', default=None, help='external mask file (or directory) for multi latent blending')
parser.add_argument('--nXY', '-n', default='1-1', help='multi latent frame split count by X (width) and Y (height)')
parser.add_argument('--target_fps', help='target fps', type=float, default=60)

parser.add_argument('--projection', help='project image', action='store_true')


args = parser.parse_args()


#---Parameters
# control
seeds = args.seeds
psi = [args.psi, args.psi]
speed = [args.speed, args.speed]
lat_x = 0.
lat_xb = 0.
lat_y = 0.
translate = args.translate
rotate = args.rotate
interpolation = 'slerp'
projection = args.projection
projection_strength = 1.

counter = 0
spout_size = (1024, 1024)

if args.size is not None: 
    args.size = [int(s) for s in args.size.split('-')][::-1]
    spout_size = (args.size[0], args.size[1])



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
dispatcher.map("/projection", get_interpolation)

ip = "127.0.0.1"
port = 7000



async def loop():
    global args, counter, lat_x, lat_xb, lat_y, G, spout
    run = True
    device = torch.device('cuda')

    #---Spout
    if not args.save_imgs:
        #---Spout
        # create spout object
        spout = Spout(silent = True, width = spout_size[1], height = spout_size[0], n_rec= 2)
        # create sender
        spout.createSender('input_gan')
        spout.createReceiver('output_gan', id = 0)
        spout.createReceiver('target_gan', id = 1)



    # setup generator   #ADA

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.verbose = True
    Gs_kwargs.size = args.size
    print(args.size)
    Gs_kwargs.scale_type = args.scale_type

    # setup aux 
    dconst = np.zeros([2, 1, 1, 1, 1])
    lmask_base = None
    dconst = torch.from_numpy(dconst).to(device)
    n_mult = 1

    # mask/blend latents with external latmask or by splitting the frame
    if args.latmask is None:
        nHW = [int(s) for s in args.nXY.split('-')][::-1]
        assert len(nHW)==2, ' Wrong count nXY: %d (must be 2)' % len(nHW)
        n_mult = nHW[0] * nHW[1]
        if n_mult > 1: print(' Latent blending w/split frame %d x %d' % (nHW[1], nHW[0]))
        lmask_base = np.tile(np.asarray([[[[1]]]]), (1,n_mult,1,1))
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
        lmask_base = np.concatenate((lmask, 1 - lmask), 1) # [frm,2,h,w]
    lmask = torch.from_numpy(lmask_base).to(device)

    #ADA

    #Load Network
    print('Loading networks from "%s"...' % args.network)        
    if (args.extended_gen):
        with dnnlib.util.open_url(args.network) as fs:
            custom = True
            G = legacy.load_network_pkl(fs, custom=custom, **Gs_kwargs)['G_ema'].to(device) # type: ignore
    else:
        with dnnlib.util.open_url(args.network) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
            
    print(' out shape', G.output_shape[1:])


    if args.save_imgs:
        os.makedirs(args.outdir, exist_ok=True)

    # setup seeds
    print(f"seeds: {len(seeds)}")
    seed = 0
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    label = torch.zeros([1, G.c_dim], device=device)
    cubic_key_latents = np.array([(np.random.RandomState(seeds[i]).randn(1, G.z_dim)) for i in range(len(seeds))])
    cs = cublerp_spline(cubic_key_latents)
    # cubic_latents = cublerp(cubic_key_latents, len(seeds), transit)


    #---PROJECTION
    w_avg_samples           = 10000
    target                  = None
    initial_noise_factor    = 0.05
    regularize_noise_weight = 1e5
    initial_learning_rate   = 0.025

    if projection:

        # Compute w stats.
        z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
        w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
        print('Projecting in W+ latent space...')
        w_avg = torch.mean(w_samples, dim=0, keepdim=True)  # [1, L, C]
        w_std = (torch.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
        # Setup noise inputs (only for StyleGAN2 models)
        noise_buffs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}
        w_noise_scale = w_std * initial_noise_factor # noise scale is constant
        lr = initial_learning_rate # learning rate is constant

        # Load the VGG16 feature detector.
        # url = 'e:/sg3-pretrained/metrics/vgg16.pkl'
        # Load VGG16 feature detector.
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            vgg16 = torch.jit.load(f).eval().to(device)
        print("Loaded vgg16...")

        w_opt = w_avg.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([w_opt] + list(noise_buffs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Init noise.
        for buf in noise_buffs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True       
        # == End setup from project() == #




    # setup time
    current_time = datetime.datetime.now()
    last_time = datetime.datetime.now()
    fps = 0

    
    #---TEST
    z2 = np.random.RandomState(1000).randn(1, G.z_dim)
    z3 = np.random.RandomState(1000).randn(1, G.z_dim)

    zb = slerp_single(z2, z3, 0.)[0]


    #---UPDATE
    while run: # Ctrl+C to stop

        if not args.save_imgs:
            # check on close window
            spout.check()

        for i in range (128):
            await asyncio.sleep(0.0)

        # update time
        current_time = datetime.datetime.now()
        time_diff = current_time - last_time
        time_diff = time_diff.total_seconds()
        last_time = datetime.datetime.now()
        if (time_diff > 0.):
            fps = fps * .98 + (1. / time_diff) * .02


        #---UPDATE SEEDS    
        if args.save_imgs:
            lat_x += (1. / args.target_fps) * speed[0]
        else:
            lat_x += time_diff * speed[0]

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

        #---test
        if args.extended_gen and n_mult > 1:
            lat_xb += time_diff * speed[0] * .05

            last_val = np.floor(lat_xb)
            frac_val = abs(lat_xb - np.trunc(lat_xb))
            if lat_xb < 0.:
                frac_val = 1. - frac_val
            next_val = np.ceil(lat_xb)

            last_item = int(last_val)%len(seeds)
            next_item = int(next_val)%len(seeds)

            z2 = np.random.RandomState(last_item + 1000).randn(1, G.z_dim)
            z3 = np.random.RandomState(next_item + 1000).randn(1, G.z_dim)

            zb = slerp_single(z2, z3, frac_val)[0]
            if interpolation == 'cubic':
                zb = cs(last_item + frac_val) 

            z = np.concatenate((z, zb), axis=0)
            # z.append(zb)

        z = torch.from_numpy(z).to(device)

        w = G.mapping(z, label, truncation_psi=psi[0])


        #---PROJECTION
        if projection:
            target_spout = spout.receive(id = 1)
            if target_spout.shape[0] > 0:
                target = torch.tensor(target_spout.transpose([2, 0, 1]), device=device)
                target = target.unsqueeze(0).to(device).to(torch.float32)
                if target.shape[2] > 256:
                    target = F.interpolate(target, size=(256, 256), mode='area')
                target_features = vgg16(target, resize_images=False, return_lpips=True) 

                # Synth images from opt_w.
                w_noise = torch.randn_like(w_opt) * w_noise_scale
                ws = w_opt + w_noise
                synth_images = G.synthesis(ws, noise_mode='const')
                # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
                synth_images = (synth_images + 1) * (255/2)
                if synth_images.shape[2] > 256:
                    synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

                # Features for synth images.
                synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
                dist = (target_features - synth_features).square().sum()

                # Noise regularization.
                reg_loss = 0.0
                for v in noise_buffs.values():
                    noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                    while True:
                        reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                        reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                        if noise.shape[2] <= 8:
                            break
                        noise = F.avg_pool2d(noise, kernel_size=2)
                loss = dist + reg_loss * regularize_noise_weight

                if (counter % 60 == 0 and counter > 0):
                    print(f"loss: {loss:0.4f}")


                # Step
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                
                # Normalize noise.
                with torch.no_grad():
                    for buf in noise_buffs.values():
                        buf -= buf.mean()
                        buf *= buf.square().mean().rsqrt()
                

                w = w_opt.detach()[0]
                w = w_avg + (w - w_avg) * psi[0]
                
        # else:
            # w = G.mapping(z, label, truncation_psi=psi[0])

        # print('w: ', w.shape)

        #---GENERATION
        # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))


        #---TEST
        data_mask = spout.receive(id = 0)
        if data_mask.shape[0] > 0:
            # print('data_mask: ', data_mask.shape)
            data_mask = np.asarray([[data_mask[:,:,0] / 255.]]) # [1,1,h,w]
            data_mask = np.concatenate((data_mask, 1 - data_mask), 1) # [frm,2,h,w]

            lmask = torch.from_numpy(data_mask).to(device)


        if len(w.shape) == 2:
            w = w.unsqueeze(0)  # An individual dlatent => [1, G.mapping.num_ws, G.mapping.w_dim]
        # print('w:', w.shape)

        if not args.extended_gen:
            # img = G(z, label, truncation_psi=psi, noise_mode=args.noise_mode)
            img = G.synthesis(w, noise_mode=args.noise_mode)
        else:
            dc      = dconst[0]
            latmask = lmask[last_item % len(lmask)] if lmask is not None else [None]
            
            # img = G(z, label, latmask, dc, truncation_psi=psi, noise_mode=args.noise_mode)
            img = G.synthesis(w, latmask, dc, noise_mode=args.noise_mode)
            # # print all the G **kwargs for debugging
            # print('G_kwargs:', G.synthesis.__dict__)

            # # print all the G *input for debugging



        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        data = img[0].cpu().numpy()


        #---SEND
        # if (args.send_texture):
        if not args.save_imgs:
            spout.send(data)
            # image = PIL.Image.fromarray(data, 'RGB')
            # spout.send(image)


        if (counter % 600 == 0 and counter > 0):
            print(f"avg_fps: {fps:0.4f}")

        counter += 1
        if args.save_imgs:
            # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed_{seed:04d}_t_{psi:0.2f}.png')
            PIL.Image.fromarray(data, 'RGB').save(f'{args.outdir}/seed_{counter:04d}_t_{psi[0]:0.2f}.png')
            if next_val > len(seeds):
                run = False



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
