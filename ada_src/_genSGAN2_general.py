import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import os.path as osp
import argparse
import numpy as np
from imageio import imsave

import torch

import dnnlib
import legacy
import imageio
import math

import time
import pandas as pd

from util.utilgan import latent_anima, basename, img_read, img_list, latent_timeline
# try: # progress bar for notebooks 
#     get_ipython().__class__.__name__
#     from util.progress_bar import ProgressIPy as ProgressBar
# except: # normal console
#     from util.progress_bar import ProgressBar

desc = "Customized StyleGAN2-ada on PyTorch"
parser = argparse.ArgumentParser(description=desc)
# general
parser.add_argument('--out_dir', default='_out', help='output directory')
parser.add_argument('--gen_name', default='test', help='gen_name')
parser.add_argument('--model', default='models/network-snapshot-010990-Gs.pkl', help='path to pkl checkpoint file')
parser.add_argument('--labels', '-l', type=int, default=None, help='labels/categories for conditioning')
# custom
parser.add_argument('--size', default=None, help='output resolution, set in X-Y format')
parser.add_argument('--scale_type', default='pad', help="main types: pad, padside, symm, symmside")
parser.add_argument('--latmask', default=None, help='external mask file (or directory) for multi latent blending')
parser.add_argument('--nXY', '-n', default='1-1', help='multi latent frame split count by X (width) and Y (height)')
parser.add_argument('--splitfine', type=float, default=0, help='multi latent frame split edge sharpness (0 = smooth, higher => finer)')
parser.add_argument('--trunc', type=float, default=0.8, help='truncation psi 0..1 (lower = stable, higher = various)')
parser.add_argument('--digress', type=float, default=0, help='distortion technique by Aydao (strength of the effect)') 
parser.add_argument('--save_lat', action='store_true', help='save latent vectors to file')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--verbose', '-v', action='store_true')
# animation
parser.add_argument('--frames', default='200-25', help='total frames to generate, length of interpolation step')
parser.add_argument("--cubic_poly", action='store_true', help="use cubic splines for poly")
parser.add_argument("--lerp", action='store_true')
parser.add_argument("--slerp", action='store_true')
parser.add_argument("--cubic", action='store_true', help="use cubic splines for smoothing")
parser.add_argument("--gauss", action='store_true', help="use Gaussian smoothing")
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--no_image_save", action='store_true')
parser.add_argument("--gen_imgs", action='store_true')
parser.add_argument("--explore_psi", type=int, default=0)
parser.add_argument("--seeds_len", type=int, default=20)
parser.add_argument("--start_point", type=float, default=0.)
parser.add_argument("--end_point", type=float, default=1.)
parser.add_argument("--real_time_fps", type=float, default=0)
parser.add_argument("--inertia", type=float, default=.5)


# csv
parser.add_argument('--use_csv', action='store_true', help='use_csv')
parser.add_argument("--seeds_path", default=None, help='___')
parser.add_argument("--params_path", default=None, help='___')
parser.add_argument('--export_csv', action='store_true')

a = parser.parse_args()

if a.size is not None: a.size = [int(s) for s in a.size.split('-')][::-1]
# a.size = [1024, 2048]
[a.frames, a.fstep] = [int(s) for s in a.frames.split('-')]


#---UTILITY FUNCTIONS



#---MAIN FUNCTION
def generate():
    out_path = osp.join(a.out_dir, a.gen_name)

    os.makedirs(out_path, exist_ok=True)
    if a.seed==0: a.seed = None
    np.random.seed(seed=a.seed)
    device = torch.device('cuda')

    # setup generator
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.verbose = a.verbose
    Gs_kwargs.size = a.size
    # print(a.size)
    Gs_kwargs.scale_type = a.scale_type

    # csv setup
    lantent_params = [.85, .5, .6]      # inertia, speed, psi
    seeds = []
    data = []

    if a.use_csv and a.seeds_path is not None and a.params_path is not None:
        data = pd.read_csv(a.params_path, header=None)
        if a.verbose: print(data)

        seeds_data = pd.read_csv(a.seeds_path, header=None)

        for i in range(2):
            row = []
            for j in range(len(seeds_data.iloc[0])):
                row.append(seeds_data[j][i])
            
            if a.verbose: print(row)
            seeds.append(row)

        print(seeds_data.iloc[0])






    # mask/blend latents with external latmask or by splitting the frame
    if a.latmask is None:
        nHW = [int(s) for s in a.nXY.split('-')][::-1]
        assert len(nHW)==2, ' Wrong count nXY: %d (must be 2)' % len(nHW)
        n_mult = nHW[0] * nHW[1]
        if a.verbose is True and n_mult > 1: print(' Latent blending w/split frame %d x %d' % (nHW[1], nHW[0]))
        lmask = np.tile(np.asarray([[[[1]]]]), (1,n_mult,1,1))
        Gs_kwargs.countHW = nHW
        Gs_kwargs.splitfine = a.splitfine
    else:
        if a.verbose is True: print(' Latent blending with mask', a.latmask)
        n_mult = 2
        if osp.isfile(a.latmask): # single file
            lmask = np.asarray([[img_read(a.latmask)[:,:,0] / 255.]]) # [1,1,h,w]
        elif osp.isdir(a.latmask): # directory with frame sequence
            lmask = np.expand_dims(np.asarray([img_read(f)[:,:,0] / 255. for f in img_list(a.latmask)]), 1) # [n,1,h,w]
        else:
            print(' !! Blending mask not found:', a.latmask); exit(1)
        if a.verbose is True: print(' latmask shape', lmask.shape)
        lmask = np.concatenate((lmask, 1 - lmask), 1) # [frm,2,h,w]
    lmask = torch.from_numpy(lmask).to(device)
    


    # load base or custom network
    pkl_name = osp.splitext(a.model)[0]
    if '.pkl' in a.model.lower():
        custom = False
        print(' .. Gs from pkl ..', basename(a.model))
    else:
        custom = True
        print(' .. Gs custom ..', basename(a.model))
    with dnnlib.util.open_url(pkl_name + '.pkl') as f:
        custom = True
        Gs = legacy.load_network_pkl(f, custom=custom, **Gs_kwargs)['G_ema'].to(device) # type: ignore

    if a.verbose is True: print(' out shape', Gs.output_shape[1:])
    if a.verbose is True: print(' making timeline..')


    # lats = [] # list of [frm,1,512]
    # for i in range(n_mult):
    #     lat_tmp = latent_anima((1, Gs.z_dim), a.frames, a.fstep, cubic=a.cubic, gauss=a.gauss, seed=a.seed, verbose=False) # [frm,1,512]
    #     lats.append(lat_tmp) # list of [frm,1,512]
    # latents = np.concatenate(lats, 1) # [frm,X,512]
    # print(' latents', latents.shape)
    # latents = torch.from_numpy(latents).to(device)
    # frame_count = latents.shape[0]

    test_psi = None
    test_speed = None
    

    lats = [] # list of [frm,1,512]
    print(' n_mult: ', n_mult)
    for i in range(n_mult):
        seeds_row = []
        if a.gen_imgs:
            for j in range(a.seeds_len):
                seeds_row.append(j + 100 * (i + 1))
        else:
            seeds_row = seeds_data.iloc[i]

        print(' seeds: ', seeds_row)

        test_lat, test_psi, test_speed = latent_timeline((1, Gs.z_dim), a.frames, data, seeds=seeds_row, inertia=a.inertia, seed=a.seed, cubic_poly=a.cubic_poly, slerp=a.slerp, cubic=a.cubic, transit=a.fstep, fps=a.real_time_fps) # [frm,1,512]
        lats.append(test_lat) # list of [frm,1,512]
    latents = np.concatenate(lats, 1) # [frm,X,512]
    print(' latents: ', latents.shape)
    latents = torch.from_numpy(latents).to(device)
    frame_count = latents.shape[0]

    print("printing_tests: ")
    print(test_lat)
    print(test_psi)
    

    if a.export_csv:
        # data_list = []
        # data_list.append(test_speed)
        # data_list.append(test_psi)
        # df = pd.DataFrame(data_list)
        df_speed = pd.DataFrame(test_speed)
        df_psi = pd.DataFrame(test_psi)

        csv_speed = df_speed.to_csv(osp.join(out_path, a.gen_name + "_speed.txt"), sep='\t', index=False, header=None)
        csv_psi = df_psi.to_csv(osp.join(out_path, a.gen_name + "_psi.txt"), sep='\t', index=False, header=None)


    # distort image by tweaking initial const layer
    dconst = np.zeros([frame_count, 1, 1, 1, 1])
    dconst = torch.from_numpy(dconst).to(device)
    if a.digress > 0:
        try: init_res = Gs.init_res
        except: init_res = (4,4) # default initial layer size 
        dconst = []
        for i in range(n_mult):
            dc_tmp = a.digress * latent_anima([1, Gs.z_dim, *init_res], a.frames, a.fstep, cubic=True, seed=a.seed, verbose=False)
            dconst.append(dc_tmp)
        dconst = np.concatenate(dconst, 1)
    else:
        dconst = np.zeros([frame_count, 1, 1, 1, 1])
    dconst = torch.from_numpy(dconst).to(device)


    # labels / conditions
    labels = [None]
    # label_size = Gs.c_dim
    # if label_size > 0:
    #     labels = torch.zeros((frame_count, n_mult, label_size), device=device) # [frm,X,lbl]
    #     if a.labels is None:
    #         label_ids = []
    #         for i in range(n_mult):
    #             label_ids.append(random.randint(0, label_size-1))
    #     else:
    #         label_ids = [int(x) for x in a.labels.split('-')]
    #         label_ids = label_ids[:n_mult] # ensure we have enough labels
    #     for i, l in enumerate(label_ids):
    #         labels[:,i,l] = 1
    # else:
    #     labels = [None]



    # setup record
    video_out = None
    if not a.gen_imgs:
        video_out = imageio.get_writer(str(osp.join(a.out_dir, a.gen_name + ".mp4")), mode='I', fps=a.fps, codec='libx264') #video

    # generate images from latent timeline
    start_frame = math.floor(frame_count * a.start_point)
    end_frame = math.ceil(frame_count * a.end_point)

    print(f"start: {start_frame}, end: {end_frame} | total frames: {end_frame - start_frame}")


    # for i in range(frame_count):
    for i in range(start_frame, end_frame):
    
        latent  = latents[i] # [X,512]
        label   = labels[i % len(labels)]
        latmask = lmask[i % len(lmask)] if lmask is not None else [None] # [X,h,w]
        dc      = dconst[i % len(dconst)] # [X,512,4,4]

        if a.explore_psi > 0:
            for j in range(a.explore_psi):
                psi_val = 1. / float(a.explore_psi) * (j + 1)
                if a.explore_psi == 1:
                    psi_val = .75
                output = Gs(latent, label, latmask, dc, truncation_psi=psi_val, noise_mode='const')
                output = (output.permute(0,2,3,1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()

                # save image
                ext = 'png' if output.shape[3]==4 else 'jpg'
                filename = osp.join(out_path, "%06d_%.2f.%s" % (i,psi_val,ext))
                imsave(filename, output[0])

        else:
            # generate multi-latent result
            if custom:
                if a.use_csv:
                    # current_psi = test_psi[i]
                    # if (end_frame - i < 2):
                    #     current_psi = test_psi[0]
                    output = Gs(latent, label, latmask, dc, truncation_psi=test_psi[i], noise_mode='const')

                else:
                    output = Gs(latent, label, latmask, dc, truncation_psi=a.trunc, noise_mode='const')
            else:
                output = Gs(latent, label, truncation_psi=a.trunc, noise_mode='const')
            output = (output.permute(0,2,3,1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()

            # save image
            ext = 'png' if output.shape[3]==4 else 'jpg'
            filename = osp.join(out_path, "%06d.%s" % (i,ext))
            if not a.no_image_save:
                imsave(filename, output[0])


        if not a.gen_imgs:
            video_out.append_data(output[0]) #video

        if (i % math.floor(frame_count / 10) == 0):
            print("%01d/%01d" % (i,frame_count))

    if not a.gen_imgs:
        video_out.close()


    # convert latents to dlatents, save them
    if a.save_lat is True:
        latents = latents.squeeze(1) # [frm,512]
        # dlatents = Gs.mapping(latents, label) # [frm,18,512]
        # if a.size is None: a.size = ['']*2
        # filename = '{}-{}-{}.npy'.format(basename(a.model), a.size[1], a.size[0])
        # filename = osp.join(osp.dirname(a.out_dir), filename)
        # dlatents = dlatents.cpu().numpy()
        # np.save(filename, dlatents)
        # print('saved dlatents', dlatents.shape, 'to', filename)


if __name__ == '__main__':
    generate()
