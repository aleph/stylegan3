# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import click
import os

import multiprocessing
import numpy as np
# from sympy import Min
import imgui
import dnnlib
from gui_utils import imgui_window
from gui_utils import imgui_utils
from gui_utils import gl_utils
from gui_utils import text_utils
from viz import renderer
from viz import pickle_widget
from viz import latent_widget
from viz import stylemix_widget
from viz import trunc_noise_widget
from viz import performance_widget
from viz import capture_widget
from viz import layer_widget
from viz import equivariance_widget
from viz import osc_widget

from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient
import asyncio
from typing import List, Any
import datetime
import csv
import random

#----------------------------------------------------------------------------

class Visualizer(imgui_window.ImguiWindow):
    def __init__(self, capture_dir=None):
        super().__init__(title='GAN Visualizer', window_width=3840, window_height=2160, window_monitor=True)

        # Internals.
        self._last_error_print  = None
        self._async_renderer    = AsyncRenderer()
        self._defer_rendering   = 0
        self._tex_img           = None
        self._tex_obj           = None

        # Widget interface.
        self.args               = dnnlib.EasyDict()
        self.result             = dnnlib.EasyDict()
        self.pane_w             = 0
        self.label_w            = 0
        self.button_w           = 0

        # Osc.
        self.down = True
        # self.drag = 0.05
        # self.inertia = 0.9

        # Widgets.
        self.pickle_widget      = pickle_widget.PickleWidget(self)
        self.latent_widget      = latent_widget.LatentWidget(self)
        self.stylemix_widget    = stylemix_widget.StyleMixingWidget(self)
        self.trunc_noise_widget = trunc_noise_widget.TruncationNoiseWidget(self)
        self.perf_widget        = performance_widget.PerformanceWidget(self)
        self.capture_widget     = capture_widget.CaptureWidget(self)
        self.layer_widget       = layer_widget.LayerWidget(self)
        self.eq_widget          = equivariance_widget.EquivarianceWidget(self)
        self.osc_widget         = osc_widget.OscWidget(self)


        if capture_dir is not None:
            self.capture_widget.path = capture_dir

        # Initialize window.
        io = imgui.get_io()
        self.set_window_size(3860., 2860.)  ##NO_GUI
        # self.set_window_size(2860., 3960.)
        

        self.set_position(0, 0)
        self._adjust_font_size()
        self.skip_frame() # Layout may change after first frame.

    def close(self):
        super().close()
        if self._async_renderer is not None:
            self._async_renderer.close()
            self._async_renderer = None

    def add_recent_pickle(self, pkl, ignore_errors=False):
        self.pickle_widget.add_recent(pkl, ignore_errors=ignore_errors)

    def load_pickle(self, pkl, ignore_errors=False):
        self.pickle_widget.load(pkl, ignore_errors=ignore_errors)

    def print_error(self, error):
        error = str(error)
        if error != self._last_error_print:
            print('\n' + error + '\n')
            self._last_error_print = error

    def defer_rendering(self, num_frames=1):
        self._defer_rendering = max(self._defer_rendering, num_frames)

    def clear_result(self):
        self._async_renderer.clear_result()

    def set_async(self, is_async):
        if is_async != self._async_renderer.is_async:
            self._async_renderer.set_async(is_async)
            self.clear_result()
            if 'image' in self.result:
                self.result.message = 'Switching rendering process...'
                self.defer_rendering()

    def _adjust_font_size(self):
        old = self.font_size
        self.set_font_size(min(self.content_width / 120, self.content_height / 60))
        if self.font_size != old:
            self.skip_frame() # Layout changed.

    def draw_frame(self):
        self.begin_frame()
        self.args = dnnlib.EasyDict()
        # self.pane_w = self.font_size * 45
        self.pane_w = self.font_size * 45
        self.button_w = self.font_size * 5
        self.label_w = round(self.font_size * 4.5)


        # Detect mouse dragging in the result area.
        dragging, dx, dy = imgui_utils.drag_hidden_window('##result_area', x=self.pane_w, y=0, width=self.content_width-self.pane_w, height=self.content_height)
        if dragging:
            self.latent_widget.drag(dx, dy)

        # Begin control pane.
        imgui.set_next_window_position(0, 0)
        # imgui.set_next_window_size(self.pane_w, self.content_height)
        imgui.set_next_window_size(self.pane_w, self.button_w * 2)
        imgui.set_next_window_position(self.window_width, self.window_height) ##NO_GUI
        # imgui.set_next_window_size(1., 1.)
        imgui.begin('##control_pane', closable=True, flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE))

        # Widgets.
        expanded, _visible = imgui_utils.collapsing_header('Network & latent', default=True)
        self.pickle_widget(expanded)
        self.latent_widget(expanded)
        self.stylemix_widget(expanded)
        self.trunc_noise_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Performance & capture', default=True)
        self.perf_widget(expanded)
        self.capture_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Layers & channels', default=True)
        self.layer_widget(expanded)
        with imgui_utils.grayed_out(not self.result.get('has_input_transform', False)):
            expanded, _visible = imgui_utils.collapsing_header('Equivariance', default=True)
            self.eq_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Osc', default=True)
        self.osc_widget(expanded)

        # Render.
        if self.is_skipping_frames():
            pass
        elif self._defer_rendering > 0:
            self._defer_rendering -= 1
        elif self.args.pkl is not None:
            self._async_renderer.set_args(**self.args)
            result = self._async_renderer.get_result()
            if result is not None:
                self.result = result



        max_w = self.content_width
        max_h = self.content_height
        pos = np.array([max_w / 2, max_h / 2])

        if 'image' in self.result:
            if self._tex_img is not self.result.image:
                self._tex_img = self.result.image
                if self._tex_obj is None or not self._tex_obj.is_compatible(image=self._tex_img):
                    self._tex_obj = gl_utils.Texture(image=self._tex_img, bilinear=False, mipmap=False)
                else:
                    self._tex_obj.update(self._tex_img)
            zoom = min(max_w / self._tex_obj.width, max_h / self._tex_obj.height)
            zoom = np.floor(zoom) if zoom >= 1 else zoom
            self._tex_obj.draw(pos=pos, zoom=zoom*.95, align=0.5, rint=True)
        if 'error' in self.result:
            self.print_error(self.result.error)
            if 'message' not in self.result:
                self.result.message = str(self.result.error)
        if 'message' in self.result:
            tex = text_utils.get_texture(self.result.message, size=self.font_size, max_width=max_w, max_height=max_h, outline=2)
            tex.draw(pos=pos, align=0.5, rint=True, color=1)

        # End frame.
        self._adjust_font_size()
        imgui.end()
        self.end_frame()

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------

class AsyncRenderer:
    def __init__(self):
        self._closed        = False
        self._is_async      = False
        self._cur_args      = None
        self._cur_result    = None
        self._cur_stamp     = 0
        self._renderer_obj  = None
        self._args_queue    = None
        self._result_queue  = None
        self._process       = None

    def close(self):
        self._closed = True
        self._renderer_obj = None
        if self._process is not None:
            self._process.terminate()
        self._process = None
        self._args_queue = None
        self._result_queue = None

    @property
    def is_async(self):
        return self._is_async

    def set_async(self, is_async):
        self._is_async = is_async

    def set_args(self, **args):
        assert not self._closed
        if args != self._cur_args:
            if self._is_async:
                self._set_args_async(**args)
            else:
                self._set_args_sync(**args)
            self._cur_args = args

    def _set_args_async(self, **args):
        if self._process is None:
            self._args_queue = multiprocessing.Queue()
            self._result_queue = multiprocessing.Queue()
            try:
                multiprocessing.set_start_method('spawn')
            except RuntimeError:
                pass
            self._process = multiprocessing.Process(target=self._process_fn, args=(self._args_queue, self._result_queue), daemon=True)
            self._process.start()
        self._args_queue.put([args, self._cur_stamp])

    def _set_args_sync(self, **args):
        if self._renderer_obj is None:
            self._renderer_obj = renderer.Renderer()
        self._cur_result = self._renderer_obj.render(**args)

    def get_result(self):
        assert not self._closed
        if self._result_queue is not None:
            while self._result_queue.qsize() > 0:
                result, stamp = self._result_queue.get()
                if stamp == self._cur_stamp:
                    self._cur_result = result
        return self._cur_result

    def clear_result(self):
        assert not self._closed
        self._cur_args = None
        self._cur_result = None
        self._cur_stamp += 1

    @staticmethod
    def _process_fn(args_queue, result_queue):
        renderer_obj = renderer.Renderer()
        cur_args = None
        cur_stamp = None
        while True:
            args, stamp = args_queue.get()
            while args_queue.qsize() > 0:
                args, stamp = args_queue.get()
            if args != cur_args or stamp != cur_stamp:
                result = renderer_obj.render(**args)
                if 'error' in result:
                    result.error = renderer.CapturedException(result.error)
                result_queue.put([result, stamp])
                cur_args = args
                cur_stamp = stamp

#----------------------------------------------------------------------------

# osc server setup
count_size = [0, 0]

ip = "127.0.0.1"
port = 8000

hands_vec = [False, False]
screen_size = [1., 1.]
hands_id = [-1, -1]
hands_time = [0., 0.]
hands_confidence = [0., 0.]
hands_palm_pos = [[0., 0., 0.], [0., 0., 0.]]
hands_palm_vel = [[0., 0., 0.], [0., 0., 0.]]
hands_palm_norm = [[0., 0., 0.], [0., 0., 0.]]
hands_palm_dir = [[0., 0., 0.], [0., 0., 0.]]
hands_palm_pos_stab = [[0., 0., 0.], [0., 0., 0.]]


# osc sender setup
# ip_send = "127.0.0.1"
ip_send = "192.168.1.122"
port_send = 7400

client = SimpleUDPClient(ip_send, port_send)  # Create client



# dispatch and handler functions
def filter_handler(address, *args: List[Any]) -> None:
    count_size[1] += 1
    if count_size[1] % 30 == 0:
        current_time = datetime.datetime.now()
        print(f"{address}: {args} ---> {count_size[0]} / {current_time}")


def global_handler(address: str, *args: List[Any]) -> None:
    # # We expect two float arguments
    # if not len(args) == 2 or type(args[0]) is not float or type(args[1]) is not float:
    #     return

    hands_vec[0] = bool(args[0])
    hands_vec[1] = bool(args[1])
    screen_size[0] = float(args[2])
    screen_size[1] = float(args[3])

    
    count_size[0] += 1
    if count_size[0] % 5000 == 0:
        current_time = datetime.datetime.now()
        print(f"Setting global values: {hands_vec[0]}, {hands_vec[1]}. Screen sizes: {screen_size[0]}, {screen_size[1]} ---> {count_size[0]} / {current_time}")


def hands_handler(address: str, *args: List[Any]) -> None:
    # print(address[26:30])
    if not (address[26:30] == "hand" or address[27:31] == "hand"):
        return


    chirality_indx = 0
    if address[21:26] == "right":       #test: finding the position of the hand indicator
        chirality_indx = 1

    hands_id[chirality_indx] = int(args[0])
    hands_time[chirality_indx] = float(args[1])
    hands_confidence[chirality_indx] = float(args[2])

    curr_vec = [0., 0., 0.]
    bias = 3

    # get hand data
    for i in range(3):
        curr_vec[i] = args[i + bias]
    bias += 3
    hands_palm_pos[chirality_indx] = curr_vec.copy()
    # print(f"{curr_vec[0]} | {curr_vec[1]} | {curr_vec[2]}")

    for i in range(3):
        curr_vec[i] = args[i + bias]
    bias += 3
    hands_palm_vel[chirality_indx] = curr_vec.copy()
    # print(f"{curr_vec[0]} | {curr_vec[1]} | {curr_vec[2]}")

    for i in range(3):
        curr_vec[i] = args[i + bias]
    bias += 3
    hands_palm_norm[chirality_indx] = curr_vec.copy()
    # print(f"{curr_vec[0]} | {curr_vec[1]} | {curr_vec[2]}")

    for i in range(3):
        curr_vec[i] = args[i + bias]
    bias += 3
    hands_palm_dir[chirality_indx] = curr_vec.copy()
    # print(f"{curr_vec[0]} | {curr_vec[1]} | {curr_vec[2]}")

    for i in range(3):
        curr_vec[i] = args[i + bias]
    bias += 3
    hands_palm_pos_stab[chirality_indx] = curr_vec
    # print(f"{curr_vec[0]} | {curr_vec[1]} | {curr_vec[2]}")
    # print({bias})
    # print(f"{hands_palm_pos[chirality_indx][0]} | {hands_palm_pos[chirality_indx][1]} | {hands_palm_pos[chirality_indx][2]}")
    # print("------------------")


    
    count_size[1] += 1
    if count_size[1] % 300 == 0:
        current_time = datetime.datetime.now()
        print(f"Setting global values: {chirality_indx}, {hands_palm_pos[chirality_indx]}, { hands_palm_vel[chirality_indx]} ---> {count_size[1]} / {current_time}")


dispatcher = Dispatcher()
dispatcher.map("/hands_global", global_handler)
dispatcher.map("/hand_data_projected*", hands_handler)

# control functions
def osc_setup(viz):
    viz.osc_widget.params.a = 0.025    #drag
    viz.osc_widget.params.b = 0.85      #inertia
    viz.osc_widget.params.c = 1.5       #speed_mult
    viz.osc_widget.params.d = 7.5       #max_speed
    viz.osc_widget.params.e = 5.        #psi_mult
    viz.osc_widget.params.f = .66       #y_mult


def osc_control(viz):
    ch_indx = -1
    if hands_vec[0] == True:
        ch_indx = 0
    elif hands_vec[1] == True:
        ch_indx = 1

    # params
    target_speed = 0.05
    target_psi = -.7
    target_latent_y = .25

    # speed_mult = 1.5
    # max_speed = 7.5
    # psi_mult = 5.
    # y_mult = .66
    drag = viz.osc_widget.params.a
    inertia = viz.osc_widget.params.b
    speed_mult = viz.osc_widget.params.c
    max_speed = viz.osc_widget.params.d
    psi_mult = viz.osc_widget.params.e
    y_mult = viz.osc_widget.params.f

    speed = float(viz.latent_widget.latent.speed)
    psi = -float(viz.trunc_noise_widget.trunc_psi)
    latent_y = float(viz.latent_widget.latent.y)


    # update ui
    values = [0., 0., 0., 0.]
    if (ch_indx >= 0):
        # values[0] = pow(min(hands_palm_vel[ch_indx][0], max_speed), 1.5)
        values[0] = min(hands_palm_vel[ch_indx][0], max_speed)
        values[1] = hands_palm_pos[ch_indx][0]
        # values[2] = hands_palm_pos[ch_indx][1]
        values[2] = y_mult * (hands_palm_pos[ch_indx][1] + .5)
        values[3] = hands_palm_pos[ch_indx][2]

    
    viz.osc_widget.val_0 = values[0]
    viz.osc_widget.val_1 = values[1]
    viz.osc_widget.val_2 = values[2]
    viz.osc_widget.val_3 = values[3]

    # update gan
    if (ch_indx >= 0):
        viz.latent_widget.latent.anim = True

        speed = speed_mult * values[0] * (1 - inertia) + speed * inertia
        psi = psi_mult * values[3] * (1 - inertia) + psi * inertia
        # latent_y = y_mult * (values[2] + 1.5)
        # latent_y = y_mult * (values[2] + 1.5) * (1 - inertia) + latent_y * inertia
        latent_y = values[2] * (1 - inertia) + latent_y * inertia

    else:
        latent_y = latent_y * (1 - drag) + target_latent_y * drag
        psi += random.randrange(-1., 1) * .002  ##NO_GUI


    # else:
    target_speed_sign = 1.
    if speed < 0.:
        target_speed_sign = -1.

    speed = speed * (1 - drag) + target_speed_sign * target_speed * drag
    psi = psi * (1 - drag) + target_psi * drag

    
    viz.latent_widget.latent.speed = speed
    viz.trunc_noise_widget.trunc_psi = -psi
    viz.latent_widget.latent.y = latent_y


    # send osc values
    # client.send_message("/interaction/speed", values[0])   # Send float message
    # client.send_message("/interaction/psi", values[3])
    # client.send_message("/interaction/latent_y", values[2])

    # client.send_message("/interaction/speed", (speed + 5.) * .1 )   # Send float message
    client.send_message("/interaction/speed", min(abs(speed), 5.) * .2 + (-psi * .1))   # Send float message
    client.send_message("/interaction/psi", min(-psi * .5, .99))
    client.send_message("/interaction/latent_y", latent_y)


    # data rows of csv file 
    # row = [values[0], values[3], values[2]]     ##NO_GUI
    
    # with open('osc_fake_data.csv', 'a', newline='') as f:
        
    #     # using csv.writer method from CSV package
    #     write = csv.writer(f)
        
    #     write.writerow(row)



async def loop(viz):
    while not viz.should_close():
        osc_control(viz)
        viz.draw_frame()

        for i in range (500):
            await asyncio.sleep(0)



# @click.command()
# @click.argument('pkls', metavar='PATH', nargs=-1)
# @click.option('--capture-dir', help='Where to save screenshot captures', metavar='PATH', default=None)
# @click.option('--browse-dir', help='Specify model path for the \'Browse...\' button', metavar='PATH')
async def main(
    # pkls,
    # capture_dir,
    # browse_dir
):

    browse_dir = None
    capture_dir = "./out"
    pkls = []
    pkls = ["C:/Users/aless/tensor/stylegan3/models/network-snapshot-010990.pkl"]    ##NO_GUI
    
    """Interactive model visualizer.

    Optional PATH argument can be used specify which .pkl file to load.
    """
    viz = Visualizer(capture_dir=capture_dir)
    osc_setup(viz)
    viz.latent_widget.latent.classes = False     ##NO_GUI

    if browse_dir is not None:
        viz.pickle_widget.search_dirs = [browse_dir]

    # List pickles.
    if len(pkls) > 0:
        for pkl in pkls:
            viz.add_recent_pickle(pkl)
        viz.load_pickle(pkls[0])
    else:
        pretrained = [
            "C:/Users/aless/tensor/stylegan3/models/network-snapshot-010990.pkl",
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-256x256.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-metfaces-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-metfacesu-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-afhqv2-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfaces-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqcat-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqdog-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqv2-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqwild-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-brecahad-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-celebahq-256x256.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-cifar10-32x32.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhqu-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhqu-256x256.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-lsundog-256x256.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-metfaces-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-metfacesu-1024x1024.pkl'
        ]

        # Populate recent pickles list with pretrained model URLs.
        for url in pretrained:
            viz.add_recent_pickle(url)

    server = AsyncIOOSCUDPServer((ip, port), dispatcher, asyncio.get_event_loop())
    transport, protocol = await server.create_serve_endpoint()  # Create datagram endpoint and start serving

    # Run.
    # while not viz.should_close():
    #     viz.draw_frame()
    await loop(viz)  # Enter main loop of program

    transport.close()  # Clean up serve endpoint
    viz.close()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
    # main()

#----------------------------------------------------------------------------
