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

import pandas as pd
from pythonosc.udp_client import SimpleUDPClient


#----------------------------------------------------------------------------

class Visualizer(imgui_window.ImguiWindow):
    def __init__(self, capture_dir=None):
        super().__init__(title='GAN Visualizer', window_width=3840, window_height=2160)

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
        self.set_position(0, 0)
        self._adjust_font_size()
        self.skip_frame() # Layout may change after first frame.


        # Import data
        self.data = None
        self.counter = 0
        self.client = None


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
        self.pane_w = self.font_size * 45
        self.button_w = self.font_size * 5
        self.label_w = round(self.font_size * 4.5)

        # Detect mouse dragging in the result area.
        dragging, dx, dy = imgui_utils.drag_hidden_window('##result_area', x=self.pane_w, y=0, width=self.content_width-self.pane_w, height=self.content_height)
        if dragging:
            self.latent_widget.drag(dx, dy)

        # Begin control pane.
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(self.pane_w, self.content_height)
        imgui.begin('##control_pane', closable=False, flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE))

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

        # Display.
        max_w = self.content_width - self.pane_w
        max_h = self.content_height
        pos = np.array([self.pane_w + max_w / 2, max_h / 2])
        if 'image' in self.result:
            if self._tex_img is not self.result.image:
                self._tex_img = self.result.image
                if self._tex_obj is None or not self._tex_obj.is_compatible(image=self._tex_img):
                    self._tex_obj = gl_utils.Texture(image=self._tex_img, bilinear=False, mipmap=False)
                else:
                    self._tex_obj.update(self._tex_img)
            zoom = min(max_w / self._tex_obj.width, max_h / self._tex_obj.height)
            zoom = np.floor(zoom) if zoom >= 1 else zoom
            self._tex_obj.draw(pos=pos, zoom=zoom, align=0.5, rint=True)
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

        self.counter += 1

#----------------------------------------------------------------------------

    # control functions
    def interpolate_data(self, x_remap, row):
        next_pos = -1.
        next_idx = -1
        while(x_remap > next_pos):

            next_idx += 1
            next_pos = self.data[next_idx][row]

            if (next_pos > 1.):
                next_idx -= 1
                next_pos = self.data[next_idx][row]
                break


        prev_id = next_idx - 1

        if prev_id < 0:
            prev_id = len(self.data.iloc[0]) - 2

        prev_pos = self.data[prev_id][row]
        next_val = self.data[next_idx][row + 1]
        prev_val = self.data[prev_id][row + 1]

        curr_pos = (x_remap - prev_pos) / (next_pos - prev_pos)
        if prev_pos > next_pos:
            (x_remap + 1. - prev_pos) / (next_pos + 1. - prev_pos)

        curr_val = prev_val * (1. - curr_pos) + next_val * curr_pos

        return curr_val


    def osc_setup(self):
        # osc sender setup
        ip_send = "127.0.0.1"
        # ip_send = "192.168.1.122"
        port_send = 7400

        self.client = SimpleUDPClient(ip_send, port_send)  # Create client


    def csv_setup(self):
        self.osc_widget.params.a = 0.85     #inertia
        self.osc_widget.params.b = .5       #speed
        self.osc_widget.params.c = .6       #psi
        self.osc_widget.params.d = 0.025    #y
        self.osc_widget.params.e = .5       #xxx
        self.osc_widget.params.f = .66      #xxx

        self.latent_widget.latent.anim = True
        self.latent_widget.latent.use_list = True
        self.osc_widget.csv = True

        if self.data is not None:
            seeds_data = pd.read_csv(r" _in\_gen\seeds_5.csv", header=None)
            self.latent_widget.seeds = []

            for i in range(2):
                row = []
                for j in range(len(seeds_data.iloc[0])):
                    row.append(seeds_data[j][i])
                
                print(row)
                self.latent_widget.seeds.append(row)


    def csv_control(self):

        # data
        if self.counter % 60 == 0 and self.osc_widget.csv:
            # self.data = pd.read_csv(r"C:\Users\aless\Documents\__Fuse\Orto Botanico Padova\Visual\_gen\params.csv", header=None)
            self.data = pd.read_csv(r"_in\_gen\params_2b.csv", header=None)
            print(self.data)

        data_len = 1
        if (self.latent_widget.latent.use_list and (self.data is not None)):
            data_len = len(self.latent_widget.seeds[0])

        x_remap = abs(self.latent_widget.latent.x - np.trunc(self.latent_widget.latent.x))
        if (data_len > 1):
            x_val = self.latent_widget.latent.x % data_len
            x_remap = x_val / data_len


        self.osc_widget.values.a = self.latent_widget.latent.x % data_len
        self.osc_widget.values.b = x_remap
        self.osc_widget.values.c = self.latent_widget.latent.y


        if (self.data is not None) and (self.osc_widget.csv):
            
            self.osc_widget.params.b = self.interpolate_data(x_remap, 0)
            self.osc_widget.params.c = self.interpolate_data(x_remap, 2)


        # update values
        inertia = self.osc_widget.params.a
        speed_mult = self.osc_widget.params.b
        psi_mult = self.osc_widget.params.c
        y_mult = self.osc_widget.params.d

        speed = float(self.latent_widget.latent.speed)
        psi = float(self.trunc_noise_widget.trunc_psi)
        latent_y = float(self.latent_widget.latent.y)

        # speed_dist = float(self.osc_widget.params.e)
        # speed_mult = speed_mult * (1. - speed_dist) + speed_mult / (.001 + abs(psi))


        # update gan
        speed = speed_mult * (1 - inertia) + speed * inertia
        psi = psi_mult * (1 - inertia) + psi * inertia
        latent_y = y_mult * (1 - inertia) + latent_y * inertia

        
        self.latent_widget.latent.speed = speed
        self.trunc_noise_widget.trunc_psi = psi
        self.latent_widget.latent.y = latent_y


        # send osc
        if (self.client is not None and self.osc_widget.send_osc):
            self.client.send_message("/interaction/speed", speed)
            if self.counter % 2 == 0:
                self.client.send_message("/interaction/psi", abs(psi))
        



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



@click.command()
@click.argument('pkls', metavar='PATH', nargs=-1)
@click.option('--capture-dir', help='Where to save screenshot captures', metavar='PATH', default=None)
@click.option('--browse-dir', help='Specify model path for the \'Browse...\' button', metavar='PATH')
def main(
    pkls,
    capture_dir,
    browse_dir
):
    """Interactive model visualizer.

    Optional PATH argument can be used specify which .pkl file to load.
    """
    viz = Visualizer(capture_dir=capture_dir)
    viz.osc_setup()
    viz.csv_setup()

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
            "C:/Users/aless/tensor/stylegan3/models/00024-stylegan2-incisioni-h3_m-1024x1024-gpus1-batch32-gamma6.6/network-snapshot-000896.pkl",
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

    # Run.
    while not viz.should_close():
        viz.csv_control()
        viz.draw_frame()
    viz.close()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
