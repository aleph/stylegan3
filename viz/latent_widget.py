# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from itertools import count
import numpy as np
import imgui
import dnnlib
from gui_utils import imgui_utils

#----------------------------------------------------------------------------

class LatentWidget:
    def __init__(self, viz):
        self.viz        = viz
        self.latent     = dnnlib.EasyDict(x=0, y=0, anim=False, speed=0.25, classes=False, use_list=False)
        self.latent_def = dnnlib.EasyDict(self.latent)
        self.step_y     = 100
        self.counter    = 0
        self.seeds      = []
        self.populate_seeds()

    def populate_seeds(self):
        self.seeds =    [[1003, 1016, 1005, 1015, 1020, 1038, 1023, 1022, 1001, 1031, 1017, 1009, 1007], 
                        [255, 329, 556, 1066, 220, 951, 1010, 1006, 275, 300, 1061, 1044, 1063]]

    def drag(self, dx, dy):
        viz = self.viz
        self.latent.x += dx / viz.font_size * 4e-2
        self.latent.y += dy / viz.font_size * 4e-2

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            imgui.text('Latent')
            imgui.same_line(viz.label_w)
            seed = round(self.latent.x) + round(self.latent.y) * self.step_y
            with imgui_utils.item_width(viz.font_size * 8):
                changed, seed = imgui.input_int('##seed', seed)
                if changed:
                    self.latent.x = seed
                    self.latent.y = 0
            imgui.same_line(viz.label_w + viz.font_size * 8 + viz.spacing)
            frac_x = self.latent.x - round(self.latent.x)
            frac_y = self.latent.y - round(self.latent.y)
            with imgui_utils.item_width(viz.font_size * 5):
                changed, (new_frac_x, new_frac_y) = imgui.input_float2('##frac', frac_x, frac_y, format='%+.2f', flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                if changed:
                    self.latent.x += new_frac_x - frac_x
                    self.latent.y += new_frac_y - frac_y
            imgui.same_line(viz.label_w + viz.font_size * 13 + viz.spacing * 2)
            _clicked, dragging, dx, dy = imgui_utils.drag_button('Drag', width=viz.button_w)
            if dragging:
                self.drag(dx, dy)
            imgui.same_line(viz.label_w + viz.font_size * 13 + viz.button_w + viz.spacing * 3)
            _clicked, self.latent.anim = imgui.checkbox('Anim', self.latent.anim)
            imgui.same_line(round(viz.font_size * 27.7))
            with imgui_utils.item_width(-1 - viz.button_w * 2 - viz.spacing * 2), imgui_utils.grayed_out(not self.latent.anim):
                changed, speed = imgui.slider_float('##speed', self.latent.speed, -5, 5, format='Speed %.3f', power=3)
                if changed:
                    self.latent.speed = speed
            imgui.same_line()
            # snapped = dnnlib.EasyDict(self.latent, x=round(self.latent.x), y=round(self.latent.y))
            # if imgui_utils.button('Snap', width=viz.button_w, enabled=(self.latent != snapped)):
            #     self.latent = snapped
            _clicked, self.latent.classes = imgui.checkbox('Classes', self.latent.classes)
            _clicked, self.latent.use_list = imgui.checkbox('List', self.latent.use_list)
            imgui.same_line()
            if imgui_utils.button('Reset', width=-1, enabled=(self.latent != self.latent_def)):
                self.latent = dnnlib.EasyDict(self.latent_def)


        if self.latent.anim:
            self.latent.x += viz.frame_delta * self.latent.speed


        if self.latent.classes:
            seed_list = self.seeds
            
            seed_plane = 0
            if self.latent.y > .66:
                seed_plane = 1

            next_plane = min(seed_plane + 1, 1)

            frac_h = 1.
            if self.latent.y > .33 and self.latent.y < .66:
                frac_h = 1. - ((self.latent.y - .33) * 3.)


            last_val = np.floor(self.latent.x)
            frac_val = 1 - abs(self.latent.x - np.trunc(self.latent.x))
            if self.latent.x < 0.:
                frac_val = 1 - frac_val
            next_val = np.ceil(self.latent.x)

            last_item = int(last_val)%len(seed_list[seed_plane])
            next_item = int(next_val)%len(seed_list[seed_plane])

            viz.args.w0_seeds = [[seed_list[seed_plane][last_item], frac_val * frac_h], [seed_list[seed_plane][next_item], (1. - frac_val) * frac_h]] # [[seed, weight], ...]
            viz.args.w0_seeds.append([seed_list[next_plane][last_item], frac_val * (1 - frac_h)]) # [[seed, weight], ...]
            viz.args.w0_seeds.append([seed_list[next_plane][next_item], (1. - frac_val) * (1 - frac_h)]) # [[seed, weight], ...]


            # viz.args.w0_seeds = [[seed_list[seed_plane][last_item], frac_val], [seed_list[seed_plane][next_item], (1. - frac_val)]] # [[seed, weight], ...]
            # print(f"self.latent.x: {self.latent.x} | last_val: {last_val} | next_val: {next_val} | [{seed_list[seed_plane][last_item]}, {frac_val}], [{seed_list[seed_plane][next_item]}, {(1. - frac_val)}]")
            # print(f"self.latent.x: {self.latent.y} | seed_plane: {seed_plane} | next_plane: {next_plane} | [{seed_list[next_plane][last_item]}, {frac_h}], [{seed_list[next_plane][next_item]}, {(1 - frac_h)}]")
            # print(f"-------------")


        elif self.latent.use_list:
            seed_list = self.seeds
            
            seed_plane = 0

            next_plane = min(seed_plane + 1, 1)

            frac_h = 1.
            # if self.latent.y > .33 and self.latent.y < .66:
            #     frac_h = 1. - ((self.latent.y - .33) * 3.)
            # frac_h = 1. - self.latent.y


            last_val = np.floor(self.latent.x)
            frac_val = 1 - abs(self.latent.x - np.trunc(self.latent.x))
            if self.latent.x < 0.:
                frac_val = 1 - frac_val
            next_val = np.ceil(self.latent.x)

            last_item = int(last_val)%len(seed_list[seed_plane])
            next_item = int(next_val)%len(seed_list[seed_plane])

            viz.args.w0_seeds = [[seed_list[seed_plane][last_item], frac_val * frac_h], [seed_list[seed_plane][next_item], (1. - frac_val) * frac_h]] # [[seed, weight], ...]
            viz.args.w0_seeds.append([seed_list[next_plane][last_item], frac_val * (1 - frac_h)]) # [[seed, weight], ...]
            viz.args.w0_seeds.append([seed_list[next_plane][next_item], (1. - frac_val) * (1 - frac_h)]) # [[seed, weight], ...]


        else:
            viz.args.w0_seeds = [] # [[seed, weight], ...]
            for ofs_x, ofs_y in [[0, 0], [1, 0], [0, 1], [1, 1]]:
                seed_x = np.floor(self.latent.x) + ofs_x
                seed_y = np.floor(self.latent.y) + ofs_y
                seed = (int(seed_x) + int(seed_y) * self.step_y) & ((1 << 32) - 1)
                weight = (1 - abs(self.latent.x - seed_x)) * (1 - abs(self.latent.y - seed_y))
                if weight > 0:
                    viz.args.w0_seeds.append([seed, weight])

#----------------------------------------------------------------------------
