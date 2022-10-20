# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import imgui
import dnnlib
from gui_utils import imgui_utils

#----------------------------------------------------------------------------

class OscWidget:

    
    def __init__(self, viz):
        self.viz            = viz
        self.listen = True

        self.values = dnnlib.EasyDict(a=0., b=0., c=0., d=0., e=0., f=0.)
        self.params = dnnlib.EasyDict(a=0., b=0., c=0., d=0., e=0., f=0.)
        self.values_def = dnnlib.EasyDict(self.values)
        self.params_def = dnnlib.EasyDict(self.params)

        # self.val_0 = 0.
        # self.val_1 = 0.
        # self.val_2 = 0.
        # self.val_3 = 0.
        # self.param_0 = 0.
        # self.param_1 = 0.
        # self.param_2 = 0.
        # self.param_3 = 0.


    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        
        if show:
            imgui.text('osc_values: ')          
            with imgui_utils.item_width(viz.font_size * 6):
                imgui.same_line(viz.label_w * 1.5)
                _changed, (self.values.a) = imgui.input_float('##value_0', self.values.a, format='%.3f')
                imgui.same_line()
                _changed, (self.values.b) = imgui.input_float('##value_1', self.values.b, format='%.3f')
                imgui.same_line()
                _changed, (self.values.c) = imgui.input_float('##value_2', self.values.c, format='%.3f')
                imgui.same_line()
                _changed, (self.values.d) = imgui.input_float('##value_3', self.values.d, format='%.3f')
                imgui.same_line()
                _changed, (self.values.e) = imgui.input_float('##value_4', self.values.e, format='%.3f')
                imgui.same_line()
                _changed, (self.values.f) = imgui.input_float('##value_5', self.values.f, format='%.3f')
           
            imgui.text('osc_params: ')          
            with imgui_utils.item_width(viz.font_size * 6):                
                imgui.same_line(viz.label_w * 1.5)
                _changed, self.params.a = imgui.input_float('##param_0', self.params.a, format='%.3f')
                imgui.same_line()
                _changed, self.params.b = imgui.input_float('##param_1', self.params.b, format='%.3f')
                imgui.same_line()
                _changed, self.params.c = imgui.input_float('##param_2', self.params.c, format='%.3f')
                imgui.same_line()
                _changed, self.params.d = imgui.input_float('##param_3', self.params.d, format='%.3f')
                imgui.same_line()
                _changed, self.params.e = imgui.input_float('##param_4', self.params.e, format='%.3f')
                imgui.same_line()
                _changed, self.params.f = imgui.input_float('##param_5', self.params.f, format='%.3f')
                # _changed, (self.val_0, self.val_1, self.val_2, self.val_3, self.val_4, self.val_5) = imgui.input_float('##values', self.val_0, self.val_1, self.val_2, self.val_3, self.val_4, self.val_5, format='%.3f')

        # if show:
        #     imgui.text('Translate')
        #     imgui.same_line(viz.label_w)
        #     with imgui_utils.item_width(viz.font_size * 8):
        #         _changed, (self.xlate.x, self.xlate.y) = imgui.input_float2('##xlate', self.xlate.x, self.xlate.y, format='%.4f')
        #     imgui.same_line(viz.label_w + viz.font_size * 8 + viz.spacing)
        #     _clicked, dragging, dx, dy = imgui_utils.drag_button('Drag fast##xlate', width=viz.button_w)
        #     if dragging:
        #         self.xlate.x += dx / viz.font_size * 2e-2
        #         self.xlate.y += dy / viz.font_size * 2e-2
        #     imgui.same_line()
        #     _clicked, dragging, dx, dy = imgui_utils.drag_button('Drag slow##xlate', width=viz.button_w)
        #     if dragging:
        #         self.xlate.x += dx / viz.font_size * 4e-4
        #         self.xlate.y += dy / viz.font_size * 4e-4
        #     imgui.same_line()
        #     _clicked, self.xlate.anim = imgui.checkbox('Anim##xlate', self.xlate.anim)
        #     imgui.same_line()
        #     _clicked, self.xlate.round = imgui.checkbox('Round##xlate', self.xlate.round)
        #     imgui.same_line()
        #     with imgui_utils.item_width(-1 - viz.button_w - viz.spacing), imgui_utils.grayed_out(not self.xlate.anim):
        #         changed, speed = imgui.slider_float('##xlate_speed', self.xlate.speed, 0, 0.5, format='Speed %.5f', power=5)
        #         if changed:
        #             self.xlate.speed = speed
        #     imgui.same_line()
        #     if imgui_utils.button('Reset##xlate', width=-1, enabled=(self.xlate != self.xlate_def)):
        #         self.xlate = dnnlib.EasyDict(self.xlate_def)

        # if show:
            # imgui.text('paper')
            # imgui.same_line(viz.label_w)
            # with imgui_utils.item_width(viz.font_size * 8):
            #     _changed, self.rotate.val = imgui.input_float('##rotate', self.rotate.val, format='%.4f')
            # imgui.same_line(viz.label_w + viz.font_size * 8 + viz.spacing)
            # _clicked, dragging, dx, _dy = imgui_utils.drag_button('Drag fast##rotate', width=viz.button_w)
            # if dragging:
            #     self.rotate.val += dx / viz.font_size * 2e-2
            # imgui.same_line()
            # _clicked, dragging, dx, _dy = imgui_utils.drag_button('Drag slow##rotate', width=viz.button_w)
            # if dragging:
            #     self.rotate.val += dx / viz.font_size * 4e-4
            # imgui.same_line()
            # _clicked, self.rotate.anim = imgui.checkbox('Anim##rotate', self.rotate.anim)
            # imgui.same_line()
            # with imgui_utils.item_width(-1 - viz.button_w - viz.spacing), imgui_utils.grayed_out(not self.rotate.anim):
            #     changed, speed = imgui.slider_float('##rotate_speed', self.rotate.speed, -1, 1, format='Speed %.4f', power=3)
            #     if changed:
            #         self.rotate.speed = speed
            # imgui.same_line()
            # if imgui_utils.button('Reset##rotate', width=-1, enabled=(self.rotate != self.rotate_def)):
            #     self.rotate = dnnlib.EasyDict(self.rotate_def)

        # if self.xlate.anim:
        #     c = np.array([self.xlate.x, self.xlate.y], dtype=np.float64)
        #     t = c.copy()
        #     if np.max(np.abs(t)) < 1e-4:
        #         t += 1
        #     t *= 0.1 / np.hypot(*t)
        #     t += c[::-1] * [1, -1]
        #     d = t - c
        #     d *= (viz.frame_delta * self.xlate.speed) / np.hypot(*d)
        #     self.xlate.x += d[0]
        #     self.xlate.y += d[1]

        # if self.rotate.anim:
        #     self.rotate.val += viz.frame_delta * self.rotate.speed

        # pos = np.array([self.xlate.x, self.xlate.y], dtype=np.float64)
        # if self.xlate.round and 'img_resolution' in viz.result:
        #     pos = np.rint(pos * viz.result.img_resolution) / viz.result.img_resolution
        # angle = self.rotate.val * np.pi * 2

        # viz.args.input_transform = [
        #     [np.cos(angle),  np.sin(angle), pos[0]],
        #     [-np.sin(angle), np.cos(angle), pos[1]],
        #     [0, 0, 1]]

        # viz.args.update(untransform=self.opts.untransform)

#----------------------------------------------------------------------------
