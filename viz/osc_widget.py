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
        self.viz    = viz
        self.listen = True
        self.csv    = False
        self.send_osc    = False

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
            
            _clicked, self.csv  = imgui.checkbox('csv', self.csv)
            imgui.same_line(viz.label_w * 1.5)
            _clicked, self.send_osc  = imgui.checkbox('send osc', self.send_osc)



#----------------------------------------------------------------------------
