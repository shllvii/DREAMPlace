##
# @file   ml_congestion.py
# @author Yibo Lin
# @date   Oct 2022
#

import math
import torch
from torch import nn
from torch.autograd import Function
import matplotlib.pyplot as plt
import pdb
import numpy as np

import dreamplace.ops.rudy.rudy as rudy
import dreamplace.ops.pinrudy.pinrudy as pinrudy
############## Your code block begins here ##############
# import your ML model 
from dreamplace.gpdl import GPDL
from scipy import ndimage

from collections import OrderedDict

class MLCongestion(nn.Module):
    """
    @brief compute congestion map based on a neural network model 
    @param fixed_node_map_op an operator to compute fixed macro map given node positions 
    @param rudy_utilization_map_op an operator to compute RUDY map given node positions
    @param pinrudy_utilization_map_op an operator to compute pin RUDY map given node positions 
    @param pin_pos_op an operator to compute pin positions given node positions 
    @param xl left boundary 
    @param yl bottom boundary 
    @param xh right boundary 
    @param yh top boundary 
    @param num_bins_x #bins in horizontal direction, assume to be the same as horizontal routing grids 
    @param num_bins_y #bins in vertical direction, assume to be the same as vertical routing grids 
    @param unit_horizontal_capacity amount of routing resources in horizontal direction in unit distance
    @param unit_vertical_capacity amount of routing resources in vertical direction in unit distance
    @param pretrained_ml_congestion_weight_file file path for pretrained weights of the machine learning model 
    """
    def __init__(self,
                 fixed_node_map_op,
                 rudy_utilization_map_op, 
                 pinrudy_utilization_map_op, 
                 pin_pos_op, 
                 xl,
                 xh,
                 yl,
                 yh,
                 num_bins_x,
                 num_bins_y,
                 unit_horizontal_capacity,
                 unit_vertical_capacity,
                 pretrained_ml_congestion_weight_file):
        super(MLCongestion, self).__init__()
        ############## Your code block begins here ##############
        self.fixed_node_map_op=fixed_node_map_op
        self.rudy_utilization_map_op=rudy_utilization_map_op
        self.pinrudy_utilization_map_op=pinrudy_utilization_map_op
        self.pin_pos_op = pin_pos_op
        self.xl, self.xh, self.yl, self.yh = xl, xh, yl, yh
        self.num_bins_x=num_bins_x
        self.num_bins_y=num_bins_y
        self.unit_horizontal_capacity=unit_horizontal_capacity
        self.unit_vertical_capacity=unit_vertical_capacity
        self.pretrained_ml_congestion_weight_file=pretrained_ml_congestion_weight_file

        self.gpdl = GPDL()
        self.gpdl.init_weights(pretrained=pretrained_ml_congestion_weight_file)
        self.gpdl.eval()
        
        ############## Your code block ends here ################

    def __call__(self, pos):
        return self.forward(pos)

    def resize(self, input):
        dimension = input.shape
        result = ndimage.zoom(input, (256 / dimension[0], 256 / dimension[1]), order=3)
        return result

    def std(self, input):
        if input.max() == 0:
            return torch.from_numpy(input)
        else:
            result = (input-input.min()) / (input.max()-input.min())
            return torch.from_numpy(result)

    def forward(self, pos):
        ############## Your code block begins here ##############
        
        macro_map = self.fixed_node_map_op(pos)
        rudy_map = self.rudy_utilization_map_op(pos)
        rudy_pin_map = self.pinrudy_utilization_map_op(pos)

        # print(macro_map.shape, rudy_map.shape, rudy_pin_map.shape)
        # print(self.gpdl.state_dict())
        # feature_list = [self.std(self.resize(macro_map.cpu())),
        #            self.std(self.resize(rudy_map.cpu())),
        #            self.std(self.resize(rudy_pin_map.cpu()))]
        #feature_map = np.transpos(np.array(feature_list), (1, 2, 0)) 
        #feature_map = np.array(feature_list)
        feature_list = [macro_map, rudy_map, rudy_pin_map]
        feature_map = torch.stack(feature_list, dim=0).unsqueeze(0)
        self.gpdl.to(feature_map.device)
        cong_map = torch.squeeze(self.gpdl(feature_map)).add_(1)
        return cong_map
        #return self.gpdl(feature_map)
        #return None
        ############## Your code block ends here ################
