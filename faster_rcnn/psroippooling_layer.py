import os
import sys
import torch
from torch.autograd import Function

class RSPooling(Function):
    def __init__(self, spatial_scale=0.0625, output_dim=21, group_size=7):
        super(RSPooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.output_dim = output_dim
        self.group_size = group_size
        self.pooled_height_ = group_size
        self.pooled_width_ = group_size
        self.output = None

    def forward(self, features, rois):
        num_rois = rois.size()[0]
        rois[:, 1:] = rois[:, 1:] * self.spatial_scale  
        output = torch.zeros(num_rois, self.output_dim, self.pooled_height_, self.pooled_width_)
        mappingchannel = torch.IntTensor(num_rois, self.output_dim, self.pooled_height_, self.pooled_width_).zero_()
        output = output.cuda(); mappingchannel = mappingchannel.cuda()
          
    def backward(self, grad_output):
        
