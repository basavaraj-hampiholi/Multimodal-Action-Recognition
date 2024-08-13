import torch
import torch.nn as nn
from .i3dpt import Unit3Dpy
import torch.nn.functional as F

class EarlyFusion(nn.Module):
    def __init__(self, i3d_mm, class_nb=25):
        super(EarlyFusion, self).__init__()
        
        self.i3d_mm = i3d_mm
        self.n_conv = Unit3Dpy(in_channels=6, out_channels=64, kernel_size=[7, 7, 7],
                                            stride=(2, 2, 2), padding='SAME')       
        self.trained_kernel = self.i3d_mm.conv3d_1a_7x7.conv3d.weight

        for i in range(3):
            self.n_conv.conv3d.weight.data[:, i, :, :, :] = self.trained_kernel.data[:, i, :, :, :]
        self.n_conv.conv3d.weight.data[:,3,:,:,:] = (self.trained_kernel.data[:, 0, :, :, :] + self.trained_kernel.data[:, 1, :, :, :])/2
        self.n_conv.conv3d.weight.data[:,4,:,:,:] = (self.trained_kernel.data[:, 1, :, :, :] + self.trained_kernel.data[:, 2, :, :, :])/2
        self.n_conv.conv3d.weight.data[:,5,:,:,:] = (self.trained_kernel.data[:, 0, :, :, :] + self.trained_kernel.data[:, 2, :, :, :])/2

        self.i3d_mm.conv3d_1a_7x7 = self.n_conv
     
    def forward(self, x_rgb, x_depth):

        rgbd = torch.cat((x_rgb, x_depth), dim=1)

        out = self.i3d_mm(rgbd)    

        return out