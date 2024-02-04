import torch
import torch.nn as nn
from .eformer import EfficientAttention, PositionwiseConv, PositionwiseConv_Cat
import torch.nn.functional as F
import numpy as np


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batchsize, num_channels, vid_len, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, vid_len, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, vid_len, height, width)

    return x

class ResNet_Single_Attn(nn.Module):
    def __init__(self, channels, n_heads):
        super(ResNet_Single_Attn, self).__init__()
        self.ch = channels*2
        self.slf_attn = EfficientAttention(self.ch, self.ch, self.ch, n_heads)
        self.pos_ffn = PositionwiseConv(self.ch, self.ch)
        self.conv = nn.Conv3d(self.ch, channels, kernel_size=1, groups=4)


    def forward(self, x_fused):
        shuffle_data = channel_shuffle(x_fused, 2)
        fused_output = self.slf_attn(shuffle_data)
        ffn_out = self.pos_ffn(fused_output)
        out = self.conv(ffn_out)

        return out

class ResNet_Attn(nn.Module):
    def __init__(self, channels, n_heads):
        super(ResNet_Attn, self).__init__()
        
        self.slf_attnr = EfficientAttention(channels, channels, channels, n_heads)
        self.slf_attnf = EfficientAttention(channels, channels, channels, n_heads)
        self.pos_ffn = PositionwiseConv(channels, channels)
        #self.conv = nn.Conv3d(channels*2, channels, kernel_size=1, groups=4)

    def forward(self, x_rgb, x_flow):
        rgb_output = self.slf_attnr(x_rgb)
        flow_output = self.slf_attnf(x_flow)
        
        add_attn = rgb_output+flow_output
        ffn_out = self.pos_ffn(add_attn)

        return ffn_out

class TransFusion(nn.Module):
    def __init__(self, i3d_rgb, i3d_flow, class_nb=101):
        super(TransFusion, self).__init__()
        
        self.inchannels1 = 192
        self.inchannels2 = 480
        self.inchannels3 = 832
        self.inchannels4 = 1024

        self.rgb_inp = nn.Sequential(i3d_rgb.conv3d_1a_7x7, i3d_rgb.maxPool3d_2a_3x3, i3d_rgb.conv3d_2b_1x1, i3d_rgb.conv3d_2c_3x3, i3d_rgb.maxPool3d_3a_3x3)
        self.flow_inp = nn.Sequential(i3d_flow.conv3d_1a_7x7, i3d_flow.maxPool3d_2a_3x3, i3d_flow.conv3d_2b_1x1, i3d_flow.conv3d_2c_3x3, i3d_flow.maxPool3d_3a_3x3)

        self.rgb_l1 = nn.Sequential(i3d_rgb.mixed_3b, i3d_rgb.mixed_3c, i3d_rgb.maxPool3d_4a_3x3)
        self.rgb_l2 = nn.Sequential(i3d_rgb.mixed_4b, i3d_rgb.mixed_4c, i3d_rgb.mixed_4d, i3d_rgb.mixed_4e, i3d_rgb.mixed_4f, i3d_rgb.maxPool3d_5a_2x2)
        self.rgb_l3 = nn.Sequential(i3d_rgb.mixed_5b, i3d_rgb.mixed_5c)

        self.flow_l1 = nn.Sequential(i3d_flow.mixed_3b, i3d_flow.mixed_3c, i3d_flow.maxPool3d_4a_3x3)
        self.flow_l2 = nn.Sequential(i3d_flow.mixed_4b, i3d_flow.mixed_4c, i3d_flow.mixed_4d, i3d_flow.mixed_4e, i3d_flow.mixed_4f, i3d_flow.maxPool3d_5a_2x2)
        self.flow_l3 = nn.Sequential(i3d_flow.mixed_5b, i3d_flow.mixed_5c)

        #self.att1 = ResNet_Attn(self.inchannels1, n_heads=4)
        #self.att2 = ResNet_Attn(self.inchannels2, n_heads=8)
        #self.att3 = ResNet_Attn(self.inchannels3, n_heads=8)
        self.att4 = ResNet_Attn(self.inchannels4, n_heads=8)
        #self.conv = nn.Conv3d(self.inchannels4*2, self.inchannels4, kernel_size=1)
        #self.cos_sim = nn.CosineSimilarity(dim=1)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        #self.drop = nn.Dropout(p=0.5)
        self.ln = nn.Linear(1024, class_nb)

    def forward(self, x_rgb, x_depth):

        rgb_in = self.rgb_inp(x_rgb)
        depth_in = self.flow_inp(x_depth)   

        #att1 = self.att1(rgb_in,depth_in)

        rgb1 = self.rgb_l1(rgb_in)
        depth1 = self.flow_l1(depth_in)

        #att2 = self.att2(rgb1, depth1)

        rgb2 = self.rgb_l2(rgb1)
        depth2 = self.flow_l2(depth1)

        #att3 = self.att3(rgb2,depth2)
        
        rgb3 = self.rgb_l3(rgb2)
        depth3 = self.flow_l3(depth2)

        att4 = self.att4(rgb3,depth3)

        pool = self.avgpool(att4)

        out = self.ln(torch.flatten(pool,1))

        return out