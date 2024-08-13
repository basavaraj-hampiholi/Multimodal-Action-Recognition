import torch
import torch.nn as nn
from .eformer import EfficientAttention, FeedForwardNet
import torch.nn.functional as F

class CTFB(nn.Module):

    """

    Implementation of Convolutional Transformer Fusion Block (CTFB).
    It computes local attention maps of individual modalities and fuse them with addition operation.
    The fused information is passed to a Feedforward network to compute global features.

    """

    def __init__(self, channels, n_heads):
        super(CTFB, self).__init__()
        
        self.slf_attnr = EfficientAttention(channels, channels, channels, n_heads)
        self.slf_attnf = EfficientAttention(channels, channels, channels, n_heads)
        self.pos_ffn = FeedForwardNet(channels, channels)

    def forward(self, x_rgb, x_flow):

        rgb_output = self.slf_attnr(x_rgb)
        flow_output = self.slf_attnf(x_flow)
        
        add_attn = rgb_output+flow_output 
        ffn_out = self.pos_ffn(add_attn)

        return ffn_out

class ConvTransformerFusion(nn.Module):
    
    """

    Multi-level fusion in two-stream network. It uses pre-trained I3D models for both RGB and Depth videos.
    Pre-trained I3D layers are segregated into four parts (at every maxpool layer).
    CTFBs can placed at each of the four fusion points, but it can differ for each dataset.
    For NVGesture (small dataset), we used only one fusion block at the end.

    """

    def __init__(self, i3d_rgb, i3d_flow, class_nb=101):
        super(ConvTransformerFusion, self).__init__()
        
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

        self.att4 = CTFB(self.inchannels4, n_heads=16)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.ln1 = nn.Linear(1024, class_nb)

    def forward(self, x_rgb, x_flow):

        rgb_in = self.rgb_inp(x_rgb)
        flow_in = self.flow_inp(x_flow)   

        rgb1 = self.rgb_l1(rgb_in)
        flow1 = self.flow_l1(flow_in)
    
        rgb2 = self.rgb_l2(rgb1)
        flow2 = self.flow_l2(flow1)


        rgb3 = self.rgb_l3(rgb2)
        flow3 = self.flow_l3(flow2)
      
        attn4 = self.att4(rgb3, flow3)  
        
        pool = self.avgpool(attn4)  

        flatten = torch.flatten(pool,1)
        out = self.ln1(flatten)

        return out