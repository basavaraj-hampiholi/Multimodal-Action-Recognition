import torch
import torch.nn as nn
from .eformer import EfficientAttention, PositionwiseConv
import torch.nn.functional as F

class ResNet_Attn(nn.Module):
    def __init__(self, channels, n_heads):
        super(ResNet_Attn, self).__init__()
        
        self.slf_attnr = EfficientAttention(channels, channels, channels, n_heads)
        self.slf_attnf = EfficientAttention(channels, channels, channels, n_heads)
        self.pos_ffn = PositionwiseConv(channels, channels)
        #self.conv = nn.Conv3d(channels*2, channels, kernel_size=1)

    def forward(self, x_rgb, x_flow):
        rgb_output = self.slf_attnr(x_rgb)
        flow_output = self.slf_attnf(x_flow)
        
        #cat_attn = torch.cat((rgb_output, flow_output), dim=1)
        add_attn = rgb_output+flow_output
        ffn_out = self.pos_ffn(add_attn)
        #out = self.conv(ffn_out)

        return ffn_out

class TransFusion(nn.Module):
    def __init__(self, i3d_rgb, i3d_flow, i3d_depth, class_nb=101):
        super(TransFusion, self).__init__()
        
        self.inchannels1 = 192
        self.inchannels2 = 480
        self.inchannels3 = 832
        self.inchannels4 = 1024

        self.rgb_inp = nn.Sequential(i3d_rgb.conv3d_1a_7x7, i3d_rgb.maxPool3d_2a_3x3, i3d_rgb.conv3d_2b_1x1, i3d_rgb.conv3d_2c_3x3, i3d_rgb.maxPool3d_3a_3x3)
        self.flow_inp = nn.Sequential(i3d_flow.conv3d_1a_7x7, i3d_flow.maxPool3d_2a_3x3, i3d_flow.conv3d_2b_1x1, i3d_flow.conv3d_2c_3x3, i3d_flow.maxPool3d_3a_3x3)
        self.depth_inp = nn.Sequential(i3d_depth.conv3d_1a_7x7, i3d_depth.maxPool3d_2a_3x3, i3d_depth.conv3d_2b_1x1, i3d_depth.conv3d_2c_3x3, i3d_depth.maxPool3d_3a_3x3)

        self.rgb_l1 = nn.Sequential(i3d_rgb.mixed_3b, i3d_rgb.mixed_3c, i3d_rgb.maxPool3d_4a_3x3)
        self.rgb_l2 = nn.Sequential(i3d_rgb.mixed_4b, i3d_rgb.mixed_4c, i3d_rgb.mixed_4d, i3d_rgb.mixed_4e, i3d_rgb.mixed_4f, i3d_rgb.maxPool3d_5a_2x2)
        self.rgb_l3 = nn.Sequential(i3d_rgb.mixed_5b, i3d_rgb.mixed_5c)

        self.flow_l1 = nn.Sequential(i3d_flow.mixed_3b, i3d_flow.mixed_3c, i3d_flow.maxPool3d_4a_3x3)
        self.flow_l2 = nn.Sequential(i3d_flow.mixed_4b, i3d_flow.mixed_4c, i3d_flow.mixed_4d, i3d_flow.mixed_4e, i3d_flow.mixed_4f, i3d_flow.maxPool3d_5a_2x2)
        self.flow_l3 = nn.Sequential(i3d_flow.mixed_5b, i3d_flow.mixed_5c)

        self.depth_l1 = nn.Sequential(i3d_depth.mixed_3b, i3d_depth.mixed_3c, i3d_depth.maxPool3d_4a_3x3)
        self.depth_l2 = nn.Sequential(i3d_depth.mixed_4b, i3d_depth.mixed_4c, i3d_depth.mixed_4d, i3d_depth.mixed_4e, i3d_depth.mixed_4f, i3d_depth.maxPool3d_5a_2x2)
        self.depth_l3 = nn.Sequential(i3d_depth.mixed_5b, i3d_depth.mixed_5c)

        #self.att1 = ResNet_Attn(self.inchannels1, n_heads=3)
        #self.att2 = ResNet_Attn(self.inchannels2, n_heads=8)
        #self.att3 = ResNet_Attn(self.inchannels3, n_heads=13)
        self.att4 = ResNet_Attn(self.inchannels4, n_heads=16)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.drop = nn.Dropout(p=0.5)
        self.ln1 = nn.Linear(1024, class_nb)

    def forward(self, x_rgb, x_flow):

        rgb_in = self.rgb_inp(x_rgb)
        flow_in = self.flow_inp(x_flow)   

        #att1 = self.att1(rgb_in, flow_in)

        #rgb1_fuse = rgb_in+att1
        #flow1_fuse = flow_in+att1
        rgb1 = self.rgb_l1(rgb_in)
        flow1 = self.flow_l1(flow_in)

        #att2 = self.att2(rgb1, flow1)

        #rgb2_fuse = rgb1+att2
        #flow2_fuse = flow1+att2
        rgb2 = self.rgb_l2(rgb1)
        flow2 = self.flow_l2(flow1)

        #att3 = self.att3(rgb2, flow2)
        
        #rgb3_fuse = rgb2+att3
        #flow3_fuse = flow2+att3
        rgb3 = self.rgb_l3(rgb2)
        flow3 = self.flow_l3(flow2)
      
        att4 = self.att4(rgb3, flow3)
        
        pool = self.avgpool(att4)
        x_flatten = torch.flatten(pool,1)

        out = self.ln1(x_flatten)
        
        return out