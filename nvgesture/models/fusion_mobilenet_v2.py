import torch
import torch.nn as nn
from .eformer import EfficientAttention, PositionwiseConv, PositionwiseConv_Cat
import torch.nn.functional as F
import numpy as np


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
        
        self.inchannels1 = 24
        self.inchannels2 = 64
        self.inchannels3 = 96
        self.inchannels4 = 160
        self.inchannels5 = 1280

        self.rgb_l1 = i3d_rgb.module.features[0:4]
        self.rgb_l2 = i3d_rgb.module.features[4:8]
        self.rgb_l3 = i3d_rgb.module.features[8:12]
        self.rgb_l4 = i3d_rgb.module.features[12:16]
        self.rgb_l5 = i3d_rgb.module.features[17:19]


        self.flow_l1 = i3d_flow.module.features[0:4]
        self.flow_l2 = i3d_flow.module.features[4:8]
        self.flow_l3 = i3d_flow.module.features[8:12]
        self.flow_l4 = i3d_flow.module.features[12:16]
        self.flow_l5 = i3d_flow.module.features[17:19]


        self.att1 = ResNet_Attn(self.inchannels1, n_heads=2)
        self.att2 = ResNet_Attn(self.inchannels2, n_heads=4)
        self.att3 = ResNet_Attn(self.inchannels3, n_heads=4)
        self.att4 = ResNet_Attn(self.inchannels4, n_heads=8)
        self.att5 = ResNet_Attn(self.inchannels5, n_heads=16)
        

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        #self.drop = nn.Dropout(p=0.5)
        self.ln = nn.Linear(1280, class_nb)

    def forward(self, x_rgb, x_depth):

        rgb1 = self.rgb_l1(x_rgb)
        depth1 = self.flow_l1(x_depth)
        att1 = self.att1(rgb1, depth1)

        rgb2 = self.rgb_l2(rgb1+att1)
        depth2 = self.flow_l2(depth1+att1)
        att2 = self.att2(rgb2, depth2)
        
        rgb3 = self.rgb_l3(rgb2+att2)
        depth3 = self.flow_l3(depth2+att2)
        att3 = self.att3(rgb3, depth3)

        rgb4 = self.rgb_l4(rgb3+att3)
        depth4 = self.flow_l4(depth3+att3)
        att4 = self.att4(rgb4, depth4) 

        rgb5 = self.rgb_l5(rgb4+att4)
        depth5 = self.flow_l5(depth4+att4)

        att5 = self.att5(rgb5, depth5) 
        
        #cat_feats = torch.cat((rgb3, depth3), dim=1)
        #fused_feats = self.conv(cat_feats)         

        pool = self.avgpool(att5)
        flatten = torch.flatten(pool,1)

        out = self.ln(flatten)

        return out