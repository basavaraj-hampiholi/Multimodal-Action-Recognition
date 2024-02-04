import torch
import torch.nn as nn
from .qkv_eformer import EfficientAttention, PositionwiseConv
import torch.nn.functional as F

class ResNet_Attn(nn.Module):
    def __init__(self, channels, n_heads):
        super(ResNet_Attn, self).__init__()
        self.slf_attn = EfficientAttention(channels, channels, channels, n_heads)
        self.pos_ffn = PositionwiseConv(channels, channels)
        self.conv = nn.Conv3d(channels*2, channels, kernel_size=1)

    def forward(self, x_rgb, x_flow):
        attn_output = self.slf_attn(x_rgb, x_flow)
        ffn_out = self.pos_ffn(attn_output)
        out = self.conv(ffn_out)
        
        return out

class TransFusion(nn.Module):
    def __init__(self, i3d_rgb, i3d_flow, class_nb=101):
        super(TransFusion, self).__init__()
        self.inchannels1 = 192
        self.inchannels2 = 480
        self.inchannels3 = 832
        self.inchannels4 = 1024

        self.rgb_inp = nn.Sequential(i3d_rgb.Conv3d_1a_7x7, i3d_rgb.MaxPool3d_2a_3x3, i3d_rgb.Conv3d_2b_1x1, i3d_rgb.Conv3d_2c_3x3, i3d_rgb.MaxPool3d_3a_3x3)
        self.flow_inp = nn.Sequential(i3d_flow.Conv3d_1a_7x7, i3d_flow.MaxPool3d_2a_3x3, i3d_flow.Conv3d_2b_1x1, i3d_flow.Conv3d_2c_3x3, i3d_flow.MaxPool3d_3a_3x3)

        self.rgb_l1 = nn.Sequential(i3d_rgb.Mixed_3b, i3d_rgb.Mixed_3c, i3d_rgb.MaxPool3d_4a_3x3)
        self.rgb_l2 = nn.Sequential(i3d_rgb.Mixed_4b, i3d_rgb.Mixed_4c, i3d_rgb.Mixed_4d, i3d_rgb.Mixed_4e, i3d_rgb.Mixed_4f, i3d_rgb.MaxPool3d_5a_2x2)
        self.rgb_l3 = nn.Sequential(i3d_rgb.Mixed_5b, i3d_rgb.Mixed_5c)
        self.rgb_logits = i3d_rgb.logits

        self.flow_l1 = nn.Sequential(i3d_flow.Mixed_3b, i3d_flow.Mixed_3c, i3d_flow.MaxPool3d_4a_3x3)
        self.flow_l2 = nn.Sequential(i3d_flow.Mixed_4b, i3d_flow.Mixed_4c, i3d_flow.Mixed_4d, i3d_flow.Mixed_4e, i3d_flow.Mixed_4f, i3d_flow.MaxPool3d_5a_2x2)
        self.flow_l3 = nn.Sequential(i3d_flow.Mixed_5b, i3d_flow.Mixed_5c)
        self.flows_logits = i3d_flow.logits

        self.att_in = ResNet_Attn(self.inchannels1, n_heads=4)
        self.att1 = ResNet_Attn(self.inchannels2, n_heads=8)
        self.att2 = ResNet_Attn(self.inchannels3, n_heads=16)
        self.att3 = ResNet_Attn(self.inchannels4, n_heads=16)

        #self.conv1 = nn.Conv3d(self.inchannels1*2, self.inchannels1, kernel_size=1)
        #self.conv2 = nn.Conv3d(self.inchannels2*2, self.inchannels2, kernel_size=1)
        #self.conv3 = nn.Conv3d(self.inchannels3*2, self.inchannels3, kernel_size=1)
        #self.conv4 = nn.Conv3d(self.inchannels4*2, self.inchannels4, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.drop = nn.Dropout(p=0.5)
        self.ln1 = nn.Linear(1024, class_nb)

    def forward(self, x_rgb, x_flow):

        rgb_in = self.rgb_inp(x_rgb)
        flow_in = self.flow_inp(x_flow)   

        att_in = self.att_in(rgb_in, flow_in)

        rgb1_fuse = rgb_in+att_in
        flow1_fuse = flow_in+att_in
        rgb1 = self.rgb_l1(rgb1_fuse)
        flow1 = self.flow_l1(flow1_fuse)

        att1 = self.att1(rgb1, flow1)

        rgb2_fuse = rgb1+att1
        flow2_fuse = flow1+att1
        rgb2 = self.rgb_l2(rgb2_fuse)
        flow2 = self.flow_l2(flow2_fuse)

        att2 = self.att2(rgb2, flow2)
        
        rgb3_fuse = rgb2+att2
        flow3_fuse = flow2+att2
        rgb3 = self.rgb_l3(rgb3_fuse)
        flow3 = self.flow_l3(flow3_fuse)
      
        att3 = self.att3(rgb3, flow3)
        
        pool = self.avgpool(att3)
        x_flatten = torch.flatten(pool,1)
        x_flatten = self.drop(x_flatten)

        out = self.ln1(x_flatten)
        return out