import torch
import torch.nn as nn
from .eformer import EfficientAttention, PositionwiseConv
import torch.nn.functional as F

class ResNet_Attn(nn.Module):
    def __init__(self, channels, n_heads):
        super(ResNet_Attn, self).__init__()

        self.slf_attnr = EfficientAttention(channels, channels, channels, n_heads)
        self.slf_attnf = EfficientAttention(channels, channels, channels, n_heads)
        self.pos_ffnr = PositionwiseConv(channels, channels)
        self.pos_ffnf = PositionwiseConv(channels, channels)

    def forward(self, x_rgb, x_flow):
        rgb_output = self.slf_attnr(x_rgb)
        flow_output = self.slf_attnf(x_flow)
        
        ffn_outr = self.pos_ffnr(rgb_output)
        ffn_outf = self.pos_ffnf(flow_output)

        return ffn_outr, ffn_outf

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

        self.att1 = ResNet_Attn(self.inchannels1, n_heads=3)
        self.att2 = ResNet_Attn(self.inchannels2, n_heads=8)
        self.att3 = ResNet_Attn(self.inchannels3, n_heads=13)
        self.att4 = ResNet_Attn(self.inchannels4, n_heads=16)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.drop = nn.Dropout(p=0.5)
        self.ln1 = nn.Linear(1024, class_nb)

    def forward(self, x_rgb, x_flow):

        rgb_in = self.rgb_inp(x_rgb)
        flow_in = self.flow_inp(x_flow)   

        att_r1, att_f1 = self.att1(rgb_in, flow_in)

        rgb1 = self.rgb_l1(rgb_in+att_f1)
        flow1 = self.flow_l1(flow_in+att_r1)

        att_r2, att_f2 = self.att2(rgb1, flow1)

        rgb2 = self.rgb_l2(rgb1+att_f2)
        flow2 = self.flow_l2(flow1+att_r2)

        att_r3, att_f3 = self.att3(rgb2, flow2)

        rgb3 = self.rgb_l3(rgb2+att_f3)
        flow3 = self.flow_l3(flow2+att_r3)
      
        att_r4, att_f4 = self.att4(rgb3, flow3)
        att4 = (att_r4+att_f4)/2

        pool = self.avgpool(att4)
        x_flatten = torch.flatten(pool,1)

        out = self.ln1(x_flatten)
        
        return out