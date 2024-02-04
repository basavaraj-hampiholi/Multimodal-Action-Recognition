import torch
import torch.nn as nn
from .eformer import EfficientAttention, PositionwiseConv
import torch.nn.functional as F

class ResNet_Attn(nn.Module):
    def __init__(self, channels, n_heads):
        super(ResNet_Attn, self).__init__()
        self.ch = channels//n_heads
        self.d_q = self.ch
        self.d_k = self.ch
        self.d_v = self.ch
        self.d_inner = self.ch
        self.slf_attnr = EfficientAttention(channels, channels, channels, n_heads)
        self.slf_attnf = EfficientAttention(channels, channels, channels, n_heads)
        self.pos_ffn = PositionwiseConv(channels, channels)

    def forward(self, x_rgb, x_flow):
        rgb_output = self.slf_attnr(x_rgb)
        flow_output = self.slf_attnf(x_flow)
        
        cat_attn = torch.cat((rgb_output, flow_output), dim=1)
        ffn_out = self.pos_ffn(cat_attn)
        
        return ffn_out

class TransFusion(nn.Module):
    def __init__(self, i3d_rgb, i3d_flow, class_nb=101):
        super(TransFusion, self).__init__()
        self.inchannels1 = 192
        self.inchannels2_1 = 480
        self.inchannels2_2 = 512
        self.inchannels2_3 = 528
        self.inchannels3 = 832
        self.inchannels4 = 1024

        self.rgb_inp = nn.Sequential(i3d_rgb.Conv3d_1a_7x7, i3d_rgb.MaxPool3d_2a_3x3, i3d_rgb.Conv3d_2b_1x1, i3d_rgb.Conv3d_2c_3x3, i3d_rgb.MaxPool3d_3a_3x3)
        self.flow_inp = nn.Sequential(i3d_flow.Conv3d_1a_7x7, i3d_flow.MaxPool3d_2a_3x3, i3d_flow.Conv3d_2b_1x1, i3d_flow.Conv3d_2c_3x3, i3d_flow.MaxPool3d_3a_3x3)
      
        self.rgb_l1 = nn.Sequential(i3d_rgb.Mixed_3b, i3d_rgb.Mixed_3c, i3d_rgb.MaxPool3d_4a_3x3)
        self.rgb_l2 = nn.Sequential(i3d_rgb.Mixed_4b, i3d_rgb.Mixed_4c) 
        self.rgb_l3 = nn.Sequential(i3d_rgb.Mixed_4d, i3d_rgb.Mixed_4e)
        self.rgb_l4 = nn.Sequential(i3d_rgb.Mixed_4f, i3d_rgb.MaxPool3d_5a_2x2) 
        self.rgb_l5 = nn.Sequential(i3d_rgb.Mixed_5b, i3d_rgb.Mixed_5c)
        self.rgb_logits = i3d_rgb.logits

        self.flow_l1 = nn.Sequential(i3d_flow.Mixed_3b, i3d_flow.Mixed_3c, i3d_flow.MaxPool3d_4a_3x3)
        self.flow_l2 = nn.Sequential(i3d_flow.Mixed_4b, i3d_flow.Mixed_4c)
        self.flow_l3 = nn.Sequential(i3d_flow.Mixed_4d, i3d_flow.Mixed_4e)
        self.flow_l4 = nn.Sequential(i3d_flow.Mixed_4f, i3d_flow.MaxPool3d_5a_2x2)
        self.flow_l5 = nn.Sequential(i3d_flow.Mixed_5b, i3d_flow.Mixed_5c)
        self.flows_logits = i3d_flow.logits

        self.att1 = ResNet_Attn(self.inchannels1, n_heads=3)
        self.att2_1 = ResNet_Attn(self.inchannels2_1, n_heads=8)
        self.att2_2 = ResNet_Attn(self.inchannels2_2, n_heads=8)
        self.att2_3 = ResNet_Attn(self.inchannels2_3, n_heads=16)
        self.att3 = ResNet_Attn(self.inchannels3, n_heads=16)
        self.att4 = ResNet_Attn(self.inchannels4, n_heads=16)

        self.conv1 = nn.Conv3d(self.inchannels1*2, self.inchannels1, kernel_size=1)
        self.conv2_1 = nn.Conv3d(self.inchannels2_1*2, self.inchannels2_1, kernel_size=1)
        self.conv2_2 = nn.Conv3d(self.inchannels2_2*2, self.inchannels2_2, kernel_size=1)
        self.conv2_3 = nn.Conv3d(self.inchannels2_3*2, self.inchannels2_3, kernel_size=1)
        self.conv3 = nn.Conv3d(self.inchannels3*2, self.inchannels3, kernel_size=1)
        self.conv4 = nn.Conv3d(self.inchannels4*2, self.inchannels4, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.drop = nn.Dropout(p=0.5)
        self.ln1 = nn.Linear(1024, class_nb)

    def forward(self, x_rgb, x_flow):

        rgb_in = self.rgb_inp(x_rgb)
        flow_in = self.flow_inp(x_flow)   

        att1 = self.conv1(self.att1(rgb_in, flow_in))

        rgb1_fuse = rgb_in+att1
        flow1_fuse = flow_in+att1
        rgb1 = self.rgb_l1(rgb1_fuse)
        flow1 = self.flow_l1(flow1_fuse)

        att2_1 = self.conv2_1(self.att2_1(rgb1, flow1))

        rgb2_fuse = rgb1+att2_1
        flow2_fuse = flow1+att2_1
        rgb2 = self.rgb_l2(rgb2_fuse)
        flow2 = self.flow_l2(flow2_fuse)

        att2_2 = self.conv2_2(self.att2_2(rgb2, flow2))
        
        rgb3_fuse = rgb2+att2_2
        flow3_fuse = flow2+att2_2
        rgb3 = self.rgb_l3(rgb3_fuse)
        flow3 = self.flow_l3(flow3_fuse)
     
        att2_3 = self.conv2_3(self.att2_3(rgb3, flow3))

        rgb4_fuse = rgb3+att2_3
        flow4_fuse = flow3+att2_3
        rgb4 = self.rgb_l4(rgb4_fuse)
        flow4 = self.flow_l4(flow4_fuse)

        att3 = self.conv3(self.att3(rgb4, flow4))

        rgb5_fuse = rgb4+att3
        flow5_fuse = flow4+att3
        rgb5 = self.rgb_l5(rgb5_fuse)
        flow5 = self.flow_l5(flow5_fuse)

        att4 = self.conv4(self.att4(rgb5, flow5))

        pool = self.avgpool(att4)
        x_flatten = torch.flatten(pool,1)
        #x_flatten = self.drop(x_flatten)

        out = self.ln1(x_flatten)
        return out