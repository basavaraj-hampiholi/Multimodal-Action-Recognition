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
        
        self.inchannels1 = 64
        self.inchannels2 = 256
        self.inchannels3 = 512
        self.inchannels4 = 1024

        self.rgb_inp = nn.Sequential(i3d_rgb.conv1, i3d_rgb.bn1, i3d_rgb.relu, i3d_rgb.maxpool)
        self.flow_inp = nn.Sequential(i3d_flow.conv1, i3d_flow.bn1, i3d_flow.relu, i3d_flow.maxpool)

        self.rgb_l1 = i3d_rgb.layer1
        self.rgb_l2 = i3d_rgb.layer2
        self.rgb_l3 = i3d_rgb.layer3
        self.rgb_l4 = i3d_rgb.layer4

        self.flow_l1 = i3d_flow.layer1
        self.flow_l2 = i3d_flow.layer2
        self.flow_l3 = i3d_flow.layer3
        self.flow_l4 = i3d_flow.layer4

        self.att1 = ResNet_Attn(self.inchannels1, n_heads=4)
        self.att2 = ResNet_Attn(self.inchannels2, n_heads=8)
        self.att3 = ResNet_Attn(self.inchannels3, n_heads=16)
        self.att4 = ResNet_Attn(self.inchannels4, n_heads=16)

        self.conv1 = nn.Conv3d(self.inchannels1*2, self.inchannels1, kernel_size=1)
        self.conv2 = nn.Conv3d(self.inchannels2*2, self.inchannels2, kernel_size=1)
        self.conv3 = nn.Conv3d(self.inchannels3*2, self.inchannels3, kernel_size=1)
        self.conv4 = nn.Conv3d(self.inchannels4*2, self.inchannels4, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.drop = nn.Dropout(p=0.5)
        self.ln1 = nn.Linear(2048, class_nb)

    def forward(self, x_rgb, x_flow):

        rgb_in = self.rgb_inp(x_rgb)
        flow_in = self.flow_inp(x_flow)   

        att1 = self.conv1(self.att1(rgb_in, flow_in))

        rgb1_fuse = rgb_in+att1
        flow1_fuse = flow_in+att1
        rgb1 = self.rgb_l1(rgb1_fuse)
        flow1 = self.flow_l1(flow1_fuse)

        att2 = self.conv2(self.att2(rgb1, flow1))

        rgb2_fuse = rgb1+att2
        flow2_fuse = flow1+att2
        rgb2 = self.rgb_l2(rgb2_fuse)
        flow2 = self.flow_l2(flow2_fuse)

        att3 = self.conv3(self.att3(rgb2, flow2))
        
        rgb3_fuse = rgb2+att3
        flow3_fuse = flow2+att3
        rgb3 = self.rgb_l3(rgb3_fuse)
        flow3 = self.flow_l3(flow3_fuse)
      
        att4 = self.conv4(self.att4(rgb3, flow3))

        out_feats = self.rgb_l4(att4)
        
        pool = self.avgpool(out_feats)
        x_flatten = torch.flatten(pool,1)
        #x_flatten = self.drop(x_flatten)

        out = self.ln1(x_flatten)
        return out