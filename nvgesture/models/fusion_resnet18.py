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
    def __init__(self, i3d_rgb, i3d_flow, class_nb=101):
        super(TransFusion, self).__init__()
        
        self.inchannels1 = 256
        self.inchannels2 = 512
        self.inchannels3 = 1024
        self.inchannels4 = 2048

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

        #self.att1 = ResNet_Attn(self.inchannels1, n_heads=4)
        #self.att2 = ResNet_Attn(self.inchannels2, n_heads=8)
        self.att3 = ResNet_Attn(self.inchannels3, n_heads=8)
        self.att4 = ResNet_Attn(self.inchannels4, n_heads=16)
        #self.conv = nn.Conv3d(self.inchannels4*2, self.inchannels4, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        #self.drop = nn.Dropout(p=0.5)
        self.ln1 = nn.Linear(2048, class_nb)

    def forward(self, x_rgb, x_flow):

        rgb_in = self.rgb_inp(x_rgb)
        flow_in = self.flow_inp(x_flow)   

        #att1 = self.att1(rgb_in, flow_in)

        rgb1 = self.rgb_l1(rgb_in)
        flow1 = self.flow_l1(flow_in)

        #att1 = self.att1(rgb1, flow1)

        rgb2 = self.rgb_l2(rgb1)
        flow2 = self.flow_l2(flow1)

        #att2 = self.att2(rgb2, flow2)
        
        rgb3 = self.rgb_l3(rgb2)
        flow3 = self.flow_l3(flow2)

        att3 = self.att3(rgb3, flow3)

        rgb4 = self.rgb_l4(rgb3+att3)
        flow4 = self.flow_l4(flow3+att3)
      
        att4 = self.att4(rgb4, flow4)  

        pool = self.avgpool(att4)      

        flatten = torch.flatten(pool,1)
        out = self.ln1(flatten)

        return out