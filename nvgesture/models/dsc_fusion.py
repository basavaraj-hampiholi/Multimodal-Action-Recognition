import torch
import torch.nn as nn
from .eformer import EfficientAttention, PositionwiseConv
import torch.nn.functional as F


class ResidualDepthwiseDilatedBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super(ResidualDepthwiseDilatedBlock,self).__init__()
        self.kerne_size = kernel_size
        self.pad = (kernel_size//2)
        self.dilation_rates = [1,2,3,4]

        self.pointwise1 = nn.Conv3d(in_channels, in_channels//2, kernel_size=1)
        self.depth_conv1 = nn.Conv3d(in_channels//2,in_channels//2,kernel_size, padding=self.pad*self.dilation_rates[0], dilation=self.dilation_rates[0], groups=in_channels//2)
        self.depth_conv2 = nn.Conv3d(in_channels//2,in_channels//2,kernel_size, padding=self.pad*self.dilation_rates[1], dilation=self.dilation_rates[1], groups=in_channels//2)
        self.depth_conv3 = nn.Conv3d(in_channels//2,in_channels//2,kernel_size, padding=self.pad*self.dilation_rates[2], dilation=self.dilation_rates[2], groups=in_channels//2)
        self.depth_conv4 = nn.Conv3d(in_channels//2,in_channels//2,kernel_size, padding=self.pad*self.dilation_rates[3], dilation=self.dilation_rates[3], groups=in_channels//2)
        self.pointwise2 = nn.Conv3d(in_channels*2, out_channels, kernel_size=1)

    def forward(self,x):
        x = self.pointwise1(x)

        x1 = self.depth_conv1(x)        
        x2 = self.depth_conv2(x)
        x3 = self.depth_conv3(x)
        x4 = self.depth_conv4(x)

        x_cat = torch.cat((x1,x2,x3,x4),dim=1)
        x = self.pointwise2(x_cat)
        out = F.layer_norm(x,x.size())
        x = out+x

        return F.relu(x_cat)

class ResNet_Attn(nn.Module):
    def __init__(self, channels, ratio):
        super(ResNet_Attn, self).__init__()
        self.rgb_msf = ResidualDepthwiseDilatedBlock(channels, channels, kernel_size=3)
        self.depth_msf = ResidualDepthwiseDilatedBlock(channels, channels, kernel_size=3)
        self.pointwise_out = nn.Conv3d(channels*4, channels, kernel_size=1)

    def forward(self, x_rgb, x_depth):

        rgb_out = self.rgb_msf(x_rgb)
        depth_out = self.depth_msf(x_depth)
        x_cat = torch.cat((rgb_out,depth_out),dim=1)
        out = F.relu(self.pointwise_out(x_cat))

        return out

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


        self.att1 = ResNet_Attn(self.inchannels1, ratio=1)
        self.att2 = ResNet_Attn(self.inchannels2, ratio=1)
        self.att3 = ResNet_Attn(self.inchannels3, ratio=1)
        self.att4 = ResNet_Attn(self.inchannels4, ratio=1)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.drop = nn.Dropout(p=0.5)
        self.ln1 = nn.Linear(1024, class_nb)

    def forward(self, x_rgb, x_depth):

        rgb_in = self.rgb_inp(x_rgb)
        depth_in = self.flow_inp(x_depth)   

        out1 = self.att1(rgb_in, depth_in)

        rgb1 = self.rgb_l1(rgb_in+out1)
        depth1 = self.flow_l1(depth_in+out1)

        out2 = self.att2(rgb1, depth1)

        rgb2 = self.rgb_l2(rgb1+out2)
        depth2 = self.flow_l2(depth1+out2)

        out3 = self.att3(rgb2, depth2)
        
        rgb3 = self.rgb_l3(rgb2+out3)
        depth3 = self.flow_l3(depth2+out3)
      
        out4 = self.att4(rgb3, depth3)
        
        pool = self.avgpool(out4)
        x_flatten = torch.flatten(pool,1)

        out = self.ln1(x_flatten)

        return out