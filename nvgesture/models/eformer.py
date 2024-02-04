import torch
from torch import nn
from torch.nn import functional as f

class DepthwiseConv(nn.Module):
    def __init__(self,in_channels,out_channels, ratio):
        super(DepthwiseConv,self).__init__()
        self.ratio=ratio
        self.pointwise1 = nn.Conv3d(in_channels, in_channels//self.ratio, kernel_size=1)
        self.depthwise = nn.Conv3d(in_channels//self.ratio, in_channels//self.ratio, kernel_size=3, padding=1, groups=in_channels//self.ratio)
        

    def forward(self,x):
        x = self.pointwise1(x)
        x = self.depthwise(x)
        #x = self.pointwise2(x)
        
        return x

class EfficientAttention(nn.Module):   
    def __init__(self, in_channels, key_channels, value_channels, head_count):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.ratio = 8

        self.dw_conv = DepthwiseConv(in_channels, key_channels, self.ratio)
        #self.queries = DepthwiseConv(in_channels, key_channels, self.ratio)
        #self.values = DepthwiseConv(in_channels, value_channels, self.ratio)
        #self.reprojection = DepthwiseConv(value_channels, in_channels, self.ratio)
        self.keys = nn.Conv3d(in_channels//self.ratio, key_channels, kernel_size=1)
        self.queries = nn.Conv3d(in_channels//self.ratio, key_channels, kernel_size=1)
        self.values = nn.Conv3d(in_channels//self.ratio, value_channels, kernel_size=1)
        self.reprojection = nn.Conv3d(value_channels, in_channels, kernel_size=1)

    def forward(self, input_):
        n, _, t, h, w = input_.size()
        residual = input_
        input_ = self.dw_conv(input_)
        keys = self.keys(input_).reshape((n, self.key_channels, t*h*w))
        queries = self.queries(input_).reshape(n, self.key_channels, t*h*w)
        values = self.values(input_).reshape((n, self.value_channels, t*h*w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            key = f.softmax(keys[:, i*head_key_channels:(i+1)*head_key_channels, :], dim=2)                                                   
            query = f.softmax(queries[:, i*head_key_channels:(i+1)*head_key_channels, :], dim=1)
            value = values[:, i*head_value_channels:(i+1)*head_value_channels, :]                                                          
            context = key@value.transpose(1, 2)         
            attended_value=(context.transpose(1, 2)@query).reshape(n, head_value_channels, t, h, w)            
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        #print(reprojected_value.shape, input_.shape)
        x = reprojected_value + residual
        x = f.layer_norm(x, x.size()) 

        return x

class PositionwiseConv(nn.Module):
    ''' A two-1x1conv layers module '''

    def __init__(self, d_in, d_hid):
        super().__init__()
        self.ratio=4
        self.c_1 = nn.Linear(d_in, d_hid//self.ratio)
        self.c_2 = nn.Linear(d_hid//self.ratio, d_in)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        residual = x       
        b,c,t,h,w = x.size()
        x = x.view(b, t*h*w, c)
        x = f.relu(self.dropout(self.c_1(x)))
        x = self.c_2(x)
        x = x.view(residual.size())
        x += residual
        x = f.layer_norm(x, x.size()) 

        return x

class PositionwiseConv_Cat(nn.Module):
    ''' A two-1x1conv layers module '''

    def __init__(self, d_in, rgb_hid, sk_hid):
        super().__init__()
        self.ratio=8
        self.c_1 = nn.Linear(d_in*2, d_in//self.ratio)
        self.c_rgb = nn.Linear(d_in//self.ratio, rgb_hid)
        self.c_sk = nn.Linear(d_in//self.ratio, sk_hid)
        self.dropout = nn.Dropout(0.1)

    def forward(self, rgb_flat, sk_flat):
        residual_rgb = rgb_flat     
        residual_sk = sk_flat  
        cat_attn = torch.cat((rgb_flat, sk_flat), dim=1)
        b,c,t,h,w = cat_attn.size()
        cat_attn = cat_attn.view(b, t, h, w, c)

        cat_x = f.relu(self.dropout(self.c_1(cat_attn)))

        x_rgb = self.c_rgb(cat_x)
        x_sk = self.c_sk(cat_x)

        x_rgb = x_rgb.view(residual_rgb.size())
        x_rgb += residual_rgb

        x_sk = x_sk.view(residual_sk.size())
        x_sk += residual_sk

        x_rgb = f.layer_norm(x_rgb, x_rgb.size()) 
        x_sk = f.layer_norm(x_sk, x_sk.size()) 

        return x_rgb, x_sk