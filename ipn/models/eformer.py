import torch
from torch import nn
from torch.nn import functional as f

class DepthwiseConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DepthwiseConv,self).__init__()
        self.ratio=8
        self.pointwise1 = nn.Conv3d(in_channels, in_channels//self.ratio, kernel_size=1)
        self.depthwise = nn.Conv3d(in_channels//self.ratio, in_channels//self.ratio, kernel_size=3, padding=1, groups=in_channels//self.ratio)
        self.pointwise2 = nn.Conv3d(in_channels//self.ratio, out_channels, kernel_size=1)

    def forward(self,x):
        x = self.pointwise1(x)
        x = self.depthwise(x)
        x = self.pointwise2(x)
        
        return x

class EfficientAttention(nn.Module):   
    def __init__(self, in_channels, key_channels, value_channels, head_count):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = DepthwiseConv(in_channels, key_channels)
        self.queries = DepthwiseConv(in_channels, key_channels)
        self.values = DepthwiseConv(in_channels, value_channels)
        self.reprojection = DepthwiseConv(value_channels, in_channels)

    def forward(self, input_):
        n, _, t, h, w = input_.size()
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
        x = reprojected_value + input_
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