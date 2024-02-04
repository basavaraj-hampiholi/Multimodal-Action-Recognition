import torch
from torch import nn
from torch.nn import functional as f

class DepthwiseConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DepthwiseConv,self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self,x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class AttentionLayers(nn.Module):   
    def __init__(self, in_channels, key_channels, value_channels):
        super().__init__()
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.keys = DepthwiseConv(in_channels, key_channels)
        self.queries = DepthwiseConv(in_channels, key_channels)
        self.values = DepthwiseConv(in_channels, value_channels)
        

    def forward(self, input_):
        n, _, t, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, t*h*w))
        queries = self.queries(input_).reshape(n, self.key_channels, t*h*w)
        values = self.values(input_).reshape((n, self.value_channels, t*h*w))

        return keys, queries, values

class EfficientAttention(nn.Module):   
    def __init__(self, in_channels, key_channels, value_channels, head_count):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.attn_rgb = AttentionLayers(in_channels, key_channels, value_channels)
        self.attn_flow = AttentionLayers(in_channels, key_channels, value_channels)

        self.reprojection = DepthwiseConv(value_channels*2, in_channels*2)

    def forward(self, input_rgb, input_flow):
        fused_input = torch.cat((input_rgb, input_flow), dim=1)
        n, _, t, h, w = input_rgb.size()
        rgb_keys, rgb_queries, rgb_values = self.attn_rgb(input_rgb)
        flow_keys, flow_queries, flow_values = self.attn_flow(input_flow)
        keys = torch.cat((rgb_keys,flow_keys),dim=1)
        queries = torch.cat((rgb_queries,flow_queries),dim=1)
        values = torch.cat((rgb_values,flow_values),dim=1)

        head_key_channels = self.key_channels*2 // self.head_count
        head_value_channels = self.value_channels*2 // self.head_count
        
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
        #x_rgb = reprojected_value + input_rgb
        #x_flow = reprojected_value + input_flow
        #x = torch.cat((x_rgb, x_flow),dim=1)
        x = reprojected_value + fused_input
        attention = f.layer_norm(x, x.size()) 

        return attention

class PositionwiseConv(nn.Module):
    ''' A two-1x1conv layers module '''

    def __init__(self, d_in, d_hid):
        super().__init__()
        #self.c_1 = nn.Conv3d(d_in*2, d_hid, kernel_size=1) # position-wise
        #self.c_2 = nn.Conv3d(d_hid, d_in*2, kernel_size=1) # position-wise
        #self.conv1 = nn.Conv3d(d_in*2, d_in, kernel_size=1)
        self.c_1 = nn.Linear(d_in*2, d_hid)
        self.c_2 = nn.Linear(d_hid, d_in*2)
        #self.c_3 = nn.Linear(d_in*2, d_in)
        #self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(0.1)

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

class Positionwise(nn.Module):

    def __init__(self, d_in, d_hid):
        super().__init__()
        #self.c_1 = nn.Conv3d(d_in*2, d_hid, kernel_size=1) # position-wise
        #self.c_2 = nn.Conv3d(d_hid, d_in*2, kernel_size=1) # position-wise
        #self.conv1 = nn.Conv3d(d_in*2, d_in, kernel_size=1)
        self.c_1 = nn.Linear(d_in, d_hid)
        self.c_2 = nn.Linear(d_hid, d_in)
        #self.c_3 = nn.Linear(d_in*2, d_in)
        #self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(0.1)

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