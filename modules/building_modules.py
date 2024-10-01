import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class TimeEmbedding(nn.Module):
    
    def __init__(self, dim):
        """Time embedding layer.
        
        Parameters:
        ----------
        dim: int
            Dimension of the time embedding
        """

        super(TimeEmbedding, self).__init__()
        self.dim = dim 
        self.half_dim = dim // 2
        emb = math.log(10000) / (self.half_dim - 1)
        emb = torch.exp(torch.arange(self.half_dim, dtype=torch.float32) * - emb)
        self.register_buffer('temb', emb)
    
    def forward(self, inputs):
        inputs = inputs.to(torch.float32)
        inputs_emb = inputs.unsqueeze(1) * self.temb.unsqueeze(0)
        emb = torch.cat([torch.sin(inputs_emb), torch.cos(inputs_emb)], axis=-1)
        return emb



class TimeMLP(nn.Module):
    
    def __init__(self, input_dim, units, activation_fn=F.silu):
        """Time MLP layer.
        
        Parameters:
        ----------
        input_dim: int
            Dimension of the input
        units: int
            Number of units in the dense layers
        activation_fn: function
            Activation function to be used
        """
        super(TimeMLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, units)
        self.activation_fn = activation_fn
        self.linear2 = nn.Linear(units, units)
        
    def forward(self, inputs):
        x = self.linear1(inputs)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x
    


class ResidualBlock(nn.Module):
    
    def __init__(self, 
                 input_channels, 
                 t_dim, 
                 num_channels, 
                 groups=8, 
                 activation_fn=F.silu):
        """Residual block with GroupNormalization and time embedding.

        Parameters:
        ----------
        input_channels: int
            Number of input channels
        t_dim: int
            Dimension of the time embedding
        num_channels: int
            Number of output channels
        groups: int
            Number of groups to be used for GroupNormalization layers
        activation_fn: function
            Activation function to be used
        """
        super(ResidualBlock, self).__init__()
        self.activation_fn = activation_fn
        self.groups = groups
        self.t_dim = t_dim
        if input_channels == num_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels=input_channels, 
                                      out_channels=num_channels, 
                                      kernel_size=1)
            
        self.linear = nn.Linear(t_dim, num_channels)
        self.group_norm1 = nn.GroupNorm(groups, input_channels)
        self.group_norm2 = nn.GroupNorm(groups, num_channels)
        self.conv1 = nn.Conv2d(in_channels=input_channels, 
                               out_channels=num_channels, 
                               kernel_size=3,
                               padding='same',
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=num_channels, 
                               out_channels=num_channels, 
                               kernel_size=3,
                               padding='same',
                               bias=False)
    
    def forward(self, inputs):
        x, t = inputs
        residual = self.residual(x)
        temb = self.activation_fn(t)
        temb = self.linear(temb)
        temb.unsqueeze_(2).unsqueeze_(2)
        x = self.group_norm1(x)
        x = self.activation_fn(x)
        x = self.conv1(x)
        x = torch.add(x, temb)
        x = self.group_norm2(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        x = torch.add(x, residual)
        return x



class DownSample(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        """Downsampling layer.
        
        Parameters:
        ----------
        in_channels: int
            Number of input channels
        out_channels: int
            Number of output channels
        """
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=3,
                              stride=2,
                              padding=1)
        
    def forward(self, inputs):
        return self.conv(inputs)



class UpSample(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        """Upsampling layer.
        
        Parameters:
        ----------
        in_channels: int
            Number of input channels
        out_channels: int
            Number of output channels"""
        super(UpSample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        output_padding=1)
        
    def forward(self, x):
        return self.deconv(x)



class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads, dropout):
        """Self attention layer.

        Parameters:
        ----------
        channels: int
            Number of input channels
        num_heads: int
            Number of heads to be used
        dropout: float
            Dropout rate
        """
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.mha = nn.MultiheadAttention(embed_dim=channels, 
                                         num_heads=num_heads, 
                                         batch_first=True,
                                         dropout=dropout)
        self.layer_norm = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w).swapaxes(1, 2)
        x_norm = self.layer_norm(x)
        attention_value, _ = self.mha(x_norm, x_norm, x_norm)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(b, c, h, w)