import torch
import torch.nn as nn
from torch.nn.modules import padding
from torch.nn.modules.conv import Conv2d

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm = True, use_activation = True, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels) if use_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace = True) if use_activation else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                #nn.Conv2d(in_channels, 64, kernel_size = 4, stride = 2, padding = 1, 
            #padding_mode = 'reflect'),
                #nn.LeakyReLU(0.2, inplace = True),
                ConvBlock(in_channels, 64, use_norm = False, kernel_size = 4, stride = 2, padding = 1, 
            padding_mode = 'reflect'),
                ConvBlock(64, 128, kernel_size = 4, stride = 2, padding = 1, padding_mode = 'reflect',
            bias = True),
                ConvBlock(128, 256, kernel_size = 4, stride = 2, padding = 1, padding_mode = 'reflect',
            bias = True),
                ConvBlock(256, 512, kernel_size = 4, stride = 1, padding = 1, padding_mode = 'reflect',
            bias = True),
                ConvBlock(512, 1, use_norm = False, use_activation = False, kernel_size = 4, stride = 1,
            padding = 1, padding_mode = 'reflect'),
                #nn.Conv2d(512, 1, kernel_size = 4, stride = 1, padding = 1, padding_mode = 'reflect'),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(x)
