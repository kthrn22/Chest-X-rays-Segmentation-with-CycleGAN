import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sampling = True, use_norm = True, 
    use_activation = True, **kwargs):
        super().__init__()        
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs) if down_sampling 
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels) if use_norm else nn.Identity(),
            nn.ReLU(inplace = True) if use_activation else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, **kwargs),
            ConvBlock(in_channels, out_channels, use_activation = False, **kwargs),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels, num_residual_blocks):
        super().__init__()

        self.encoder = nn.ModuleList(
            [
                #nn.Conv2d(in_channels, 64, kernel_size = 7, stride = 1, padding = 3, 
            #padding_mode = 'reflect'),
                #nn.ReLU(inplace = True),               
                ConvBlock(in_channels, 64, kernel_size = 7, stride = 1,
            padding = 3, padding_mode = 'reflect'),        
                ConvBlock(64, 128, kernel_size = 3, stride = 2, padding = 1, padding_mode = 'reflect'),
                ConvBlock(128, 256, kernel_size = 3, stride = 2, padding = 1, padding_mode = 'reflect')
            ]
        )

        self.transformer = nn.ModuleList(
            [
                ResidualBlock(256, 256, kernel_size = 3, stride = 1, padding = 1, padding_mode = 'reflect') 
                for _ in range(num_residual_blocks)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                ConvBlock(256, 128, down_sampling = False, kernel_size = 3, stride = 2, padding = 1, 
            output_padding = 1),
                ConvBlock(128, 64, down_sampling = False, kernel_size = 3, stride = 2, padding = 1,
            output_padding = 1),
                ConvBlock(64, in_channels, use_norm = False, use_activation = False, kernel_size = 7,
            stride = 1, padding = 3, padding_mode = 'reflect'),
                #nn.Conv2d(64, in_channels, kernel_size = 7, stride = 1, padding = 3, 
            #padding_mode = 'reflect')
            ]
        )

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        for layer in self.transformer:
            x = layer(x)
        for layer in self.decoder:
            x = layer(x)

        return torch.tanh(x)
