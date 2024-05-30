import torch
from torch import nn
from torch.nn import functional as f
from Attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, n_embd*4)
        self.linear2 = nn.Linear(4*n_embd, 4*n_embd)
    def forward(self, x):
        x = self.linear1(x)
        x = f.silu(x)
        x = self.linear2(x)
        return x


# initialized with layers and depending on layer feeds them different info when called
class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNet_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNet_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)


class UNet_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim_time=1280):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(dim_time, out_channels)

        self.groupnorm_merge = nn.GroupNorm(32, out_channels)
        self.conv_merge = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels,kernel_size=1, padding=0)



class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # multiply height and width by 2
        x = f.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module([
            # starts with (batchsize, 4, height/8, width/8)
            SwitchSequential(nn.Conv2d(4,320, kernel_size=3, padding=1)),
            SwitchSequential(UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)),
            SwitchSequential(UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)),

            # height and width divided by 16 now
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNet_ResidualBlock(320, 640), UNet_AttentionBlock(8, 80)),
            SwitchSequential(UNet_ResidualBlock(640, 640), UNet_AttentionBlock(8, 40)),

            # height and width divided by 32 now
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNet_ResidualBlock(640, 1280), UNet_AttentionBlock(8, 160)),
            SwitchSequential(UNet_ResidualBlock(1280, 1280), UNet_AttentionBlock(8, 160)),

            # height and width divided by 64 now
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNet_ResidualBlock(1280, 1280)),
            SwitchSequential(UNet_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            UNet_ResidualBlock(1280, 1280),
            UNet_AttentionBlock(8, 160),
            UNet_ResidualBlock(1280, 1280),
        )

        self.decoder = nn.Module([
            # taking in residue
            SwitchSequential(UNet_ResidualBlock(2560, 1280)),
            SwitchSequential(UNet_ResidualBlock(2560, 1280)),
            SwitchSequential(UNet_ResidualBlock(2560, 1280), Upsample(1280)),

            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)),
            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)),
            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160), Upsample(1280)),

            SwitchSequential(UNet_ResidualBlock(1920, 640), UNet_AttentionBlock(8, 80)),
            SwitchSequential(UNet_ResidualBlock(1280, 640), UNet_AttentionBlock(8, 80)),
            SwitchSequential(UNet_ResidualBlock(960, 640), UNet_AttentionBlock(8, 80), Upsample(640)),

            SwitchSequential(UNet_ResidualBlock(960, 320), UNet_AttentionBlock(8, 40)),
            SwitchSequential(UNet_ResidualBlock(640, 320), UNet_AttentionBlock(8, 40)),
            SwitchSequential(UNet_ResidualBlock(640, 320), UNet_AttentionBlock(8, 40)),

        ])


class UNet_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.groupnorm(x)
        x = f.silu(x)
        x = self.conv(x)
        return x


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = UNet_OutputLayer(320, 4)

    def forward(self, latent, context, t):
        # latent (batch_size, 4, height/8, width/8) from VAE
        # context (batch_size, seq_len, dim) from CLIP
        # time (1, time_dim)

        t = self.time_embedding(t)
        output = self.UNet(latent, context, time)
        output = self.final(output)
        return output