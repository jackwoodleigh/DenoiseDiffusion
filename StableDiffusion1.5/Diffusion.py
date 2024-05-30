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

    # relating the latent with time embedding such that output is depending on the combination
    def forward(self, x, t):
        residual = x
        x = self.groupnorm(x)
        x = f.silu(x)
        x = self.conv(x)

        t = f.silu(t)
        t = self.linear_time(t)

        merge = x + t.unsqueeze(-1).unsqueeze(-1)
        merge = self.groupnorm_merge(merge)
        merge = f.silu(merge)
        merge = self.conv_merge(merge)

        return merge



class UNet_AttentionBlock(nn.Module):
    def __init__(self, n_heads, n_embd, d_context=768):
        super().__init__()
        channels = n_heads * n_embd

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_inp = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.layernorm1 = nn.LayerNorm(channels)
        self.attention1 = SelfAttention(n_heads, channels, in_projection_bias=False)
        self.layernorm2 = nn.LayerNorm(channels)
        self.attention2 = CrossAttention(n_heads, channels, d_context, in_projection_bias=False)
        self.layernorm3 = nn.LayerNorm(channels)

        self.linear_gelu1 = nn.LayerNorm(channels, 4*2*channels)
        self.linear_gelu2 = nn.LayerNorm(4*channels, channels)
        self.conv_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        long_residual = x
        x = self.groupnorm(x)
        x = self.conv_inp(x)

        n, c, h, w = x.shape

        x = x.view((n,c,h*w))
        x = x.transpose(-1, -2)

        # norm with self attention + residual
        short_residual = x
        x = self.layernorm1(x)
        self.attention1(x)
        x += short_residual

        # norm with cross attention + residual
        short_residual = x
        x = self.layernorm2(x)
        self.attention2(x, context)
        x += short_residual

        # norm with FF and gelu + residual
        short_residual = x
        x = self.layernorm3(x)
        x, gate = self.linear_gelu1(x).chunk(2, dim=-1)
        x = x * f.gelu(gate)
        x = self.linear_gelu2(x)
        x += short_residual

        # reverse transform
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        x = self.conv_out(x)
        return x + long_residual






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
            # starts with (batchsize, 4, height/8, width/8) assuming VAE latents input
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
            # taking in residual
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

    def forward(self, latent, context, time):
        # latent (batch_size, 4, height/8, width/8) from VAE
        # context (batch_size, seq_len, dim) from CLIP
        # time (1, time_dim)

        time = self.time_embedding(time)
        output = self.UNet(latent, context, time)
        output = self.final(output)
        return output