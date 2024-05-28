import torch
import torch.nn as nn
import torch.nn.functional as f
from Attention import SelfAttention

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VAE_ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),

            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        # need a residual layer to match the input channels to output channels if theyre not equal when
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residue = x
        x = self.layers(x)
        return x + self.residual_layer(residue)


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(VAE_AttentionBlock, self).__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.self_attention = SelfAttention(1, channels)


    def forward(self, x):
        residue = x

        # batch size, channels, height, width
        n, c, h, w = x.shape

        # have a sequence that represents all pixels
        x = x.veiw((n, c, h * w))

        # swap channels/feature with pixels for self attention
        x = x.transpose(-1, -2)

        x = self.self_attention(x)

        x = x.transpose(-1, -2)

        x = x.veiw((n, c, h, w))

        return x + residue



class VAE_Encoder(nn.Module):
    def __init__(self, in_channels):
        super(VAE_Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),

            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),  # original image size / 2

            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),  # original image size / 4

            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),  # original image size / 8

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            nn.GroupNorm(512, 512),
            nn.SiLU(),

            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            nn.Conv2d(8, 8, kernel_size=1, padding=0)

            # Ouputs tensor (batch_size, 8, height/8, width/8)
        )

    def forward(self, x, noise):
        for module in self.layers:
            if getattr(module, 'stride', None) == (2, 2):
                x = f.pad(x, (0, 1, 0, 1))     # pad right and top
            x = module(x)

        # Output represents the mean and log_variance
        # We split the channels that in half the first part representing mean and second log_variance
        # tensor (batch_size, 4, height/8, width/8) for each
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # limit the extremes of log variance
        log_variance = torch.clamp(log_variance, -30, 20)

        # converting to variance
        variance = log_variance.exp()

        std = variance.sqrt()

        # reparameterization trick: z = mean + std*noise
        z = mean + std * noise

        # constant is specific to stable diffusion but is used to ensure that latent space has std is close to 1
        z *= 0.18215

        return z
