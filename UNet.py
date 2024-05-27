import torch
import torch.nn as nn
import torch.nn.functional as f



class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return f.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_channels, in_channels, residual=True),
            DoubleConvBlock(in_channels, out_channels),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)

        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConvBlock(in_channels, in_channels, residual=True),
            DoubleConvBlock(in_channels, out_channels, in_channels//2)
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttentionBlock, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size*self.size).swapaxes(1,2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2,1).view(-1, self.channels, self.size, self.size)

# Conditional
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_classes=None, time_dim=256, device="cuda"):
        super(UNet, self).__init__()
        self.device = device
        self.time_dim = time_dim

        self.in_conv = DoubleConvBlock(in_channels, 64)

        self.down1 = DownBlock(64, 128)
        self.sa1 = SelfAttentionBlock(128, 32)

        self.down2 = DownBlock(128, 256)
        self.sa2 = SelfAttentionBlock(256, 16)

        self.down3 = DownBlock(256, 256)
        self.sa3 = SelfAttentionBlock(256, 8)

        self.bot1 = DoubleConvBlock(256, 512)
        self.bot2 = DoubleConvBlock(512, 512)
        self.bot3 = DoubleConvBlock(512,256)

        self.up1 = UpSampleBlock(512, 128)
        self.sa4 = SelfAttentionBlock(128,16)
        self.up2 = UpSampleBlock(256, 64)
        self.sa5 = SelfAttentionBlock(64, 32)
        self.up3 = UpSampleBlock(128, 64)
        self.sa6 = SelfAttentionBlock(64, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def time_encoding(self, t, channels):
        inv_freq = 1.0 / (1000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        time_enc_a = torch.sin(t.repeat(1, channels // 2) ** inv_freq)
        time_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([time_enc_a, time_enc_b], dim=-1)

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.time_encoding(t, self.time_dim)
        if y is not None:
            t += self.label_emb(y)


        # Encoder
        x1 = self.in_conv(x)

        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)

        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)

        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        # Bottleneck
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        # Decoder
        x = self.up1(x4, x3, t)
        x = self.sa4(x)

        x = self.up2(x, x2, t)
        x = self.sa5(x)

        x = self.up3(x, x1, t)
        x = self.sa6(x)

        out = self.out_conv(x)
        return out


