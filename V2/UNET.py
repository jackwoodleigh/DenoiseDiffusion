import torch
from torch import nn
from torch.nn import functional as f
from Attention import SelfAttention, CrossAttention

def TimeEmbedding(t, dim, max_period=1000, device="cuda"):
    half_dim = dim // 2
    frequencies = torch.exp(-torch.log(torch.tensor(max_period, device=device, dtype=torch.float32)) * torch.arange(half_dim, device=device) / half_dim)
    angles = t[:, None].float() * frequencies[None, :]
    embeddings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:  # if dim is odd, add an extra zero column
        embeddings = torch.cat([embeddings, torch.zeros_like(t[:, None])], dim=-1)
    return embeddings


class SwitchSequential(nn.Sequential):
    def forward(self, x, time, context):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embd_dim=256):
        super().__init__()
        self.input = nn.Sequential(
            nn.GroupNorm(4, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.time_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embd_dim, out_channels)
        )

        self.output = nn.Sequential(
            nn.GroupNorm(4, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x, t_emb):
        residual = x

        x = self.input(x)

        t_emb = self.time_layer(t_emb)

        x = x + t_emb.unsqueeze(-1).unsqueeze(-1)

        x = self.output(x)

        return x + self.residual_layer(residual)


class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, context_embd_dim=256):
        super().__init__()
        channels = n_head * n_embd

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)

        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, context_embd_dim, in_proj_bias=False)
        
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        # x: (Batch_Size, Features, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)

        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)

        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)

        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        # optional cross attention should context be added
        if context is not None:
            residue_short = x
            x = self.layernorm_2(x)
            x = self.attention_2(x, context)
            x += residue_short

        residue_short = x
        x = self.layernorm_3(x)
        # GeGLU as implemented in the original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        # (Batch_Size, Height * Width, Features) -> two tensors of shape (Batch_Size, Height * Width, Features * 4)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * f.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short

        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        return self.conv_output(x) + residue_long


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = f.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, context_embd_dim=256, time_embd_dim=256):
        super().__init__()
        self.context_embd_dim = context_embd_dim
        self.time_embd_dim = time_embd_dim
        self.context_emb = nn.Embedding(num_embeddings=10, embedding_dim=context_embd_dim)

        self.encoder = nn.ModuleList([
            SwitchSequential(nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(64, 64, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 16, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(64, 64, time_embd_dim=time_embd_dim),UNET_AttentionBlock(4, 16, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(64, 128, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 32, context_embd_dim=context_embd_dim)),

            # Image size / 2
            SwitchSequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(128, 128, time_embd_dim=time_embd_dim),  UNET_AttentionBlock(4, 32, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(128, 128, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 32, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(128, 256, time_embd_dim=time_embd_dim),  UNET_AttentionBlock(4, 64, context_embd_dim=context_embd_dim)),

            # Image size / 4
            SwitchSequential(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(256, 256, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 64, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(256, 256, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 64, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(256, 512, time_embd_dim=time_embd_dim),  UNET_AttentionBlock(4, 128, context_embd_dim=context_embd_dim)),

            # Image size / 8
            SwitchSequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(512, 512, time_embd_dim=time_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(512, 512, time_embd_dim=time_embd_dim)),
        ])

        self.bottleneck = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(512, 1024, time_embd_dim=time_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(1024, 1024, time_embd_dim=time_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(1024, 512, time_embd_dim=time_embd_dim)),
        ])

        self.decoder = nn.ModuleList([

            # at each step concatenating output from bottleneck with residual from encoder

            # Image size /4
            SwitchSequential(UNET_ResidualBlock(1024, 512, time_embd_dim=time_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(1024, 512, time_embd_dim=time_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(1024, 512, time_embd_dim=time_embd_dim), Upsample(512)),

            # Image size / 2
            SwitchSequential(UNET_ResidualBlock(1024, 256, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 64, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(512, 256, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 64, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(512, 256, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 64, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(512, 256, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 64, context_embd_dim=context_embd_dim), Upsample(256)),

            # Image size
            SwitchSequential(UNET_ResidualBlock(512, 128, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 32, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(256, 128, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 32, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(256, 128, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 32, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(256, 128, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 32, context_embd_dim=context_embd_dim), Upsample(128)),

            SwitchSequential(UNET_ResidualBlock(256, 64, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 16, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(128, 64, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 16, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(128, 64, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 16, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(128, 64, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 16, context_embd_dim=context_embd_dim)),

        ])

        self.output_layer = nn.Sequential(
            nn.GroupNorm(4, 64),
            nn.SiLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t, context):
        t = TimeEmbedding(t, self.time_embd_dim)

        if context is not None:
            context = self.context_emb(context)

        residuals = []
        for layer in self.encoder:
            x = layer(x, t, context)
            residuals.append(x)

        for layer in self.bottleneck:
            x = layer(x, t, context)

        for layer in self.decoder:
            x = torch.cat((x, residuals.pop()), dim=1)
            x = layer(x, t, context)

        return self.output_layer(x)

'''
n = UNET()
print(sum(p.numel() for p in n.parameters() if p.requires_grad))'''