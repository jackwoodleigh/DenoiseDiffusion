import torch
from torch import nn
from torch.nn import functional as f
from Attention import SelfAttention, CrossAttention
import math
from einops import rearrange, reduce, repeat
'''def TimeEmbedding(t, dim, max_period=1000, device="cuda"):
    half_dim = dim // 2
    frequencies = torch.exp(-torch.log(torch.tensor(max_period, device=device, dtype=torch.float32)) * torch.arange(half_dim, device=device) / half_dim)
    angles = t[:, None].float() * frequencies[None, :]
    embeddings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:  # if dim is odd, add an extra zero column
        embeddings = torch.cat([embeddings, torch.zeros_like(t[:, None])], dim=-1)
    return embeddings
'''
'''class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim=256, is_random=True):
        super().__init__()
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)
    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered'''

class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


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
    def __init__(self, in_channels, out_channels, time_embd_dim=256, dropout_rate=0.1):
        super().__init__()
        self.input = nn.Sequential(
            nn.GroupNorm(4, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout(dropout_rate),
            nn.SiLU(),
        )
        self.time_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embd_dim, out_channels)
        )

        self.output = nn.Sequential(
            nn.GroupNorm(4, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout(dropout_rate),
            nn.SiLU(),
        )

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.initialize()
    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        #nn.init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

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
        self.input = nn.Sequential(
            nn.GroupNorm(32, channels, eps=1e-6),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        )

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

        x = self.input(x)

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


class Up_Sample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.initialize()
    def initialize(self):
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = f.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class Down_Sample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, context_embd_dim=256, time_embd_dim=256):
        super().__init__()
        self.context_embd_dim = context_embd_dim
        self.time_embd_dim = time_embd_dim
        self.context_emb = nn.Embedding(num_embeddings=10, embedding_dim=context_embd_dim)

        self.encoder = nn.ModuleList([
            SwitchSequential(Down_Sample(in_channels, 64, kernel_size=3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(64, 64, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 16, context_embd_dim=context_embd_dim)),
            #SwitchSequential(UNET_ResidualBlock(64, 64, time_embd_dim=time_embd_dim),UNET_AttentionBlock(4, 16, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(64, 128, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 32, context_embd_dim=context_embd_dim)),

            # Image size / 2
            SwitchSequential(Down_Sample(128, 128, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(128, 128, time_embd_dim=time_embd_dim),  UNET_AttentionBlock(4, 32, context_embd_dim=context_embd_dim)),
            #SwitchSequential(UNET_ResidualBlock(128, 128, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 32, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(128, 256, time_embd_dim=time_embd_dim),  UNET_AttentionBlock(4, 64, context_embd_dim=context_embd_dim)),

            # Image size / 4
            SwitchSequential(Down_Sample(256, 256, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(256, 256, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 64, context_embd_dim=context_embd_dim)),
            #SwitchSequential(UNET_ResidualBlock(256, 256, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 64, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(256, 512, time_embd_dim=time_embd_dim),  UNET_AttentionBlock(4, 128, context_embd_dim=context_embd_dim)),

            # Image size / 8
            SwitchSequential(Down_Sample(512, 512, kernel_size=3, stride=2, padding=1)),
            #SwitchSequential(UNET_ResidualBlock(512, 512, time_embd_dim=time_embd_dim)),
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
            #SwitchSequential(UNET_ResidualBlock(1024, 512, time_embd_dim=time_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(1024, 512, time_embd_dim=time_embd_dim), Up_Sample(512)),

            # Image size / 2
            SwitchSequential(UNET_ResidualBlock(1024, 256, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 64, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(512, 256, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 64, context_embd_dim=context_embd_dim)),
            #SwitchSequential(UNET_ResidualBlock(512, 256, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 64, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(512, 256, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 64, context_embd_dim=context_embd_dim), Up_Sample(256)),

            # Image size
            SwitchSequential(UNET_ResidualBlock(512, 128, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 32, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(256, 128, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 32, context_embd_dim=context_embd_dim)),
            #SwitchSequential(UNET_ResidualBlock(256, 128, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 32, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(256, 128, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 32, context_embd_dim=context_embd_dim), Up_Sample(128)),

            SwitchSequential(UNET_ResidualBlock(256, 64, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 16, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(128, 64, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 16, context_embd_dim=context_embd_dim)),
            #SwitchSequential(UNET_ResidualBlock(128, 64, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 16, context_embd_dim=context_embd_dim)),
            SwitchSequential(UNET_ResidualBlock(128, 64, time_embd_dim=time_embd_dim), UNET_AttentionBlock(4, 16, context_embd_dim=context_embd_dim)),

        ])

        self.output_layer = nn.Sequential(
            nn.GroupNorm(4, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),

        )
        self.time_embedding = RandomOrLearnedSinusoidalPosEmb()

    def forward(self, x, t, context):
        t = self.time_embedding(t)

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

        x = self.output_layer(x)

        return x

'''
n = UNET()
print(sum(p.numel() for p in n.parameters() if p.requires_grad))'''