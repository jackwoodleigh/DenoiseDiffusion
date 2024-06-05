import torch
from torch import nn
import math
from torch.nn import functional as F
class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False),
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class ContextEmbedding(nn.Module):
    def __init__(self, num_labels, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.condEmbedding(t)
        return emb

class Down_Sample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 5, stride=2, padding=2)

    def forward(self, x, temb, cemb):
        x = self.conv1(x) + self.conv2(x)
        return x

class Up_Sample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, temb, cemb):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.proj_q = nn.Conv2d(channels, channels, 1, padding=0)
        self.proj_k = nn.Conv2d(channels, channels, 1, padding=0)
        self.proj_v = nn.Conv2d(channels, channels, 1, padding=0)
        self.proj = nn.Conv2d(channels, channels, 1, padding=0)

    def forward(self, x):
        residual = x
        B, C, H, W = x.shape

        x = self.group_norm(x)
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        x = torch.bmm(w, v)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.proj(x)

        return x + residual





class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, tdim, dropout_rate=0.3, self_attention=True):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout(dropout_rate),
            nn.SiLU()
        )
        self.block_2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout(dropout_rate),
            nn.SiLU()
        )
        self.t_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(tdim, out_channels),
        )
        self.context_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(tdim, out_channels),
        )

        # matching channels of residual
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        if self_attention:
            self.self_attention = SelfAttention(out_channels)
        else:
            self.self_attention = nn.Identity()

    def forward(self, x, t, context):
        residual = x
        t = self.t_proj(t)[:, :, None, None]
        context = self.context_proj(context)[:, :, None, None]

        x = self.block_1(x)
        x += t
        x += context
        x = self.block_2(x)

        x += self.residual(residual)
        x = self.self_attention(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, T, block_layout, d_model=64, block_multiplier=2):
        super().__init__()
        if block_layout is None:
            block_layout = [2, 2, 2, 2]
        self.tdim = d_model * 4
        self.time_emb = TimeEmbedding(T, d_model, self.tdim)
        self.context_emb = ContextEmbedding(10, d_model, self.tdim)

        self.input = nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1)


        # Decoder
        self.down = nn.ModuleList()
        down_channel_list = [d_model]
        previous_channels = d_model
        for i, count in enumerate(block_layout):
            current_channels = d_model * (block_multiplier**(i+1))

            # for number of blocks in current layer
            for block in range(count):
                self.down.append(ResBlock(previous_channels, current_channels, self.tdim))
                previous_channels = current_channels
                down_channel_list.append(previous_channels)

            # adding down sample blocks
            if i != len(block_layout)-1:
                self.down.append(Down_Sample(previous_channels))
                down_channel_list.append(previous_channels)

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResBlock(previous_channels, previous_channels, self.tdim),
            ResBlock(previous_channels, previous_channels, self.tdim)
        ])


        # Encoder
        self.up = nn.ModuleList()
        # Layers in encoder
        for i, count in reversed(list(enumerate(block_layout))):
            current_channels = d_model * (block_multiplier**(i+1))

            # for number of blocks in current layer
            for block in range(count + 1):
                self.up.append(ResBlock(down_channel_list.pop() + previous_channels, current_channels, self.tdim))
                previous_channels = current_channels

            # adding down sample blocks
            if i != 0:
                self.up.append(Up_Sample(previous_channels))


        self.output = nn.Sequential(
            nn.GroupNorm(32, previous_channels),
            nn.Conv2d(previous_channels, previous_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(previous_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t, context):
        t = self.time_emb(t)
        context = self.context_emb(context)

        x = self.input(x)
        skip_connect = [x]

        # Decoder
        for layer in self.down:
            x = layer(x, t, context)
            skip_connect.append(x)

        # Bottleneck
        for layer in self.bottleneck:
            x = layer(x, t, context)

        # Encoder
        for layer in self.up:
            if isinstance(layer, ResBlock):
                x = torch.cat([x, skip_connect.pop()], dim=1)
            x = layer(x, t, context)

        # Output Layer
        x = self.output(x)

        return x


model = UNet(in_channels=3, out_channels=3, T=1000, block_layout=[2, 2, 2, 2], d_model=128, block_multiplier=2)
x = torch.randn(10, 3, 32, 32)
t = torch.randint(1, 1000, (10,))
context = torch.randint(0, 10, (10,))
#print(model(x, t, context).shape)
