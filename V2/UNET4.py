import torch
from torch import nn
import math
from torch.nn import functional as F
class Embedding(nn.Module):
    def __init__(self, T, d_model, emb_dim, num_labels):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb.view(T, d_model)

        self.time_embedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False),
            nn.Linear(d_model, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.label_embedding = nn.Embedding(num_embeddings=num_labels, embedding_dim=emb_dim, padding_idx=0)

    def forward(self, time, labels):
        emb = self.time_embedding(time)
        if labels is not None:
            emb += self.label_embedding(labels)
        emb = F.silu(emb)
        return emb


class Down_Sample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 5, stride=2, padding=2)

    def forward(self, x, emb):
        x = self.conv1(x) + self.conv2(x)
        return x

class Up_Sample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, emb):
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



class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, dropout_rate=0.3, self_attention=True):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels)
            #nn.Dropout(dropout_rate),

        )
        self.block_2 = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),

        )
        self.embedding_block = nn.Linear(emb_dim, 2*out_channels)

        # matching channels of residual
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        if self_attention:
            self.self_attention = SelfAttention(out_channels)
        else:
            self.self_attention = nn.Identity()

    def forward(self, x, embedding):
        residual = x

        x = self.block_1(x)

        emb = self.embedding_block(embedding)
        emb1, emb2 = emb.chunk(2, dim=1)
        x *= emb1[:, :, None, None] + 1.
        x += emb2[:, :, None, None]

        x = self.block_2(x)

        x += self.residual(residual)

        x = self.self_attention(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, T, block_structure, block_multiplier, d_model=64):
        super().__init__()
        self.emb_dim = d_model * 4
        self.embedding = Embedding(T, d_model, self.emb_dim, 10)

        self.input = nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1)

        # Decoder
        self.down = nn.ModuleList()
        down_channel_list = [d_model]
        previous_channels = d_model
        for i, count in enumerate(block_structure):
            current_channels = d_model * (block_multiplier[i]**(i+1))
            # for number of blocks in current layer
            for block in range(count):
                self.down.append(DownBlock(previous_channels, current_channels, self.emb_dim ))
                previous_channels = current_channels
                down_channel_list.append(previous_channels)

            # adding down sample blocks
            if i != len(block_structure)-1:
                self.down.append(Down_Sample(previous_channels))
                down_channel_list.append(previous_channels)

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            DownBlock(previous_channels, previous_channels, self.emb_dim ),
            DownBlock(previous_channels, previous_channels, self.emb_dim )
        ])


        # Encoder
        self.up = nn.ModuleList()
        # Layers in encoder
        for i, count in reversed(list(enumerate(block_structure))):
            current_channels = d_model * (block_multiplier[i]**(i+1))

            # for number of blocks in current layer
            for block in range(count + 1):
                self.up.append(DownBlock(down_channel_list.pop() + previous_channels, current_channels, self.emb_dim ))
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

    def forward(self, x, t, labels):
        emb = self.embedding(t, labels)

        x = self.input(x)
        skip_connect = [x]

        # Decoder
        for layer in self.down:
            x = layer(x, emb)
            skip_connect.append(x)

        # Bottleneck
        for layer in self.bottleneck:
            x = layer(x, emb)

        # Encoder
        for layer in self.up:
            if isinstance(layer, DownBlock):
                x = torch.cat([x, skip_connect.pop()], dim=1)
            x = layer(x, emb)

        # Output Layer
        x = self.output(x)

        return x





model = UNet(in_channels=3, out_channels=3, T=1000, block_structure=[1, 1, 1], d_model=64, block_multiplier=[2, 2, 2])
x = torch.randn(10, 3, 32, 32)
t = torch.randint(1, 1000, (10,))
labels = torch.randint(0, 10, (10,))
print(model(x, t, labels).shape)


