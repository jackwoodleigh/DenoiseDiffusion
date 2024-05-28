import torch
from torch import nn
from torch.nn import functional as f
import math

class SelfAttention(nn.Module):

    def __init__(self, n_heads, d_embed, in_projection_bias=True, out_projection_bias=False):
        super().__init__()
        self.in_projection = nn.Linear(d_embed, 3*d_embed, bias=in_projection_bias)
        self.out_projection = nn.Linear(d_embed, d_embed, bias=out_projection_bias)
        self.n_heads = n_heads
        self.d_heads = d_embed // n_heads


    def forward(self, x, c_mask=False):
        input_shape = x.shape

