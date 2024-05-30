import torch
from torch import nn
from torch.nn import functional as f
import math

class SelfAttention(nn.Module):

    def __init__(self, n_heads, d_embed, in_projection_bias=True, out_projection_bias=True):
        super().__init__()
        # part used to split input into Q K V
        self.in_projection = nn.Linear(d_embed, 3*d_embed, bias=in_projection_bias)

        # weights multiplied by H (result of joined heads)
        self.out_projection = nn.Linear(d_embed, d_embed, bias=out_projection_bias)

        self.n_heads = n_heads
        self.d_heads = d_embed // n_heads

    def forward(self, x, c_mask=False):
        # input = (batch_size, sequence_length/channel, embedding/pixels)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        intermediate_shape = (batch_size, sequence_length, self.n_heads, self.d_heads)

        # split into 3 output tensors that represent q k v
        q, k, v = self.in_projection(x).chunk(3, dim=-1)

        # result: (batch_size, seq_length, h, dim/h) -> (batch_size, h, seq_length, dim/h)
        # now have q,v,k have h heads each having a size of (seq_length, dim/h)
        q = q.view(intermediate_shape).transpose(1, 2)
        k = k.view(intermediate_shape).transpose(1, 2)
        v = v.view(intermediate_shape).transpose(1, 2)

        # this is now (batch_size, h, seq_length, seq_length)
        weight = q @ k.transpose(-1, -2)


        # cannot watch future tokens only past
        if c_mask:
            # mask where everything above the principle diagonal is all 1 else 0
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)

            # make all values above diagonal neg inf such that softmax sets 0
            weight.masked_fill(mask, -torch.inf)

        weight /= math.sqrt(self.d_heads)

        weight = f.softmax(weight, dim=-1)

        # result (batch_size, h, seq_length, dim/h)
        output = weight @ v

        output = output.transpose(1, 2)
        output = output.reshape(input_shape)
        output = self.out_projection(output)
        return output
