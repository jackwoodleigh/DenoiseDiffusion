import torch
from torch import nn
from torch.nn import functional as f
from Attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab, n_embd, n_tokens):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.pos_embedding = nn.Parameter(torch.zeros(n_tokens, n_embd))

    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x += self.pos_embedding
        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_head, n_embd):
        super().__init__()
        self.layers = nn.Sequential(


        )
        self.layer_norm = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)
        self.linear1 = nn.Linear(n_embd, 4*n_embd)
        self.linear2 = nn.Linear(4*n_embd, n_embd)

    def forward(self, x):
        residual = x

        # self attention
        x = self.layer_norm(x)
        x = self.attention(x, c_mask=True)
        x += residual

        # feed forward
        residual = x
        x = self.layer_norm2(x)
        x = self.linear1(x)
        x = x * torch.sigmoid(1.702 * x)    # quick gelu
        x = self.linear2(x)
        x += residual

        return x




class CLIP(nn.Module):
    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12)
        ])
        self.layer_norm = nn.LayerNorm(768)

    def foward(self, tokens):
        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)
        for layer in self.layers:
            state = layer(state)

        output = self.layer_norm(state)

        return output