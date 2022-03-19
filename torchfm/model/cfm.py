import torch
import torch.nn as nn
from torchfm.layer import (AttentionalFactorizationMachine, FeaturesEmbedding,
                           FeaturesLinear, Conv_block1D)

class ConvFM(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, attn_size, dropouts):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_fields = len(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.afm = AttentionalFactorizationMachine(embed_dim, attn_size, dropouts)
        self.cfm = Conv_block1D([1, 32, 64, 128, 256, 512])

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        conv = self.afm(self.embedding(x)).reshape(x.shape[0], -1, self.embed_dim)
        conv = self.cfm(conv)
        x = self.linear(x) + conv
        return x.squeeze(1)
