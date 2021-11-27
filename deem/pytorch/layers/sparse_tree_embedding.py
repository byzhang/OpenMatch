import entmax
import torch
from torch import nn


class ObliviousTreeEmbedding(nn.Module):
    def __init__(self, in_features, width, depth, entmax_alpha=4.0):
        super().__init__()
        self.layers = [nn.Linear(in_features, width) for _ in range(depth)]
        self.entmax_alpha = entmax_alpha

    def forward(self, X):
        factors = [entmax.entmax_bisect(layer(X), alpha=self.entmax_alpha, dim=-1, n_iter=25) for layer in self.layers]
        output = factors[0]
        for x in range(1, len(factors)):
            output = torch.einsum('bi,bj->bij', (output, factors[x]))  # batch outer product
            output = torch.flatten(output, start_dim=1)  # flatten to batched vectors
        return output
