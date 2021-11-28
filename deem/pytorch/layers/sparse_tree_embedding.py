import entmax
import torch
from torch import nn
from ..torch_utils import set_device


class ObliviousTreeEmbedding(nn.Module):
    def __init__(self, in_features, width, depth, entmax_alpha=4.0, gpu=-1):
        super().__init__()
        self.device = set_device(gpu)
        self.layers = [nn.Linear(in_features, width) for _ in range(depth)]
        self.layers = nn.ModuleList(self.layers)
        self.entmax_alpha = entmax_alpha
        self.to(self.device)

    def forward(self, X):
        dense_factors = [layer(X) for layer in self.layers]
        factors = [entmax.entmax_bisect(factor, alpha=self.entmax_alpha, dim=-1, n_iter=25) for factor in dense_factors]
        output = factors[0]
        for x in range(1, len(factors)):
            # TODO: batch outer product on sparse tensors (broadcasting to dim 0)
            output = torch.einsum('bi,bj->bij', (output, factors[x]))  # batch outer product
            output = torch.flatten(output, start_dim=1)  # flatten to batched vectors
        return output
