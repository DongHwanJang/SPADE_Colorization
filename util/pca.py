import torch
from torch import nn

def isnan(x):
    return (x != x).any()


class PCA(nn.Module):
    def forward(self, X, k=2):
        # preprocess the data
        if isnan(X):
            raise ValueError("It founds NaN value before starting SVD.")

        X_mean = torch.mean(X, 0)
        X = X - X_mean.expand_as(X) + torch.Tensor([1e-8]).cuda().expand_as(X)

        # svd
        U, S, V = torch.svd(torch.t(X), some=False)
        return torch.mm(X, U[:, :k])