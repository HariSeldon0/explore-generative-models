import torch
from torch import nn


class PCA(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eigvecs = None  # D, H

    def forward(self, X, dim):
        assert self.eigvecs != None, "self.eigvecs is None"
        Z = X @ self.eigvecs[:, :dim]  # B, H
        X_re = Z @ self.eigvecs[:, :dim].T
        return X_re

    def fit(self, X):
        """calculate self.W

        Args:
            X (Tensor): shape = (N, D)
        """
        X_mean = torch.mean(X, 0, dtype=torch.float)  # D
        X_centered = X - X_mean
        covariance = torch.matmul(X_centered.T, X_centered) / (X_centered.shape[0] - 1)
        eigvals, eigvecs = torch.linalg.eigh(covariance)

        sorted_idx = torch.argsort(eigvals, descending=True)
        self.eigvecs = eigvecs[:, sorted_idx]


class MLP_AE(nn.Module):
    def __init__(self, input_dim, hidden_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y, z


class CNN_AE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = nn.Sequential()
