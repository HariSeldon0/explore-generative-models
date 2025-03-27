import torch
from torch import nn


class MNIST_MLP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.flatten = nn.Flatten()
        self.hidden_layers = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU(),
        )
        self.output_layer = nn.Linear(512, 256)

    def forward(self, x, mask):
        # x.size = mask.size = (batch, 28*28)
        x = self.flatten(x)
        x = x * mask  # element-wise product
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
