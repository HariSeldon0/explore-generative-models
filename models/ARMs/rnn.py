import torch
from torch import nn


class MNIST_LSTM(nn.Module):
    def __init__(self, hidden_size=100, num_layers=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_size = 1  # grayscale
        self.output_size = 256  # 0 - 255
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            self.input_size, self.hidden_size, self.num_layers, batch_first=True
        )
        self.output = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden):
        outs, hidden = self.lstm(input, hidden)  # hidden: (batch, seq, hidden)
        logits = self.output(outs)
        return logits, hidden

    def get_init_hidden(self, batch_size):
        return (
            torch.zeros(
                self.num_layers, batch_size, self.hidden_size, dtype=torch.float
            ),
            torch.zeros(
                self.num_layers, batch_size, self.hidden_size, dtype=torch.float
            ),
        )
