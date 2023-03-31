import torch
from torch import nn

class LinearModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.lr = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_size, 2)
        )

    def forward(self, x):
        return self.lr(x)
