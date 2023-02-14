import torch
from torch import nn

class LinearModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.lr = nn.Linear(input_size, 2)

    def forward(x):
        return self.lr(x)
