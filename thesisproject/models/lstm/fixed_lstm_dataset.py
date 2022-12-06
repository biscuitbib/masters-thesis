import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path
from contextlib import contextmanager
from typing import Union, List

class FixedLSTMDataset(Dataset):
    """
    Args:
        Dataset: A serial dataset with outputs K x C x H x W, and binary targets
        Encoder: A U-Net encoder
    """
    def __init__(self, dataset, encoder):
        self.dataset = dataset
        self.encoder = encoder

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        series = self.dataset.__next__()
        self.encoder.eval()
        with torch.no_grad():
            encodings = self.encoder(series)
            return encodings