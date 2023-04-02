import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path
from contextlib import contextmanager
from typing import Union, List

class BiomarkerLSTMDataset(Dataset):
    def __init__(self, subjects_df):
        self.subjects_df = subjects_df
        self.subject_id_and_knees = subjects_df["subject_id_and_knee"].unique()

    def __len__(self):
        return len(self.subject_id_and_knees)

    def __getitem__(self, index):
        subject_id_and_knee = self.subject_id_and_knees[index]
        rows = self.subjects_df[self.subjects_df["subject_id_and_knee"] == subject_id_and_knee].fillna(0.0)

        rows = rows.sort_values("visit")
        TKR = int(rows["TKR"].values[0])
        features = rows.loc[:, ~rows.columns.isin(["subject_id_and_knee", "TKR", "filename", "is_right"])]
        features["visit"] -= features["visit"].min()
        features = features.values

        return torch.from_numpy(features).float(), TKR, subject_id_and_knee