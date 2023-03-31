import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler

from .image_series_dataset import ImageSeriesDataset
from thesisproject.data.slice_series_loader import SliceSeriesLoader
from .biomarker_lstm_dataset import BiomarkerLSTMDataset

class BiomarkerLSTMDataModule(pl.LightningDataModule):
    """
    batch[0] is batch of image slice series B x 1 x C x H x W
    batch[1] is batch of TKR labels B x 1
    output of model is batch of softmax predictions B x 2
    """
    def __init__(
        self,
        subjects_df,
        batch_size: int=8,
        max_n: int=None,
        train_indices=None,
        val_indices=None,
        test_indices=None
    ):
        super().__init__()
        self.batch_size = batch_size

        self.subjects_df = subjects_df.sample(frac=1)
        subjects_n = self.subjects_df.shape[0]
        max_n = max(subjects_n, max_n) if max_n is not None else subjects_n
        self.subjects_df = self.subjects_df.head(max_n)

        self.num_workers = min(4, os.cpu_count())

        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

    def collate_fn(self, batch):
        """
        data is list[tuple(input, label)]
        """
        input_list, target_list = [], []
        for [features, label] in batch:
            input_list.append(features)
            target_list.append(label)
        input_list = pad_sequence(input_list, batch_first=True)
        target_list = torch.tensor(target_list)
        return [input_list, target_list]

    def setup(self, stage: str):
        if self.train_indices is None or self.val_indices is None or self.test_indices is None:
            total_train_df, test_df = train_test_split(self.subjects_df, test_size=0.2)
            train_df, val_df = train_test_split(total_train_df, test_size=0.2)
        else:
            train_df = self.subjects_df[self.subjects_df["subject_id_and_knee"].isin(self.train_indices)]
            val_df = self.subjects_df[self.subjects_df["subject_id_and_knee"].isin(self.val_indices)]
            test_df = self.subjects_df[self.subjects_df["subject_id_and_knee"].isin(self.test_indices)]

        # Scale feature columns wrt. training data.
        cols = train_df.columns[~train_df.columns.isin(["subject_id_and_knee", "TKR", "filename", "is_right", "visit"])]
        scaler = StandardScaler().fit(train_df[cols])
        train_df[cols] = scaler.transform(train_df[cols])
        val_df[cols] = scaler.transform(val_df[cols])
        test_df[cols] = scaler.transform(test_df[cols])

        # train data
        self.train_dataset = BiomarkerLSTMDataset(train_df)

        # val data
        self.val_dataset = BiomarkerLSTMDataset(val_df)

        # test data
        self.test_dataset = BiomarkerLSTMDataset(test_df)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)