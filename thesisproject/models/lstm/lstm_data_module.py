import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .image_series_dataset import ImageSeriesDataset
from thesisproject.data.slice_series_loader import SliceSeriesLoader
from .fixed_lstm_dataset import FixedLSTMDataset

class SquarePad:
    def __call__(self, image: torch.Tensor):
        imsize = image.shape
        max_edge = np.argmax(imsize)
        pad_amounts = [imsize[max_edge] - imsize[0], imsize[max_edge] - imsize[1], imsize[max_edge] - imsize[2]]

        padding = [int(np.floor(pad_amounts[0] / 2)),
                    int(np.ceil(pad_amounts[0] / 2)),
                    int(np.floor(pad_amounts[1] / 2)),
                    int(np.ceil(pad_amounts[1] / 2)),
                    int(np.floor(pad_amounts[2] / 2)),
                    int(np.ceil(pad_amounts[2] / 2)),] #left, right, top, bottom, front, back
        padding = tuple(padding[::-1])

        padded_im = F.pad(image, padding, "constant", 0)
        return padded_im

class LSTMDataModule(pl.LightningDataModule):
    """
    batch[0] is batch of image slice series B x 1 x C x H x W
    batch[1] is batch of TKR labels B x 1
    output of model is batch of softmax predictions B x 2
    """
    def __init__(
        self,
        data_dir: str,
        subjects_csv: str,
        batch_size: int=8,
        max_n: int=None,
        train_slices_per_epoch: int=2000,
        val_slices_per_epoch: int=1000,
        test_slices_per_epoch: int=1000,
        train_indices=None,
        val_indices=None,
        test_indices=None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.subjects_df = pd.read_csv(subjects_csv).sample(frac=1)
        subjects_n = self.subjects_df.shape[0]
        max_n = max(subjects_n, max_n) if max_n is not None else subjects_n
        self.subjects_df = self.subjects_df.head(max_n)

        self.train_slices_per_epoch = train_slices_per_epoch
        self.val_slices_per_epoch = val_slices_per_epoch
        self.test_slices_per_epoch = test_slices_per_epoch
        self.num_workers = min(4, os.cpu_count())

        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

    def collate_fn(self, batch):
        """
        data is list[tuple(input, label)]
        """
        input_list, target_list, timedeltas_list = [], [], []
        for [image_series, label, timedeltas] in batch:
            if image_series.ndim == 3:
                image_series = image_series.unsqueeze(0)
            input_list.append(image_series)
            target_list.append(label)
            timedeltas_list.append(timedeltas)
        input_list = pad_sequence(input_list, batch_first=True)
        target_list = torch.tensor(target_list)
        timedeltas_list = pad_sequence(timedeltas_list, batch_first=True)
        return [input_list, target_list, timedeltas_list]

    def setup(self, stage: str):
        if self.train_indices is None or self.val_indices is None or self.test_indices is None:
            total_train_df, test_df = train_test_split(self.subjects_df, test_size=0.2)
            train_df, val_df = train_test_split(total_train_df, test_size=0.2)
        else:
            train_df = self.subjects_df[self.subjects_df["subject_id_and_knee"].isin(self.train_indices)]
            val_df = self.subjects_df[self.subjects_df["subject_id_and_knee"].isin(self.val_indices)]
            test_df = self.subjects_df[self.subjects_df["subject_id_and_knee"].isin(self.test_indices)]

        # train data
        self.train_dataset = ImageSeriesDataset(self.data_dir, train_df, image_transform=SquarePad())

        self.train_slice_loader = SliceSeriesLoader(self.train_dataset, slices_per_epoch=self.train_slices_per_epoch)

        # val data
        val_dataset = ImageSeriesDataset(self.data_dir, val_df, image_transform=SquarePad())

        self.val_slice_loader = SliceSeriesLoader(val_dataset, slices_per_epoch=self.train_slices_per_epoch, elastic_deform=False)

        # test data
        self.test_dataset = ImageSeriesDataset(self.data_dir, test_df, image_transform=SquarePad())

        #self.test_slice_loader = SliceSeriesLoader(test_dataset, slices_per_epoch=self.test_slices_per_epoch, elastic_deform=False)

    def train_dataloader(self):
        return DataLoader(self.train_slice_loader, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_slice_loader, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        #TODO check that using correct dataset
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers, pin_memory=True, collate_fn=lambda imageseries: imageseries)