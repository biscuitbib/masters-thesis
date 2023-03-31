import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .image_tkr_dataset import ImageTKRDataset
from thesisproject.data.slice_series_loader import SliceSeriesLoader

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

class EncoderDataModule(pl.LightningDataModule):
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
        train_slices_per_epoch: int=2000,
        val_slices_per_epoch: int=1000,
        test_slices_per_epoch: int=1000,
        train_indices=None,
        val_indices=None,
        test_indices=None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.subjects_csv = subjects_csv
        self.batch_size = batch_size
        self.train_slices_per_epoch = train_slices_per_epoch
        self.val_slices_per_epoch = val_slices_per_epoch
        self.test_slices_per_epoch = test_slices_per_epoch
        self.num_workers = min(4, os.cpu_count())

        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

        self.collate_fn = None

    def setup(self, stage: str):
        subjects_df = pd.read_csv(self.subjects_csv).sample(frac=1)

        if self.train_indices is None or self.val_indices is None:
            train_df, val_df = train_test_split(subjects_df, test_size=0.2)
        else:
            train_df = subjects_df[subjects_df["subject_id_and_knee"].isin(self.train_indices)]
            val_df = subjects_df[subjects_df["subject_id_and_knee"].isin(self.val_indices)]

        train_dataset = ImageTKRDataset(self.data_dir, train_df, image_transform=SquarePad())

        self.train_slice_loader = SliceSeriesLoader(train_dataset, slices_per_epoch=self.train_slices_per_epoch)

        val_dataset = ImageTKRDataset(self.data_dir, val_df, image_transform=SquarePad())

        self.val_slice_loader = SliceSeriesLoader(val_dataset, slices_per_epoch=self.val_slices_per_epoch, elastic_deform=False)

        if self.test_indices is not None:
            test_df = subjects_df[subjects_df["subject_id_and_knee"].isin(self.test_indices)]
            test_dataset = ImageTKRDataset(self.data_dir, test_df, image_transform=SquarePad())
            self.test_slice_loader = SliceSeriesLoader(test_dataset, slices_per_epoch=self.val_slices_per_epoch, elastic_deform=False)
        else:
            self.test_slice_loader = self.val_slice_loader


    def train_dataloader(self):
        return DataLoader(self.train_slice_loader, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_slice_loader, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        if self.test_slice_loader is not None:
            return DataLoader(self.test_slice_loader, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)
        else:
            return self.val_dataloader()