import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from thesisproject.data.image_tkr_dataset import ImageTKRDataset
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
        train_val_test_ratios = [0.6, 0.2, 0.2]
    ):
        super().__init__()
        self.data_dir = data_dir
        self.subjects_csv = subjects_csv
        self.batch_size = batch_size
        self.train_slices_per_epoch = train_slices_per_epoch
        self.val_slices_per_epoch = val_slices_per_epoch
        self.test_slices_per_epoch = test_slices_per_epoch
        self.num_workers = min(4, os.cpu_count())

        assert len(train_val_test_ratios) == 3 and sum(train_val_test_ratios) == 1
        self.train_ratio, self.val_ratio, self.test_ratio = train_val_test_ratios

        self.collate_fn = None

    """
    def collate_fn(self, data):
        data = [(input.squeeze(0), target) for input, target in data]
        return data
    """

    def setup(self, stage: str):
        """
        TODO split into train and val
        """
        subjects_df = pd.read_csv(self.subjects_csv).sample(frac=1).head(20)

        total_train_df, test_df = train_test_split(subjects_df, test_size=self.test_ratio)
        train_df, val_df = train_test_split(total_train_df, test_size=self.val_ratio)

        train_dataset = ImageTKRDataset(self.data_dir, train_df, image_transform=SquarePad())

        self.train_slice_loader = SliceSeriesLoader(train_dataset, slices_per_epoch=self.train_slices_per_epoch)

        val_dataset = ImageTKRDataset(self.data_dir, val_df, image_transform=SquarePad())

        self.val_slice_loader = SliceSeriesLoader(val_dataset, slices_per_epoch=self.train_slices_per_epoch, elastic_deform=False)

        test_dataset = ImageTKRDataset(self.data_dir, test_df, image_transform=SquarePad())

        self.test_slice_loader = SliceSeriesLoader(test_dataset, slices_per_epoch=self.test_slices_per_epoch, elastic_deform=False)


    def train_dataloader(self):
        return DataLoader(self.train_slice_loader, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_slice_loader, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_slice_loader, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)