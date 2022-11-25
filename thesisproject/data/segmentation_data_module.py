import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from thesisproject.data.image_pair_dataset import ImagePairDataset
from thesisproject.data.image_queue import ImageQueue
from thesisproject.data.loading_pool import LoadingPool
from thesisproject.data.slice_loader import SliceLoader
from torch.utils.data import DataLoader


def square_pad(image: torch.Tensor):
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

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 10,
        train_slices_per_epoch=2000,
        val_slices_per_epoch=1000
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_slices_per_epoch = train_slices_per_epoch
        self.val_slices_per_epoch = val_slices_per_epoch
        self.num_workers = min(4, os.cpu_count())

    def setup(self, stage: str):
        self.train_dataset = ImagePairDataset(
            self.data_dir + "train",
            image_transform=square_pad,
            label_transform=square_pad)

        self.train_slice_loader = SliceLoader(self.train_dataset, slices_per_epoch=self.train_slices_per_epoch)

        self.val_dataset = ImagePairDataset(
            self.data_dir + "val",
            image_transform=square_pad,
            label_transform=square_pad)

        self.val_slice_loader = SliceLoader(self.val_dataset, slices_per_epoch=self.val_slices_per_epoch, elastic_deform=False)

        self.test_dataset = ImagePairDataset(self.data_dir + "test", predict_mode=False, image_transform=square_pad, label_transform=square_pad)

    def train_dataloader(self):
        return DataLoader(self.train_slice_loader, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_slice_loader, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers, pin_memory=True, collate_fn=lambda imagepair: imagepair)