import os
import numpy as np

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from skimage.io import imread


class Dataset2D(Dataset):
    def __init__(self, img_dir: str, labels_dir: str, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.target_transform = target_transform
        self._img_names = sorted(os.listdir(img_dir))
        self._label_names = sorted(os.listdir(labels_dir))
        self._len = len(self._img_names)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self._img_names[index])
        label_path = os.path.join(self.labels_dir, self._label_names[index])

        image = imread(img_path)
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)

        label = imread(label_path)#torch.from_numpy(imread(label_path)).unsqueeze(dim=0)
        label = torch.from_numpy(label.astype(np.long)).unsqueeze(dim=0)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label