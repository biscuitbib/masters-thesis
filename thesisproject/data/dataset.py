import os
import numpy as np

import torch
import torchio as tio
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

class SegmentationData:
    """
    Class for dataset of 2D slices of 3D images.
    3D images are in .nii.gz file format.
    Currently only suports single plane slices (sagittal plane)
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.train_dir = os.path.join(data_dir, "train")
        self.train_filenames = os.listdir(
            os.path.join(self.train_dir, "images")
            )

        tmp_image = tio.ScalarImage(os.path.join(self.train_dir, "images", self.train_filenames[0]))

        self.max_side = max(tmp_image.shape)

        self.test_dir = os.path.join(data_dir, "test")
        self.test_filenames = os.listdir(
            os.path.join(self.test_dir, "images")
            )

        self.val_dir = os.path.join(data_dir, "val")
        self.val_filenames = os.listdir(
            os.path.join(self.val_dir, "images")
            )

        self.train_data = self._images_to_subjects(self.train_dir, self.train_filenames)
        self.test_data = self._images_to_subjects(self.test_dir, self.test_filenames)
        self.val_data = self._images_to_subjects(self.val_dir, self.val_filenames)

        self.slice_shape = (self.max_side, self.max_side, 1)

    def _images_to_subjects(self, path, filenames):
        subject_list = []
        for filename in filenames:
            subject_dict = {
                'image': tio.ScalarImage(os.path.join(path, "images", filename)),
                'label': tio.LabelMap(os.path.join(path, "labels", filename)),
            }
            subject = tio.Subject(subject_dict)
            subject_list.append(subject)

        return tio.SubjectsDataset(subject_list, transform=tio.CropOrPad(self.max_side))

    def get_queue(self, dataset, max_queue_length=16, slices_per_volume=2):
        sampler = tio.UniformSampler(self.slice_shape)
        queue = tio.Queue(dataset, max_queue_length, slices_per_volume, sampler)
        return queue
