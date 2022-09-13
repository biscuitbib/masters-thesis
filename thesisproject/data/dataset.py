import os
import sys
import numpy as np

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from skimage.io import imread

"""
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
"""

class ImageData(Dataset):
    """
    Class for dataset of 2D slices of 3D images.
    3D images are in .nii.gz file format.
    
    TODO
    Add queue to load more image pairs into memory, instead of loading as needed.
    """
    def __init__(self, base_dir, transform=None, target_transform=None, num_access=1):
        self.img_dir = os.path.join(base_dir, "images")
        self.label_dir = os.path.join(base_dir, "labels")
        self.filenames = sorted(os.listdir(self.img_dir))
        
        self.transform = transform
        self.target_transform = target_transform
        
        self.dataset_len = len(self.filenames)
        self.num_access = num_access
        
    def __len__(self):
        return self.dataset_len * self.num_access

    def __getitem__(self, idx):
        file_idx = idx % self.dataset_len
        
        img_path = os.path.join(self.img_dir, self.filenames[file_idx])
        label_path = os.path.join(self.label_dir, self.filenames[file_idx])
        
        image = nib.load(img_path).get_fdata()
        label = nib.load(label_path).get_fdata()
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label


class SliceLoader(DataLoader):
    def __init__(self, dataset, slices_per_batch=16, volumes_per_batch=8, *dataloader_args, **dataloader_kwargs):
        self.slices_per_batch = slices_per_batch
        self.volumes_per_batch = volumes_per_batch
        self.transform = T.Resize(150)
        super().__init__(
            dataset,
            *dataloader_args, 
            **dataloader_kwargs, 
            batch_size=volumes_per_batch, 
            collate_fn=self.collate_slices
        )
    
    def collate_slices(self, batch):
        """
        Get n slices for randomly selected m volumes, along random axes.
        Image batch must have shape: B x C x H x W
        Label batch must have shape: B x H x W
        """
        image_slices = []
        label_slices = []
        while len(image_slices) < self.slices_per_batch:
            idx = min(len(batch) - 1, np.random.randint(self.volumes_per_batch))
            imagepair = batch[idx]
            image, label = imagepair[0], imagepair[1]
            
            permute_idx = np.random.choice(3)
            axis_to_permute = [[0, 1, 2], [1, 0, 2], [2, 0, 1]][permute_idx]
            
            image = image.permute(axis_to_permute)
            label = label.permute(axis_to_permute)
            
            slice_depth = np.random.randint(image.shape[0])
            
            image = image[slice_depth, :, :]
            label = label[slice_depth, :, :]
            
            if torch.sum(label) == 0:
                continue
                
            image_slices.append(image.unsqueeze(dim=0))
            label_slices.append(label.unsqueeze(dim=0))
            
        image_slices = torch.cat(image_slices).unsqueeze(dim=1).float()
        label_slices = torch.cat(label_slices).long()
        
        return image_slices, label_slices
            
        