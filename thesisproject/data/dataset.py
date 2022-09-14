import os
import random
import numpy as np
from queue import Queue
import torch
import torchvision.transforms as T
from torch.utils.data import IterableDataset,  Dataset, DataLoader
import nibabel as nib
from skimage.io import imread
import elasticdeform.torch as etorch

class ImagePair:
    def __init__(self, img_path, label_path)

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

        if np.random.random > 1/3:
            [image, label] = etorch.deform_random_grid([image, label], sigma=np.random.uniform(20, 30))

        return image, label

class ImageQueue(IterableDataset):
    def __init__(self, base_dir, transform=None, target_transform=None, queue_length=16, max_access=10, max_iterations=200):
        self.dataset = ImageData(base_dir, transform=transform, target_transform=target_transform)

        self.queue_length = min(queue_length, len(self.dataset))
        self.max_access = max_access
        self.max_iterations = max_iterations

        self.image_queue = Queue(maxsize=queue_length)

        self._load_queue_full()

        self.non_loaded_indices = set(np.arange(len(self.dataset)))
        self.loaded_indices = set()

        self.total_iterations = 0


    def _add_to_queue(self, idx):
        image, label = self.dataset[idx]
        self.image_queue.put((image, label, idx, 0))

    def _load_queue_full(self):
        random_indices = set(np.random.choice(self.queue_length))
        self.loaded_indices = random_indices
        self.non_loaded_indices = self.non_loaded_indices - self.loaded_indices
        for idx in random_indices:
            self._add_to_queue(idx)

    def __len__(self):
        return self.total_iterations

    def __iter__(self):
        self.max_iterations += 1

        image, label, cur_idx, n_access = self.image_queue.get()

        if n_access <= self.max_access:
            self.image_queue.put((image, label, n_access + 1))
        else:
            new_idx = np.random.choice(self.non_loaded_indices)
            self.non_loaded_indices.remove(new_idx)
            self.non_loaded_indices.add(cur_idx)
            self.loaded_indices.remove(cur_idx)
            self.loaded_indices.add(new_idx)

            self._add_to_queue(new_idx)

        yield image, label


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
