import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
import elasticdeform.torch as etorch
from typing import Union


class SliceSeriesLoader(IterableDataset):
    def __init__(
        self,
        dataset_queue,
        slices_per_epoch=2500,
        image_transform=None,
        elastic_deform=True
        ):

        self.dataset_queue = dataset_queue
        self.slices_per_epoch = slices_per_epoch
        self.image_transform = image_transform
        self.elastic_deform = elastic_deform

    def __len__(self):
        return self.slices_per_epoch

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            len = self.slices_per_epoch
        else:
            num_workers = worker_info.num_workers
            len = self.slices_per_epoch // num_workers

        for i in range(len):
            with self.dataset_queue.get_random_image() as imagepair:
                yield self._get_random_slice(imagepair)

    def _get_random_slice(self, imageseries):
        """
        Get slice for randomly selected volume, along random axes.
        Image volumes have shape K x C x H x H x H (cubic images)
        Labels are K x C
        Image slices have shape K x C x H x H
        """
        image_volumes: torch.Tensor = imageseries.image
        label: torch.Tensor = imageseries.label

        has_fg = False
        while not has_fg:
            permute_idx = np.random.choice(3)
            axis_to_permute = [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4], [0, 1, 4, 2, 3]][permute_idx]

            # transpose volume to make new first axis
            image_transpose = image_volumes.permute(axis_to_permute)

            slice_depth = np.random.randint(image_transpose.shape[2])

            image_slices = image_transpose[:, :, slice_depth, :, :]

            if torch.all(torch.tensor([image_slice.max() > 0 for image_slice in image_slices])):
                has_fg = True

        if self.elastic_deform and np.random.random() <= 1/3:
            image_chunks = [image_slice for image_slice in image_slices]
            deform_axis = [(1, 2) for _ in range(image_slices.shape[0])]
            displacement_val = np.random.randn(2, 5, 5) * 5.
            displacement = torch.tensor(displacement_val)
            image_chunks = etorch.deform_grid(image_chunks, displacement, order=0, axis=deform_axis)
            image_slices = torch.stack(image_chunks, dim=0)

        image_slices = torch.stack([image_slice - image_slice.min() for image_slice in image_slices], dim=0)
        image_slices = torch.stack([image_slice / image_slice.max() if image_slice.max() > 0 else image_slice for image_slice in image_slices], dim=0)
        #image_slices -= torch.min(image_slices, dim=0).values
        #image_slices /= torch.max(image_slices, dim=0).values

        if image_slices.shape[0] == 1:
            image_slices = image_slices.squeeze(0)

        return image_slices, label