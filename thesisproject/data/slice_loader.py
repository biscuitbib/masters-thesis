import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
import elasticdeform.torch as etorch

from thesisproject.data.image_queue import ImageQueue

class SliceLoader(IterableDataset):
    def __init__(
        self,
        dataset_queue: ImageQueue,
        slices_per_epoch=2500,
        image_transform=None,
        label_transform=None,
        only_fg=False
        ):

        self.dataset_queue = dataset_queue
        self.slices_per_epoch = slices_per_epoch
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.only_fg = only_fg

    def __len__(self):
        return self.slices_per_epoch

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            len = self.slices_per_epoch
        else:
            num_workers = worker_info.num_workers
            len = self.slices_per_epoch // num_workers

        for i in range(len):
            with self.dataset_queue.get_random_image() as imagepair:
                yield self._get_random_slice(imagepair)

    def _get_random_slice(self, imagepair):
        """
        Get slice for randomly selected volume, along random axes.
        Image batch must have shape: B x C x H x W
        Label batch must have shape: B x H x W
        """
        image_volume, label_volume = imagepair.image, imagepair.label

        has_fg = False
        i = 0
        while not has_fg and i < 1000:
            permute_idx = np.random.choice(3)
            axis_to_permute = [[0, 1, 2], [1, 0, 2], [2, 0, 1]][permute_idx]

            # transpose volume to make new first axis
            image_transpose = image_volume.permute(axis_to_permute)
            label_transpose = label_volume.permute(axis_to_permute)

            slice_depth = np.random.randint(image_transpose.shape[0])

            image_slice = image_transpose[slice_depth, :, :]
            label_slice = label_transpose[slice_depth, :, :]

            valid_labels = (label_slice != 0) * (label_slice < 8)
            if True or self.only_fg and torch.sum(valid_labels > 0):
                has_fg = True

            i += 1


        if np.random.random() <= 1/3:
            displacement_val = np.random.randn(2, 5, 5) * 5.
            displacement = torch.tensor(displacement_val)
            [image_slice, label_slice] = etorch.deform_grid([image_slice, label_slice], displacement, order=0)
            #image_slice = etorch.deform_grid(image_slice, displacement, order=0)


        image_slice -= image_slice.min()
        image_slice /= image_slice.max()

        image_slice = image_slice.unsqueeze(0)

        return image_slice, label_slice