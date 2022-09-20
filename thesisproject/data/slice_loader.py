import numpy as np
import torch
from torch.utils.data import DataLoader
import elasticdeform.torch as etorch

from thesisproject.data.image_queue import ImageQueue

class SliceLoader(DataLoader):
    def __init__(
        self,
        dataset_queue: ImageQueue,
        slices_per_batch=16,
        slices_per_epoch=2500,
        volumes_per_batch= 8,
        image_transform=None,
        label_transform=None,
        *dataloader_args,
        **dataloader_kwargs
        ):
        self.dataset_queue = dataset_queue
        self.slices_per_batch = slices_per_batch
        self.slices_per_epoch = slices_per_epoch
        self.volumes_per_batch = volumes_per_batch
        self.image_transform = image_transform
        self.label_transform = label_transform

        total_image_access = (self.slices_per_epoch // slices_per_batch) * self.volumes_per_batch

        self.dataset_queue.set_len(total_image_access)

        super().__init__(
            dataset_queue,
            *dataloader_args,
            **dataloader_kwargs,
            batch_size=self.volumes_per_batch,
            collate_fn=self.collate_slices
        )

    def collate_slices(self, batch):
        """
        Get n slices for randomly selected m volumes, along random axes.
        Image batch must have shape: B x C x H x W
        Label batch must have shape: B x H x W
        """
        print(batch)
        image_slices = []
        label_slices = []
        while len(image_slices) < self.slices_per_batch:
            idx = min(len(batch) - 1, np.random.randint(self.volumes_per_batch))
            imagepair = batch[idx]
            print(imagepair)
            image, label = imagepair.image, imagepair.label
            weight = 1.

            permute_idx = np.random.choice(3)
            axis_to_permute = [[0, 1, 2], [1, 0, 2], [2, 0, 1]][permute_idx]

            image = image.permute(axis_to_permute)
            label = label.permute(axis_to_permute)

            slice_depth = np.random.randint(image.shape[0])

            image = image[slice_depth, :, :]
            label = label[slice_depth, :, :]

            if torch.sum(label) == 0:
                continue

            if True:
                if np.random.random() <= 1/3:
                    displacement_val = np.random.randn(2, 5, 5) * 5.
                    displacement = torch.tensor(displacement_val)
                    [image, label] = etorch.deform_grid([image, label], displacement, order=0)
                    weight = 1/3

                image -= image.min()
                image /= image.max()

            image_slices.append(image.unsqueeze(dim=0))
            label_slices.append(label.unsqueeze(dim=0))
            weights.append(weight)

        image_slices = torch.cat(image_slices).unsqueeze(dim=1).float()
        label_slices = torch.cat(label_slices).long()
        weights = torch.tensor(weights)

        return image_slices, label_slices