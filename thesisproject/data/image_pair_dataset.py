import os
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from .image_pair import ImagePair

class ImagePairDataset(Dataset):
    def __init__(self, dir, predict_mode=False, image_transform=None, label_transform=None):
        self.dir = Path(dir)
        self.predict_mode = predict_mode

        self.image_transform = image_transform
        self.label_transform = label_transform

        self.image_dir = self.dir / "images"
        self.label_dir = None
        if not self.predict_mode:
            self.label_dir = self.dir / "labels"

        self.image_paths = self._get_paths(self.image_dir)
        self.label_paths = None
        if not self.predict_mode:
            self.label_paths = self._get_paths(self.label_dir)

        self.images = self._get_image_objects()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        return self.images[index]

    def _get_paths(self, dir):
        filenames = sorted(os.listdir(dir))
        paths = [Path(p) for p in filenames]
        return paths

    def _get_image_objects(self):
        """
        Initialize all ImagePair objects from paths at self.image_paths and
        self.label_paths (if labels exist). Note that data is not loaded
        eagerly.

        Returns:
            A list of initialized ImagePairs
        """
        image_objects = []
        if self.predict_mode:
            for img_path in self.image_paths:
                image = ImagePair(self.image_dir / img_path, sample_weight=1.0)
                image_objects.append(image)
        else:
            for img_path, label_path in zip(self.image_paths, self.label_paths):
                image = ImagePair(self.image_dir / img_path, label_path=self.label_dir / label_path, sample_weight=1.0)
                image_objects.append(image)

        return image_objects

    def get_random(self, N=1, unique=False):
        """
        Return N random images, with or without re-sampling

        Args:
            N:      Int, number of randomly sampled images to return
            unique: Bool, whether the sampled images should be all unique

        Returns:
            A list of ImagePair objects
        """
        returned = []
        while len(returned) < N:
            if self.queue:
                with self.queue.get() as image:
                    if unique and image in returned:
                        continue
                    else:
                        returned.append(image)
                        yield image
            else:
                image = self.images[np.random.randint(len(self))]
                if unique and image in returned:
                    continue
                else:
                    returned.append(image)
                    yield image
