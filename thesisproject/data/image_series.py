from contextlib import contextmanager
import nibabel as nib
import torch
from pathlib import Path
from typing import List

class ImageSeries:
    """
    Image series and medical outcome label
    """
    def __init__(self, identifier, image_paths: List[Path], label=None,sample_weight=1.0, image_transform=None):
        self.predict_mode = label is not None
        self.identifier = identifier
        self.sample_weight = sample_weight
        self.image_transform = image_transform

        self.image_paths = image_paths

        self._image_objs = [nib.load(filename) for filename in self.image_paths]

        self._images = None
        self._label = label

        #TODO implement view interpolator
        self._interpolator = None

        self.im_dtype = torch.float32
        self.lab_dtype = torch.uint8

    def _load_image_objects(self):
        return [nib.load(filename) for filename in self.image_paths]

    @property
    def is_loaded(self):
        return self._image is not None

    @property
    def image(self):
        if self._images is None:
            self._images = []
            for obj in self._image_objs:
                image= torch.from_numpy(obj.get_fdata(caching='unchanged')).type(self.im_dtype)

                if self.image_transform:
                    image = self.image_transform(image)

                if image.ndim == 3:
                    image.unsqueeze(0)

                self._images.append(image)

        return self._image

    @property
    def label(self):
        return self._label

    def load(self):
        self._images

    def unload(self):
        self._images = None

    @contextmanager
    def loaded_in_context(self):
        """
        Context manager which keeps this ImagePair loaded in the context
        and unloads it at exit.
        """
        try:
            yield self.load()
        finally:
            self.unload()
