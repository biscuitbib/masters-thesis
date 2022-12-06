from contextlib import contextmanager
import nibabel as nib
import torch
from pathlib import Path

class ImageTKR:
    """
    Image and medical outcome label
    """
    def __init__(self, identifier, image_path: Path, label=None, sample_weight=1.0, image_transform=None):
        self.predict_mode = label is not None
        self.identifier = identifier
        self.sample_weight = sample_weight
        self.image_transform = image_transform

        self.image_path = image_path

        self._image_obj = nib.load(self.image_path)

        self._image = None
        self._label = torch.tensor(label)

        #TODO implement view interpolator
        self._interpolator = None

        self.im_dtype = torch.float32
        self.lab_dtype = torch.uint8

    @property
    def is_loaded(self):
        return self._image is not None

    @property
    def image(self):
        if self._image is None:
            try:
                image = torch.from_numpy(self._image_obj.get_fdata(caching="unchanged")).type(self.im_dtype)
            except:
                raise IOError(f"{self.image_path} is broken!")

            if self.image_transform:
                image = self.image_transform(image)

            if image.ndim == 3:
                image = image.unsqueeze(0)

            self._image = image.unsqueeze(0)

        return self._image

    @property
    def label(self):
        return self._label

    def load(self):
        self._image

    def unload(self):
        self._image = None

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
