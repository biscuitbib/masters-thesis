from contextlib import contextmanager
import nibabel as nib
import torch
from pathlib import Path

class ImageSeries:
    """
    Image series and medical outcome label
    """
    def __init__(self, image_paths: list(str), label, sample_weight=1.0, image_transform=None):
        self.predict_mode = not label

        self.sample_weight = sample_weight
        self.image_transform = image_transform

        self.image_paths = [Path(image_path) for image_path in image_paths]

        # Patient identifier
        self.identifier = self._get_identifier()
        self._image_objs = self._load_image_objects()

        self._images = None
        self._label = label

        #TODO implement view interpolator
        self._interpolator = None

        self.im_dtype = torch.float32
        self.lab_dtype = torch.uint8

    def _get_identifier(self):
        img_id = self.image_path.stem.split('.')[0]
        if not self.predict_mode:
            label_id = self.label_path.stem.split('.')[0]
            if img_id != label_id:
                raise ValueError("Image identifier '%s' does not match labels identifier '%s'"
                                 % (img_id, label_id))
        return img_id

    @property
    def is_loaded(self):
        return self._image is not None

    @property
    def image(self):
        if self._image is None:
            self._image = torch.from_numpy(self._image_obj.get_fdata(caching='unchanged')).type(self.im_dtype)

            if self.image_transform:
                self._image = self.image_transform(self._image)

        if self._image.ndim == 3:
            self._image.unsqueeze(0)

        return self._image

    @property
    def label(self):
        if self._label is None:
            try:
                self._label = torch.from_numpy(self._label_obj.get_fdata(caching="unchanged")).type(self.lab_dtype)
                if self.label_transform:
                    self._label = self.label_transform(self._label)

            except AttributeError:
                return None

        return self._label

    def load(self):
        self.image
        self.label

    def unload(self):
        self._image = None
        self._label = None

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
