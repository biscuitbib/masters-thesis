import numpy as np
import torch
from contextlib import contextmanager
import torch
from pathlib import Path
from typing import List

class FeatureExtractVector:
    """
    Imaging biomarker series and medical outcome label.
    """
    def __init__(self, identifier, feature_vectors, label=None):
        self.predict_mode = label is not None
        self.identifier = identifier

        self.feature_vectors = feature_vectors
        self.vector_size = feature_vectors[0].shape[0]

        self._images = None
        self._label = label

        #TODO implement view interpolator
        self._interpolator = None

        self.im_dtype = torch.float32
        self.lab_dtype = torch.uint8

    @property
    def is_loaded(self):
        return self._images is not None

    @property
    def image(self):
        return self.feature_vectors

    @property
    def label(self):
        return self._label

    def load(self):
        pass

    def unload(self):
        pass

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
