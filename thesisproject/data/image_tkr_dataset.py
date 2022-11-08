import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path
from contextlib import contextmanager

from .image_tkr import ImageTKR

class Flip:
    def __init__(self, is_right):
        self.is_right = is_right

    def __call__(self, image):
         # Flip coronal plane
        image = torch.flip(image, dims=(1))

        if self.is_right:
            image = torch.flip(image, dims=(2))

        return image

class ImageTKRDataset(Dataset):
    def __init__(self, image_base_dir, subjects_csv, predict_mode=False, image_transform=None):
        """
        image_base_dir is the folder containing the images
        subjects_csv contains the fields: filename, subject_id_and_knee, is_right, TKR, visit

        The samples are individual knees of individual subjects
        """
        self.image_base_dir = Path(image_base_dir)
        self.subjects_df = pd.read_csv(subjects_csv)
        self.predict_mode = predict_mode

        self.image_transform = image_transform

        self._subjects = self._get_image_tkr_objects()

    def __len__(self):
        return len(self._subjects)

    def __getitem__(self, index):
        return self._subjects[index]

    def __iter__(self):
        for subject in self._subjects:
            yield subject

    def load(self):
        for subject in self._subjects:
            subject.load()

    def unload(self):
        for subject in self._subjects:
            subject.unload()

    def _get_image_tkr_objects(self):
        """
        Initialize all ImageSeries objects, with unloaded image-files
        """
        subjects = []
        for i, row in self.subjects_df.iterrows():
            is_right = row["is_right"]
            TKR = int(row["TKR"])
            filename = row["filename"]
            subject_id_and_knee = row["subject_id_and_knee"]

            transform = T.Compose([self.image_transform])#, Flip(is_right)])

            image_tkr = ImageTKR(
                subject_id_and_knee,
                self.image_base_dir / Path(filename),
                label=TKR,
                image_transform=transform
            )

            subjects.append(image_tkr)

        return subjects
    
    @contextmanager
    def get_random_image(self):
        idx = np.random.randint(len(self))
        try:
            yield self._subjects[idx]
        finally:
            self._subjects[idx].unload()

    @contextmanager
    def get_image_at_index(self, idx):
        try:
            yield self._subjects[idx]
        finally:
            self._subjects[idx].unload()