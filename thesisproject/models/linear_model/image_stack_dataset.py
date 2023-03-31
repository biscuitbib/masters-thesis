import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path
from contextlib import contextmanager

from .image_stack import ImageStack

class Flip:
    def __init__(self, is_right):
        self.is_right = is_right

    def __call__(self, image):
         # Flip coronal plane
        image = torch.flip(image, dims=(1,))

        if self.is_right:
            image = torch.flip(image, dims=(2,))

        return image

class ImageStackDataset(Dataset):
    def __init__(self, image_base_dir, subjects_df, predict_mode=False, image_transform=None, n_visits=1):
        """
        image_base_dir is the folder containing the images
        subjects_csv contains the fields: filename, subject_id_and_knee, is_right, TKR, visit

        The samples are individual knees of individual subjects
        """
        self.image_base_dir = Path(image_base_dir)
        self.subjects_df = subjects_df
        self.predict_mode = predict_mode

        self.n_visits = n_visits

        self.image_transform = image_transform

        self._subjects = self._get_image_series_objects()

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

    def _get_image_series_objects(self):
        """
        Initialize all ImageSeries objects, with unloaded image-files
        """
        subjects = []
        subject_id_and_knees = self.subjects_df["subject_id_and_knee"].unique()
        for subject_id_and_knee in subject_id_and_knees:
            rows = self.subjects_df[self.subjects_df["subject_id_and_knee"] == subject_id_and_knee]
            rows = rows.sort_values("visit")
            is_right = rows["is_right"].values[0]
            TKR = int(rows["TKR"].values[0])
            filenames = rows["filename"].values
            timedeltas = rows["visit"].values
            timedeltas -= timedeltas[0]

            transform = T.Compose([self.image_transform, Flip(is_right)])

            image_stack = ImageStack(
                subject_id_and_knee,
                [self.image_base_dir / Path(filename) for filename in filenames],
                timedeltas=timedeltas,
                label=TKR,
                image_transform=transform,
                n_visits=self.n_visits
            )

            subjects.append(image_stack)

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