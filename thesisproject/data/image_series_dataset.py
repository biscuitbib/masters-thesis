import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path
from contextlib import contextmanager

from .image_series import ImageSeries

class Flip:
    def __init__(self, is_right):
        self.is_right = is_right

    def __call__(self, image):
         # Flip coronal plane
        image = torch.flip(image, dims=(1))

        if self.is_right:
            image = torch.flip(image, dims=(2))

        return image

class ImageSeriesDataset(Dataset):
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
        subject_knee_images_dict = self._get_subject_knee_images_dict()
        for subject_id_and_knee, filenames in subject_knee_images_dict.items():
            row = self.subjects_df(self.subjects_df["subject_id_and_knee"] == subject_id_and_knee)
            is_right = row["is_right"].values[0]
            TKR = int(row["TKR"].values[0])

            transform = T.Compose([self.image_transform, Flip(is_right)])

            image_series = ImageSeries(
                subject_id_and_knee,
                [self.image_base_dir + Path(filename) for filename in filenames],
                label=TKR,
                image_transform=transform
            )

            subjects.append(image_series)

        return subjects

    def _get_subject_knee_images_dict(self):
        subject_id_and_knee = self.subjects_df.sort_values("subject_id_and_knee")["subject_id_and_knee"].values

        subject_id_and_knee_filenames = self.subjects_df.sort_values("subject_id_and_knee").groupby("subject_id_and_knee")["filename"].apply(list).values

        return dict(zip(subject_id_and_knee, subject_id_and_knee_filenames))

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