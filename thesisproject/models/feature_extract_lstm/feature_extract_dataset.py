import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path
from contextlib import contextmanager

from .feature_extract_vector import FeatureExtractVector

class ImageSeriesDataset(Dataset):
    def __init__(self, subjects_df, predict_mode=False):
        """
        image_base_dir is the folder containing the images
        subjects_csv contains the fields: filename, subject_id_and_knee, is_right, TKR, visit

        The samples are individual knees of individual subjects
        """
        self.subjects_df = subjects_df
        self.predict_mode = predict_mode

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
        Initialize all feature vectors
        """
        subjects = []
        subject_id_and_knees = self.subjects_df["subject_id_and_knee"].unique()
        for subject_id_and_knee in subject_id_and_knees:
            # Extract feature vectors and timedeltas
            rows = self.subjects_df[self.subjects_df["subject_id_and_knee"] == subject_id_and_knee]
            rows = rows.sort_values("visit")
            TKR = int(rows["TKR"].values[0])
            features = rows.loc[:, ~rows.columns.isin(["subject_id_and_knee", "TKR", "filename", "is_right"])]
            features["visit"] -= features["visit"].min()
            features = features.values # list of feature vectors

            feature_vector = FeatureExtractVector(subject_id_and_knee, features, TKR)

            subjects.append(feature_vector)

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