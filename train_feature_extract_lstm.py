import os

import numpy as np
import pytorch_lightning as pl
import torch
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from sklearn.utils.estimator_checks import check_estimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
from tqdm import tqdm

from thesisproject.models.lstm import BiomarkerLSTMDataModule, LitBiomarkerLSTM, LSTM

train_indices = np.load("/home/blg515/train_ids.npy", allow_pickle=True).astype(str)
val_indices = np.load("/home/blg515/val_ids.npy", allow_pickle=True).astype(str)
test_indices = np.load("/home/blg515/test_ids.npy", allow_pickle=True).astype(str)

df0 = pd.read_csv("/home/blg515/masters-thesis/feature_extract_0.csv")
df1 = pd.read_csv("/home/blg515/masters-thesis/feature_extract_1.csv")
df2 = pd.read_csv("/home/blg515/masters-thesis/feature_extract_2.csv")
df3 = pd.read_csv("/home/blg515/masters-thesis/feature_extract_3.csv")

subjects_df = pd.concat([df0, df1, df2, df3], ignore_index=True).sample(frac=1)
n_features = subjects_df.shape[1] - 4 # Exclude columns: filename, subject_id_and_knee, TKR, is_right

lstm_data = BiomarkerLSTMDataModule(
    subjects_df, batch_size=8,
    train_indices=train_indices,
    val_indices=val_indices,
    test_indices=test_indices
    )

lstm = LSTM(n_features, 1000, 2)
model = LitBiomarkerLSTM(lstm)

# Callbacks
early_stopping = EarlyStopping("loss/val_loss", mode="min", min_delta=0.0, patience=15)
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# Training
checkpoint_path = None

num_gpus = torch.cuda.device_count()

print("num_gpus: ", num_gpus)

trainer = pl.Trainer(
    accelerator='gpu',
    devices=num_gpus,
    strategy="ddp" if num_gpus > 1 else None,
    callbacks=[early_stopping, lr_monitor],
    default_root_dir="model_saves/biomarker-lstm/",
    profiler="simple",
    auto_lr_find=True,
    auto_scale_batch_size=False,
    enable_progress_bar=True,
    accumulate_grad_batches=2,
    max_epochs=100
)

#trainer.tune(model, datamodule=lstm_data)

trainer.fit(
    model=model,
    ckpt_path=checkpoint_path,
    datamodule=lstm_data
)