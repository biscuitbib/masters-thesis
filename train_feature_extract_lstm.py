import os
import sys

import yaml
import numpy as np
import re
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

with open("/home/blg515/masters-thesis/hparams.yaml", "r") as stream:
    hparams = yaml.safe_load(stream)

args = sys.argv[1:]
hidden_size_index = int(args[0])

train_indices = np.load("/home/blg515/masters-thesis/train_ids.npy", allow_pickle=True).astype(str)
val_indices = np.load("/home/blg515/masters-thesis/val_ids.npy", allow_pickle=True).astype(str)
test_indices = np.load("/home/blg515/masters-thesis/test_ids.npy", allow_pickle=True).astype(str)

regex = re.compile("feature_extract(_\d+)?\.csv")
csv_files = [file for file in os.listdir("/home/blg515/masters-thesis/") if regex.match(file)]

subjects_df = pd.concat([pd.read_csv(csv) for csv in csv_files], ignore_index=True).sample(frac=1)
n_features = subjects_df.shape[1] - 4 # Exclude columns: filename, subject_id_and_knee, TKR, is_right

lstm_data = BiomarkerLSTMDataModule(
    subjects_df,
    batch_size=8,
    train_indices=train_indices,
    val_indices=val_indices,
    test_indices=test_indices
    )

hidden_size = hparams["lstm"]["hidden_size"][hidden_size_index]
weight_decay = [0.01, 0.001, 0.01][hidden_size_index]
lstm = LSTM(n_features, hidden_size, 2)
model = LitBiomarkerLSTM(lstm, lr=1e-5, weight_decay=weight_decay)

print(f"Training imaging biomarker LSTM with hidden_size={hidden_size}")

# Callbacks
early_stopping = EarlyStopping("loss/val_loss", mode="min", min_delta=0.0, patience=30)
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
    #accumulate_grad_batches=2,
    max_epochs=2000
)

print(f"saving model to {trainer.logger.log_dir}")

trainer.fit(
    model=model,
    ckpt_path=checkpoint_path,
    datamodule=lstm_data
)