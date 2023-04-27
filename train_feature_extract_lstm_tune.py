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
from sklearn.model_selection import train_test_split, KFold
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

weight_decay = [1e-6, 1e-5, 1e-6][hidden_size_index]

kf = KFold(n_splits=5)

wds = [10**(float(i)) for i in np.arange(-4, 2, 1)]
best_acc = 0
best_wd = 0
for wd in wds:
    avg_acc = 0
    for train_index, val_index in kf.split(train_indices):
        train_indices_fold = train_indices[train_index]
        val_indices_fold = train_indices[val_index]

        lstm_data = BiomarkerLSTMDataModule(
            subjects_df,
            batch_size=8,
            train_indices=train_indices_fold,
            val_indices=val_indices_fold,
            test_indices=test_indices
            )

        hidden_size = hparams["lstm"]["hidden_size"][hidden_size_index]
        lstm = LSTM(n_features, hidden_size, 2)
        model = LitBiomarkerLSTM(lstm, lr=1e-4, weight_decay=wd)

        #print(f"Training imaging biomarker LSTM with hidden_size={hidden_size}")

        # Callbacks
        early_stopping = EarlyStopping("loss/val_loss", mode="min", min_delta=0.0, patience=30)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        # Training
        checkpoint_path = None

        num_gpus = torch.cuda.device_count()

        #print("num_gpus: ", num_gpus)

        trainer = pl.Trainer(
            accelerator='gpu',
            devices=num_gpus,
            strategy="ddp" if num_gpus > 1 else None,
            callbacks=[early_stopping, lr_monitor],
            default_root_dir="model_saves/biomarker-lstm-tune/",
            profiler=None,
            auto_lr_find=True,
            auto_scale_batch_size=False,
            enable_progress_bar=False,
            max_epochs=50
        )

        print(f"saving model to {trainer.logger.log_dir}")

        trainer.fit(
            model=model,
            ckpt_path=checkpoint_path,
            datamodule=lstm_data
        )

        results = trainer.validate(model=model, datamodule=lstm_data)
        avg_acc += results[0]["val/accuracy"] / 5

    if avg_acc > best_acc:
        best_acc = avg_acc
        best_wd = wd

print(f"Best weight decay of {best_wd} for hidden size {hidden_size}")