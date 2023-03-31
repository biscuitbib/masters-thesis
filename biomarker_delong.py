import os
import sys

import yaml
import pickle
import numpy as np
import pandas as pd
import re
import pytorch_lightning as pl
import torch
from torch.nn.functional import softmax
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import Normalizer, StandardScaler
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from thesisproject.models.mpu import UNet, LitMPU
from thesisproject.models.encoder import Encoder, LitEncoder, EncoderDataModule
from thesisproject.models.lstm import LSTMDataModule, LSTM, LitFixedLSTM,BiomarkerLSTMDataModule, LitBiomarkerLSTM
from thesisproject.models.linear_model import LinearDataModule, LinearModel, LitFixedLinearModel
from thesisproject.utils import delong_roc_test, zero_oner

with open("/home/blg515/masters-thesis/hparams.yaml", "r") as stream:
    hparams = yaml.safe_load(stream)

args = sys.argv[1:]

if len(args) > 0:
    index = int(args[0])

    hidden_size_index = index % 3
    n_visits_index = index // 3

n_visits = hparams["linear_regression"]["n_visits"][n_visits_index]


# data
train_indices = np.load("/home/blg515/masters-thesis/train_ids.npy", allow_pickle=True).astype(str)
test_indices = np.load("/home/blg515/masters-thesis/test_ids.npy", allow_pickle=True).astype(str)

regex = re.compile("feature_extract(_\d+)?\.csv")
csv_files = [file for file in os.listdir("/home/blg515/masters-thesis/") if regex.match(file)]

subjects_df = pd.concat([pd.read_csv(csv) for csv in csv_files], ignore_index=True).sample(frac=1)
subjects_df = subjects_df.drop_duplicates().reset_index(drop=True)

subject_id_and_knees = subjects_df["subject_id_and_knee"].unique()

def create_data(indices, n_visits=1):
    max_visits = subjects_df.groupby("subject_id_and_knee").size().max()

    subjects = []
    labels = []
    print("Creating dataset...")
    for subject_id_and_knee in indices:
        # Extract feature vectors and timedeltas
        rows = subjects_df[subjects_df["subject_id_and_knee"] == subject_id_and_knee].fillna(0.0)
        if rows.shape[0] < n_visits:
            continue

        rows = rows.sort_values("visit")
        TKR = int(rows["TKR"].values[0])
        features = rows.loc[:, ~rows.columns.isin(["subject_id_and_knee", "TKR", "filename", "is_right"])]
        features["visit"] -= features["visit"].min()
        features = features.values # list of feature vectors
        last_feature = features[-1,:]
        subject_visits = features.shape[0]
        padding = np.zeros((max_visits - subject_visits, features.shape[1]))
        features_padded = np.concatenate([features, padding], axis=None)

        subjects.append(features[-n_visits:, :].reshape(-1))
        labels.append(TKR)

    subjects = np.array(subjects)
    labels = np.array(labels)
    return subjects, labels

X_train, y_train = create_data(test_indices, n_visits=n_visits)
Xs, ys = create_data(test_indices, n_visits=n_visits)

# standardize
normalizer = StandardScaler().fit(X_train)
X_normalized = normalizer.transform(Xs)

# imaging biomarker linear
linear_model = pickle.load(open(f"biomarker-linear-n{n_visits}.pickle", "rb"))

# imaging biomarker lstm
regex = re.compile("feature_extract(_\d+)?\.csv")
csv_files = [file for file in os.listdir("/home/blg515/masters-thesis/") if regex.match(file)]

subjects_df = pd.concat([pd.read_csv(csv) for csv in csv_files], ignore_index=True).sample(frac=1)
n_features = subjects_df.shape[1] - 4 # Exclude columns: filename, subject_id_and_knee, TKR, is_right

hidden_size = hparams["lstm"]["hidden_size"][hidden_size_index]
lstm_checkpoint = hparams["lstm"]["biomarker_lstm_path"][hidden_size_index]

lstm = LSTM(n_features, hidden_size, 2)
lstm_model = LitBiomarkerLSTM(lstm).load_from_checkpoint(lstm_checkpoint, lstm=lstm)
lstm_model.eval()

device = lstm_model.device

# compute predictions
linear_predictions = []
lstm_predictions = []
print(len(list(zip(X_normalized, ys))))
for X, y in tqdm(zip(X_normalized, ys)):
    linear_input = X.reshape((1, -1))[:, np.arange(X.shape[0]) % 39 != 0]
    lstm_input = torch.tensor(X.reshape((1, n_visits, -1))).to(device).float()

    linear_output = linear_model.predict(linear_input)[0]
    with torch.no_grad():
        lstm_output = lstm_model.predict(lstm_input)[:,1]

    linear_predictions.append(linear_output)
    lstm_predictions.append(lstm_output)

ys = np.array(ys)
linear_predictions = np.array(linear_predictions)
lstm_predictions = np.array(lstm_predictions)

fpr_lr, tpr_lr, _ = roc_curve(ys, linear_predictions)
auc_score_lr = auc(fpr_lr, tpr_lr)

fpr_lstm, tpr_lstm, _ = roc_curve(ys, lstm_predictions)
auc_score_lstm = auc(fpr_lstm, tpr_lstm)

print(auc_score_lr, auc_score_lstm)

p_delong = 10**(delong_roc_test(ys, zero_oner(linear_predictions), zero_oner(lstm_predictions)))
print(f"p-Value for linear n={n_visits} and lstm h={hidden_size}: ", p_delong)