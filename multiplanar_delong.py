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
from thesisproject.models.lstm import LSTM, LitFullLSTM
from thesisproject.models.linear_model import LinearDataModule, LinearModel, LitFullLinearModel
from thesisproject.utils import delong_roc_test, zero_oner

assert torch.cuda.is_available()

with open("/home/blg515/masters-thesis/hparams.yaml", "r") as stream:
    hparams = yaml.safe_load(stream)

args = sys.argv[1:]

if len(args) > 0:
    index = int(args[0])

    encoding_size_index = index % 3
    hidden_size_index = index // 3
    n_visits_index = index // 3
else:
    encoding_size_index = 0
    hidden_size_index = 0
    n_visits_index = 0

# Data
image_path = "/home/blg515/masters-thesis/oai_images" #"/home/blg515/ucph-erda-home/OsteoarthritisInitiative/NIFTY/"
subjects_csv = "/home/blg515/masters-thesis/image_samples.csv"

# data
train_indices = np.load("/home/blg515/masters-thesis/train_ids.npy", allow_pickle=True).astype(str)
val_indices = np.load("/home/blg515/masters-thesis/val_ids.npy", allow_pickle=True).astype(str)
test_indices = np.load("/home/blg515/masters-thesis/test_ids.npy", allow_pickle=True).astype(str)

n_visits = hparams["linear_regression"]["n_visits"][n_visits_index]
# linear
data = LinearDataModule(
    image_path,
    subjects_csv,
    batch_size=1,
    n_visits=n_visits,
    train_slices_per_epoch=1000,
    val_slices_per_epoch=500,
    train_indices=train_indices,
    val_indices=val_indices,
    test_indices=test_indices
)
data.setup("test")

## Models
mpu_checkpoint = hparams["mpu_path"]
unet = UNet(1, 9, 384)
lit_mpu = LitMPU(unet).load_from_checkpoint(mpu_checkpoint, unet=unet)

encoding_size = hparams["encoder"]["encoding_size"][encoding_size_index]
encoder_checkpoint = hparams["encoder"]["encoding_path"][encoding_size_index]

encoder = Encoder(1448, unet.fc_in, encoding_size)

hidden_size = hparams["lstm"]["hidden_size"][hidden_size_index]
lstm_checkpoint = hparams["lstm"]["fixed_lstm_path"][encoding_size_index * 3 + hidden_size_index]

lstm = LSTM(encoder.vector_size + 1, hidden_size, 2) #input size is vector size + 1 if adding dt

linear_checkpoint = hparams["linear_regression"]["fixed_linear_path"][encoding_size_index * 3 + n_visits_index]

linear = LinearModel(encoding_size * n_visits)

full_linear_checkpoint = hparams["linear_regression"]["full_linear_path"]
linear_model = LitFullLinearModel(unet, encoder, linear, n_visits=n_visits).load_from_checkpoint(full_linear_checkpoint, unet=unet, encoder=encoder, linear=linear, n_visits=n_visits)

full_lstm_checkpoint = hparams["lstm"]["full_lstm_path"]
lstm_model = LitFullLSTM(unet, encoder, lstm).load_from_checkpoint(full_lstm_checkpoint, unet=unet, encoder=encoder, lstm=lstm)

dataloader = data.test_dataloader()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

linear_model.to(device)
lstm_model.to(device)

# compute predictions
true_labels = []
linear_predictions = []
lstm_predictions = []

for batch in tqdm(dataloader):
    imageObject = batch[0]
    with imageObject.loaded_in_context():
        images, labels, timedeltas = imageObject.image.to(device), imageObject.label, imageObject.timedeltas[-n_visits:].to(device)

        with torch.no_grad():
            lstm_pred, _ = lstm_model.predict(images, timedeltas)#[:, 1]
            linear_pred = linear_model.predict(images)

            lstm_predictions.append(lstm_pred.detach().cpu().clone())
            linear_predictions.append(linear_pred.detach().cpu().clone())
            true_labels.append(labels)

ys = np.array(true_labels)
linear_predictions = np.array(linear_predictions)
lstm_predictions = np.array(lstm_predictions)

fpr_lr, tpr_lr, _ = roc_curve(ys, linear_predictions)
auc_score_lr = auc(fpr_lr, tpr_lr)

fpr_lstm, tpr_lstm, _ = roc_curve(ys, lstm_predictions)
auc_score_lstm = auc(fpr_lstm, tpr_lstm)

print(auc_score_lr, auc_score_lstm)

p_delong = 10**(delong_roc_test(ys, zero_oner(linear_predictions), zero_oner(lstm_predictions)))
print(f"p-Value for linear n={n_visits} and lstm h={hidden_size}: ", p_delong)