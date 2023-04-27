import os
import sys
import gc
import psutil

import json
import yaml
import numpy as np
import pandas as pd
import re
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay, roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from thesisproject.models.mpu import UNet, LitMPU
from thesisproject.models.encoder import Encoder, LitEncoder, EncoderDataModule
from thesisproject.models.lstm import LSTMDataModule, LSTM, LitFixedLSTM, LitFullLSTM
from thesisproject.models.linear_model import LinearDataModule, LinearModel, LitFixedLinearModel, LitFullLinearModel

with open("/home/blg515/masters-thesis/hparams.yaml", "r") as stream:
    hparams = yaml.safe_load(stream)

args = sys.argv[1:]

if len(args) > 0:
    index = int(args[0])
    model_type = args[1]
    is_test = args[2] == "test"

    encoding_size_index = index % 3
    hidden_size_index = index // 3
    n_visits_index = index // 3
else:
    model_name = "lr"
    is_test = True

    encoding_size_index = 0
    hidden_size_index = 0
    n_visits_index = 0

# Data
image_path = "/home/blg515/masters-thesis/oai_images" #"/home/blg515/ucph-erda-home/OsteoarthritisInitiative/NIFTY/"
subjects_csv = "/home/blg515/masters-thesis/image_samples.csv"

test_df = pd.read_csv("/home/blg515/masters-thesis/results.csv")

train_indices = np.load("/home/blg515/masters-thesis/train_ids.npy", allow_pickle=True).astype(str)
val_indices = np.load("/home/blg515/masters-thesis/val_ids.npy", allow_pickle=True).astype(str)
#test_indices = np.load("/home/blg515/masters-thesis/test_ids.npy", allow_pickle=True).astype(str)
test_indices = np.array(["9066155-R", "9672573-R", "9975485-R", "9638123-R", "9708289-R", "9800285-R", "9456233-R", "9102858-R", "9297051-R", "9922855-L", "9689788-R", "9732727-R", "9676101-L", "9485359-L", "9684122-L", "9053047-R", "9850238-R", "9588436-R", "9561770-R", "9732751-R", "9627172-R", "9368395-R"])

n_visits = hparams["linear_regression"]["n_visits"][n_visits_index]

if model_type == "lstm":
    data = LSTMDataModule(
        image_path,
        subjects_csv,
        batch_size=1,
        train_slices_per_epoch=1000,
        val_slices_per_epoch=500,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices
    )
else:
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

if model_type == "lstm":
    full_lstm_checkpoint = hparams["lstm"]["full_lstm_path"]
    model = LitFullLSTM(unet, encoder, lstm).load_from_checkpoint(full_lstm_checkpoint, unet=unet, encoder=encoder, lstm=lstm)
else:
    full_linear_checkpoint = hparams["linear_regression"]["full_linear_path"]
    model = LitFullLinearModel(unet, encoder, linear, n_visits=n_visits).load_from_checkpoint(full_linear_checkpoint, unet=unet, encoder=encoder, linear=linear, n_visits=n_visits)

print(model.lr, model.weight_decay)
exit()
dataloader = data.test_dataloader()

#Change depending on model
if model_type == "lstm":
    model_name = f"full_lstm_encoding_size={encoding_size}_hidden_size={hidden_size}_" + ("test" if is_test else "train")
    col_name = f"full_lstm_l{encoding_size}_h{hidden_size}_" + ("test" if is_test else "train")
else:
    model_name = f"full_lr_encoding_size={encoding_size}_n_visits={n_visits}_" + ("test" if is_test else "train")
    col_name = f"full_lr_l{encoding_size}_n{n_visits}_" + ("test" if is_test else "train")

if col_name not in test_df.columns:
    test_df[col_name] = np.nan

print("Calculating AUC for " + model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

true_volume_predictions = None
n_true = 0
false_volume_predictions = None
n_false = 0

predictions = []
true_labels = []
for batch in tqdm(dataloader):
    imageObject = batch[0]
    with imageObject.loaded_in_context():
        images, labels, timedeltas = imageObject.image.to(device), imageObject.label, imageObject.timedeltas.to(device)

        with torch.no_grad():
            if model_type == "lstm":
                output, volume_prediction = model.predict(images, timedeltas)#[:, 1]
            else:
                output, volume_prediction = model.predict(images)#[:, 1]

            predictions.append(output.item())
            true_labels.append(labels)

            # Add prediction value to test_df
            identifier = imageObject.identifier
            index = test_df[test_df["subject_id_and_knee"] == identifier].index[0]
            test_df.at[index, col_name] = output.item()

    volume_prediction = np.stack([view_prediction.detach().cpu().numpy() for view_prediction in volume_prediction])
    if output >= 0.5 and labels == 1:
        np.save(f"/home/blg515/masters-thesis/prediction_volumes/lstm_{identifier}", volume_prediction)

test_df.to_csv(f"/home/blg515/masters-thesis/results_{model_name}.csv", index=False)
