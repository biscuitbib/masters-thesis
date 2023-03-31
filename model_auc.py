import os
import sys
import gc
import psutil

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
from thesisproject.models.lstm import LSTMDataModule, LSTM, LitFixedLSTM,BiomarkerLSTMDataModule, LitBiomarkerLSTM
from thesisproject.models.linear_model import LinearDataModule, LinearModel, LitFixedLinearModel, LitFullLinearModel

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

train_indices = np.load("/home/blg515/masters-thesis/train_ids.npy", allow_pickle=True).astype(str)
val_indices = np.load("/home/blg515/masters-thesis/val_ids.npy", allow_pickle=True).astype(str)
test_indices = np.load("/home/blg515/masters-thesis/test_ids.npy", allow_pickle=True).astype(str)

"""
# encoder
data = EncoderDataModule(
    image_path,
    subjects_csv,
    batch_size=8,
    train_slices_per_epoch=1000,
    val_slices_per_epoch=1000,
    train_indices=train_indices,
    val_indices=val_indices,
    test_indices=val_indices
)
# lstm
data = LSTMDataModule(
    image_path,
    subjects_csv,
    batch_size=8,
    train_slices_per_epoch=1000,
    val_slices_per_epoch=500,
    train_indices=train_indices,
    val_indices=val_indices,
    test_indices=val_indices
)
"""
# linear
n_visits = hparams["linear_regression"]["n_visits"][n_visits_index]
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
#model = LitEncoder(unet, encoder).load_from_checkpoint(encoder_checkpoint, unet=unet, encoder=encoder)

hidden_size = hparams["lstm"]["hidden_size"][hidden_size_index]
lstm_checkpoint = hparams["lstm"]["fixed_lstm_path"][encoding_size_index * 3 + hidden_size_index]

lstm = LSTM(encoder.vector_size + 1, hidden_size, 2) #input size is vector size + 1 if adding dt
#model = LitFixedLSTM(unet, encoder, lstm).load_from_checkpoint(lstm_checkpoint, unet=unet, encoder=encoder, lstm=lstm)

linear_checkpoint = hparams["linear_regression"]["fixed_linear_path"][encoding_size_index * 3 + n_visits_index]

linear = LinearModel(encoding_size * n_visits)
#model = LitFixedLinearModel(unet, encoder, linear, n_visits=n_visits).load_from_checkpoint(linear_checkpoint, unet=unet, encoder=encoder, linear=linear, n_visits=n_visits)

full_linear_checkpoint = hparams["linear_regression"]["full_linear_path"]
model = LitFullLinearModel(unet, encoder, linear, n_visits=n_visits).load_from_checkpoint(full_linear_checkpoint, unet=unet, encoder=encoder, linear=linear, n_visits=n_visits)

"""
# imaging biomarker lstm
regex = re.compile("feature_extract(_\d+)?\.csv")
csv_files = [file for file in os.listdir("/home/blg515/masters-thesis/") if regex.match(file)]

subjects_df = pd.concat([pd.read_csv(csv) for csv in csv_files], ignore_index=True).sample(frac=1)
n_features = subjects_df.shape[1] - 4 # Exclude columns: filename, subject_id_and_knee, TKR, is_right

lstm_index = int(args[0])
hidden_size = hparams["lstm"]["hidden_size"][lstm_index]
lstm_checkpoint = hparams["lstm"]["biomarker_lstm_path"][lstm_index]

data = BiomarkerLSTMDataModule(
    subjects_df,
    batch_size=8,
    train_indices=train_indices,
    val_indices=val_indices,
    test_indices=test_indices
    )
data.setup("test")

lstm = LSTM(n_features, hidden_size, 2)
model = LitBiomarkerLSTM(lstm).load_from_checkpoint(lstm_checkpoint, lstm=lstm)
"""

dataloader = data.test_dataloader()

predictions = []
true_labels = []

#Change depending on model
model_name = f"full_lr_encoding_size={encoding_size}_n_visits={n_visits}_test"
print("Calculating AUC for " + model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

for batch in tqdm(dataloader):
    images, labels = batch[0].to(device), batch[1]

    with torch.no_grad():
        outputs = model.predict(images)#[:, 1]

        print(outputs, labels)

        predictions.append(outputs.detach().cpu().clone())
        true_labels.append(labels)

    del images
    del outputs

    gc.collect()
    torch.cuda.empty_cache()

    """
    if predictions is None:
        predictions = outputs
    else:
        predictions = torch.cat([predictions, outputs])

    if true_labels is None:
        true_labels = labels
    else:
        true_labels = torch.cat([true_labels, labels])
    """

fpr, tpr, _ = roc_curve(true_labels, predictions)
auc_score = auc(fpr, tpr)

plt.style.use("seaborn")
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color="blue", label=f"Encoder (AUC={round(auc_score, 3)})")

plt.xlim(-0.01, 1.01)
plt.ylim(-0.01, 1.01)
ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
plt.plot([0, 1], [0, 1], linestyle="dashed", color="black", alpha=0.5, label="Random Classifier")
plt.legend()
plt.savefig(model_name + "_auc.png")
