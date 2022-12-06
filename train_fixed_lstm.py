import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from thesisproject.data import FixedLstmDataModule
from thesisproject.models import LitEncoder, LitFixedLSTM, UNet, LSTM

# Data
image_path = "/home/blg515/ucph-erda-home/OsteoarthritisInitiative/NIFTY/"
subjects_csv = "/home/blg515/image_samples.csv"

## Model
encoder_checkpoint = "/home/blg515/masters-thesis/model_saves/encoder/lightning_logs/version_7447/checkpoints/encoder.ckpt"

label_keys = ["Lateral femoral cart.",
                "Lateral meniscus",
                "Lateral tibial cart.",
                "Medial femoral cartilage",
                "Medial meniscus",
                "Medial tibial cart.",
                "Patellar cart.",
                "Tibia"]

unet = UNet(1, 9, 384, class_names=label_keys)
lit_encoder = LitEncoder(unet).load_from_checkpoint(encoder_checkpoint, unet=unet)

lstm_data = FixedLstmDataModule(
    image_path,
    subjects_csv,
    lit_encoder,
    batch_size=8,
    train_slices_per_epoch=2000,
    val_slices_per_epoch=1000
)

lstm = LSTM(unet.encoding_size, 1000, 2)
model = LitFixedLSTM(lstm)

# Training
checkpoint_path = None

# Callbacks
early_stopping = EarlyStopping("loss/val_loss", mode="min", min_delta=0.0, patience=15)
lr_monitor = LearningRateMonitor(logging_interval='epoch')

num_gpus = torch.cuda.device_count()

trainer = pl.Trainer(
    fast_dev_run=False,
    accelerator='gpu',
    devices=num_gpus,
    strategy="ddp" if num_gpus > 1 else None,
    callbacks=[early_stopping, lr_monitor],
    default_root_dir="model_saves/fixed-lstm/",
    profiler="simple",
    auto_lr_find=True,
    auto_scale_batch_size=False,
    enable_progress_bar=True,
    max_epochs=100
)

trainer.tune(model, datamodule=lstm_data)

trainer.fit(
    model=model,
    ckpt_path=checkpoint_path,
    datamodule=lstm_data
)
