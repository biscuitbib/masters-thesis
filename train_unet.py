import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from pytorch_lightning.callbacks import EarlyStopping
from thesisproject.data.utils import get_slice_loaders
from thesisproject.models import LitUNet
from torch.utils.data import DataLoader


class SquarePad:
    def __call__(self, image: torch.Tensor):
        imsize = image.shape
        max_edge = np.argmax(imsize)
        pad_amounts = [imsize[max_edge] - imsize[0], imsize[max_edge] - imsize[1], imsize[max_edge] - imsize[2]]

        padding = [int(np.floor(pad_amounts[0] / 2)),
                   int(np.ceil(pad_amounts[0] / 2)),
                   int(np.floor(pad_amounts[1] / 2)),
                   int(np.ceil(pad_amounts[1] / 2)),
                   int(np.floor(pad_amounts[2] / 2)),
                   int(np.ceil(pad_amounts[2] / 2)),] #left, right, top, bottom, front, back
        padding = tuple(padding[::-1])

        padded_im = F.pad(image, padding, "constant", 0)
        return padded_im

#path = "../toy-data/"
path = "../knee_data/"

volume_transform = SquarePad()

train_data, val_data = get_slice_loaders(
    path,
    volume_transform=volume_transform,
    n_train_slices=2000,
    n_val_slices=1000
)

train_loader = DataLoader(train_data, batch_size=8, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=8, num_workers=4, pin_memory=True)

## Train

label_keys = ["Lateral femoral cart.",
                "Lateral meniscus",
                "Lateral tibial cart.",
                "Medial femoral cartilage",
                "Medial meniscus",
                "Medial tibial cart.",
                "Patellar cart.",
                "Tibia"]
"""
label_keys = ["Sphere"]
"""

unet_args = (1, 9, 384) # input channels, output classes, image size
unet_kwargs = {"class_names": label_keys}

unet = LitUNet(
    train_data,
    val_dataset=val_data,
    unet_args=unet_args,
    unet_kwargs=unet_kwargs
)
 #unet = UNet(1, 2, 100, class_names=label_keys)

cont = False

early_stopping = EarlyStopping('val_loss')

trainer = pl.Trainer(
    fast_dev_run=True, #Disable when training
    num_sanity_val_steps=2,
    callbacks=[early_stopping]
)

trainer.fit(
    model=unet,
    default_root_dir="model_saves/unet/",
    ckpt_path="model_saves/unet/checkpoint.ckpt" if cont else None
)