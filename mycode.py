import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import pad
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from skimage.io import imread, imsave
from tqdm import tqdm

from thesisproject.data import ImageData, SliceLoader
from thesisproject.models import UNet
from thesisproject.utils import get_metrics, mask_to_rgb, segmentation_to_rgb
from thesisproject.train import training_loop
from thesisproject.test import test_loop

class Square_pad:
    def __init__(self, fill=0):
        self.fill=fill

    def __call__(self, image: Tensor):
        imsize = image.shape
        max_edge = np.argmax(imsize)
        pad_amounts = [imsize[max_edge] - imsize[0], imsize[max_edge] - imsize[1], imsize[max_edge] - imsize[2]]

        padding = [pad_amounts[0], 0, pad_amounts[1], 0, pad_amounts[2], 0] #left, right, top, bottom, front, back
        padding = tuple(padding[::-1])

        padded_im = F.pad(image, padding, "constant", self.fill)
        return padded_im


if __name__ == "__main__":
    path = "../ScanManTrain61_knee_data/"

    volume_transform = Square_pad()

    train_data = ImageData(
        path + "train",
        transform=volume_transform,
        target_transform=volume_transform,
        num_access=3
    )
    val_data = ImageData(
        path + "val",
        transform=volume_transform,
        target_transform=volume_transform,
        num_access=1
    )

    train_loader = SliceLoader(
        train_data,
        slices_per_batch=8,
        volumes_per_batch=2,
        augment=True,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    val_loader = SliceLoader(
        val_data,
        slices_per_batch=8,
        volumes_per_batch=2,
        augment=True,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    ## Train
    net = UNet(1, 10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=5e-5)

    #torch.backends.cudnn.enabled = False
    training_loop(net, criterion, optimizer, train_loader, val_loader, num_epochs=100, cont=False)

    """
    with torch.no_grad():
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net.to(device)

        checkpoint_path = os.path.join("model_saves", "model_checkpoint_first.pt")
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])

        path = "../ScanManTrain61_knee_data/"

        test_data = ImageData(
            path + "test",
            transform=volume_transform,
            target_transform=volume_transform,
            num_access=1
        )

        test_loader = DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )

        test_loop(net, test_loader)
    """
