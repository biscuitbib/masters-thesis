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

    transform = Square_pad()

    train_data = ImageData(path + "train", transform=transform, target_transform=transform, num_access=5)
    val_data = ImageData(path + "val", transform=transform, target_transform=transform, num_access=5)

    train_loader = SliceLoader(train_data, slices_per_batch=8, volumes_per_batch=4, shuffle=True, num_workers=4, pin_memory=False)
    val_loader = SliceLoader(train_data, slices_per_batch=8, volumes_per_batch=4, shuffle=False, num_workers=4, pin_memory=False)

    ## Train
    net = UNet(1, 10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=5e-5)

    training_loop(net, criterion, optimizer, train_loader, val_loader, num_epochs=100, cont=True)