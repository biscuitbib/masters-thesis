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

from thesisproject.data import Dataset2D, ImageData, SliceLoader
from thesisproject.models import UNet
from thesisproject.utils import get_metrics, mask_to_rgb, segmentation_to_rgb
from thesisproject.train import training_loop


if __name__ == "__main__":
    path = "../toy_data/"

    train_data = ImageData(path + "train")
    val_data = ImageData(path + "val")
    
    train_loader = SliceLoader(train_data, slices_per_batch=32, volumes_per_batch=8, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = SliceLoader(train_data, slices_per_batch=32, volumes_per_batch=8, shuffle=False, num_workers=1, pin_memory=True)

    ## Train
    net = UNet(1, 4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    training_loop(net, criterion, optimizer, train_loader, val_loader, num_epochs=20)