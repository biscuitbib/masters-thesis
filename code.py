#!/usr/local/bin/python3
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch import Tensor
import torchvision
import torchvision.transforms as T
from torchvision.transforms.functional import pad
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from thesisproject.data import Dataset2D
from thesisproject.models import UNet
from PIL import Image
from skimage.io import imread, imsave
from tqdm import tqdm

def prepare_data():
    if not os.path.exists("pet_data"):
        os.mkdir("pet_data")

    if not os.path.exists(os.path.join("pet_data", "images")):
        os.mkdir(os.path.join("pet_data", "images"))

        img_path = "../oxford-iiit-pet/images"

        for filename in tqdm(os.listdir(img_path)):
            if filename.endswith(".jpg"):
                im = imread(os.path.join(img_path, filename))
                name = filename[:-4]
                path = os.path.join("pet_data/images/" , name + '.png')
                imsave(path, im)
            else:
                continue

    if not os.path.exists(os.path.join("pet_data", "labels")):
        os.mkdir(os.path.join("pet_data", "labels"))

        labels_path = "../oxford-iiit-pet/annotations/trimaps"

        for filename in tqdm(os.listdir(labels_path)):
            im = imread(os.path.join(labels_path, filename))
            path = os.path.join("pet_data/labels/" , filename)
            imsave(path, im)
            
class Square_pad:
    def __init__(self, fill=0):
        self.fill=fill

    def __call__(self, image: Tensor):
        imsize = image.size()[-2:]
        pad_amount = np.max(imsize) - np.min(imsize)
        pad_edge = np.argmax(imsize)

        padding = [0, 0, 0, 0]
        padding[pad_edge] = pad_amount
        
        padded_im = pad(image, padding, fill=self.fill)
        return padded_im
    
class Normalize:
    def __call__(self, image):
        return image / torch.max(image)
    
class Squeeze:
    def __call__(self, image):
        return image.squeeze()
    
colors = np.array([[np.random.randint(255), np.random.randint(255), np.random.randint(255)] for _ in range(59)])

def segmentation_to_rgb(image):
    preds = torch.argmax(image, dim=1)
    pred_imgs = torch.tensor([colors[p] for p in preds]).permute(0,3,1,2)
    return pred_imgs

def mask_to_rgb(image):
    mask_imgs = torch.tensor([colors[p] for p in image]).permute(0,3,1,2)
    return mask_imgs

if __name__ == "__main__":
    writer = SummaryWriter()
    
    transform = T.Compose([Square_pad(), T.Resize((256, 256)), Normalize()])
    target_transform = T.Compose([Square_pad(), T.Resize((256, 256)), Squeeze()])

    data = Dataset2D("../clothes_data/png_images/IMAGES/", "../clothes_data/png_masks/MASKS/", transform=transform,
    target_transform=target_transform)
    
    # 8-2 train test split
    train_len = int(len(data) * 0.3)
    test_len = len(data) - train_len
    train, test = random_split(data, [train_len, test_len])

    train_loader = DataLoader(train, shuffle=True, batch_size=16, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test, shuffle=False, batch_size=16, num_workers=1, pin_memory=True)

    ## Train
    width_out = height_out = width_in = height_in = 256
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    net = UNet(3, 59)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    num_iters = 0
    for epoch in range(10):  # loop over the dataset multiple times
        pbar = tqdm(total=train_len, position=0, leave=True)
        running_loss = 0.0
        num_batches = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            #print(outputs.shape, outputs.dtype)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            batch_samples = inputs.shape[0]
            current_loss = loss.item() / batch_samples
            running_loss += current_loss
            num_batches += 1
                
            pbar.update(inputs.shape[0])
        else:
            input_imgs = inputs.cpu()
            pred_imgs = segmentation_to_rgb(outputs.cpu())
            target_imgs = mask_to_rgb(labels.cpu())
            writer.add_images("images/inputs", input_imgs, epoch)
            writer.add_images("images/targets", target_imgs, epoch)
            writer.add_images("images/predictions", pred_imgs, epoch)
            
        writer.add_scalar("loss/train", running_loss/num_batches, epoch)
            
    print('Finished Training')