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
from thesisproject.data import Dataset2D
from thesisproject.models import UNet
from PIL import Image
from skimage.io import imread
from tqdm import tqdm

def prepare_data():
    if not os.path.exists("pet_data"):
        os.mkdir("pet_data")

    if not os.path.exists(os.path.join("pet_data", "images")):
        os.mkdir(os.path.join("pet_data", "images"))

        img_path = "../oxford-iiit-pet/images"

        for filename in tqdm(os.listdir(img_path)):
            if filename.endswith(".jpg"):
                im = Image.open(os.path.join(img_path, filename))
                name = filename[:-4]
                path = os.path.join("pet_data/images/" , name + '.png')
                im.save(path)
                im.close()
            else:
                continue

    if not os.path.exists(os.path.join("pet_data", "labels")):
        os.mkdir(os.path.join("pet_data", "labels"))

        labels_path = "../oxford-iiit-pet/annotations/trimaps"

        for filename in tqdm(os.listdir(labels_path)):
            im = Image.open(os.path.join(labels_path, filename))
            path = os.path.join("pet_data/labels/" , filename)
            im.save(path)
            im.close()

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

class Subtract:
    def __call__(self, image: Tensor):
        return image - 1

if __name__ == "__main__":
    prepare_data()

    transform = T.Compose([Square_pad(), T.Resize((256, 256), torchvision.transforms.InterpolationMode.NEAREST)])
    target_transform = T.Compose([Square_pad(fill=2), T.Resize((256, 256), torchvision.transforms.InterpolationMode.NEAREST), Subtract()])

    data = Dataset2D("pet_data/images", "pet_data/labels", transform=transform,
    target_transform=target_transform)
    train_len = int(len(data) * 0.8)
    test_len = len(data) - train_len
    train, test = random_split(data, [train_len, test_len])

    train_loader = DataLoader(train, shuffle=True, batch_size=128, num_workers=4)
    test_loader = DataLoader(test, shuffle=False, batch_size=128, num_workers=4)

    ## Train
    width_out = height_out = width_in = height_in = 256
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = UNet(3, 3)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in tqdm(range(10)):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].float().to(device), data[1].float().to(device)
            print(inputs.shape, labels.shape)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')