import numpy as np
from torch import Tensor, nn, optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import pad
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

from thesisproject.data.utils import get_slice_loaders
from thesisproject.models import UNet
from thesisproject.train import training_loop

class Square_pad:
    def __call__(self, image: Tensor):
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

if __name__ == "__main__":
    path = "../knee_data/"

    volume_transform = Square_pad()

    train_data, val_data = get_slice_loaders(path, volume_transform=volume_transform)

    train_loader = DataLoader(train_data, batch_size=8, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=8, num_workers=4)

    ## Train
    label_keys = ["Lateral femoral cart.",
                  "Lateral meniscus",
                  "Lateral tibial cart.",
                  "Medial femoral cartilage",
                  "Medial meniscus",
                  "Medial tibial cart.",
                  "Patellar cart.",
                  "Tibia"]
    net = UNet(1, 9, class_names=label_keys)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=5e-5)
    
    training_loop(
        net,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        num_epochs=100,
        cont=True
    )