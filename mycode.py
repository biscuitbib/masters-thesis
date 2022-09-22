import numpy as np
from torch import Tensor, nn, optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import pad
from torch.utils.data import DataLoader, random_split
from skimage.io import imread, imsave
from multiprocessing import cpu_count

from thesisproject.data.utils import get_slice_loaders
from thesisproject.models import UNet
from thesisproject.train import training_loop
from thesisproject.test import test_loop

class Square_pad:
    def __call__(self, image: Tensor):
        imsize = image.shape
        max_edge = np.argmax(imsize)
        pad_amounts = [imsize[max_edge] - imsize[0], imsize[max_edge] - imsize[1], imsize[max_edge] - imsize[2]]

        padding = [pad_amounts[0], 0, pad_amounts[1], 0, pad_amounts[2], 0] #left, right, top, bottom, front, back
        padding = tuple(padding[::-1])

        padded_im = F.pad(image, padding, "constant", 0)
        return padded_im

if __name__ == "__main__":
    path = "../ScanManTrain61_knee_data/"

    volume_transform = Square_pad()

    train_data, val_data = get_slice_loaders(path, volume_transform=volume_transform)

    num_cpus = cpu_count()
    print("num cpus: ", num_cpus)
    train_loader = DataLoader(train_data, batch_size=8, num_workers=1)
    val_loader = DataLoader(val_data, batch_size=8, num_workers=1)

    ## Train
    net = UNet(1, 9)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=5e-5)

    #torch.backends.cudnn.enabled = False
    training_loop(
        net,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        num_epochs=1000,
        cont=False
    )

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
