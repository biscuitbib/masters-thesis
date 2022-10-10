import os
import numpy as np
import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import pad
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

from thesisproject.models import UNet
from thesisproject.data import ImagePairDataset
from thesisproject.test import Test

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
    
def test_collate(image):
    return image

if __name__ == "__main__":
    path = "../knee_data/"

    volume_transform = Square_pad()

    test_data = ImagePairDataset(path + "val", predict_mode=False, image_transform=volume_transform, label_transform=volume_transform)

    test_loader = DataLoader(test_data, batch_size=1, num_workers=4, collate_fn=test_collate)

    ## Train
    label_keys = [
        "Lateral femoral cart.",
        "Lateral meniscus",
        "Lateral tibial cart.",
        "Medial femoral cartilage",
        "Medial meniscus",
        "Medial tibial cart.",
        "Patellar cart.",
        "Tibia"]
    net = UNet(1, 9, class_names=label_keys)

    with torch.no_grad():
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net.to(device)

        checkpoint_path = os.path.join("model_saves", "model_checkpoint.pt")
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])

        test = Test(net, test_loader, save_preds=False)
        test()
