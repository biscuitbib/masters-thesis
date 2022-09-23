import os
import torch
import nibabel as nib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.decomposition import PCA
from scipy import ndimage

from tqdm import trange

from thesisproject.predict import predict_volume
from thesisproject.models import UNet
from thesisproject.data import extract_features

"""
for every file in dataset
    create prediction volume
    plot image, label, prediction

"""

if __name__ == "__main__":
    path = "/datasets/oai/"
    files = os.listdir(path)

    net = UNet(1, 9)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    checkpoint_path = os.path.join("model_saves", "model_checkpoint.pt")
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])

    for filename in files:
        nii_file = nib.load(path + filename)

        scan = nii_file.get_fdata()

        scan_flip = np.flip(scan, axis=0)

        scan_tensor = torch.from_numpy(scan_flip.copy()).float().to(device)

        scan_tensor -= scan_tensor.min()
        scan_tensor /= scan_tensor.max()

        prediction = predict_volume(net, scan_tensor)

        extracted_features = extract_features(scan_tensor, prediction)

        print({"filename": filename, **extracted_features})