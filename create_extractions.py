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

"""
for every file in dataset
    create prediction volume
    plot image, label, prediction
    
"""

def create_animation(image, label, dim=0):
    c, h, w, d = image.shape

    fig, ax = plt.subplots()

    frames = []
    if dim == 0:
        for i in range(h):
            im_slice = image[i, :, :, :]
            lab_slice = label[i, :, :, :]
            im = ax.imshow(torch.cat([im_slice, lab_slice], dim=1))
            frames.append([im])
    elif dim == 1:
        for i in range(w):
            im_slice = image[:, i, :, :]
            lab_slice = label[:, i, :, :]
            im = ax.imshow(torch.cat([im_slice, lab_slice], dim=1))
            frames.append([im])
    elif dim == 2:
        for i in range(d):
            im_slice = image[:, :, i, :]
            lab_slice = label[:, :, i, :]
            im = ax.imshow(torch.cat([im_slice, lab_slice], dim=1))
            frames.append([im])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)

    ani.save(f"pred_dim{dim}.mp4")
    
    
def mask_to_rgb(image):
    colors = np.array([
        [0, 0, 0],
        [228, 146, 244],
        [229, 124, 117],
        [184, 139, 229],
        [118, 226, 196],
        [159, 226, 111],
        [35, 211, 150],
        [123, 224, 219],
        [31, 154, 173],
        [81, 214, 74]])
    mask_imgs = np.array([colors[p] for p in image])
    mask_imgs = torch.tensor(mask_imgs).permute(3,0,1,2)
    return mask_imgs

def grayscale_to_rgb(image):
    image = image.repeat(1, 3, 1, 1)
    return image
    
if __name__ == "__main__":
    path = "/datasets/oai/"
    files = os.listdir(path)

    net = UNet(1, 10)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    checkpoint_path = os.path.join("model_saves", "model_checkpoint_first.pt")
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])

    scan = nib.load(path + files[0]).get_fdata()
    scan_flip = np.flip(scan, axis=0)

    scan_tensor = torch.from_numpy(scan_flip.copy()).float().to(device)
    
    scan_tensor -= scan_tensor.min()
    scan_tensor /= scan_tensor.max()

    #prediction = predict_volume(net, scan_tensor)
    h, w, d = scan_tensor.shape
    with torch.no_grad():
        prediction_image = torch.zeros((net.n_classes, h, w, d), dtype=scan_tensor.dtype)
        for i in trange(h):
            image_slice = scan_tensor[i, :, :]
            image_slice = image_slice.unsqueeze(0).unsqueeze(0).to(device)
            prediction = net(image_slice)

            prediction_image[:, i, :, :] += prediction.squeeze().detach().cpu()
            
    prediction_volume = torch.argmax(prediction_image, dim=0)
    
    img = scan_tensor.detach().cpu().squeeze().unsqueeze(0).repeat(3, 1, 1, 1).permute(1, 2, 3, 0)
    lab = mask_to_rgb(prediction_volume.detach().cpu().squeeze()).permute(1, 2, 3, 0)
    create_animation(img, lab, dim=2)

