import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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

def segmentation_to_rgb(image):
    preds = torch.argmax(image, dim=1)
    pred_imgs = np.array([colors[p] for p in preds])
    pred_imgs = torch.tensor(pred_imgs).permute(0,3,1,2)
    return pred_imgs

def mask_to_rgb(image):
    mask_imgs = np.array([colors[p] for p in image])
    mask_imgs = torch.tensor(mask_imgs).permute(0,3,1,2)
    return mask_imgs

def grayscale_to_rgb(image):
    image = image.repeat(1, 3, 1, 1)
    return image

def create_animation(image, dim=0):
    h, w, d = image.shape

    fig, ax = plt.subplots()

    frames = []
    if dim == 0:
        for i in range(h):
            im = ax.imshow(image[i, :, :])
            frames.append([im])
    elif dim == 1:
        for i in range(w):
            im = ax.imshow(image[:, i, :])
            frames.append([im])
    elif dim == 2:
        for i in range(d):
            im = ax.imshow(image[:, :, i])
            frames.append([im])


    ani = animation.ArtistAnimation(
        fig,
        frames,
        interval=50,
        blit=True,
        repeat_delay=1000
    )

    ani.save(f"dim_{dim}.mp4")