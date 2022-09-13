import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation


colors = np.array([[np.random.randint(255), np.random.randint(255), np.random.randint(255)] for _ in range(10)])
colors[0] = np.array([255, 255, 255])

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

    x = np.arange(0, 2*np.pi, 0.01)


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


    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)

    ani.save(f"dim_{dim}.mp4")