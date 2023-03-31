import numpy as np
import torch
import itertools
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

colors = np.array([
    [0, 0, 0],       #background
    [212, 54, 0], #Lateral Femoral Cartilage
    [20, 60, 179], #Lateral Meniscus
    [7, 148, 25], #Lateral Tibial Cartilage
    [255, 99, 46], #Medial Femoral Cartilage
    [65, 81, 204], #Medial Meniscus
    [56, 209, 76],  #Medial Tibial Cartilage
    [252, 250, 99], #Patellar Cartilage
    [112, 200, 219]]) / 255. #Tibia

def segmentation_to_rgb(image):
    preds = torch.argmax(image, dim=1)
    pred_imgs = np.array([colors[p] for p in preds])
    return pred_imgs

def mask_to_rgb(image):
    mask_imgs = np.array([colors[p] for p in image])
    return mask_imgs

def grayscale_to_rgb(image):
    image = image.repeat(1, 3, 1, 1).permute(0, 2, 3, 1)
    return image

def create_overlay_figure(inputs, labels, outputs, images_per_batch=4):
    images_per_batch = min(inputs.shape[0], images_per_batch)
    input_imgs = inputs.detach().cpu()
    input_imgs -= torch.min(input_imgs)
    input_imgs /= torch.max(input_imgs)
    input_imgs = grayscale_to_rgb(input_imgs)

    b, h, w, c = input_imgs.shape

    pred_imgs = segmentation_to_rgb(outputs.detach().cpu())
    pred_overlay = (input_imgs / 2) + (pred_imgs / 2)
    pred_overlay = torch.cat([pred_overlay[i, ...] for i in range(pred_overlay.shape[0])], dim=1).numpy()

    target_imgs = mask_to_rgb(labels.detach().cpu())
    target_overlay = (input_imgs / 2) + (target_imgs / 2)
    target_overlay = torch.cat([target_overlay[i, ...] for i in range(target_overlay.shape[0])], dim=1).numpy()

    fig, (ax_gt, ax_pred) = plt.subplots(2, 1, figsize=(20, 20))
    ax_gt.imshow(target_overlay[:, :w * images_per_batch, ...])
    ax_gt.set_title("Ground truth")
    ax_gt.set_xticks([])
    ax_gt.set_yticks([])

    ax_pred.imshow(pred_overlay[:, :w * images_per_batch, ...])
    ax_pred.set_title("Prediction")
    ax_pred.set_xticks([])
    ax_pred.set_yticks([])

    fig.tight_layout()

    return fig, [ax_gt, ax_pred]

def create_confusion_matrix_figure(cm, class_names=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion matrix")
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks, class_names, rotation=45)
    ax.set_yticks(tick_marks, class_names, rotation=45)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    fig.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    return fig, ax

def create_animation(image, label, dim=0):
    h, w, d = image.shape
    image = np.expand_dims(image, axis=-1)
    image -= np.min(image)
    image /= np.max(image)
    image *= 255
    image_rgb = np.repeat(image, 3, axis=-1).astype(np.uint8)

    label_rgb = np.zeros((*label.shape, 3), dtype=np.uint8)

    for gray, rgb in enumerate(colors):
        label_rgb[label == gray, :] = rgb

    fig, ax = plt.subplots()

    frames = []
    if dim == 0:
        for i in range(h):
            im_slice = image_rgb[i, :, :, :]
            lab_slice = label_rgb[i, :, :, :]
            im = ax.imshow(np.concatenate([im_slice, lab_slice], axis=1))
            frames.append([im])
    elif dim == 1:
        for i in range(w):
            im_slice = image_rgb[:, i, :, :]
            lab_slice = label_rgb[:, i, :, :]
            im = ax.imshow(np.concatenate([im_slice, lab_slice], axis=1))
            frames.append([im])
    elif dim == 2:
        for i in range(d):
            im_slice = image_rgb[:, :, i, :]
            lab_slice = label_rgb[:, :, i, :]
            im = ax.imshow(np.concatenate([im_slice, lab_slice], axis=1))
            frames.append([im])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)

    ani.save(f"dim_{dim}.mp4")

if __name__ == "__main__":
    import nibabel as nib
    import os

    filename = "9625955_20040810_SAG_3D_DESS_RIGHT_016610679011.nii.gz"

    img_path = "../../../knee_data/test/images/" + filename
    label_path = "../../predictions/" + filename

    image = nib.load(img_path).get_fdata()
    label = nib.load(label_path).get_fdata()

    label_keys = ["LateralFemoralCartilage",
                      "LateralMeniscus",
                      "LateralTibialCartilage",
                      "MedialFemoralCartilage",
                      "MedialMeniscus",
                      "MedialTibialCartilage",
                      "PatellarCartilage",
                      "Tibia"]

    fig, axs = plt.subplots(colors.shape[0]- 1, 1)

    for i, color in enumerate(colors):
        if i == 0:
            continue
        color_img = np.zeros((10, 10, 3), dtype=np.uint8)
        color_img[:, :] = color

        axs[i - 1].imshow(color_img)
        axs[i - 1].set_title(label_keys[i-1])
        axs[i - 1].set_xticks([])
        axs[i - 1].set_yticks([])


    plt.tight_layout()
    plt.savefig("labels.jpg")

    create_animation(image, label, dim=1)