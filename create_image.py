import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

image_path = "/home/blg515/masters-thesis/oai_images/"

image_paths = [
    "/home/blg515/masters-thesis/oai_images/9000798-Right-V00.nii.gz",
    "/home/blg515/masters-thesis/oai_images/9000798-Right-V03.nii.gz",
    "/home/blg515/masters-thesis/oai_images/9000798-Right-V06.nii.gz"
    ]

for i, image in enumerate(image_paths):
    nii_image = nib.load(image).get_fdata()

    fig, ax = plt.subplots(figsize=(20, 20))
    plt.axis("off")
    ax.imshow(nii_image[:, 210, :], cmap="gray")
    fig.savefig(f"slice_im_{i}.png", bbox_inches="tight")
