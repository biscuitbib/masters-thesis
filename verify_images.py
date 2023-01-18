import os

import numpy as np
import nibabel as nib
from tqdm import tqdm
import pandas as pd

image_samples = pd.read_csv("/home/blg515/image_samples.csv")

image_path = "/home/blg515/ucph-erda-home/OsteoarthritisInitiative/NIFTY/"

image_files = image_samples["filename"].values

b_to_mb = 0.000001

failed_images = []
image_sizes = []
for image_file in tqdm(image_files):
    size = os.path.getsize(image_path + image_file)
    image_sizes.append(size * b_to_mb)

    nii = nib.load(image_path + image_file)
    try:
        nii_image = nii.get_fdata(caching="unchanged")
    except KeyboardInterrupt:
        print("Shutting down.")
        exit()
    except:
        failed_images.append(image_file)

if len(failed_images) == 0:
    print("All images were loaded succesfully!")
else:
    print("Found issues with the following images:")
    [print(image_file) for image_file in failed_images]


image_sizes = np.array(image_sizes)
_, unique_indices = np.unique(image_sizes, return_index=True)
print(f"""
Image sizes:
min:  {np.min(image_sizes)}
max:  {np.max(image_sizes)}
mean: {np.mean(image_sizes)}
std:  {np.std(image_sizes)}

""")