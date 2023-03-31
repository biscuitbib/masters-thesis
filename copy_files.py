import os
import shutil
import numpy as np
from tqdm import tqdm

image_path = "/home/blg515/ucph-erda-home/OsteoarthritisInitiative/NIFTY/"

assert os.path.exists(image_path), f"Path to images ({image_path}) does not exist."

new_path = "/home/blg515/masters-thesis/oai_images/"

image_filenames = np.loadtxt("/home/blg515/masters-thesis/subject_images.txt", dtype=str)

files_copied = os.listdir(new_path)

files_to_copy = list(set(image_filenames) - set(files_copied))

for filename in tqdm(files_to_copy):
    shutil.copy(image_path + filename, new_path + filename)