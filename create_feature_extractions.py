import os
import sys

from queue import Queue
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from thesisproject.data import extract_features
from thesisproject.models.mpu import LitMPU, UNet
from tqdm import tqdm
from time import time

args = sys.argv[1:]

num_jobs = 1
if len(args) == 2:
    job_index = int(args[0])
    num_jobs = int(args[1])

image_path = "/home/blg515/ucph-erda-home/OsteoarthritisInitiative/NIFTY/"

subjects_df = pd.read_csv("/home/blg515/image_samples_edit.csv")
image_files = subjects_df["filename"].values #np.loadtxt("/home/blg515/masters-thesis/subject_images.txt", dtype="str")

assert os.path.exists(image_path)

if num_jobs > 1:
    image_files = np.array_split(image_files, num_jobs)[job_index]

class Square_pad:
    def __call__(self, image: torch.Tensor):
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

def filename_to_subject_info(filename):
    subject_id = int(filename[:7])
    is_right = False
    if filename[8] == "R":
        is_right = True
        visit = int(filename[15:17])
    else:
        visit = int(filename[14:16])
    return subject_id, is_right, visit

volume_transform = Square_pad()

label_keys = ["Lateral femoral cart.",
              "Lateral meniscus",
              "Lateral tibial cart.",
              "Medial femoral cartilage",
              "Medial meniscus",
              "Medial tibial cart.",
              "Patellar cart.",
              "Tibia"]

unet = UNet(1, 9, 384, class_names=label_keys)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "/home/blg515/masters-thesis/model_saves/unet/lightning_logs/version_9098/checkpoints/epoch=38-step=6513.ckpt"
print(f"Trying to load from checkpoint:\n{checkpoint_path}")

litunet: LitMPU = LitMPU.load_from_checkpoint(checkpoint_path, unet=unet)
litunet.eval()
litunet.to(device)

computed_files = []
if os.path.exists("feature_extract.csv"):
    df = pd.read_csv("feature_extract.csv")
    for _, row in df.iterrows():
        computed_files.append(row["filename"])
else:
    df = pd.DataFrame()

files_to_compute = list(set(image_files) - set(computed_files))

image_q = Queue()
[image_q.put((filename, nib.load(image_path + filename))) for filename in files_to_compute]

pbar = tqdm(total=len(image_files))
pbar.update(len(computed_files))
with torch.no_grad():
    #for filename in tqdm(files_to_compute):
    while not image_q.empty():
        filename, nii_file = image_q.get()
        #pbar.set_description(f"{filename} (load)")
        #nii_file = nib.load(f"{image_path}{filename}")

        pbar.set_description(f"{filename}")
        subject_id, is_right, visit = filename_to_subject_info(filename)

        scan = nii_file.get_fdata(caching='unchanged')

        # Flip coronal plane
        scan = np.flip(scan, axis=1).copy()

        if is_right:
            scan = np.flip(scan, axis=2).copy()

        scan_tensor = volume_transform(torch.from_numpy(scan).float())

        scan_tensor -= scan_tensor.min()
        scan_tensor /= max(1, scan_tensor.max())
        scan_tensor = scan_tensor.to(device)

        #TODO fix misuse of prediction for lightning module
        start = time()
        prediction = litunet.predict_step(scan_tensor)
        end = time()

        pbar.set_description(f"{filename} (pred: {round(end - start, 4)}s)")

        start = time()
        try:
            extracted_features = extract_features(scan_tensor.detach().cpu().numpy(), prediction.detach().cpu().numpy())
        except KeyboardInterrupt:
            print("Shutting down.")
            exit()
        except:
            print(f"Failed feature extraction for image: {filename}")
            continue
        end = time()

        pbar.set_description(f"{filename} (extract: {round(end - start, 4)}s)")

        subject_id_and_knee =  str(subject_id) + ("-R" if is_right else "-L")
        subject_row = subjects_df[subjects_df["filename"] == filename].head(1)

        row_df = pd.DataFrame([{
            "subject_id_and_knee": subject_id_and_knee,
            "is_right": is_right,
            "visit": visit,
            "filename": filename,
            "TKR": subject_row["TKR"].iloc[0],
            **extracted_features}])

        df = pd.concat([df, row_df])
        if num_jobs > 1:
            df.to_csv(f"feature_extract_{job_index}.csv", index=False)
        else:
            df.to_csv(f"feature_extract.csv", index=False)
        pbar.update(1)
