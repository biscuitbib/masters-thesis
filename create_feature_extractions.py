import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import DataLoader
from tqdm import tqdm

from thesisproject.models import UNet
from thesisproject.predict import Predict
from thesisproject.data import ImagePairDataset, extract_features

image_path = "ucph-erda-home/Osteoarthritis-initiative/NIFTY/"
image_files = np.loadtxt("subject_images.txt", dtype="str")


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
        knee = filename[8:13]
        visit = int(filename[15:17])
    else:
        knee = filename[8:12]
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
net = UNet(1, 9, 384, class_names=label_keys)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

checkpoint = torch.load("model_saves/unet-checkpoint.pt")
net.load_state_dict(checkpoint["model_state_dict"])

predict = Predict(net, batch_size=8, show_progress=False)

computed_files = []
if os.path.exists("feature_extract.csv"):
    df = pd.read_csv("feature_extract.csv")
    for _, row in df.iterrows():
        computed_files.append(row["filename"])
else:
    df = pd.DataFrame()

files_to_compute = list(set(image_files) - set(computed_files))

#print(f"{len(files_to_compute)}/{len(image_files)} files left for feature extraction.")

#pbar = tqdm(total=len(image_files), unit="images")
#pbar.update(len(computed_files))
for filename in files_to_compute:
    #pbar.set_description(f"{filename} (prediction)")

    nii_file = nib.load(f"{image_path}{filename}")

    subject_id, is_right, visit = filename_to_subject_info(filename)

    scan = nii_file.get_fdata()

    # Flip coronal plane
    scan = np.flip(scan, axis=1).copy()

    if isright:
        scan = np.flip(scan, axis=2).copy()

    scan_tensor = volume_transform(torch.from_numpy(scan).float().to(device))

    scan_tensor -= scan_tensor.min()
    scan_tensor /= scan_tensor.max()

    prediction = predict(scan_tensor)

    pbar.set_description(f"{filename} (extract)")
    extracted_features = extract_features(scan_tensor.detach().cpu().numpy(), prediction.detach().cpu().numpy())

    subject_id_and_knee =  str(subject_id) + ("-R" if is_right else "-L"),
    subject_row = subjects_df.loc[subject_id_and_knee]

    row_df = pd.DataFrame([{
        "subject_id_and_knee": subject_id_and_knee,
        "is_right": is_right,
        "visit": visit,
        "filename": filename,
        "TKR": subject_row["TKR"],
        **extracted_features}])

    df = pd.concat([df, row_df])
    df.to_csv("feature_extract.csv", index=False)
    #pbar.update(1)

#pbar.close()
