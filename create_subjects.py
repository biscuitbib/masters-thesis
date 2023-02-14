import os
import nibabel as nib
import numpy as np
import pandas as pd
from typing import List, Set
import re

def get_patient_images_from_id(filenames: Set[str], subject_id: int, last_visit_right=12, last_visit_left: int=12):
    possible_right_images = {str(subject_id) + "-" + "Right" + "-" + f"V{visit:02}.nii.gz" for visit in range(last_visit_right)}
    possible_left_images = {str(subject_id) + "-" + "Left" + "-" + f"V{visit:02}.nii.gz" for visit in range(last_visit_left)}
    return list(possible_right_images & filenames), list(possible_left_images & filenames)

"""
Creating subjects dataset
Steps:
- Filter all subjects with images
- Find all subjects with left or right TKR
- For each TKR knee subject
    - Filter all subjects without TKR
    - Filter all subjects with bmi +- 2
    - Filter all subjects with age +- 4
    - Take random (max 2)
    - If less than 4, discard TKR patient
    - For TKR patient take last 3 visits before TKR
- Filter all without medical insurance
- Filter all where no problems with MRI
- Filter all who could not walk 400m at baseline
"""

path = "/home/blg515/OAIdata21/"
files = os.listdir(path)
reg = re.compile(r'^AllClinical.*txt')
all_clinical = [file for file in files if reg.match(file)]

all_clinical_df = [pd.read_csv(path + file, sep='|', index_col="ID") for file in all_clinical]
outcomes_df = pd.read_csv(path + "Outcomes99.txt", sep='|', index_col="id")
outcomes_df.index.names = ["ID"]
subjects_df = pd.concat([*all_clinical_df, outcomes_df], axis=1)
subjects_df = subjects_df.reset_index()

"""
Filter out subjects
"""
# Formatting columns and cleaning
df = subjects_df.copy()
print(f"Initial subject count: {df.shape[0]}")

# Keep subjects with medical insurance
df = df[df["V00MEDINS"] == "1: Yes"]
print(f"Subject count with insurance: {df.shape[0]}")

# Keep subjects without MRI problems
df = df[df["P01MRPRBCV"] == "0: No"]
print(f"Subject count without MRI problems: {df.shape[0]}")

# Keep subjects that completed 400m walk
df = df[df["V00400MCMP"].isin(["1: Completed test without stopping", "2: Completed test with one or more rests"])]
print(f"Subject count that could walk 400m: {df.shape[0]}")

df["TKR-R"] = np.where(df["V99ERKDAYS"] == ".: Missing Form/Incomplete Workbook", 0, 1)
df["TKR-L"] = np.where(df["V99ELKDAYS"] == ".: Missing Form/Incomplete Workbook", 0, 1)

last_visit_dict = {
    ".: Missing Form/Incomplete Workbook": -1,
    "10: 96-month": 10,
    "9: 84-month": 9,
    "8: 72-month": 8,
    "7: 60-month": 7,
    "6: 48-month": 6,
    "1: 12-month": 1,
    "5: 36-month": 5,
    "3: 24-month": 3,
    "0: Baseline": 0,
    "4: 30-month": 4,
    "2: 18-month": 2
}
df = df.replace({"V99ERKVSPR": last_visit_dict, "V99ELKVSPR": last_visit_dict})

# Columns: Subject id, Age at baseline, BMI at baseline, last visit before right tkr, last visit before left tkr, right tkr, left tkr
columns = ["ID", "V00AGE", "P01BMI", "V99ERKVSPR", "V99ELKVSPR", "TKR-R", "TKR-L"]
df = df[columns]
print(f"Went from {subjects_df.shape[0]} subjects, to {df.shape[0]} subjects.")

"""
Split each subject into left and right knee
"""
new_cols = ["subject_id_and_knee", "age", "BMI", "last_visit_before_tkr", "TKR", "is_right"]

right_df = df.copy()
right_df = right_df.rename(columns={"V00AGE": "age", "P01BMI": "BMI", "V99ERKVSPR": "last_visit_before_tkr", "TKR-R": "TKR"})
right_df["is_right"] = True
right_df = right_df

left_df = df.copy()
left_df = left_df.rename(columns={"V00AGE": "age", "P01BMI": "BMI", "V99ELKVSPR": "last_visit_before_tkr", "TKR-L": "TKR"})
left_df["is_right"] = False
left_df = left_df

split_df = pd.concat([right_df, left_df], ignore_index=True)
split_df["subject_id_and_knee"] = split_df.apply(lambda row: str(row["ID"]) + ("-R" if row["is_right"] else "-L"), axis=1)

df = split_df[new_cols]
df.set_index("subject_id_and_knee")

TKR_df = df[(df["TKR"] == True)]
non_TKR_df = df.drop(TKR_df.index)

"""
Create dataset of TKR samples and matched non-TKR samples
"""
dataset_df = pd.DataFrame(columns=new_cols)

potential_matches_df = non_TKR_df.copy()
exclude_TKR_indices = []

# "9671958-Right-V03.nii.gz"
image_filenames = set(os.listdir("/home/blg515/ucph-erda-home/OsteoarthritisInitiative/NIFTY"))
image_subject_id_and_knees = set([file[:9] for file in image_filenames])

potential_matches_df = potential_matches_df[potential_matches_df["subject_id_and_knee"].isin(image_subject_id_and_knees)]

for i, p in TKR_df.iterrows():
    age = p["age"]
    bmi = p["BMI"]

    age_match_df = potential_matches_df[potential_matches_df["age"].between(age - 4, age + 4, inclusive="both")]
    bmi_match_df = age_match_df[age_match_df["BMI"].between(bmi - 2, bmi + 2, inclusive="both")]
    match_df = bmi_match_df

    n = min(match_df.shape[0], 2)
    if n == 0:
        match_df = age_match_df
        n = min(match_df.shape[0], 2)
        if n == 0:
            exclude_TKR_indices.append(i)
            continue

    matches = match_df.sample(n=n)
    potential_matches_df.drop(matches.index.values, inplace=True)

    dataset_df = pd.concat([dataset_df, matches], axis=0)

print(f"Excluded {len(exclude_TKR_indices)} TKR sample.")

dataset_df = pd.concat([dataset_df, TKR_df.drop(exclude_TKR_indices)], axis=0)

n_positive = dataset_df[dataset_df["TKR"] == 1].shape[0]
n_negative = dataset_df[dataset_df["TKR"] == 0].shape[0]

print(n_positive, n_negative)

# save samples
dataset_df.to_csv("subjects.csv", index=False)

"""
Get images from subjects
"""
#image_filenames = set(np.loadtxt("image_filenames.txt", dtype="str"))
image_df = None
subject_id_and_knees = dataset_df["subject_id_and_knee"].unique()

def filenames_from_subject_id_and_knee(subject_id_and_knee, files):
    subject_files = [file for file in files if file[:9] == subject_id_and_knee]
    return subject_files

no_images = []

for subject_id_and_knee in subject_id_and_knees:
    tkr = dataset_df[dataset_df["subject_id_and_knee"] == subject_id_and_knee]["TKR"].iloc[0]
    is_right = subject_id_and_knee[:-1] == "R"
    subject_filenames = filenames_from_subject_id_and_knee(subject_id_and_knee, image_filenames)
    if len(subject_filenames) == 0:
        no_images.append(subject_id_and_knee)

    dicts = []
    for filename in subject_filenames:
        visit = int(filename[-9:-7])
        dicts.append({
            "filename": filename,
            "TKR": tkr,
            "visit": visit,
            "subject_id_and_knee": subject_id_and_knee,
            "is_right": is_right
        })
    df = pd.DataFrame.from_dict(dicts)
    image_df = pd.concat([image_df, df], axis=0)


print(f"{len(no_images)} had no images")

image_df.to_csv("image_samples.csv", index=False)

image_files = image_df["filename"].unique()
print(f"Found {len(image_files)} images")
with open("subject_images.txt", "w") as f:
    for file in image_files:
        f.write(file + "\n")