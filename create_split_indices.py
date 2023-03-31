import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

samples = pd.read_csv("/home/blg515/masters-thesis/image_samples.csv").sample(frac=1)

subject_id_and_knees = samples["subject_id_and_knee"].unique()

train_val, test = train_test_split(subject_id_and_knees, test_size=0.2)
train, val = train_test_split(train_val, test_size=0.2)

train_ratio = samples[samples["subject_id_and_knee"].isin(train)]["TKR"].mean()
val_ratio = samples[samples["subject_id_and_knee"].isin(val)]["TKR"].mean()
test_ratio = samples[samples["subject_id_and_knee"].isin(test)]["TKR"].mean()

print(f"""
Training split:   class balance = {train_ratio}, N = {len(train)}
Validation split: class balance = {val_ratio}, N = {len(val)}
Test split:       class balance = {test_ratio}, N = {len(test)}
""")

np.save("train_ids", train)
np.save("val_ids", val)
np.save("test_ids", test)