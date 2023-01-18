import pandas as pd
"""
Step 1: Join all feature extraction inputs to a single dataframe
"""
image_df = pd.read_csv("/home/blg515/image_samples_edit.csv")
print(image_df.shape[0])

df0 = pd.read_csv("/home/blg515/masters-thesis/feature_extract_0.csv")
df1 = pd.read_csv("/home/blg515/masters-thesis/feature_extract_1.csv")
df2 = pd.read_csv("/home/blg515/masters-thesis/feature_extract_2.csv")
df3 = pd.read_csv("/home/blg515/masters-thesis/feature_extract_3.csv")
print(df0.shape[0])
print(df1.shape[0])
print(df2.shape[0])
print(df3.shape[0])

df = pd.concat([df0, df1, df2, df3], ignore_index=True)
print(df.shape[0])

df = df.groupby(["subject_id_and_knee", "visit"]).first()
print(df.shape[0])