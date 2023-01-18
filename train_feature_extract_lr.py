import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tqdm import tqdm

df0 = pd.read_csv("/home/blg515/masters-thesis/feature_extract_0.csv")
df1 = pd.read_csv("/home/blg515/masters-thesis/feature_extract_1.csv")
df2 = pd.read_csv("/home/blg515/masters-thesis/feature_extract_2.csv")
df3 = pd.read_csv("/home/blg515/masters-thesis/feature_extract_3.csv")

subjects_df = pd.concat([df0, df1, df2, df3], ignore_index=True).sample(frac=1)

max_visits = subjects_df.groupby("subject_id_and_knee").size().max()

subjects = []
labels = []
subject_id_and_knees = subjects_df["subject_id_and_knee"].unique()
print("Creating dataset...")
for subject_id_and_knee in tqdm(subject_id_and_knees):
    # Extract feature vectors and timedeltas
    rows = subjects_df[subjects_df["subject_id_and_knee"] == subject_id_and_knee].fillna(0.0)
    rows = rows.sort_values("visit")
    TKR = int(rows["TKR"].values[0])
    features = rows.loc[:, ~rows.columns.isin(["subject_id_and_knee", "TKR", "filename", "is_right"])]
    features["visit"] -= features["visit"].min()
    features = features.values # list of feature vectors
    n_visits = features.shape[0]
    padding = np.zeros((max_visits - n_visits, features.shape[1]))
    features_padded = np.concatenate([features, padding], axis=None)

    subjects.append(features_padded)
    labels.append(TKR)

subjects = np.array(subjects)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(subjects, labels, test_size=0.25)
print(f"{y_train.shape[0]} training samples and {y_test.shape[0]} test samples.")

print(f"""Class balance:
Training data: {np.mean(y_train)}
Test data:     {np.mean(y_test)}
""")

# To normalize or not?
normalizer = Normalizer().fit(X_train)
X_train_normalized = normalizer.transform(X_train)
X_test_normalized = normalizer.transform(X_test)

# Fit
reg = LinearRegression().fit(X_train_normalized, y_train)
y_pred = reg.predict(X_test_normalized)
y_pred = (y_pred >= 0.5).astype(int)

# Test
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

eps = 1e-8
accuracy = (tp + tn) / (tn + fp + fn + tp + eps)
precision = tp / (tp + fp + eps)
recall = tp / (tp + fn + eps)
specificity = tn / (tn + fp + eps)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc_score = auc(fpr, tpr)

print(f"""
Test metrics for linear regression classifier ({y_test.shape[0]} test samples):
Accuracy:    {accuracy:.4f}
Precision:   {precision:.4f}
Sensitivity: {recall:.4f}
Specificity: {specificity:.4f}
AUC:         {auc_score:.4f}
"""
)

RocCurveDisplay.from_predictions(y_test, y_pred)
plt.save
