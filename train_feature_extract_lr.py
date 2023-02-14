import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.utils.estimator_checks import check_estimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
from tqdm import tqdm

train_indices = np.load("/home/blg515/train_ids.npy", allow_pickle=True).astype(str)
val_indices = np.load("/home/blg515/val_ids.npy", allow_pickle=True).astype(str)
test_indices = np.load("/home/blg515/test_ids.npy", allow_pickle=True).astype(str)

df0 = pd.read_csv("/home/blg515/masters-thesis/feature_extract_0.csv")
df1 = pd.read_csv("/home/blg515/masters-thesis/feature_extract_1.csv")
df2 = pd.read_csv("/home/blg515/masters-thesis/feature_extract_2.csv")
df3 = pd.read_csv("/home/blg515/masters-thesis/feature_extract_3.csv")

subjects_df = pd.concat([df0, df1, df2, df3], ignore_index=True).sample(frac=1)

def create_data(indices):
    max_visits = subjects_df.groupby("subject_id_and_knee").size().max()
    n_visits = 2

    subjects = []
    labels = []
    print("Creating dataset...")
    for subject_id_and_knee in indices:
        # Extract feature vectors and timedeltas
        rows = subjects_df[subjects_df["subject_id_and_knee"] == subject_id_and_knee].fillna(0.0)
        if rows.shape[0] < n_visits:
            continue

        rows = rows.sort_values("visit")
        TKR = int(rows["TKR"].values[0])
        features = rows.loc[:, ~rows.columns.isin(["subject_id_and_knee", "TKR", "filename", "is_right", "visit"])]
        #features["visit"] -= features["visit"].min()
        features = features.values # list of feature vectors
        last_feature = features[-1,:]
        subject_visits = features.shape[0]
        padding = np.zeros((max_visits - subject_visits, features.shape[1]))
        features_padded = np.concatenate([features, padding], axis=None)

        subjects.append(features[-n_visits:, :].reshape(-1))
        labels.append(TKR)

    subjects = np.array(subjects)
    labels = np.array(labels)
    return subjects, labels

X_train, y_train = create_data(train_indices)
X_test, y_test = create_data(val_indices)

print(f"{X_train.shape[0]} training samples out of {len(train_indices)} possible.")
print(f"{X_test.shape[0]} test samples out of {len(val_indices)} possible.")


# To normalize or not?
normalizer = StandardScaler().fit(X_train)
X_train_normalized = normalizer.transform(X_train)
X_test_normalized = normalizer.transform(X_test)

# Fit
reg = LinearRegression().fit(X_train_normalized, y_train)
y_pred = reg.predict(X_test_normalized)
y_pred_thresh = (y_pred >= 0.5).astype(int)

# Test
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh, labels=[0, 1]).ravel()

eps = 1e-8
accuracy = (tp + tn) / (tn + fp + fn + tp + eps)
precision = tp / (tp + fp + eps)
recall = tp / (tp + fn + eps)
specificity = tn / (tn + fp + eps)

fpr, tpr, _ = roc_curve(y_test, y_pred)
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

plt.style.use("seaborn")
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color="blue", label=f"Linear Regression Classifier (AUC={round(auc_score, 2)})")

plt.xlim(-0.01, 1.01)
plt.ylim(-0.01, 1.01)
ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
plt.plot([0, 1], [0, 1], linestyle="dashed", color="black", alpha=0.5, label="Random Classifier")
plt.legend()
plt.savefig("feature_extract_lr_roc.png")