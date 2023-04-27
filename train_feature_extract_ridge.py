import os
import sys
import gc

import yaml
import pickle
import numpy as np
import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt
from sklearn.utils.estimator_checks import check_estimator
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
from tqdm import tqdm

with open("/home/blg515/masters-thesis/hparams.yaml", "r") as stream:
    hparams = yaml.safe_load(stream)

args = sys.argv[1:]
n_visits_index = int(args[0])

n_visits = hparams["linear_regression"]["n_visits"][n_visits_index]

train_indices = np.load("/home/blg515/masters-thesis/train_ids.npy", allow_pickle=True).astype(str)
val_indices = np.load("/home/blg515/masters-thesis/val_ids.npy", allow_pickle=True).astype(str)
test_indices = np.load("/home/blg515/masters-thesis/test_ids.npy", allow_pickle=True).astype(str)

test_df = pd.read_csv("/home/blg515/masters-thesis/results.csv")

regex = re.compile("feature_extract(_\d+)?\.csv")
csv_files = [file for file in os.listdir("/home/blg515/masters-thesis/") if regex.match(file)]

subjects_df = pd.concat([pd.read_csv(csv) for csv in csv_files], ignore_index=True).sample(frac=1)
subjects_df = subjects_df.drop_duplicates().reset_index(drop=True)

subject_id_and_knees = subjects_df["subject_id_and_knee"].unique()

print(f"{subjects_df.shape[0]} samples, with {len(subject_id_and_knees)} unique subjects and knees")

def create_data(indices, n_visits=1):
    max_visits = subjects_df.groupby("subject_id_and_knee").size().max()

    identifiers = []
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
        identifier = rows["subject_id_and_knee"].values[0]
        features = rows.loc[:, ~rows.columns.isin(["subject_id_and_knee", "TKR", "filename", "is_right", "visit"])]
        #features["visit"] -= features["visit"].min()
        features = features.values # list of feature vectors
        last_feature = features[-1,:]
        subject_visits = features.shape[0]
        padding = np.zeros((max_visits - subject_visits, features.shape[1]))
        features_padded = np.concatenate([features, padding], axis=None)

        identifiers.append(identifier)
        subjects.append(features[-n_visits:, :].reshape(-1))
        labels.append(TKR)

    subjects = np.array(subjects)
    labels = np.array(labels)
    return subjects, labels, identifiers

print(f"Ridge regression with n_visits={n_visits}")

X_train, y_train, identifiers_train = create_data(train_indices, n_visits=n_visits)
X_val, y_val, _ = create_data(val_indices, n_visits=n_visits)
X_test, y_test, identifiers_test = create_data(test_indices, n_visits=n_visits)

print(f"{X_train.shape[0]} training samples out of {len(train_indices)} possible.")
print(f"{X_test.shape[0]} test samples out of {len(test_indices)} possible.")

col_name_test = f"lr_n{n_visits}_test"
if col_name_test not in test_df.columns:
    test_df[col_name_test] = np.nan

col_name_train = f"lr_n{n_visits}_train"
if col_name_train not in test_df.columns:
    test_df[col_name_train] = np.nan

# To normalize or not?
normalizer = StandardScaler().fit(X_train)
X_train_normalized = normalizer.transform(X_train)
X_val_normalized = normalizer.transform(X_val)
X_test_normalized = normalizer.transform(X_test)

# tune alpha parameter for L2 regularization
alphas = [10**(i) for i in np.arange(-5, 4, 0.25)]
auc_scores = []

skf = StratifiedKFold(n_splits=5)

for alpha in alphas:
    avg_auc = 0
    for train_index, test_index in skf.split(X_train_normalized, y_train):
        train_X, train_y = X_train_normalized[train_index], y_train[train_index]
        test_X, test_y = X_train_normalized[test_index], y_train[test_index]

        model = Ridge(alpha=alpha).fit(train_X, train_y)
        y_pred = model.predict(test_X)

        fpr, tpr, _ = roc_curve(test_y, y_pred)
        avg_auc += auc(fpr, tpr) / 5

    del model
    gc.collect()
    auc_scores.append(avg_auc)

print(list(zip(alphas, auc_scores)))

best_alpha = alphas[np.argmax(auc_scores)]

print(f"Found best alpha={best_alpha} for L2 regularization")


# Fit
reg = Ridge(alpha=best_alpha).fit(X_train_normalized, y_train)
y_pred = reg.predict(X_test_normalized)
y_pred_thresh = (y_pred >= 0.5).astype(int)

# Save model
pickle.dump(reg, open(f"biomarker-linear-n{n_visits}.pickle", "wb"))

# Test
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh, labels=[0, 1]).ravel()

eps = 1e-8
accuracy = (tp + tn) / (tn + fp + fn + tp + eps)
precision = tp / (tp + fp + eps)
recall = tp / (tp + fn + eps)
specificity = tn / (tn + fp + eps)

fpr, tpr, _ = roc_curve(y_test, y_pred)
auc_score = auc(fpr, tpr)

# Training performance
y_pred_train = reg.predict(X_train_normalized)
fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train)
auc_score_train = auc(fpr_train, tpr_train)


# Save predictions to test_df
for identifier, pred in zip(identifiers_test, y_pred):
    index = test_df[test_df["subject_id_and_knee"] == identifier].index[0]
    test_df.at[index, col_name_test] = pred

for identifier, pred in zip(identifiers_train, y_pred_train):
    index = test_df[test_df["subject_id_and_knee"] == identifier].index[0]
    test_df.at[index, col_name_train] = pred

test_df.to_csv("/home/blg515/masters-thesis/results.csv", index=False)


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
# Test plot
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color="blue", label=f"Ridge Regression Classifier (AUC={round(auc_score, 3)})")

plt.xlim(-0.01, 1.01)
plt.ylim(-0.01, 1.01)
ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
plt.plot([0, 1], [0, 1], linestyle="dashed", color="black", alpha=0.5, label="Random Classifier")
plt.legend()
plt.savefig(f"biomarker_ridge_roc_n={n_visits}_test.png")

# Training plot
fig, ax = plt.subplots()
ax.plot(fpr_train, tpr_train, color="blue", label=f"Ridge Regression Classifier (AUC={round(auc_score_train, 3)})")

plt.xlim(-0.01, 1.01)
plt.ylim(-0.01, 1.01)
ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
plt.plot([0, 1], [0, 1], linestyle="dashed", color="black", alpha=0.5, label="Random Classifier")
plt.legend()
plt.savefig(f"biomarker_ridge_roc_n={n_visits}_train.png")

exit()
# Check coefficients of model
features = subjects_df.loc[:, ~subjects_df.columns.isin(["subject_id_and_knee", "TKR", "filename", "is_right", "visit"])].columns
features = [" ".join(["cart." if word == "cartilage" else word for word in feature.split("_")]).capitalize() for feature in features]

coef = reg.coef_

n_features = len(features)
bars = { f"Visit {i + 1}": coef[n_features*i:n_features*(i+1)] for i in range(n_visits)}

fig, ax = plt.subplots()
bases = np.zeros((n_features, 2)) # min, max
for i, (visit, coefs) in enumerate(bars.items()):
    base = [bases[i,0] if coefs[i] < 0 else bases[i,1] for i in range(n_features)]
    p = ax.bar(features, coefs, 0.5, label=visit, bottom=base)
    bases[:,0] += [min(0, coefs[i]) for i in range(n_features)]
    bases[:,1] += [max(0, coefs[i]) for i in range(n_features)]

ax.set_title("Feature coefficients")
plt.xticks(rotation=80, ha='right')
ax.legend()
fig.savefig(f"lr_coefs_n={n_visits}.png", bbox_inches="tight")
