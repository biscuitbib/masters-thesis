import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
from tqdm import tqdm

df0 = pd.read_csv("/home/blg515/masters-thesis/feature_extract_0.csv")
df1 = pd.read_csv("/home/blg515/masters-thesis/feature_extract_1.csv")
df2 = pd.read_csv("/home/blg515/masters-thesis/feature_extract_2.csv")
df3 = pd.read_csv("/home/blg515/masters-thesis/feature_extract_3.csv")

subjects_df = pd.concat([df0, df1, df2, df3], ignore_index=True).sample(frac=1)

print(f"{subjects_df.shape[0]} rows.")

max_visits = subjects_df.groupby("subject_id_and_knee").size().max()
n_visits = 3

subjects = []
labels = []
subject_id_and_knees = subjects_df["subject_id_and_knee"].unique()
print("Creating dataset...")
for subject_id_and_knee in subject_id_and_knees:
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

#X_train, X_test, y_train, y_test = train_test_split(subjects, labels, test_size=0.25)
print(f"{subjects.shape[0]} samples out of {subject_id_and_knees.shape[0]} total.")

n_folds = 10
folds = StratifiedKFold(n_splits=n_folds, shuffle=False).split(subjects, labels)

print(f"{n_folds} fold cross validation")
mean_accuracy = 0
mean_auc = 0

for i, (train_idx, test_idx) in tqdm(enumerate(folds)):
    X_train, y_train = subjects[train_idx], labels[train_idx]
    X_test, y_test = subjects[test_idx], labels[test_idx]

    # To normalize or not?
    normalizer = StandardScaler().fit(X_train)
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

    mean_accuracy += accuracy / n_folds
    mean_auc += auc_score / n_folds

    print(f"""
    Test metrics for linear regression classifier ({y_test.shape[0]} test samples):
    Accuracy:    {accuracy:.4f}
    Precision:   {precision:.4f}
    Sensitivity: {recall:.4f}
    Specificity: {specificity:.4f}
    AUC:         {auc_score:.4f}
    """
    )

    RocCurveDisplay.from_predictions(y_test, y_pred, pos_label=1, name="Imaging Biomarker Linear Regression")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], linestyle="dashed", color="black", alpha=0.5, label="Random classifier")
    plt.savefig(f"feature_extract_lr_roc_{i}.png")

print(f"Mean accuracy: {mean_accuracy}\nMean AUC score: {mean_auc}")
