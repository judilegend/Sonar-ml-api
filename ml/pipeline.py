import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

import warnings
warnings.filterwarnings("ignore")

# 
#  STEP 1 — DATA ACQUISITION
# 
print("=" * 60)
print("STEP 1 — Sonar Dataset Acquisition")
print("=" * 60)

# Loading from OpenML (identical to the UCI Sonar dataset)
sonar = fetch_openml(name="sonar", version=1, as_frame=True, parser="auto")
df = sonar.frame.copy()

# Target column is named "Class": M = Mine, R = Rock
print(f"\nDataset Dimensions: {df.shape}")
print(f"Columns            : {df.columns.tolist()}")
print(f"\nClass Distribution:\n{df['Class'].value_counts()}")
print(f"\nFirst 5 rows:")
print(df.head())


# 
#  STEP 2 — EXPLORATORY DATA ANALYSIS (EDA)
# 
print("\n" + "=" * 60)
print("STEP 2 — Exploratory Data Analysis (EDA)")
print("=" * 60)

# 2a. Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe().round(3))

# 2b. Missing values
print(f"\nMissing values: {df.isnull().sum().sum()} (total)")

# 2c. Class distribution
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Class distribution plot
df["Class"].value_counts().plot(kind="bar", ax=axes[0], color=["steelblue", "salmon"])
axes[0].set_title("Class Distribution")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Number of Samples")
axes[0].tick_params(axis="x", rotation=0)

# Boxplot of the first 10 features by class
df_melt = df.melt(id_vars="Class", value_vars=df.columns[:10], var_name="Feature", value_name="Value")
sns.boxplot(data=df_melt, x="Feature", y="Value", hue="Class", ax=axes[1], palette=["steelblue", "salmon"])
axes[1].set_title("Distribution of the First 10 Features")
axes[1].tick_params(axis="x", rotation=90)

# Correlation matrix (features 1-20)
numeric_features = df.drop(columns="Class").iloc[:, :20]
corr_matrix = numeric_features.corr()
sns.heatmap(corr_matrix, ax=axes[2], cmap="coolwarm", center=0,
            xticklabels=False, yticklabels=False, cbar_kws={"shrink": 0.8})
axes[2].set_title("Correlation Matrix (First 20 Features)")

plt.tight_layout()
plt.savefig("eda_sonar.png", dpi=120, bbox_inches="tight")
plt.show()
print("→ EDA Plot saved: eda_sonar.png")


# 
#  STEP 3 — PREPROCESSING
# 
print("\n" + "=" * 60)
print("STEP 3 — Preprocessing")
print("=" * 60)

# Features / Target separation
X = df.drop(columns="Class").astype(float).values
y = LabelEncoder().fit_transform(df["Class"])   # M=0, R=1 (or inverse)

print(f"\nX shape: {X.shape}  |  y shape: {y.shape}")
print(f"Encoding: {dict(zip(df['Class'].unique(), LabelEncoder().fit(df['Class']).transform(df['Class'].unique())))}")

# Train / Test split (80% / 20%), stratified to maintain balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# Normalization — IMPORTANT for SVM (sensitive to scale)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit on train ONLY
X_test_sc  = scaler.transform(X_test)        # applied to test

print("StandardScaler normalization applied.")
print(f"  Train mean (feature 0) before: {X_train[:, 0].mean():.4f}  after: {X_train_sc[:, 0].mean():.4f}")
print(f"  Train std  (feature 0) before: {X_train[:, 0].std():.4f}  after: {X_train_sc[:, 0].std():.4f}")


# 
#  STEP 4 — SVM TRAINING (with GridSearchCV)
# 
print("\n" + "=" * 60)
print("STEP 4 — SVM Training + GridSearchCV")
print("=" * 60)

# Hyperparameter grid to explore
param_grid = [
    {
        "kernel": ["rbf"],
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.001, 0.01],
    },
    {
        "kernel": ["linear"],
        "C": [0.01, 0.1, 1, 10],
    },
    {
        "kernel": ["poly"],
        "C": [0.1, 1, 10],
        "degree": [2, 3],
        "gamma": ["scale"],
    },
]

# GridSearch with 5-fold cross-validation
print("\nSearching for best hyperparameters (5-fold GridSearchCV)…")
grid_search = GridSearchCV(
    SVC(probability=True),  # probability=True for the ROC curve
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_train_sc, y_train)

print(f"\nBest parameters  : {grid_search.best_params_}")
print(f"Best CV score     : {grid_search.best_score_:.4f}")

# Final model = best estimator found
best_svm = grid_search.best_estimator_


# 
#  STEP 5 — SVM MODEL EVALUATION
# 
print("\n" + "=" * 60)
print("STEP 5 — SVM Model Evaluation")
print("=" * 60)

y_pred   = best_svm.predict(X_test_sc)
y_proba  = best_svm.predict_proba(X_test_sc)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Mine (M)", "Rock (R)"]))

# Cross-validation on the train set (robustness check)
cv_scores = cross_val_score(best_svm, X_train_sc, y_train, cv=10, scoring="accuracy")
print(f"10-fold Cross-Validation (train): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# --- Evaluation Figures ------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Mine (M)", "Rock (R)"])
disp.plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title(f"Confusion Matrix — SVM ({grid_search.best_params_['kernel']})")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc_val = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color="steelblue", lw=2, label=f"SVM (AUC = {roc_auc_val:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", lw=1)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend()

plt.tight_layout()
plt.savefig("evaluation_svm.png", dpi=120, bbox_inches="tight")
plt.show()
print("→ Evaluation Plot saved: evaluation_svm.png")


# 
#  STEP 6 — COMPARISON WITH OTHER MODELS
# 
print("\n" + "=" * 60)
print("STEP 6 — Model Comparison")
print("=" * 60)

models = {
    "SVM (best)"         : best_svm,
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN (k=5)"          : KNeighborsClassifier(n_neighbors=5),
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    acc_test  = model.score(X_test_sc, y_test)
    cv_mean   = cross_val_score(model, X_train_sc, y_train, cv=10, scoring="accuracy").mean()
    results[name] = {"Test Accuracy": acc_test, "CV Mean (10-fold)": cv_mean}

df_results = pd.DataFrame(results).T.sort_values("Test Accuracy", ascending=False)
print("\nPerformance Summary:")
print(df_results.round(4).to_string())

# Comparative Chart
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(df_results))
width = 0.35
bars1 = ax.bar(x - width/2, df_results["Test Accuracy"],   width, label="Test Accuracy",   color="steelblue")
bars2 = ax.bar(x + width/2, df_results["CV Mean (10-fold)"], width, label="CV Mean (10-fold)", color="salmon", alpha=0.85)

ax.set_xlabel("Model")
ax.set_ylabel("Accuracy")
ax.set_title("Model Comparison — Sonar Dataset")
ax.set_xticks(x)
ax.set_xticklabels(df_results.index, rotation=15, ha="right")
ax.set_ylim(0, 1.1)
ax.legend()

# Adding value labels on top of bars
for bar in bars1:
    ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=120, bbox_inches="tight")
plt.show()
print("→ Comparison Plot saved: model_comparison.png")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE — Happy Analyzing!")
print("=" * 60)