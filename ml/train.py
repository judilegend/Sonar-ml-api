import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# ── 1. Load the dataset ───────────────────────────────────────────────────────
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"

print("[1/5] Loading the Sonar dataset...")
df = pd.read_csv(URL, header=None)
columns = [f"F{i}" for i in range(60)] + ["Label"]
df.columns = columns

X = df.drop(columns="Label").values.astype(float)
y_raw = df["Label"].values

# ── 2. Encode labels ─────────────────────────────────────────────────────────
print("[2/5] Encoding labels (M/R → 1/0)...")
le = LabelEncoder()
y = le.fit_transform(y_raw)
# Save the mapping for the API (e.g. {0: 'M', 1: 'R'})
label_map = {int(i): str(c) for i, c in zip(le.transform(le.classes_), le.classes_)}
print(f"       Mapping: {label_map}")

# ── 3. Train/test split + normalization ───────────────────────────────────────
print("[3/5] Splitting train/test and normalizing...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ── 4. GridSearchCV ─────────────────────────────────────────────────────────
print("[4/5] Running GridSearchCV (please wait)...")
param_grid = [
    {"kernel": ["rbf"],    "C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.001, 0.01]},
    {"kernel": ["linear"], "C": [0.01, 0.1, 1, 10]},
]
grid = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train_sc, y_train)

best_svm = grid.best_estimator_
print(f"       Best parameters: {grid.best_params_}")
print(f"       Best CV score: {grid.best_score_:.4f}")

# ── 5. Final evaluation ──────────────────────────────────────────────────────
print("[5/5] Evaluating on the test set...")
y_pred = best_svm.predict(X_test_sc)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Save the model bundle ────────────────────────────────────────────────────
# Save scaler + model + mapping in a single pickle file.
# The API will load this bundle at startup.
os.makedirs("models", exist_ok=True)

bundle = {
    "model": best_svm,
    "scaler": scaler,
    "label_map": label_map,  # {0: 'M', 1: 'R'}
    "best_params": grid.best_params_,
    "cv_score": round(grid.best_score_, 4),
}

with open("models/sonar_model.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("\n✅ Model saved: models/sonar_model.pkl")
print(f"   Bundle contents: {list(bundle.keys())}")
