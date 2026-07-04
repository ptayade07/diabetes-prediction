"""
Step 5: Model 1 - Logistic Regression (baseline)
- Train/test split (80/20, stratified)
- Feature scaling (required for linear models)
- Evaluation: accuracy, precision, recall, F1 + confusion matrix
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

df = pd.read_csv("data/diabetes_clean.csv")
X = df.drop(columns="Outcome")
y = df["Outcome"]

# 80/20 split; stratify keeps the same diabetic/non-diabetic ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")

# Scale features: Logistic Regression works best when features are on
# the same scale (Age ~20-80 vs Insulin ~15-800 would otherwise dominate)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)  # NOTE: transform only - never fit on test data!

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

print("\n--- Logistic Regression Results ---")
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5.5, 5))
ConfusionMatrixDisplay(cm, display_labels=["No Diabetes", "Diabetes"]).plot(
    ax=ax, colorbar=False, cmap="Blues"
)
ax.set_title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig("outputs/confusion_logistic.png", dpi=150)
print("Saved plot -> outputs/confusion_logistic.png")