"""
Step 6: Model 2 - Random Forest + Comparison with Logistic Regression
- Trains both models on the same split (fair comparison)
- Metrics table + 5-fold cross-validation
- Side-by-side confusion matrices, ROC curves, comparison bar chart
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
)

df = pd.read_csv("data/diabetes_clean.csv")
X = df.drop(columns="Outcome")
y = df["Outcome"]

# Same split as Step 5 (random_state=42) -> fair comparison
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Model 1: Logistic Regression (needs scaled data) ---
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_s, y_train)

# --- Model 2: Random Forest (tree models don't need scaling) ---
rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# --- Evaluate both ---
def get_metrics(name, model, X_te):
    y_pred = model.predict(X_te)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
    }, y_pred, model.predict_proba(X_te)[:, 1]

m_lr, pred_lr, prob_lr = get_metrics("Logistic Regression", log_reg, X_test_s)
m_rf, pred_rf, prob_rf = get_metrics("Random Forest", rf, X_test)

# 5-fold cross-validation (more robust than a single split)
m_lr["CV Accuracy"] = cross_val_score(log_reg, scaler.transform(X), y, cv=5).mean()
m_rf["CV Accuracy"] = cross_val_score(rf, X, y, cv=5).mean()

results = pd.DataFrame([m_lr, m_rf]).set_index("Model").round(4)
print("\n===== MODEL COMPARISON =====")
print(results)
results.to_csv("outputs/model_comparison.csv")

# --- Plot 1: side-by-side confusion matrices ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, y_pred) in zip(axes, [("Logistic Regression", pred_lr),
                                     ("Random Forest", pred_rf)]):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No Diabetes", "Diabetes"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)
plt.tight_layout()
plt.savefig("outputs/confusion_matrices.png", dpi=150)

# --- Plot 2: ROC curves ---
fig, ax = plt.subplots(figsize=(7, 5.5))
for name, prob in [("Logistic Regression", prob_lr), ("Random Forest", prob_rf)]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves"); ax.legend()
plt.tight_layout()
plt.savefig("outputs/roc_curves.png", dpi=150)

# --- Plot 3: metric comparison bar chart ---
fig, ax = plt.subplots(figsize=(9, 5))
results[["Accuracy", "Precision", "Recall", "F1-score"]].T.plot.bar(
    ax=ax, color=["#264653", "#e76f51"], rot=0)
ax.set_title("Logistic Regression vs Random Forest")
ax.set_ylim(0, 1); ax.set_ylabel("Score")
for c in ax.containers:
    ax.bar_label(c, fmt="%.3f", fontsize=8)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150)

print("\nSaved: confusion_matrices.png, roc_curves.png, model_comparison.png")