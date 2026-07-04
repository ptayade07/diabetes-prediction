"""
Step 2: Data Loading & Exploration
Dataset: Pima Indians Diabetes Database (768 patients, 8 features)
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The CSV has no header row, so we name the columns ourselves
COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome",
]

df = pd.read_csv("data/diabetes.csv", header=None, names=COLUMNS)

print("Shape (rows, columns):", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nStatistical summary:")
print(df.describe())

print("\nClass balance (0 = No Diabetes, 1 = Diabetes):")
print(df["Outcome"].value_counts())

# --- Spot the hidden problem: impossible zero values ---
print("\nZero values per column (many of these are medically impossible!):")
for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    n = (df[col] == 0).sum()
    print(f"  {col:>15}: {n} zeros ({n/len(df):.1%})")

# --- Save a visual overview ---
df.hist(figsize=(12, 9), bins=25, color="#2a9d8f", edgecolor="black")
plt.suptitle("Feature Distributions - Pima Diabetes Dataset")
plt.tight_layout()
plt.savefig("outputs/eda_distributions.png", dpi=150)
print("\nSaved plot -> outputs/eda_distributions.png")