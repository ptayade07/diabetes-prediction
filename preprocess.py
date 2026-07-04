"""
Step 3: Data Preprocessing
- Replace medically impossible zeros with NaN (they are hidden missing values)
- Impute missing values with the column median
- Save the cleaned dataset for the modeling steps
"""

import numpy as np
import pandas as pd

COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome",
]

df = pd.read_csv("data/diabetes.csv", header=None, names=COLUMNS)
print("Loaded:", df.shape)

# Columns where a value of 0 is physically impossible
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

print("\nImputing hidden missing values (zeros) with the median:")
for col in zero_as_missing:
    n_zero = (df[col] == 0).sum()
    median = df.loc[df[col] != 0, col].median()  # median of the VALID values only
    df[col] = df[col].replace(0, np.nan).fillna(median)
    print(f"  {col:>15}: {n_zero:3d} zeros -> replaced with median {median}")

# Sanity check: no zeros should remain in those columns
remaining = (df[zero_as_missing] == 0).sum().sum()
print(f"\nRemaining zeros in medical columns: {remaining} (should be 0)")

df.to_csv("data/diabetes_clean.csv", index=False)
print("Saved cleaned dataset -> data/diabetes_clean.csv")