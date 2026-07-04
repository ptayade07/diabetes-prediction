"""
Step 4: Feature Selection
Rank features by Mutual Information - how much each feature tells us
about the Outcome. Helps justify which features the models rely on.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif

df = pd.read_csv("data/diabetes_clean.csv")
X = df.drop(columns="Outcome")
y = df["Outcome"]

mi = mutual_info_classif(X, y, random_state=42)
mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)

print("Feature importance (Mutual Information with Outcome):\n")
print(mi_series)

# Bar chart for the report
fig, ax = plt.subplots(figsize=(8, 5))
mi_series.sort_values().plot.barh(ax=ax, color="#2a9d8f")
ax.set_title("Feature Importance (Mutual Information)")
ax.set_xlabel("Mutual information with Outcome")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
print("\nSaved plot -> outputs/feature_importance.png")

print("\nConclusion: Glucose and BMI carry the strongest signal, which is")
print("medically sensible. All 8 features show some signal, so we keep all")
print("of them - with only 8 features there's no need to drop any.")