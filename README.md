# Diabetes Prediction using Machine Learning

End-to-end classification project predicting diabetes onset from diagnostic
measurements. Built for the SpacECE India Foundation AI/ML assessment task.

## Dataset
**Pima Indians Diabetes Database** — 768 patient records, 8 clinical features
(Glucose, BMI, Blood Pressure, Age, etc.) and a binary outcome
(diabetic / non-diabetic). Source: National Institute of Diabetes and
Digestive and Kidney Diseases.

## Project Pipeline (step by step)

| Step | Script | What it does |
|---|---|---|
| 1 | — | Project setup, dependencies, dataset download |
| 2 | `explore_data.py` | EDA: shape, class balance, discovery of hidden missing values |
| 3 | `preprocess.py` | Replace impossible zeros with median imputation |
| 4 | `feature_selection.py` | Mutual information ranking (Glucose & BMI strongest) |
| 5 | `train_logistic.py` | Model 1: Logistic Regression baseline |
| 6 | `train_forest_compare.py` | Model 2: Random Forest + full comparison |

## Key Preprocessing Insight
Five columns contain physically impossible zeros (e.g. `BloodPressure = 0`) —
these are hidden missing values, affecting up to 48.7% of the `Insulin` column.
Deleting those rows would lose half the dataset, so they are imputed with the
median of the valid values instead.

## Results

| Model | Test Accuracy | Precision | Recall | F1 | CV Accuracy (5-fold) |
|---|---|---|---|---|---|
| Logistic Regression | 70.78% | 0.6000 | 0.5000 | 0.5455 | 77.22% |
| Random Forest | 73.38% | 0.6512 | 0.5185 | 0.5773 | 76.18% |

Random Forest outperforms the baseline on the held-out test set across all
metrics, as it can capture non-linear feature interactions. Both models reach
~76-77% on 5-fold cross-validation, in line with published benchmarks for
this dataset.

## How to Run
```bash
pip install -r requirements.txt
python explore_data.py
python preprocess.py
python feature_selection.py
python train_logistic.py
python train_forest_compare.py
```
All plots and metrics are saved to `outputs/`.