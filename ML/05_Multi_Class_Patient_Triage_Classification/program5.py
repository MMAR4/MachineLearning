# ============================================================
# PROJECT 5 — MULTI-CLASS PATIENT TRIAGE CLASSIFICATION
# WEEK 4 — DAY 17
# ============================================================

import pandas as pd
import numpy as np
from io import StringIO


# ============================================================
# TASK 1 — DATASET CONSTRUCTION
# ============================================================

raw_data = """Patient_ID,Heart_Rate_BPM,Systolic_BP,Oxygen_Sat_Pct,Age,Triage_Category
P_501,72,120,98,35,General
P_502,125,165,89,68,ICU
P_503,95,138,94,45,Urgent
P_504,68,115,99,28,General
P_505,140,180,85,75,ICU
P_506,88,130,95,52,Urgent
P_507,75,122,97,40,General
P_508,130,170,88,62,ICU
P_509,92,135,93,50,Urgent
P_510,70,118,98,30,General"""

df = pd.read_csv(StringIO(raw_data))

print("========== EMERGENCY PATIENT TRIAGE ANALYSIS ==========")
print("\nDataset:")
print(df)

print("\nDataset Shape:", df.shape)


# ============================================================
# TASK 2 — SUPERVISED PARADIGM IDENTIFICATION
# ============================================================

paradigm = "Multi-Class Classification"
target_type = "Discrete Categorical"

print("\nSupervised Paradigm :", paradigm)
print("Target Type        :", target_type)


# ============================================================
# TASK 3 — INPUT-OUTPUT SPLITTING
# ============================================================

# X = input features
# Patient_ID is excluded because it is only an identifier
# Triage_Category is the target, so it is also excluded from X

X = df.drop(columns=["Patient_ID", "Triage_Category"])

# y = target
y = df["Triage_Category"]

print("\nFeatures (X):")
print(X)

print("\nTarget (y):")
print(y)

print("\nFeature Columns:")
print(*X.columns, sep="\n")

print("\nTarget Column:")
print(y.name)


# ============================================================
# TASK 4 — MULTI-CLASS TARGET DISTRIBUTION
# ============================================================

class_counts = y.value_counts()
class_percentages = y.value_counts(normalize=True) * 100

print("\nClass Distribution:")

for category in ["General", "Urgent", "ICU"]:
    print(
        f"{category:8} : "
        f"{class_counts[category]} "
        f"({class_percentages[category]:.1f}%)"
    )


# ============================================================
# TASK 5 — GROUP-WISE FEATURE MEANS
# ============================================================

group_means = df.groupby("Triage_Category")[
    ["Heart_Rate_BPM", "Systolic_BP", "Oxygen_Sat_Pct"]
].mean()

print("\nTriage Level Group Averages:")
print(group_means)


# ============================================================
# TASK 6 — SUPERVISED ALGORITHM SELECTION GUIDE
# ============================================================

print("\nAlgorithm Guidance:")

print("\n1. Logistic Regression")
print("- Can handle multi-class classification.")
print("- Learns relationships between patient features and triage categories.")
print("- Provides a relatively interpretable classification approach.")

print("\n2. Decision Tree")
print("- Can create decision boundaries using feature conditions.")
print("- Easy to interpret.")
print("- Suitable when rules such as high heart rate or low oxygen")
print("  saturation help distinguish triage categories.")

print("\n3. K-Nearest Neighbors (KNN)")
print("- Classifies a new patient based on similar patients in the dataset.")
print("- Simple to understand.")
print("- Can be affected by feature scales and the choice of K.")

print("\nRecommended for interpretable clinical rule boundaries:")
print("Decision Tree or Multi-Class Logistic Regression")