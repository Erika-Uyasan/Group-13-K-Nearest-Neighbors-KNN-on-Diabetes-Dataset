import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# =========================
# 1. LOAD DATASET
# =========================
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

df = pd.read_csv(url, names=columns)

# =========================
# 2. REPLACE INVALID ZEROS
# =========================
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols] = df[cols].replace(0, np.nan)

# =========================
# 3. HANDLE MISSING VALUES (MEDIAN IMPUTATION)
# =========================
for col in cols:
    df[col].fillna(df[col].median(), inplace=True)

# =========================
# 4. FEATURE SCALING (STANDARDIZATION)
# =========================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
df_scaled["Outcome"] = y

# =========================
# 5. GRAPH 1: OUTCOME DISTRIBUTION
# =========================
plt.figure(figsize=(6,4))
plt.hist(df["Outcome"], bins=2, edgecolor="black")
plt.xticks([0, 1])
plt.title("Diabetes Outcome Distribution")
plt.xlabel("Outcome (0 = Non-Diabetic, 1 = Diabetic)")
plt.ylabel("Count")
plt.show()

# =========================
# 6. GRAPH 2: GLUCOSE BEFORE VS AFTER CLEANING
# =========================
df_raw = pd.read_csv(url, names=columns)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(df_raw["Glucose"], bins=20, edgecolor="black")
plt.title("Before Cleaning (Glucose)")

plt.subplot(1,2,2)
plt.hist(df["Glucose"], bins=20, edgecolor="black")
plt.title("After Cleaning (Glucose)")

plt.tight_layout()
plt.show()

# =========================
# 7. OUTPUT CHECK
# =========================
print("\nScaled Data Sample:")
print(df_scaled.head())