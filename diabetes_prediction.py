import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# LOAD DATASET
# -------------------------------
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

data = pd.read_csv(url, header=None, names=columns)

# -------------------------------
# CLEAN DATA (BASED ON YOUR EXPLANATION)
# -------------------------------
# Replace unrealistic zeros with NaN
invalid_zero_cols = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

data[invalid_zero_cols] = data[invalid_zero_cols].replace(0, np.nan)

# Fill missing values with median
data.fillna(data.median(numeric_only=True), inplace=True)

# -------------------------------
# SAMPLE DATA (FOR GRAPH)
# -------------------------------
sample = data.head(80)

# -------------------------------
# GRAPH 1: IMPORTANT FEATURES (CLEANED DATA)
# -------------------------------
plt.figure(figsize=(12,6))

plt.plot(sample.index, sample["Glucose"], label="Glucose (Blood Sugar)")
plt.plot(sample.index, sample["BloodPressure"], label="Blood Pressure")
plt.plot(sample.index, sample["SkinThickness"], label="Skin Thickness")
plt.plot(sample.index, sample["Insulin"], label="Insulin")
plt.plot(sample.index, sample["BMI"], label="BMI")
plt.plot(sample.index, sample["Age"], label="Age")

plt.title("Important Diabetes Features (After Cleaning Zero Values)")
plt.xlabel("Patient Index (Rows)")
plt.ylabel("Values")
plt.legend()
plt.show()

# -------------------------------
# GRAPH 2: OUTCOME (0 vs 1)
# -------------------------------
plt.figure(figsize=(12,4))

plt.scatter(
    sample.index,
    sample["Outcome"],
    c=sample["Outcome"],
    cmap="bwr",
    s=80
)

plt.title("Diabetes Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.xlabel("Patient Index (Rows)")
plt.ylabel("Outcome")
plt.yticks([0, 1])
plt.show()

# -------------------------------
# GRAPH 3: OUTCOME COUNT (BALANCE CHECK)
# -------------------------------
plt.figure(figsize=(6,4))

data["Outcome"].value_counts().sort_index().plot(
    kind="bar",
    color=["blue", "red"]
)

plt.title("Distribution of Diabetes Outcome")
plt.xlabel("Outcome (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)
plt.show()