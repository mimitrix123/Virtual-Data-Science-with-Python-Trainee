"""Week 1 - Data Acquisition, Cleaning, and Preprocessing
Virtual Data Science with Python Trainee

Expected input: train.csv from the Kaggle Titanic dataset.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Load raw data
df = pd.read_csv("train.csv")
print("Raw shape:", df.shape)

# 2. Preserve raw data by working on a copy
clean = df.copy()

# 3. Remove complete duplicate rows
clean = clean.drop_duplicates()

# 4. Standardize categorical fields
clean["Sex"] = clean["Sex"].astype(str).str.strip().str.lower()
clean["Embarked"] = clean["Embarked"].astype(str).str.strip().str.upper()

# 5. Impute Age using Pclass + Sex group medians
age_median = clean.groupby(["Pclass", "Sex"])["Age"].transform("median")
clean["Age"] = clean["Age"].fillna(age_median)
clean["Age"] = clean["Age"].fillna(clean["Age"].median())

# 6. Impute Embarked using mode
clean["Embarked"] = clean["Embarked"].fillna(clean["Embarked"].mode()[0])

# 7. Drop high-missingness Cabin in the baseline workflow
clean = clean.drop(columns=["Cabin"])

# 8. Validate numeric values
clean = clean[clean["Age"].between(0, 100)]
clean = clean[clean["Fare"] >= 0]

# 9. Create baseline modeling matrix
model_df = clean.drop(columns=["PassengerId", "Name", "Ticket"])
model_df = pd.get_dummies(model_df, columns=["Sex", "Embarked"], drop_first=True)

# 10. Optional scaling for numeric features
numeric_features = ["Age", "Fare", "SibSp", "Parch"]
scaler = StandardScaler()
model_df[numeric_features] = scaler.fit_transform(model_df[numeric_features])

# 11. Final validation
assert model_df.isna().sum().sum() == 0
assert model_df.duplicated().sum() == 0
assert (model_df["Fare"] >= 0).all()

print("Final shape:", model_df.shape)
print("Total missing values:", model_df.isna().sum().sum())
print(model_df.head())
