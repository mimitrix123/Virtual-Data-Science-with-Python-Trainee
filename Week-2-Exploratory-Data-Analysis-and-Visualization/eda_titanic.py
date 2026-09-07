"""Week 2 - Exploratory Data Analysis and Visualization
Virtual Data Science with Python Trainee
Expected input: train.csv from the Kaggle Titanic dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("train.csv")

# Initial inspection
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe(include="all"))

# Missing values
missing = df.isna().sum().sort_values(ascending=False)
missing = missing[missing > 0]
print("Missing values:\n", missing)

sns.barplot(x=missing.values, y=missing.index)
plt.title("Missing Values by Variable")
plt.xlabel("Missing Records")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()

# Overall survival
sns.countplot(data=df, x="Survived")
plt.title("Overall Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Passenger Count")
plt.tight_layout()
plt.show()

# Survival rates by selected categorical variables
for column, title in [
    ("Sex", "Survival Rate by Sex"),
    ("Pclass", "Survival Rate by Passenger Class"),
    ("Embarked", "Survival Rate by Embarkation Port"),
]:
    rates = df.groupby(column)["Survived"].mean().mul(100)
    print(f"\n{column} survival rates:\n", rates)
    sns.barplot(x=rates.index, y=rates.values)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Survival Rate (%)")
    plt.tight_layout()
    plt.show()

# Fare distribution and outliers
print("\nFare summary:\n", df["Fare"].describe())
sns.boxplot(x=df["Fare"])
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.tight_layout()
plt.show()

# Optional correlation view for numerical variables
numeric_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Numerical Variables")
plt.tight_layout()
plt.show()
