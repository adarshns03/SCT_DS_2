import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
gender = pd.read_csv("gender_submission.csv")

# Quick cleaning (safe way, no warnings)
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])

# --- 1. Correlation heatmap ---
plt.figure(figsize=(6,5))
sns.heatmap(train.corr(numeric_only=True), annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()

# --- 2. Group survival rates ---
print("Survival by Sex:\n", train.groupby("Sex")["Survived"].mean())
print("Survival by Class:\n", train.groupby("Pclass")["Survived"].mean())
print("Survival by Embarked:\n", train.groupby("Embarked")["Survived"].mean())

# --- 3. Barplots ---
sns.barplot(x="Sex", y="Survived", data=train)
plt.title("Survival Rate by Gender")
plt.show()

sns.barplot(x="Pclass", y="Survived", data=train)
plt.title("Survival Rate by Class")
plt.show()

# --- 4. Age distribution vs survival ---
sns.histplot(data=train, x="Age", hue="Survived", bins=20, multiple="stack")
plt.title("Age vs Survival")
plt.show()
