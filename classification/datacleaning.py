import pandas as pd

df = pd.read_csv("F:/mlproject/classification/datasets/Titanic-Dataset.csv")

print("Before cleaning:")
print(df.isnull().sum())

df = df.drop_duplicates()

df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

df["Age"] = df["Age"].fillna(df["Age"].median())

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df = pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int)

print("\nAfter cleaning:")
print(df.isnull().sum())

print("\nCleaned data:")
print(df.head())

df.to_csv("F:/mlproject/classification/datasets/titanic_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully.")