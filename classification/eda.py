import pandas as pd

df = pd.read_csv(r"F:\mlproject\classification\datasets\Titanic-Dataset.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("F:/mlproject/classification/datasets/Titanic-Dataset.csv")

sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.show()

sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class")
plt.show()

sns.histplot(df["Age"].dropna(), bins=30)
plt.title("Age Distribution")
plt.show()