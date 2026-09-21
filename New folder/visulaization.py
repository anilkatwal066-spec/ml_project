import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("dataset/insurance_cleaned.csv")
# 1. AGE VS CHARGES

plt.figure(figsize=(8, 5))

sns.scatterplot(x="age", y="charges", data=df)

plt.title("Age vs Insurance Charges")
plt.xlabel("Age")
plt.ylabel("Charges")

plt.show()
# 2. BMI VS CHARGES

plt.figure(figsize=(8, 5))

sns.scatterplot(x="bmi", y="charges", data=df)

plt.title("BMI vs Insurance Charges")
plt.xlabel("BMI")
plt.ylabel("Charges")

plt.show()
# 3. SMOKER VS CHARGES
plt.figure(figsize=(8, 5))

sns.boxplot(x="smoker", y="charges", data=df)

plt.title("Smoker vs Insurance Charges")
plt.xlabel("Smoker")
plt.ylabel("Charges")

plt.show()
# 4. REGION VS CHARGES
plt.figure(figsize=(8, 5))

sns.boxplot(x="region", y="charges", data=df)

plt.title("Region vs Insurance Charges")
plt.xlabel("Region")
plt.ylabel("Charges")

plt.show()

print("Visualization completed successfully!")