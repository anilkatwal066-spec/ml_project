import pandas as pd

df = pd.read_csv(r"F:\mlproject\New folder\dataset\insurance.csv")

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()


# Numerical columns
numeric_columns = [
    "age",
    "bmi",
    "children",
    "charges"
]

# Convert numerical columns
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column])


# Fill missing numerical values
for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )


# Categorical columns
categorical_columns = [
    "sex",
    "smoker",
    "region"
]

# Clean categorical values
for column in categorical_columns:
    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

# Fill missing categorical values
for column in categorical_columns:
    df[column] = df[column].fillna(
        df[column].mode()[0]
    )


# Remove invalid target values
df = df[df["charges"] > 0]


# Final check
print("\n")
print("=" * 60)
print("DATA CLEANING - AFTER")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# Display clean data
print("\nCleaned Data:")
print(df.head(10))


# Save cleaned dataset
df.to_csv(
    r"F:\mlproject\New folder\dataset\insurance_cleaned.csv",
    index=False
)


print("\nCleaned dataset saved successfully!")
print("File: dataset/insurance1_cleaned.csv")