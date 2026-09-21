import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load cleaned dataset

df = pd.read_csv(
    r"F:\mlproject\New folder\dataset\insurance_cleaned.csv"
)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# Separate input and target

X = df.drop("charges", axis=1)
y = df["charges"]

print("\nInput columns:")
print(X.columns.tolist())

print("\nTarget column:")
print(y.name)


# Convert categorical columns into numbers

X = pd.get_dummies(X, dtype=int)

print("\nCategorical data converted into numbers!")

print("\nModel input columns:")
print(X.columns.tolist())


# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)


# Create pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])


# Hyperparameter tuning

parameters = {
    "model__fit_intercept": [True, False]
}

grid_search = GridSearchCV(
    pipeline,
    parameters,
    cv=5,
    scoring="r2"
)


# Train the model

grid_search.fit(X_train, y_train)

print("\nModel training completed!")


# Display best parameters

print("\nBest Parameters:")
print(grid_search.best_params_)


# Display best cross-validation score

print("\nBest CV R2 Score:")
print(grid_search.best_score_)


# Get the best trained model

best_model = grid_search.best_estimator_


# Make predictions

y_pred = best_model.predict(X_test)


# Evaluate model

mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(y_test, y_pred) ** 0.5

r2 = r2_score(y_test, y_pred)


print("\nFinal Model Performance:")

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# Save the trained model

joblib.dump(
    best_model,
    r"F:\mlproject\New folder\insurance_model.pkl"
)

print("\nModel saved successfully!")
