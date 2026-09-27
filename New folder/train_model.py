import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv(
    r"F:\mlproject\New folder\dataset\insurance_cleaned.csv"
)

X = df.drop("charges", axis=1)
y = df["charges"]

X = pd.get_dummies(X, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

parameters = {
    "model__fit_intercept": [True, False]
}

grid_search = GridSearchCV(
    pipeline,
    parameters,
    cv=5,
    scoring="r2"
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("Best Parameters:", grid_search.best_params_)
print("Best CV R2 Score:", grid_search.best_score_)

print("\nModel Performance:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

joblib.dump(
    best_model,
    r"F:\mlproject\New folder\insurance_model.pkl"
)

print("\nModel saved successfully.")