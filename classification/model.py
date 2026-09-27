import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("F:/mlproject/classification/datasets/titanic_cleaned.csv")

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

models = {
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ]),
    "SVC": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC())
    ])
}

param_grids = {
    "KNN": {
        "model__n_neighbors": [3, 5, 7, 9, 11],
        "model__weights": ["uniform", "distance"]
    },
    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__penalty": ["l2"]
    },
    "SVC": {
        "model__C": [0.1, 1, 10, 100],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", "auto"]
    }
}

results = {}
best_models = {}

for name in models:
    grid = GridSearchCV(
        models[name],
        param_grids[name],
        cv=5,
        scoring="accuracy"
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy
    best_models[name] = best_model

    print("\n", name)
    print("Best Parameters:", grid.best_params_)
    print("CV Accuracy:", grid.best_score_)
    print("Test Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

best_model_name = max(results, key=results.get)
best_model = best_models[best_model_name]

print("\nModel Comparison:")
for name, accuracy in results.items():
    print(f"{name}: {accuracy:.2f}")

print("\nBest Model:", best_model_name)
print("Best Accuracy:", f"{results[best_model_name]:.2f}")

joblib.dump(best_model, "F:/mlproject/classification/titanic_model.pkl")

print("\nBest model saved successfully.")