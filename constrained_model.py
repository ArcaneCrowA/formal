import time

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from z3_model import fair_robust_predict, tree_constraints

# Dataset loading and preprocessing
df = pd.read_csv("datasets/adult.csv").dropna()
df["income_label"] = df["income"].map({"<=50K": 0, ">50K": 1})

categorical = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country",
]
for col in categorical:
    df[col] = LabelEncoder().fit_transform(df[col])

features = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education.num",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
    "native.country",
]

X = df[features]
y = df["income_label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Get Z3 constraints
tree_cons = tree_constraints(clf.tree_, features)

# Define perturbations
deltas = {
    "age": 1,
    "capital.gain": 1000,
    "capital.loss": 1000,
    "hours.per.week": 1,
}

# Generate predictions
samples = X_test.to_dict(orient="records")
constrained_preds = []
start_time_constrained = time.time()
for sample in samples:
    constrained_preds.append(
        fair_robust_predict(sample, tree_cons, features, "sex", deltas)
    )
end_time_constrained = time.time()

print("\nConstrained Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, constrained_preds):.4f}")
print(f"Precision: {precision_score(y_test, constrained_preds):.4f}")
print(
    f"Time for constrained model predictions: {end_time_constrained - start_time_constrained:.4f} seconds"
)

print("\nClassification Report for Constrained Model:")
print(classification_report(y_test, constrained_preds))
start_time_original = time.time()
original_preds = clf.predict(X_test)
end_time_original = time.time()
print(f"Accuracy: {accuracy_score(y_test, original_preds):.4f}")
print(f"Precision: {precision_score(y_test, original_preds):.4f}")
print(
    f"Time for original model predictions: {end_time_original - start_time_original:.4f} seconds"
)
