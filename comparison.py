import time

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
)
from sklearn.tree import DecisionTreeClassifier

from dataset import load_and_preprocess_dataset
from z3_model import fair_robust_predict, tree_constraints

# CHOOSE DATASET HERE
dataset_name = "adult"
# dataset_name = "german"
# dataset_name = "loan_data"

# Load data
X_train, X_test, y_train, y_test, features, deltas, sensitive = (
    load_and_preprocess_dataset(dataset_name)
)

# Train model
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Get Z3 constraints
tree_cons = tree_constraints(clf.tree_, features)


# Generate predictions
samples = X_test.to_dict(orient="records")
constrained_preds = []
start_time_constrained = time.time()
for sample in samples:
    constrained_preds.append(
        fair_robust_predict(sample, tree_cons, features, sensitive, deltas)
    )
end_time_constrained = time.time()

print("\nClassification Report for Constrained Model:")
print(classification_report(y_test, constrained_preds))


print("\nConstrained Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, constrained_preds):.4f}")
print(f"Precision: {precision_score(y_test, constrained_preds):.4f}")
print(
    "Time for constrained model predictions:"
    f" {end_time_constrained - start_time_constrained:.4f} seconds"
)

start_time_original = time.time()
original_preds = clf.predict(X_test)
end_time_original = time.time()
print("\nOriginal Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, original_preds):.4f}")
print(f"Precision: {precision_score(y_test, original_preds):.4f}")
print(
    f"Time for original model predictions: {end_time_original - start_time_original:.4f} seconds"
)
