import time

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
)
from sklearn.tree import DecisionTreeClassifier

from src.utils.dataset import load_and_preprocess_dataset
from src.verification.z3_model import fair_robust_predict, tree_constraints


def load_data_and_train_model(dataset_name):
    """
    Load the dataset and train a Decision Tree model.

    Args:
        dataset_name (str): Name of the dataset to load.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, features, deltas, sensitive, clf)
    """
    X_train, X_test, y_train, y_test, features, deltas, sensitive = (
        load_and_preprocess_dataset(dataset_name)
    )

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    return X_train, X_test, y_train, y_test, features, deltas, sensitive, clf


def evaluate_model(
    model,
    X_test,
    y_test,
    features,
    sensitive,
    deltas,
    tree_cons=None,
    use_z3=False,
):
    """
    Evaluate the model's performance and return predictions, metrics, and violations.

    Args:
        model: Trained model to evaluate.
        X_test: Test features.
        y_test: True labels.
        features: List of feature names.
        sensitive: Sensitive attribute name.
        deltas: Dictionary of perturbation deltas for robustness checks.
        tree_cons: Precomputed Z3 constraints for the model.
        use_z3 (bool): Whether to use Z3 for fairness/robustness checks.

    Returns:
        tuple: (predictions, accuracy, precision, time_taken, violations)
    """
    samples = X_test.to_dict(orient="records")
    predictions = []
    violations = []
    start_time = time.time()

    if use_z3:
        for sample in samples:
            pred, has_violation = fair_robust_predict(
                sample, tree_cons, features, sensitive, deltas
            )
            predictions.append(pred)
            violations.append(has_violation)
    else:
        predictions = model.predict(X_test)
        violations = [False] * len(
            predictions
        )  # No violations for the original model

    end_time = time.time()
    time_taken = end_time - start_time

    # Adjust predictions for samples with violations
    adjusted_predictions = []
    for i, has_violation in enumerate(violations):
        if has_violation:
            # Treat as misclassification by flipping the prediction
            adjusted_predictions.append(1 - predictions[i])
        else:
            adjusted_predictions.append(predictions[i])

    accuracy = accuracy_score(y_test, adjusted_predictions)
    precision = precision_score(y_test, adjusted_predictions)

    return predictions, accuracy, precision, time_taken, violations


def print_metrics(
    model_name,
    predictions,
    y_test,
    accuracy,
    precision,
    time_taken,
    violations=None,
):
    """
    Print performance metrics for the model.

    Args:
        model_name (str): Name of the model.
        predictions: Predicted labels.
        y_test: True labels.
        accuracy (float): Accuracy score.
        precision (float): Precision score.
        time_taken (float): Time taken for predictions.
        violations (list): List of violation flags (optional).
    """
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, predictions))

    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(
        f"Time for {model_name.lower()} predictions: {time_taken:.4f} seconds"
    )

    if violations is not None:
        violation_rate = sum(violations) / len(violations)
        print(f"Violation Rate: {violation_rate:.4f}")


def main():
    # CHOOSE DATASET HERE
    dataset_name = "adult"
    # dataset_name = "german"
    # dataset_name = "loan_data"

    # Load data and train model
    (
        X_train,
        X_test,
        y_train,
        y_test,
        features,
        deltas,
        sensitive,
        clf,
    ) = load_data_and_train_model(dataset_name)

    # Get Z3 constraints
    tree_cons = tree_constraints(clf.tree_, features)

    # Evaluate constrained model (using Z3)
    (
        constrained_preds,
        constrained_accuracy,
        constrained_precision,
        constrained_time,
        constrained_violations,
    ) = evaluate_model(
        clf,
        X_test,
        y_test,
        features,
        sensitive,
        deltas,
        tree_cons,
        use_z3=True,
    )

    # Evaluate original model (without Z3)
    (
        original_preds,
        original_accuracy,
        original_precision,
        original_time,
        _,
    ) = evaluate_model(
        clf,
        X_test,
        y_test,
        features,
        sensitive,
        deltas,
        use_z3=False,
    )

    # Print metrics
    print_metrics(
        "Constrained Model",
        constrained_preds,
        y_test,
        constrained_accuracy,
        constrained_precision,
        constrained_time,
        violations=constrained_violations,
    )

    print_metrics(
        "Original Model",
        original_preds,
        y_test,
        original_accuracy,
        original_precision,
        original_time,
    )


if __name__ == "__main__":
    main()
