import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
)
from sklearn.tree import DecisionTreeClassifier

from config import DATASET_NAME, MODEL_PARAMETERS, VERIFICATION_PARAMETERS
from src.utils.dataset import load_and_preprocess_dataset
from src.verification.z3_model import (
    fair_robust_predict,
    tree_constraints,
    verify_global_properties,
)


def load_data_and_train_model(dataset_name=DATASET_NAME):
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

    clf = DecisionTreeClassifier(
        max_depth=MODEL_PARAMETERS["max_depth"],
        random_state=MODEL_PARAMETERS["random_state"],
    )
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
    global_verification_result=None,
    use_z3=VERIFICATION_PARAMETERS["use_z3"],
):
    """
    Evaluate the model's performance and return predictions, metrics, and per-instance timing.

    Args:
        model: Trained model to evaluate.
        X_test: Test features.
        y_test: True labels.
        features: List of feature names.
        sensitive: Sensitive attribute name.
        deltas: Dictionary of perturbation deltas for robustness checks.
        tree_cons: Precomputed Z3 constraints for the model.
        global_verification_result: Result from verify_global_properties().
                                    If verified, uses fast path.
        use_z3 (bool): Whether to use Z3 for fairness/robustness checks.

    Returns:
        dict: {
            "predictions": list,
            "accuracy": float,
            "precision": float,
            "total_inference_time": float,
            "per_instance_times": list[float],
            "violations": list[bool]
        }
    """
    samples = X_test.to_dict(orient="records")
    predictions = []
    violations = []
    per_instance_times = []

    is_verified = (
        global_verification_result is not None
        and global_verification_result.get("is_verified", False)
    )

    start_time = time.time()

    if use_z3:
        for sample in samples:
            instance_start = time.time()

            if is_verified:
                # Fast path: globally verified, use tree traversal directly
                pred = int(model.predict([list(sample.values())])[0])
                has_violation = False
            else:
                # Slow path: per-instance SMT coercion
                pred, has_violation = fair_robust_predict(
                    sample, tree_cons, features, sensitive, deltas
                )

            instance_time = time.time() - instance_start
            predictions.append(pred)
            violations.append(has_violation)
            per_instance_times.append(instance_time)
    else:
        # Standard sklearn prediction (no verification)
        instance_start = time.time()
        predictions = model.predict(X_test).tolist()
        violations = [False] * len(predictions)
        base_time = (time.time() - instance_start) / len(predictions)
        per_instance_times = [base_time] * len(predictions)

    total_inference_time = time.time() - start_time

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)

    return {
        "predictions": predictions,
        "accuracy": accuracy,
        "precision": precision,
        "total_inference_time": total_inference_time,
        "per_instance_times": per_instance_times,
        "violations": violations,
    }


def print_timing_report(
    model_name,
    verification_time=None,
    total_inference_time=None,
    per_instance_times=None,
    violations=None,
):
    """
    Print the three-part timing breakdown for the article.

    Args:
        model_name (str): Name of the model.
        verification_time (float): Pre-deployment verification time (seconds).
        total_inference_time (float): Total time to process all test instances.
        per_instance_times (list[float]): Per-instance inference times.
        violations (list[bool]): List of violation flags.
    """
    print(f"\n{'=' * 60}")
    print(f"Timing Report: {model_name}")
    print(f"{'=' * 60}")

    if verification_time is not None:
        print(
            f"Verification time (pre-deployment): {verification_time:.4f} seconds"
        )

    if per_instance_times and len(per_instance_times) > 0:
        times = np.array(per_instance_times)
        print("Per-instance inference time:")
        print(f"  Mean:   {times.mean() * 1000:.4f} ms")
        print(f"  Median: {np.median(times) * 1000:.4f} ms")
        print(f"  P95:    {np.percentile(times, 95) * 1000:.4f} ms")
        print(f"  P99:    {np.percentile(times, 99) * 1000:.4f} ms")
        print(f"  Min:    {times.min() * 1000:.4f} ms")
        print(f"  Max:    {times.max() * 1000:.4f} ms")

    if total_inference_time is not None:
        print(f"Total inference time: {total_inference_time:.4f} seconds")

    if violations is not None:
        violation_rate = sum(violations) / len(violations)
        print(f"Coercion/Violation rate: {violation_rate:.4f}")

    print(f"{'=' * 60}\n")


def print_metrics(
    model_name,
    predictions,
    y_test,
    accuracy,
    precision,
    total_inference_time,
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
        total_inference_time (float): Total inference time.
        violations (list): List of violation flags (optional).
    """
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, predictions))

    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Total inference time: {total_inference_time:.4f} seconds")

    if violations is not None:
        violation_rate = sum(violations) / len(violations)
        print(f"Coercion Rate: {violation_rate:.4f}")


def main():
    dataset_name = DATASET_NAME

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

    print(
        f"\n=== Formal Verification Results for {dataset_name.upper()} Dataset ==="
    )

    # Step 1: Encode tree into Z3 constraints (part of verification)
    print("\nEncoding tree into SMT constraints...")
    encode_start = time.time()
    tree_cons = tree_constraints(clf.tree_, features)
    encode_time = time.time() - encode_start
    print(f"Encoding time: {encode_time:.4f} seconds")

    # Step 2: Global verification (pre-deployment)
    print("\nRunning global verification (pre-deployment)...")
    verification_result = verify_global_properties(
        tree_cons, features, sensitive, deltas
    )
    verification_time = verification_result["total_time"]
    is_verified = verification_result["is_verified"]

    print(
        f"Global verification result: {'VERIFIED (UNSAT)' if is_verified else 'NOT VERIFIED (SAT)'}"
    )
    print(f"Verification time: {verification_time:.4f} seconds")

    # Step 3: Evaluate constrained model (with verification awareness)
    print("\nEvaluating constrained model...")
    constrained_results = evaluate_model(
        clf,
        X_test,
        y_test,
        features,
        sensitive,
        deltas,
        tree_cons,
        global_verification_result=verification_result,
        use_z3=VERIFICATION_PARAMETERS["use_z3"],
    )

    # Step 4: Evaluate original model (baseline, no verification)
    print("Evaluating original model (baseline)...")
    original_results = evaluate_model(
        clf,
        X_test,
        y_test,
        features,
        sensitive,
        deltas,
        use_z3=False,
    )

    # Print detailed timing reports
    print_timing_report(
        "Constrained Model (Verified)"
        if is_verified
        else "Constrained Model (Unverified)",
        verification_time=encode_time + verification_time,
        total_inference_time=constrained_results["total_inference_time"],
        per_instance_times=constrained_results["per_instance_times"],
        violations=constrained_results["violations"],
    )

    print_timing_report(
        "Original Model (Baseline)",
        total_inference_time=original_results["total_inference_time"],
        per_instance_times=original_results["per_instance_times"],
    )

    # Print classification metrics
    print_metrics(
        "Constrained Model",
        constrained_results["predictions"],
        y_test,
        constrained_results["accuracy"],
        constrained_results["precision"],
        constrained_results["total_inference_time"],
        violations=constrained_results["violations"],
    )

    print_metrics(
        "Original Model",
        original_results["predictions"],
        y_test,
        original_results["accuracy"],
        original_results["precision"],
        original_results["total_inference_time"],
    )

    # Summary for article
    print("\n" + "=" * 60)
    print("SUMMARY FOR ARTICLE")
    print("=" * 60)
    print(
        f"Verification time (pre-deployment): {encode_time + verification_time:.4f} seconds"
    )
    print(f"  - Encoding: {encode_time:.4f} seconds")
    print(f"  - Global SMT check: {verification_time:.4f} seconds")
    print(
        f"  - Result: {'Model is globally verified (fast inference path enabled)' if is_verified else 'Violations detected (per-instance SMT required)'}"
    )

    constrained_times = np.array(constrained_results["per_instance_times"])
    print("\nPer-instance inference time (constrained):")
    print(f"  Mean: {constrained_times.mean() * 1000:.4f} ms")
    print(f"  P95: {np.percentile(constrained_times, 95) * 1000:.4f} ms")

    original_times = np.array(original_results["per_instance_times"])
    print("\nPer-instance inference time (original):")
    print(f"  Mean: {original_times.mean() * 1000:.4f} ms")

    print(
        f"\nTotal inference time (constrained): {constrained_results['total_inference_time']:.4f} seconds"
    )
    print(
        f"Total inference time (original): {original_results['total_inference_time']:.4f} seconds"
    )
    print(
        f"Speedup from verification: {original_results['total_inference_time'] / constrained_results['total_inference_time']:.2f}x"
        if constrained_results["total_inference_time"] > 0
        else ""
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
