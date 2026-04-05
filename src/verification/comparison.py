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
    verify_global_fairness,
    verify_global_robustness,
)


def load_data_and_train_model(dataset_name=DATASET_NAME, random_state=42):
    """
    Load the dataset and train a Decision Tree model.

    Args:
        dataset_name (str): Name of the dataset to load.
        random_state (int): Random seed for model training.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, features, deltas, sensitive, clf)
    """
    X_train, X_test, y_train, y_test, features, deltas, sensitive = (
        load_and_preprocess_dataset(dataset_name, random_state=random_state)
    )

    clf = DecisionTreeClassifier(
        max_depth=MODEL_PARAMETERS["max_depth"],
        random_state=random_state,
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
        global_verification_result: Dict with "fairness_verified" and "robustness_verified" flags.
        use_z3 (bool): Whether to use Z3 for fairness/robustness checks.

    Returns:
        dict: {
            "predictions": list,
            "accuracy": float,
            "precision": float,
            "total_inference_time": float,
            "per_instance_times": list[float],
            "fairness_violations": list[bool],
            "robustness_violations": list[bool]
        }
    """
    samples = X_test.to_dict(orient="records")
    predictions = []
    fairness_violations = []
    robustness_violations = []
    per_instance_times = []

    is_fair = (
        global_verification_result is not None
        and global_verification_result.get("fairness_verified", False)
    )
    is_robust = (
        global_verification_result is not None
        and global_verification_result.get("robustness_verified", False)
    )

    start_time = time.time()

    if use_z3:
        for sample in samples:
            instance_start = time.time()

            if is_fair and is_robust:
                # Fast path: globally verified, no per-instance SMT needed
                pred = int(model.predict([list(sample.values())])[0])
                fairness_violations.append(False)
                robustness_violations.append(False)
            else:
                # Slow path: per-instance SMT checks
                pred, fair_viol, robust_viol = fair_robust_predict(
                    sample, tree_cons, features, sensitive, deltas
                )
                fairness_violations.append(fair_viol)
                robustness_violations.append(robust_viol)

            instance_time = time.time() - instance_start
            predictions.append(pred)
            per_instance_times.append(instance_time)
    else:
        # Standard sklearn prediction (no verification)
        instance_start = time.time()
        predictions = model.predict(X_test).tolist()
        fairness_violations = [False] * len(predictions)
        robustness_violations = [False] * len(predictions)
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
        "fairness_violations": fairness_violations,
        "robustness_violations": robustness_violations,
    }


def print_timing_report(
    model_name,
    verification_time=None,
    fairness_verification_time=None,
    robustness_verification_time=None,
    total_inference_time=None,
    per_instance_times=None,
    fairness_violations=None,
    robustness_violations=None,
):
    """
    Print the three-part timing breakdown for the article.

    Args:
        model_name (str): Name of the model.
        verification_time (float): Total pre-deployment verification time.
        fairness_verification_time (float): Fairness verification time.
        robustness_verification_time (float): Robustness verification time.
        total_inference_time (float): Total time to process all test instances.
        per_instance_times (list[float]): Per-instance inference times.
        fairness_violations (list[bool]): Fairness violation flags.
        robustness_violations (list[bool]): Robustness violation flags.
    """
    print(f"\n{'=' * 60}")
    print(f"Timing Report: {model_name}")
    print(f"{'=' * 60}")

    if verification_time is not None:
        print(
            f"Verification time (pre-deployment): {verification_time:.4f} seconds"
        )
    if fairness_verification_time is not None:
        print(
            f"  Fairness verification: {fairness_verification_time:.4f} seconds"
        )
    if robustness_verification_time is not None:
        print(
            f"  Robustness verification: {robustness_verification_time:.4f} seconds"
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

    if fairness_violations is not None:
        fair_rate = sum(fairness_violations) / len(fairness_violations)
        print(f"Fairness violation rate: {fair_rate:.4f}")
    if robustness_violations is not None:
        robust_rate = sum(robustness_violations) / len(robustness_violations)
        print(f"Robustness violation rate: {robust_rate:.4f}")

    print(f"{'=' * 60}\n")


def print_metrics(
    model_name,
    predictions,
    y_test,
    accuracy,
    precision,
    total_inference_time,
    fairness_violations=None,
    robustness_violations=None,
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
        fairness_violations (list): List of fairness violation flags (optional).
        robustness_violations (list): List of robustness violation flags (optional).
    """
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, predictions))

    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Total inference time: {total_inference_time:.4f} seconds")

    if fairness_violations is not None:
        fair_rate = sum(fairness_violations) / len(fairness_violations)
        print(f"Fairness violation rate: {fair_rate:.4f}")
    if robustness_violations is not None:
        robust_rate = sum(robustness_violations) / len(robustness_violations)
        print(f"Robustness violation rate: {robust_rate:.4f}")


def main(num_runs=1):
    dataset_name = DATASET_NAME

    all_verification_times = []
    all_fairness_verification_times = []
    all_robustness_verification_times = []
    all_inference_times = []
    all_per_instance_means = []
    all_accuracies = []
    all_precisions = []
    all_fairness_violation_rates = []
    all_robustness_violation_rates = []

    for run in range(num_runs):
        seed = 42 + run
        print(f"\n{'=' * 60}")
        print(f"Run {run + 1}/{num_runs} (Seed: {seed})")
        print(f"{'=' * 60}")

        X_train, X_test, y_train, y_test, features, deltas, sensitive, clf = (
            load_data_and_train_model(dataset_name, random_state=seed)
        )

        encode_start = time.time()
        tree_cons = tree_constraints(clf.tree_, features)
        encode_time = time.time() - encode_start

        fairness_result = verify_global_fairness(tree_cons, features, sensitive)
        robustness_result = verify_global_robustness(
            tree_cons, features, sensitive, deltas
        )
        fairness_verification_time = fairness_result["total_time"]
        robustness_verification_time = robustness_result["total_time"]
        verification_time = (
            encode_time
            + fairness_verification_time
            + robustness_verification_time
        )
        is_fair = fairness_result["is_verified"]
        is_robust = robustness_result["is_verified"]

        all_verification_times.append(verification_time)
        all_fairness_verification_times.append(fairness_verification_time)
        all_robustness_verification_times.append(robustness_verification_time)

        global_verification_result = {
            "fairness_verified": is_fair,
            "robustness_verified": is_robust,
        }
        constrained_results = evaluate_model(
            clf,
            X_test,
            y_test,
            features,
            sensitive,
            deltas,
            tree_cons,
            global_verification_result=global_verification_result,
            use_z3=VERIFICATION_PARAMETERS["use_z3"],
        )

        original_results = evaluate_model(
            clf, X_test, y_test, features, sensitive, deltas, use_z3=False
        )

        all_inference_times.append(constrained_results["total_inference_time"])
        all_per_instance_means.append(
            np.mean(constrained_results["per_instance_times"])
        )
        all_accuracies.append(constrained_results["accuracy"])
        all_precisions.append(constrained_results["precision"])
        all_fairness_violation_rates.append(
            np.mean(constrained_results["fairness_violations"])
        )
        all_robustness_violation_rates.append(
            np.mean(constrained_results["robustness_violations"])
        )

        if num_runs == 1:
            status = "Verified" if (is_fair and is_robust) else "Unverified"
            print_timing_report(
                f"Constrained Model ({status})",
                verification_time=verification_time,
                fairness_verification_time=fairness_verification_time,
                robustness_verification_time=robustness_verification_time,
                total_inference_time=constrained_results[
                    "total_inference_time"
                ],
                per_instance_times=constrained_results["per_instance_times"],
                fairness_violations=constrained_results["fairness_violations"],
                robustness_violations=constrained_results[
                    "robustness_violations"
                ],
            )
            print_timing_report(
                "Original Model (Baseline)",
                total_inference_time=original_results["total_inference_time"],
                per_instance_times=original_results["per_instance_times"],
            )
            print_metrics(
                "Constrained Model",
                constrained_results["predictions"],
                y_test,
                constrained_results["accuracy"],
                constrained_results["precision"],
                constrained_results["total_inference_time"],
                fairness_violations=constrained_results["fairness_violations"],
                robustness_violations=constrained_results[
                    "robustness_violations"
                ],
            )
            print_metrics(
                "Original Model",
                original_results["predictions"],
                y_test,
                original_results["accuracy"],
                original_results["precision"],
                original_results["total_inference_time"],
            )

    if num_runs > 1:
        print("\n" + "=" * 60)
        print(f"AGGREGATED RESULTS OVER {num_runs} RUNS")
        print("=" * 60)

        def fmt(mean, std):
            return f"{mean:.4f} ± {std:.4f}"

        print(
            f"Verification time (total): {fmt(np.mean(all_verification_times), np.std(all_verification_times))} s"
        )
        print(
            f"  Fairness verification: {fmt(np.mean(all_fairness_verification_times), np.std(all_fairness_verification_times))} s"
        )
        print(
            f"  Robustness verification: {fmt(np.mean(all_robustness_verification_times), np.std(all_robustness_verification_times))} s"
        )
        print(
            f"Total inference time: {fmt(np.mean(all_inference_times), np.std(all_inference_times))} s"
        )
        print(
            f"Per-instance mean time: {fmt(np.mean(all_per_instance_means) * 1000, np.std(all_per_instance_means) * 1000)} ms"
        )
        print(
            f"Accuracy: {fmt(np.mean(all_accuracies), np.std(all_accuracies))}"
        )
        print(
            f"Precision: {fmt(np.mean(all_precisions), np.std(all_precisions))}"
        )
        print(
            f"Fairness violation rate: {fmt(np.mean(all_fairness_violation_rates), np.std(all_fairness_violation_rates))}"
        )
        print(
            f"Robustness violation rate: {fmt(np.mean(all_robustness_violation_rates), np.std(all_robustness_violation_rates))}"
        )
        print("=" * 60)


if __name__ == "__main__":
    main()
