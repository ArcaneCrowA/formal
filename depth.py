import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from config import DATASET_NAME, DEPTHS
from src.utils.dataset import load_and_preprocess_dataset
from src.verification.z3_model import (
    fair_robust_predict,
    tree_constraints,
    verify_global_fairness,
    verify_global_robustness,
)

warnings.filterwarnings("ignore", category=UserWarning)


def run_depth_experiment(depth, dataset_name="adult"):
    """
    Run verification and inference timing experiments at a specific depth.

    Returns:
        dict: Timing metrics for this depth
    """
    X_train, X_test, y_train, y_test, features, deltas, sensitive = (
        load_and_preprocess_dataset(dataset_name)
    )

    clf = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Phase 1: Pre-deployment Verification
    # Step 1a: Encode tree into SMT constraints
    encode_start = time.time()
    tree_cons = tree_constraints(clf.tree_, features)
    encode_time = time.time() - encode_start

    # Step 1b: Global fairness verification (Proposition 2)
    fairness_result = verify_global_fairness(tree_cons, features, sensitive)
    fairness_check_time = fairness_result["solving_time"]
    is_fair = fairness_result["is_verified"]

    # Step 1c: Global robustness verification (Proposition 3)
    robustness_result = verify_global_robustness(
        tree_cons, features, sensitive, deltas
    )
    robustness_check_time = robustness_result["solving_time"]
    is_robust = robustness_result["is_verified"]

    total_verification_time = (
        encode_time + fairness_check_time + robustness_check_time
    )

    # Phase 2: Per-instance Inference
    samples = X_test.to_dict(orient="records")
    predictions = []
    fairness_violations = []
    robustness_violations = []
    per_instance_times = []

    inference_start = time.time()

    for sample in samples:
        instance_start = time.time()

        if is_fair and is_robust:
            # Fast path: globally verified, use standard prediction
            pred = int(clf.predict([list(sample.values())])[0])
            fairness_viol = False
            robustness_viol = False
        else:
            # Slow path: per-instance SMT checks
            pred, fairness_viol, robustness_viol = fair_robust_predict(
                sample, tree_cons, features, sensitive, deltas
            )

            predictions.append(pred)
            fairness_violations.append(fairness_viol)
            robustness_violations.append(robustness_viol)
        instance_time = time.time() - instance_start
        per_instance_times.append(instance_time)

    total_inference_time = time.time() - inference_start

    # Calculate statistics
    times_array = np.array(per_instance_times)
    fairness_violation_rate = (
        sum(fairness_violations) / len(fairness_violations)
        if fairness_violations
        else 0.0
    )
    robustness_violation_rate = (
        sum(robustness_violations) / len(robustness_violations)
        if robustness_violations
        else 0.0
    )

    return {
        "depth": depth,
        "is_fair": is_fair,
        "is_robust": is_robust,
        "encode_time": encode_time,
        "fairness_check_time": fairness_check_time,
        "robustness_check_time": robustness_check_time,
        "total_verification_time": total_verification_time,
        "total_inference_time": total_inference_time,
        "per_instance_mean": times_array.mean(),
        "per_instance_median": np.median(times_array),
        "per_instance_p95": np.percentile(times_array, 95),
        "per_instance_p99": np.percentile(times_array, 99),
        "per_instance_min": times_array.min(),
        "per_instance_max": times_array.max(),
        "fairness_violation_rate": fairness_violation_rate,
        "robustness_violation_rate": robustness_violation_rate,
        "num_samples": len(samples),
    }


def main():
    depths = DEPTHS
    results = []

    print("Running depth experiments with separated timing metrics...")
    print("=" * 70)

    for depth in depths:
        print(f"\nRunning depth {depth}...")
        result = run_depth_experiment(depth, DATASET_NAME)
        results.append(result)

        print(
            f"  Verification: {result['total_verification_time']:.4f}s "
            f"(encode: {result['encode_time']:.4f}s, "
            f"fairness: {result['fairness_check_time']:.4f}s, "
            f"robustness: {result['robustness_check_time']:.4f}s)"
        )
        print(
            f"  Fairness: {'VERIFIED (UNSAT)' if result['is_fair'] else 'NOT VERIFIED (SAT)'}"
        )
        print(
            f"  Robustness: {'VERIFIED (UNSAT)' if result['is_robust'] else 'NOT VERIFIED (SAT)'}"
        )
        print(f"  Inference (mean): {result['per_instance_mean'] * 1000:.4f}ms")
        print(f"  Total inference: {result['total_inference_time']:.4f}s")
        print(
            f"  Fairness violation rate: {result['fairness_violation_rate']:.4f}"
        )
        print(
            f"  Robustness violation rate: {result['robustness_violation_rate']:.4f}"
        )

    # Extract data for plotting
    depth_values = [r["depth"] for r in results]
    verification_times = [r["total_verification_time"] for r in results]
    encode_times = [r["encode_time"] for r in results]
    fairness_check_times = [r["fairness_check_time"] for r in results]
    robustness_check_times = [r["robustness_check_time"] for r in results]
    inference_means = [
        r["per_instance_mean"] * 1000 for r in results
    ]  # Convert to ms
    inference_p95 = [r["per_instance_p95"] * 1000 for r in results]
    total_inference_times = [r["total_inference_time"] for r in results]
    fair_flags = [r["is_fair"] for r in results]
    robust_flags = [r["is_robust"] for r in results]

    # Plot 1: Verification Time vs Depth
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(
        depth_values,
        verification_times,
        marker="o",
        linewidth=2,
        markersize=8,
        color="blue",
        label="Total verification",
    )
    plt.plot(
        depth_values,
        encode_times,
        marker="s",
        linewidth=1.5,
        markersize=6,
        color="green",
        label="Encoding only",
    )
    plt.plot(
        depth_values,
        fairness_check_times,
        marker="^",
        linewidth=1.5,
        markersize=6,
        color="orange",
        label="Fairness SMT check",
    )
    plt.plot(
        depth_values,
        robustness_check_times,
        marker="D",
        linewidth=1.5,
        markersize=6,
        color="purple",
        label="Robustness SMT check",
    )
    plt.xlabel("Tree Depth", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.title("Verification Time (Pre-deployment) vs Depth", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(depth_values)
    plt.legend(fontsize=10)

    # Plot 2: Per-instance Inference Time vs Depth
    plt.subplot(2, 2, 2)
    plt.plot(
        depth_values,
        inference_means,
        marker="o",
        linewidth=2,
        markersize=8,
        color="red",
        label="Mean",
    )
    plt.fill_between(
        depth_values,
        [r["per_instance_min"] * 1000 for r in results],
        inference_p95,
        alpha=0.2,
        color="red",
        label="Min to P95 range",
    )
    plt.xlabel("Tree Depth", fontsize=12)
    plt.ylabel("Time (ms)", fontsize=12)
    plt.title("Per-instance Inference Time vs Depth", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(depth_values)
    plt.legend(fontsize=10)

    # Plot 3: Total Inference Time vs Depth
    plt.subplot(2, 2, 3)
    plt.plot(
        depth_values,
        total_inference_times,
        marker="o",
        linewidth=2,
        markersize=8,
        color="purple",
    )
    plt.xlabel("Tree Depth", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.title("Total Inference Time vs Depth", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(depth_values)

    # Plot 4: Verification Status vs Depth
    plt.subplot(2, 2, 4)
    fair_verified = [d for d, v in zip(depth_values, fair_flags) if v]
    fair_unverified = [d for d, v in zip(depth_values, fair_flags) if not v]
    robust_verified = [d for d, v in zip(depth_values, robust_flags) if v]
    robust_unverified = [d for d, v in zip(depth_values, robust_flags) if not v]
    plt.scatter(
        fair_verified,
        [1.1] * len(fair_verified),
        color="green",
        s=100,
        marker="o",
        label="Fairness Verified",
    )
    plt.scatter(
        fair_unverified,
        [1.1] * len(fair_unverified),
        color="red",
        s=100,
        marker="x",
        label="Fairness Not Verified",
    )
    plt.scatter(
        robust_verified,
        [0.9] * len(robust_verified),
        color="blue",
        s=100,
        marker="o",
        label="Robustness Verified",
    )
    plt.scatter(
        robust_unverified,
        [0.9] * len(robust_unverified),
        color="orange",
        s=100,
        marker="x",
        label="Robustness Not Verified",
    )
    plt.xlabel("Tree Depth", fontsize=12)
    plt.ylabel("Status", fontsize=12)
    plt.title("Global Verification Status vs Depth", fontsize=12)
    plt.yticks([0.9, 1.1], ["Robustness", "Fairness"])
    plt.grid(True, alpha=0.3)
    plt.xticks(depth_values)
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("depth_timing_separated.png", dpi=300, bbox_inches="tight")
    print("\n" + "=" * 70)
    print("Figure saved as 'depth_timing_separated.png'")

    # Print summary table for article
    print("\n" + "=" * 70)
    print("SUMMARY TABLE FOR ARTICLE")
    print("=" * 70)
    print(
        f"{'Depth':<6} {'Fair':<6} {'Robust':<7} {'Verification (s)':<18} "
        f"{'Inference Mean (ms)':<20} {'Total Inference (s)':<20}"
    )
    print("-" * 70)
    for r in results:
        fair_str = "Yes" if r["is_fair"] else "No"
        robust_str = "Yes" if r["is_robust"] else "No"
        print(
            f"{r['depth']:<6} {fair_str:<6} {robust_str:<7} "
            f"{r['total_verification_time']:<18.4f} "
            f"{r['per_instance_mean'] * 1000:<20.4f} "
            f"{r['total_inference_time']:<20.4f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
