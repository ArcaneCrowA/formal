import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from config import DATASET_NAME
from src.utils.dataset import load_and_preprocess_dataset
from src.verification.z3_model import (
    fair_robust_predict,
    tree_constraints,
    verify_global_properties,
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

    # Step 1b: Global property verification
    verification_result = verify_global_properties(
        tree_cons, features, sensitive, deltas
    )
    global_check_time = verification_result["solving_time"]
    is_verified = verification_result["is_verified"]
    total_verification_time = encode_time + global_check_time

    # Phase 2: Per-instance Inference
    samples = X_test.to_dict(orient="records")
    predictions = []
    violations = []
    per_instance_times = []

    inference_start = time.time()

    for sample in samples:
        instance_start = time.time()

        if is_verified:
            # Fast path: globally verified, use tree traversal directly
            pred = int(clf.predict([list(sample.values())])[0])
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

    total_inference_time = time.time() - inference_start

    # Calculate statistics
    times_array = np.array(per_instance_times)
    violation_rate = sum(violations) / len(violations) if violations else 0.0

    return {
        "depth": depth,
        "is_verified": is_verified,
        "encode_time": encode_time,
        "global_check_time": global_check_time,
        "total_verification_time": total_verification_time,
        "total_inference_time": total_inference_time,
        "per_instance_mean": times_array.mean(),
        "per_instance_median": np.median(times_array),
        "per_instance_p95": np.percentile(times_array, 95),
        "per_instance_p99": np.percentile(times_array, 99),
        "per_instance_min": times_array.min(),
        "per_instance_max": times_array.max(),
        "violation_rate": violation_rate,
        "num_samples": len(samples),
    }


def main():
    depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
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
            f"global check: {result['global_check_time']:.4f}s)"
        )
        print(
            f"  Result: {'VERIFIED (UNSAT)' if result['is_verified'] else 'NOT VERIFIED (SAT)'}"
        )
        print(f"  Inference (mean): {result['per_instance_mean'] * 1000:.4f}ms")
        print(f"  Total inference: {result['total_inference_time']:.4f}s")
        print(f"  Violation rate: {result['violation_rate']:.4f}")

    # Extract data for plotting
    depth_values = [r["depth"] for r in results]
    verification_times = [r["total_verification_time"] for r in results]
    encode_times = [r["encode_time"] for r in results]
    global_check_times = [r["global_check_time"] for r in results]
    inference_means = [
        r["per_instance_mean"] * 1000 for r in results
    ]  # Convert to ms
    inference_p95 = [r["per_instance_p95"] * 1000 for r in results]
    total_inference_times = [r["total_inference_time"] for r in results]
    verified_flags = [r["is_verified"] for r in results]

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
        global_check_times,
        marker="^",
        linewidth=1.5,
        markersize=6,
        color="orange",
        label="Global SMT check",
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
    verified_depths = [d for d, v in zip(depth_values, verified_flags) if v]
    unverified_depths = [
        d for d, v in zip(depth_values, verified_flags) if not v
    ]
    plt.scatter(
        verified_depths,
        [1] * len(verified_depths),
        color="green",
        s=100,
        marker="o",
        label="Verified (UNSAT)",
    )
    plt.scatter(
        unverified_depths,
        [1] * len(unverified_depths),
        color="red",
        s=100,
        marker="x",
        label="Not Verified (SAT)",
    )
    plt.xlabel("Tree Depth", fontsize=12)
    plt.ylabel("Status", fontsize=12)
    plt.title("Global Verification Status vs Depth", fontsize=12)
    plt.yticks([1], ["Verified"])
    plt.grid(True, alpha=0.3)
    plt.xticks(depth_values)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("depth_timing_separated.png", dpi=300, bbox_inches="tight")
    print("\n" + "=" * 70)
    print("Figure saved as 'depth_timing_separated.png'")

    # Print summary table for article
    print("\n" + "=" * 70)
    print("SUMMARY TABLE FOR ARTICLE")
    print("=" * 70)
    print(
        f"{'Depth':<6} {'Verified':<10} {'Verification (s)':<18} "
        f"{'Inference Mean (ms)':<20} {'Total Inference (s)':<20}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['depth']:<6} {'Yes' if r['is_verified'] else 'No':<10} "
            f"{r['total_verification_time']:<18.4f} "
            f"{r['per_instance_mean'] * 1000:<20.4f} "
            f"{r['total_inference_time']:<20.4f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
