import time

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from config import DATASET_NAME
from src.utils.dataset import load_and_preprocess_dataset
from src.verification.z3_model import fair_robust_predict, tree_constraints


def run_smt_solver_at_depth(depth, dataset_name="adult"):
    """
    Run SMT solver at a specific depth and return timing data.
    """
    # Load data and train model at specified depth
    X_train, X_test, y_train, y_test, features, deltas, sensitive = (
        load_and_preprocess_dataset(dataset_name)
    )

    clf = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Get Z3 constraints
    tree_cons = tree_constraints(clf.tree_, features)

    # Evaluate with Z3
    samples = X_test.to_dict(orient="records")
    start_time = time.time()

    for sample in samples:
        fair_robust_predict(sample, tree_cons, features, sensitive, deltas)

    end_time = time.time()
    time_taken = end_time - start_time

    return time_taken


def main():
    depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    times = []

    print("Running SMT solver at different depths...")
    for depth in depths:
        print(f"Running depth {depth}...")
        time_taken = run_smt_solver_at_depth(depth, DATASET_NAME)
        times.append(time_taken)
        print(f"Depth {depth}: {time_taken:.4f} seconds")

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(depths, times, marker="o", linewidth=2, markersize=8)
    plt.xlabel("Tree Depth", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.title(
        f"SMT Solver Performance vs Tree Depth for {DATASET_NAME.title()} dataset",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)
    plt.xticks(depths)
    plt.tight_layout()

    # Save the figure
    plt.savefig("adult_depth_timing.png", dpi=300)
    print("Figure saved as 'smt_depth_timing.png'")


if __name__ == "__main__":
    main()
