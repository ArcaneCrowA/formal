# verification_methods.py
# Python implementation of formal verification methods for decision tree classifiers using Z3

import numpy as np
import pandas as pd
import z3
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, _tree


class DecisionTreeVerifier:
    def __init__(self, model: DecisionTreeClassifier, feature_bounds: dict, epsilon: float = 0.1, delta: float = 0.02):
        """
        model: trained DecisionTreeClassifier
        feature_bounds: {feature_index: (min, max)} for continuous features
        epsilon: perturbation radius for L_inf robustness
        delta: (reserved for future fairness checks)
        """
        self.model = model
        self.bounds = feature_bounds
        self.epsilon = epsilon
        self.delta = delta

    def _gather_constraints(self):
        # Initialize list of constraints
        self.constraints = []
        # Declare input vars
        self.x = {}
        for i, (lo, hi) in self.bounds.items():
            xi = z3.Real(f"x_{i}")
            self.x[i] = xi
            self.constraints.append(xi >= lo)
            self.constraints.append(xi <= hi)
        # Output var
        self.y = z3.Bool("y_pred")

    def _translate_tree(self):
        tree: _tree.Tree = self.model.tree_

        def recurse(node, path):
            if tree.feature[node] != _tree.TREE_UNDEFINED:
                idx = tree.feature[node]
                thresh = tree.threshold[node]
                recurse(tree.children_left[node], path + [self.x[idx] <= thresh])
                recurse(tree.children_right[node], path + [self.x[idx] > thresh])
            else:
                pred = np.argmax(tree.value[node])
                cond = z3.And(*path) if path else z3.BoolVal(True)
                # assert that if cond then y_pred == (pred == 1)
                self.constraints.append(z3.Implies(cond, self.y == (bool(pred) == 1)))

        recurse(0, [])

    def add_robustness_constraints(self, x0_values: dict):
        # Perturbation constraints
        for i, x0 in x0_values.items():
            xi = self.x[i]
            self.constraints.append(xi >= x0 - self.epsilon)
            self.constraints.append(xi <= x0 + self.epsilon)
        # Fix original label
        y0 = bool(self.model.predict([list(x0_values.values())])[0])
        self.constraints.append(self.y == y0)

    def verify(self, mode="robustness", **kwargs):
        # Reset and gather
        self._gather_constraints()
        self._translate_tree()
        if mode == "robustness":
            self.add_robustness_constraints(kwargs.get("x0", {}))

        solver = z3.Solver()
        solver.add(self.constraints)
        return solver.check() == z3.sat


# --- Example: UCI Adult Dataset ---
if __name__ == "__main__":
    # Load UCI Adult
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    cols = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]
    df = pd.read_csv(url, names=cols, na_values="?", skipinitialspace=True)
    df.dropna(inplace=True)

    # Features and target
    X = df.drop("income", axis=1)
    y = (df["income"] == ">50K").astype(int)

    # Identify continuous and categorical
    cont_feats = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    cat_feats = [c for c in X.columns if c not in cont_feats]

    # Preprocessing pipeline
    ct = ColumnTransformer([("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_feats)], remainder="passthrough")
    X_proc = ct.fit_transform(X)

    # Build feature bounds
    feature_bounds = {}
    n_features = X_proc.shape[1]
    for i in range(n_features):
        col = X_proc[:, i]
        feature_bounds[i] = (float(col.min()), float(col.max()))

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    # Verify a test sample for robustness
    sample = X_test[0]
    x0 = {i: float(v) for i, v in enumerate(sample)}
    verifier = DecisionTreeVerifier(clf, feature_bounds, epsilon=0.1)
    robust = verifier.verify(mode="robustness", x0=x0)
    print(f"Sample 0 robustness under L_inf=0.1: {robust}")
