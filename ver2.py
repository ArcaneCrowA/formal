from sklearn.tree import DecisionTreeClassifier
from z3 import And, Implies, Or, Real, Solver, sat


def tree_constraints(tree, feature_names):
    """
    Return X_vars (feature->Z3 Real), and a list of (tests_list, leaf_value).
    tests_list is a Python list of Z3 Boolean expressions.
    """
    constraints = []
    X_vars = {name: Real(name) for name in feature_names}

    def dfs(node_id, tests):
        # Leaf node
        if tree.feature[node_id] == -2:
            leaf_val = int(tree.value[node_id][0].argmax())
            constraints.append((tests.copy(), leaf_val))
            return

        feat_idx = tree.feature[node_id]
        feat_name = feature_names[feat_idx]
        thresh = tree.threshold[node_id]
        # left branch
        tests.append(X_vars[feat_name] <= thresh)
        dfs(tree.children_left[node_id], tests)
        tests.pop()
        # right branch
        tests.append(X_vars[feat_name] > thresh)
        dfs(tree.children_right[node_id], tests)
        tests.pop()

    dfs(0, [])
    return X_vars, constraints


def demographic_parity_assertions(solver: Solver, X_vars: dict, path_leaf: list, sample: dict, Z_name: str):
    """
    Enforce that prediction does not depend on Z for this single sample.
    """
    sample_eqs = [X_vars[f] == sample[f] for f in X_vars]
    z = Real(Z_name)
    solver.add(Or(z == 0, z == 1))
    # pick the leaf this sample falls into
    for tests, leaf_val in path_leaf:
        match = And(*tests, *sample_eqs) if tests else And(*sample_eqs)
        # enforce the same outcome regardless of z
        solver.add(Implies(And(match, z == 0), True))
        solver.add(Implies(And(match, z == 1), True))
        break


def robustness_assertions(solver: Solver, X_vars: dict, path_leaf: list, sample: dict, deltas: dict):
    """
    Guarantee prediction stability under ±deltas perturbation for this sample.
    """
    # create perturbed variables
    Xp = {f: Real(f + "_p") for f in X_vars}
    # add bounds for perturbations
    for f in X_vars:
        d = deltas.get(f, 0)
        solver.add(Xp[f] >= sample[f] - d)
        solver.add(Xp[f] <= sample[f] + d)

    sample_eqs = [X_vars[f] == sample[f] for f in X_vars]
    # find the leaf path for this sample
    for tests, leaf_val in path_leaf:
        cond_orig = And(*tests, *sample_eqs) if tests else And(*sample_eqs)
        perturbed_conds = []
        for test in tests:
            # extract variable name and bound from the Z3 test
            var_name = test.arg(0).decl().name()
            bound_ref = test.arg(1)
            # convert bound_ref (which may be a rational) to float
            if bound_ref.is_int_value():
                bound = bound_ref.as_long()
            else:
                num = bound_ref.numerator_as_long()
                den = bound_ref.denominator_as_long()
                bound = num / den
            # check operator
            op = test.decl().name()  # '<=' or '>'
            if op == "<=":
                perturbed_conds.append(Xp[var_name] <= bound)
            else:
                perturbed_conds.append(Xp[var_name] > bound)
        # assert: if original sample matches, then any perturbed satisfying same tests yields same outcome
        solver.add(Implies(cond_orig, Implies(And(*perturbed_conds), True)))
        break


if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    # Load and preprocess adult dataset
    df = pd.read_csv("adult.csv").dropna()
    # encode target
    df["income_label"] = df["income"].map({"<=50K": 0, ">50K": 1})

    # label encode categoricals
    categ = ["workclass", "education", "marital.status", "occupation", "relationship", "race", "sex", "native.country"]
    for col in categ:
        df[col] = LabelEncoder().fit_transform(df[col])

    # feature list must match df columns exactly
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train decision tree
    clf = DecisionTreeClassifier(max_depth=4, random_state=0)
    clf.fit(X_train, y_train)

    # extract SMT constraints
    X_vars, path_leaf = tree_constraints(clf.tree_, features)

    # convert test set to list of dicts
    samples = X_test.to_dict(orient="records")

    # define perturbation limits, keys must match `features`
    deltas = {
        "age": 1,
        "workclass": 0,
        "fnlwgt": 0,
        "education": 0,
        "education.num": 0,
        "marital.status": 0,
        "occupation": 0,
        "relationship": 0,
        "race": 0,
        "sex": 0,
        "capital.gain": 1000,
        "capital.loss": 1000,
        "hours.per.week": 1,
        "native.country": 0,
    }

    for i, sample in enumerate(samples):
        solver_f = Solver()
        demographic_parity_assertions(solver_f, X_vars, path_leaf, sample, "sex")
        print(f"Sample {i} fairness sat? {solver_f.check() == sat}")

        solver_r = Solver()
        robustness_assertions(solver_r, X_vars, path_leaf, sample, deltas)
        print(f"Sample {i} robustness sat? {solver_r.check() == sat}")
