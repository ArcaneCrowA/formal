import time

from z3 import (
    And,
    BoolVal,
    Implies,
    Or,
    Real,
    RealVal,
    Solver,
    sat,
    substitute,
    unsat,
)


def tree_constraints(tree, feature_names):
    """
    Extract decision tree constraints into Z3 format.
    Returns:
        X_vars: Dictionary of Z3 variables for features
        path_leaf: List of (path_condition, leaf_value) tuples
    """
    path_leaf = []
    X_vars = {name: Real(name) for name in feature_names}

    def dfs(node_id, current_path):
        if tree.feature[node_id] == -2:  # Leaf node
            # tree.value[node_id] is an array of shape (1, n_classes)
            leaf_val = int(tree.value[node_id].argmax())
            # Ensure cond is always a Z3 expression, even for empty paths (root-only trees)
            cond = And(*current_path) if current_path else BoolVal(True)
            path_leaf.append((cond, leaf_val))
            return

        feature_id = tree.feature[node_id]
        feature_name = feature_names[feature_id]
        threshold = float(tree.threshold[node_id])

        # Left child: feature <= threshold
        left_cond = X_vars[feature_name] <= threshold
        current_path.append(left_cond)
        dfs(tree.children_left[node_id], current_path)
        current_path.pop()

        # Right child: feature > threshold
        right_cond = X_vars[feature_name] > threshold
        current_path.append(right_cond)
        dfs(tree.children_right[node_id], current_path)
        current_path.pop()

    dfs(0, [])
    return X_vars, path_leaf


def verify_global_fairness(tree_cons, features, sensitive_attr):
    """
    Pre-deployment global verification of individual fairness (Proposition 2).

    Checks if there exists ANY pair of inputs (x, x') that differ ONLY
    in the sensitive attribute but produce different predictions.

    Returns:
        dict: {
            "is_verified": bool,  # True if UNSAT (fair)
            "solving_time": float,
            "total_time": float
        }
    """
    X_vars, path_leaf = tree_cons

    Xp_vars = {f: Real(f + "_p") for f in features}
    pred_x = Real("pred_x")
    pred_xp = Real("pred_xp")

    solver = Solver()

    # Enforce tree logic for x
    for cond, leaf_val in path_leaf:
        solver.add(Implies(cond, pred_x == float(leaf_val)))

    # Enforce tree logic for x'
    subst_map = [(X_vars[f], Xp_vars[f]) for f in features]
    for cond, leaf_val in path_leaf:
        perturbed_cond = substitute(cond, *subst_map)
        solver.add(Implies(perturbed_cond, pred_xp == float(leaf_val)))

    solver.add(Or(pred_x == 0.0, pred_x == 1.0))
    solver.add(Or(pred_xp == 0.0, pred_xp == 1.0))

    # Fairness: x' differs ONLY in sensitive attribute
    for f in features:
        if f == sensitive_attr:
            solver.add(Xp_vars[f] != X_vars[f])
        else:
            solver.add(Xp_vars[f] == X_vars[f])

    solver.add(pred_x != pred_xp)

    start_time = time.time()
    result = solver.check()
    solving_time = time.time() - start_time

    return {
        "is_verified": result == unsat,
        "solving_time": solving_time,
        "total_time": solving_time,
    }


def verify_global_robustness(tree_cons, features, sensitive_attr, deltas):
    """
    Pre-deployment global verification of local robustness (Proposition 3).

    Checks if there exists ANY pair of inputs (x, x') within perturbation
    bounds that produce different predictions.

    Returns:
        dict: {
            "is_verified": bool,  # True if UNSAT (robust)
            "solving_time": float,
            "total_time": float
        }
    """
    X_vars, path_leaf = tree_cons

    Xp_vars = {f: Real(f + "_p") for f in features}
    pred_x = Real("pred_x")
    pred_xp = Real("pred_xp")

    solver = Solver()

    # Enforce tree logic for x
    for cond, leaf_val in path_leaf:
        solver.add(Implies(cond, pred_x == float(leaf_val)))

    # Enforce tree logic for x'
    subst_map = [(X_vars[f], Xp_vars[f]) for f in features]
    for cond, leaf_val in path_leaf:
        perturbed_cond = substitute(cond, *subst_map)
        solver.add(Implies(perturbed_cond, pred_xp == float(leaf_val)))

    solver.add(Or(pred_x == 0.0, pred_x == 1.0))
    solver.add(Or(pred_xp == 0.0, pred_xp == 1.0))

    # Robustness: sensitive fixed, others within delta
    for f in features:
        if f == sensitive_attr:
            solver.add(Xp_vars[f] == X_vars[f])
        elif f in deltas:
            solver.add(Xp_vars[f] >= X_vars[f] - float(deltas[f]))
            solver.add(Xp_vars[f] <= X_vars[f] + float(deltas[f]))
        else:
            solver.add(Xp_vars[f] == X_vars[f])

    solver.add(pred_x != pred_xp)

    start_time = time.time()
    result = solver.check()
    solving_time = time.time() - start_time

    return {
        "is_verified": result == unsat,
        "solving_time": solving_time,
        "total_time": solving_time,
    }


def fair_robust_predict(sample, tree_cons, features, sensitive_attr, deltas):
    """
    SMT-based prediction with separate fairness and robustness checks (Algorithm 2).

    Checks two properties independently for each input:
    - Fairness (Proposition 2): Does flipping the sensitive attribute change the prediction?
    - Robustness (Proposition 3): Do small perturbations to non-sensitive features change the prediction?

    Returns:
        tuple: (final_prediction, fairness_violation, robustness_violation)
    """
    X_vars, path_leaf = tree_cons

    # 1. Get Original Prediction
    sample_subst = [(X_vars[f], RealVal(float(sample[f]))) for f in features]

    original_pred = None
    s = Solver()
    for cond, leaf_val in path_leaf:
        s.push()
        s.add(substitute(cond, *sample_subst))
        if s.check() == sat:
            original_pred = leaf_val
            s.pop()
            break
        s.pop()

    if original_pred is None:
        return 0, False, False

    # 2. Fairness Check: Flip sensitive attribute, fix all others
    fairness_violation = False
    fairness_solver = Solver()
    Xp_fair = {f: Real(f + "_fair") for f in features}

    fairness_solver.add(
        Xp_fair[sensitive_attr] != float(sample[sensitive_attr])
    )
    for f in features:
        if f != sensitive_attr:
            fairness_solver.add(Xp_fair[f] == float(sample[f]))

    pred_fair = Real("pred_fair")
    fairness_solver.add(Or(pred_fair == 0.0, pred_fair == 1.0))

    subst_map_fair = [(X_vars[f], Xp_fair[f]) for f in features]
    for cond, leaf_val in path_leaf:
        perturbed_cond = substitute(cond, *subst_map_fair)
        fairness_solver.add(
            Implies(perturbed_cond, pred_fair == float(leaf_val))
        )

    fairness_solver.add(pred_fair != float(original_pred))

    if fairness_solver.check() == sat:
        fairness_violation = True

    # 3. Robustness Check: Fix sensitive attribute, perturb others within delta
    robustness_violation = False
    robustness_solver = Solver()
    Xp_rob = {f: Real(f + "_rob") for f in features}

    robustness_solver.add(
        Xp_rob[sensitive_attr] == float(sample[sensitive_attr])
    )
    for f in features:
        if f != sensitive_attr and f in deltas:
            robustness_solver.add(
                Xp_rob[f] >= float(sample[f]) - float(deltas[f])
            )
            robustness_solver.add(
                Xp_rob[f] <= float(sample[f]) + float(deltas[f])
            )
        elif f != sensitive_attr:
            robustness_solver.add(Xp_rob[f] == float(sample[f]))

    pred_rob = Real("pred_rob")
    robustness_solver.add(Or(pred_rob == 0.0, pred_rob == 1.0))

    subst_map_rob = [(X_vars[f], Xp_rob[f]) for f in features]
    for cond, leaf_val in path_leaf:
        perturbed_cond = substitute(cond, *subst_map_rob)
        robustness_solver.add(
            Implies(perturbed_cond, pred_rob == float(leaf_val))
        )

    robustness_solver.add(pred_rob != float(original_pred))

    if robustness_solver.check() == sat:
        robustness_violation = True

    # 4. Return original prediction with violation flags
    final_prediction = original_pred
    return final_prediction, fairness_violation, robustness_violation
