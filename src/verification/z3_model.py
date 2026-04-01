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


def verify_global_properties(tree_cons, features, sensitive_attr, deltas):
    """
    Pre-deployment global verification of fairness and robustness.

    Checks if there exists ANY pair of inputs (x, x') within perturbation bounds
    that produce different predictions. This is a one-time cost before deployment.

    Returns:
        dict: {
            "is_verified": bool,  # True if UNSAT (no violations exist globally)
            "solving_time": float,
            "total_time": float
        }
    """
    X_vars, path_leaf = tree_cons

    # Create perturbed variables x'
    Xp_vars = {f: Real(f + "_p") for f in features}

    # Define predictions for x and x'
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

    # Ensure predictions are binary
    solver.add(Or(pred_x == 0.0, pred_x == 1.0))
    solver.add(Or(pred_xp == 0.0, pred_xp == 1.0))

    # Apply perturbation bounds
    for f in features:
        if f == sensitive_attr:
            # Fairness: sensitive attribute must remain the same
            solver.add(Xp_vars[f] == X_vars[f])
        elif f in deltas:
            # Robustness: non-sensitive features within delta
            solver.add(Xp_vars[f] >= X_vars[f] - float(deltas[f]))
            solver.add(Xp_vars[f] <= X_vars[f] + float(deltas[f]))
        else:
            # Other features held constant
            solver.add(Xp_vars[f] == X_vars[f])

    # Check for violation: exists x, x' such that pred_x != pred_xp
    solver.add(pred_x != pred_xp)

    start_time = time.time()
    result = solver.check()
    solving_time = time.time() - start_time

    is_verified = result == unsat

    return {
        "is_verified": is_verified,
        "solving_time": solving_time,
        "total_time": solving_time,
    }


def fair_robust_predict(sample, tree_cons, features, sensitive_attr, deltas):
    """
    SMT-based prediction with fairness and robustness coercion (Algorithm 2).

    Implements coercion logic to find fair predictions within perturbation bounds.
    If no fair prediction is possible, falls back to the original prediction.

    Returns:
        tuple: (final_prediction, has_violation)
               - final_prediction: The coerced prediction if possible, otherwise original
               - has_violation: Boolean, True if coercion was applied
    """
    X_vars, path_leaf = tree_cons

    # 1. Get Original Prediction (fallback)
    # Convert Python values to Z3 RealVal for substitution
    sample_subst = [(X_vars[f], RealVal(float(sample[f]))) for f in features]

    original_pred = None
    s = Solver()
    for cond, leaf_val in path_leaf:
        # Check if the concrete sample satisfies this path's condition
        s.push()
        s.add(substitute(cond, *sample_subst))
        if s.check() == sat:
            original_pred = leaf_val
            s.pop()
            break
        s.pop()

    if original_pred is None:
        # Fallback if tree structure is inconsistent
        return 0, False

    # 2. Reset & Configure Solver for Coercion Logic
    coercion_solver = Solver()
    Xp_vars = {f: Real(f + "_p") for f in features}

    # 3. Apply Constraints (The "Coercion" Logic)
    # Fix Sensitive Attributes: Force x_prime's sensitive attribute to match x
    coercion_solver.add(
        Xp_vars[sensitive_attr] == float(sample[sensitive_attr])
    )

    # Set Perturbation Bounds for other features
    for f in features:
        if f != sensitive_attr and f in deltas:
            # Constrain x_prime to be close to x (within delta)
            coercion_solver.add(
                Xp_vars[f] >= float(sample[f]) - float(deltas[f])
            )
            coercion_solver.add(
                Xp_vars[f] <= float(sample[f]) + float(deltas[f])
            )
        elif f != sensitive_attr:
            # Features not being tested are held constant
            coercion_solver.add(Xp_vars[f] == float(sample[f]))

    # Define the model's output for the perturbed sample X'
    pred_p = Real("pred_p")
    coercion_solver.add(Or(pred_p == 0.0, pred_p == 1.0))

    # Enforce Tree Logic: Map original variables to perturbed variables
    subst_map = [(X_vars[f], Xp_vars[f]) for f in features]
    for cond, leaf_val in path_leaf:
        # For each path, if X' satisfies the condition, then the prediction is leaf_val
        perturbed_cond = substitute(cond, *subst_map)
        coercion_solver.add(Implies(perturbed_cond, pred_p == float(leaf_val)))

    # 4. Solve for a DIFFERENT Valid Prediction (S.check())
    # Check if a valid, fair prediction exists within bounds that is DIFFERENT from original
    # Exclude the original prediction to only count meaningful coercion
    coercion_solver.add(pred_p != float(original_pred))

    if coercion_solver.check() == sat:
        # COERCION HAPPENS HERE
        # The solver found a DIFFERENT valid, fair path. Use it.
        model = coercion_solver.model()
        final_prediction = int(model[pred_p].as_long())
        has_violation = True  # Meaningful coercion was applied
    else:
        # FALLBACK
        # No DIFFERENT fair prediction possible within bounds.
        # Revert to original (potentially biased) prediction.
        final_prediction = original_pred
        has_violation = False  # No meaningful coercion applied

    return final_prediction, has_violation
