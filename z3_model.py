from z3 import And, BoolVal, Implies, Or, Real, RealVal, Solver, sat, substitute


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


def fair_robust_predict(sample, tree_cons, features, sensitive_attr, deltas):
    """
    SMT-based violation checker that integrates fairness and robustness.

    Searches for a counter-example (violation) in the neighborhood of the sample.
    A violation exists if there's a point X' in the neighborhood such that
    Tree(X') != Tree(sample).

    Returns:
        tuple: (original_prediction, has_violation)
               - original_prediction: The class predicted by the tree for the sample.
               - has_violation: Boolean, True if a counter-example was found by Z3.
    """
    X_vars, path_leaf = tree_cons

    # 1. Identify the original prediction for the sample
    # Convert Python values to Z3 RealVal to satisfy is_expr(p[1]) in substitute()
    sample_subst = [(X_vars[f], RealVal(float(sample[f]))) for f in features]

    original_pred = None
    s = Solver()
    for cond, leaf_val in path_leaf:
        # Check if the concrete sample satisfies this path's condition
        s.push()
        # substitute(e, (old1, new1), (old2, new2), ...)
        s.add(substitute(cond, *sample_subst))
        if s.check() == sat:
            original_pred = leaf_val
            s.pop()
            break
        s.pop()

    if original_pred is None:
        # Fallback if for some reason the tree structure is inconsistent
        return 0, False

    # 2. Search for a violation (Counter-example)
    v_solver = Solver()
    Xp_vars = {f: Real(f + "_p") for f in features}

    # Neighborhood constraints for X'
    for f in features:
        if f == sensitive_attr:
            # Test for individual fairness (e.g., flipping sensitive bits 0 <-> 1)
            v_solver.add(Or(Xp_vars[f] == 0.0, Xp_vars[f] == 1.0))
        elif f in deltas:
            # Test for local robustness (perturbations within delta)
            v_solver.add(Xp_vars[f] >= float(sample[f]) - float(deltas[f]))
            v_solver.add(Xp_vars[f] <= float(sample[f]) + float(deltas[f]))
        else:
            # Features not being tested are held constant
            v_solver.add(Xp_vars[f] == float(sample[f]))

    # Define the model's output for the perturbed sample X'
    pred_p = Real("pred_p")
    v_solver.add(Or(pred_p == 0.0, pred_p == 1.0))

    # Map original variables in path conditions to the perturbed variables
    subst_map = [(X_vars[f], Xp_vars[f]) for f in features]
    for cond, leaf_val in path_leaf:
        # For each path, if X' satisfies the condition, then the prediction is leaf_val.
        perturbed_cond = substitute(cond, *subst_map)
        v_solver.add(Implies(perturbed_cond, pred_p == float(leaf_val)))

    # The violation condition: there exists X' such that its prediction != original prediction.
    v_solver.add(pred_p != float(original_pred))

    # If the solver finds such an X', then the sample violates the constraints.
    has_violation = v_solver.check() == sat

    return original_pred, has_violation
