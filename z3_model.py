from z3 import And, Implies, Or, Real, Solver, sat


def tree_constraints(tree, feature_names):
    """
    Extract decision tree constraints into Z3 format.
    Returns:
        X_vars: Dictionary of Z3 variables for features
        constraints: List of (path_conditions, leaf_value)
    """
    constraints = []
    X_vars = {name: Real(name) for name in feature_names}

    def dfs(node_id, current_path):
        if tree.feature[node_id] == -2:  # Leaf node
            leaf_val = int(tree.value[node_id].argmax())
            constraints.append((current_path.copy(), leaf_val))
            return

        feature_name = feature_names[tree.feature[node_id]]
        threshold = tree.threshold[node_id]

        # Left child (<= threshold)
        new_condition = X_vars[feature_name] <= threshold
        current_path.append(new_condition)
        dfs(tree.children_left[node_id], current_path)
        current_path.pop()

        # Right child (> threshold)
        new_condition = X_vars[feature_name] > threshold
        current_path.append(new_condition)
        dfs(tree.children_right[node_id], current_path)
        current_path.pop()

    dfs(0, [])
    return X_vars, constraints


def fair_robust_predict(sample, tree_cons, features, sensitive_attr, deltas):
    """
    Get prediction that satisfies fairness and robustness constraints.
    Returns original prediction if constraints can't be satisfied.
    """
    X_vars, path_leaf = tree_cons
    solver = Solver()
    pred = Real("pred")

    # Enforce binary classification output
    solver.add(Or(pred == 0, pred == 1))  # NEW: Force prediction to be 0/1

    # Original prediction constraints
    original_pred = None
    for tests, leaf_val in path_leaf:
        conditions = [X_vars[f] == sample[f] for f in features]
        if tests:
            conditions.extend(tests)
        solver.push()
        solver.add(And(*conditions))
        if solver.check() == sat:
            original_pred = leaf_val
            solver.pop()
            break
        solver.pop()

    # Reset solver for constraints
    solver = Solver()
    pred = Real("pred")
    solver.add(Or(pred == 0, pred == 1))  # NEW: Force prediction to be 0/1

    # Fairness constraint
    Z = X_vars[sensitive_attr]
    solver.add(Or(Z == 0, Z == 1))

    # Robustness constraints
    Xp_vars = {f: Real(f + "_p") for f in features}
    for f in features:
        if f in deltas:
            delta = deltas[f]
            solver.add(Xp_vars[f] >= sample[f] - delta)
            solver.add(Xp_vars[f] <= sample[f] + delta)
        else:
            solver.add(Xp_vars[f] == sample[f])

    # Path consistency
    for tests, leaf_val in path_leaf:
        # Original path conditions
        orig_cond = [X_vars[f] == sample[f] for f in features]
        if tests:
            orig_cond.extend(tests)
        orig_cond = And(*orig_cond)

        # Perturbed path conditions
        perturb_cond = []
        for test in tests:
            var_name = test.arg(0).decl().name()
            threshold = test.arg(1)
            if test.decl().name() == "<=":
                perturb_cond.append(Xp_vars[var_name] <= threshold)
            else:
                perturb_cond.append(Xp_vars[var_name] > threshold)
        perturb_cond = And(*perturb_cond)

        # Add implication
        solver.add(Implies(orig_cond, And(pred == leaf_val, perturb_cond)))

    # Get solution safely
    if solver.check() == sat:
        try:
            return solver.model()[pred].as_long()  # Convert to Python int
        except:
            return original_pred  # Fallback if conversion fails
    return original_pred
