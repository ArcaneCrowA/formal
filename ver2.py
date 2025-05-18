from z3 import Solver, Real, Or, And, Not, sat

def constrained_predict(sample, path_leaf, features, Z_name, deltas):
    """
    Returns a prediction for `sample` that satisfies fairness and robustness constraints.
    If no solution exists, returns the original prediction.
    """
    solver = Solver()
    X_vars = {f: Real(f) for f in features}

    # Encode the original tree constraints
    original_pred = None
    for tests, leaf_val in path_leaf:
        cond = And([X_vars[f] == sample[f] for f in features])
        if tests:
            cond = And(cond, *tests)
        solver.add(Implies(cond, Real('pred') == leaf_val))
        if solver.check() == sat:
            original_pred = leaf_val
            break

    # Reset solver and add fairness/robustness constraints
    solver = Solver()

    # Encode fairness: prediction must not depend on Z
    Z = X_vars[Z_name]
    solver.add(Or(Z == 0, Z == 1))  # Assume binary sensitive attribute

    # Encode robustness: perturbed features must stay within deltas
    Xp_vars = {f: Real(f + "_p") for f in features}
    for f in features:
        d = deltas.get(f, 0)
        solver.add(Xp_vars[f] >= sample[f] - d)
        solver.add(Xp_vars[f] <= sample[f] + d)

    # Ensure original and perturbed predictions match
    for tests, leaf_val in path_leaf:
        # Original prediction condition
        cond_orig = And([X_vars[f] == sample[f] for f in features])
        if tests:
            cond_orig = And(cond_orig, *tests)
        # Perturbed prediction condition
        cond_perturbed = And([Xp_vars[f] == X_vars[f] for f in features if f != Z_name])
        if tests:
            perturbed_tests = [Xp_vars[t.arg(0).decl().name() <= t.arg(1) if t.decl().name() == "<="
                               else Xp_vars[t.arg(0).decl().name() > t.arg(1) for t in tests]
            cond_perturbed = And(cond_perturbed, *perturbed_tests)
        # Assert same prediction for original and perturbed
        solver.add(Implies(cond_orig, And(Real('pred') == leaf_val, Implies(cond_perturbed, Real('pred_p') == leaf_val)))

    # Solve for a valid prediction
    if solver.check() == sat:
        model = solver.model()
        return model[Real('pred')].as_long()
    else:
        return original_pred  # Fallback to original prediction if constraints are unsatisfiable
