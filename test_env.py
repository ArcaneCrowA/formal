from rl_environment import FairnessEnv


def test_environment():
    """
    Test script to verify the integration between the RL Environment,
    the Decision Tree model, and the Z3-based fairness/robustness checker.
    """
    # Choose a dataset to test
    dataset_name = "german"
    print(f"--- Testing FairnessEnv with dataset: {dataset_name} ---")

    # Initialize environment
    # We use a small sample size (20) to ensure the SMT solver runs quickly during this test
    env = FairnessEnv(dataset_name=dataset_name, sample_size=20, lambd=0.1)

    # Reset environment to get initial state
    state, info = env.reset()
    print(f"Initial State (Original Accuracy, Initial DPG): {state}")
    print(f"Number of features (Actions): {len(env.features)}")
    print(f"Features: {env.features}")

    # Test a few random actions (features to evaluate for bias)
    num_steps = 3
    for i in range(num_steps):
        # Randomly select a feature index
        action = env.action_space.sample()
        feature_name = env.features[action]

        print(
            f"\nStep {i + 1}: Selecting feature index {action} ('{feature_name}')"
        )

        # Execute the action in the environment
        # This triggers fair_robust_predict (SMT solver) for each sample
        next_state, reward, done, truncated, info = env.step(action)

        print(f"  -> Reward received: {reward:.4f}")
        print(f"  -> New State [Accuracy, DPG]: {next_state}")
        print(f"  -> Episode Done: {done}")

        # In this environment, episodes are single-step. Reset for next test.
        if done:
            env.reset()

    print("\n--- Environment Test Successful ---")


if __name__ == "__main__":
    try:
        test_environment()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
