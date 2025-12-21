import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.spaces import Box, Discrete
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from config import DATASET_NAME, MODEL_PARAMETERS, RL_TRAINING_PARAMETERS
from src.utils.dataset import load_and_preprocess_dataset
from src.verification.z3_model import fair_robust_predict, tree_constraints


class FairnessEnv(gym.Env):
    """
    Reinforcement Learning Environment for Fairness and Robustness testing.
    The agent learns to select feature columns that are most likely to
    violate fairness or robustness constraints as verified by an SMT solver.
    """

    def __init__(
        self,
        dataset_name=DATASET_NAME,
        sample_size=RL_TRAINING_PARAMETERS["sample_size"],
        lambd=RL_TRAINING_PARAMETERS["lambda"],
    ):
        super().__init__()
        # Load and preprocess the dataset
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.features,
            self.deltas_info,
            self.default_sensitive,
        ) = load_and_preprocess_dataset(dataset_name)

        self.lambd = lambd
        # Limit sample size for efficiency during RL training
        self.sample_size = min(sample_size, len(self.X_test))
        self.X_test_sub = self.X_test.iloc[: self.sample_size]
        self.y_test_sub = self.y_test.iloc[: self.sample_size]

        # Train a base Decision Tree model
        self.model = DecisionTreeClassifier(
            max_depth=MODEL_PARAMETERS["max_depth"],
            random_state=MODEL_PARAMETERS["random_state"],
        )
        self.model.fit(self.X_train, self.y_train)

        # Pre-extract Z3 constraints from the decision tree
        self.tree_cons = tree_constraints(self.model.tree_, self.features)

        # Baseline performance metrics
        self.original_preds = self.model.predict(self.X_test_sub)
        self.original_accuracy = accuracy_score(
            self.y_test_sub, self.original_preds
        )

        # Action Space: Discrete set of indices corresponding to feature columns
        self.action_space = Discrete(len(self.features))

        # State Space: [accuracy, demographic_parity_gap]
        # Observation space is normalized between 0 and 1
        self.observation_space = Box(
            low=0, high=1, shape=(2,), dtype=np.float32
        )

        self.samples = self.X_test_sub.to_dict(orient="records")
        self.state = self._get_initial_state()

    def _calculate_dpg(self, preds, sensitive_attr):
        """
        Calculates the Demographic Parity Gap (DPG).
        DPG = |P(y_hat=1 | Z=0) - P(y_hat=1 | Z=1)|
        Uses binning for continuous features to ensure meaningful group rates.
        Applies Laplace smoothing to avoid division by zero and stabilize estimates.
        """
        df_temp = self.X_test_sub.copy()
        df_temp["preds"] = preds

        # Bin continuous features if they have many unique values to avoid sparse groups
        if df_temp[sensitive_attr].nunique() > 10:
            try:
                # Use quantile-based binning into 4 groups (quartiles)
                df_temp[sensitive_attr] = pd.qcut(
                    df_temp[sensitive_attr], q=4, duplicates="drop"
                )
            except ValueError:
                # Fallback if qcut fails due to lack of variation
                pass

        # Calculate positive rate for each group in the sensitive attribute with Laplace smoothing
        alpha = 1.0  # Laplace smoothing parameter
        group_counts = df_temp.groupby(sensitive_attr, observed=True)[
            "preds"
        ].count()
        group_positives = df_temp.groupby(sensitive_attr, observed=True)[
            "preds"
        ].sum()
        group_rates = (group_positives + alpha) / (group_counts + 2 * alpha)

        if len(group_rates) < 2:
            return 0.0

        # DPG is the absolute difference between the max and min positive rates across groups
        return float(np.abs(group_rates.max() - group_rates.min()))

    def _get_initial_state(self):
        """Computes the initial state using the dataset's default sensitive attribute."""
        initial_dpg = self._calculate_dpg(
            self.original_preds, self.default_sensitive
        )
        return np.array([self.original_accuracy, initial_dpg], dtype=np.float32)

    def step(self, action):
        sensitive_feature_name = self.features[action]

        # 1. Invoke SMT Fairness/Robustness Evaluator (Z3 logic)
        # We check for violations in the neighborhood of each sample.
        constrained_preds = []
        violations = []
        for sample in self.samples:
            pred, has_violation = fair_robust_predict(
                sample,
                self.tree_cons,
                self.features,
                sensitive_feature_name,
                self.deltas_info,
            )
            if has_violation:
                # Flip the prediction if a violation is found
                constrained_preds.append(1 - pred)
            else:
                constrained_preds.append(pred)
            violations.append(has_violation)

        constrained_preds = np.array(constrained_preds)

        # 2. Calculate Bias Score
        # Fraction of samples where a violation was found
        bias_score = np.mean(violations)

        # 3. Calculate Accuracy Change
        constrained_accuracy = accuracy_score(
            self.y_test_sub, constrained_preds
        )
        accuracy_change = constrained_accuracy - self.original_accuracy

        # 4. Calculate current Fairness Metric (DPG) for the state
        current_dpg = self._calculate_dpg(
            constrained_preds, sensitive_feature_name
        )

        # 5. Calculate Reward: r = bias_score(f_i) + lambda * accuracy_change
        # Rewarding the detection of bias to help the agent identify unfair features.
        reward = float(bias_score) + self.lambd * float(accuracy_change)

        # 6. Update State
        self.state = np.array(
            [constrained_accuracy, current_dpg], dtype=np.float32
        )

        # Each step is an independent evaluation of a feature (one-step episode)
        done = True
        return self.state, reward, done, False, {}

    def reset(self, seed=None, options=None):
        """Resets the environment to a random baseline state subset."""
        super().reset(seed=seed)

        # Sample a random subset of the test data for each episode
        indices = np.random.choice(
            len(self.X_test), self.sample_size, replace=False
        )
        self.X_test_sub = self.X_test.iloc[indices]
        self.y_test_sub = self.y_test.iloc[indices]
        self.samples = self.X_test_sub.to_dict(orient="records")

        # Update baseline performance metrics for this specific subset
        self.original_preds = self.model.predict(self.X_test_sub)
        self.original_accuracy = accuracy_score(
            self.y_test_sub, self.original_preds
        )

        self.state = self._get_initial_state()
        return self.state, {}
