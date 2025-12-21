# Configuration file for the formal verification framework
# This file contains all constants, parameters, and dataset selection.

# Dataset selection
DATASET_NAME = "german"  # Options: "adult", "german", "loan_data"

# RL Training Parameters
RL_TRAINING_PARAMETERS = {
    "num_episodes": 1000,
    "learning_rate": 0.01,
    "lambda": 0.1,  # Balances fairness and accuracy in the reward function
    "sample_size": 50,  # Number of samples to use for RL training
}

# Deltas for Robustness Checks
# These values define the allowed perturbations for each feature during robustness checks.
DELTAS = {
    "adult": {
        "age": 5,
        "capital.gain": 1000,
        "capital.loss": 1000,
        "hours.per.week": 5,
    },
    "german": {
        "Age_years": 5,
        "Credit_Amount": 1000,
    },
    "loan_data": {
        "age": 5,
        "income": 1000,
        "loan_amount": 1000,
    },
}

# Sensitive Attributes for Fairness Checks
# These attributes are used to test for fairness violations.
SENSITIVE_ATTRIBUTES = {
    "adult": "sex",
    "german": "Sex_Marital_Status",
    "loan_data": "gender",
}

# Model Training Parameters
MODEL_PARAMETERS = {
    "max_depth": 3,
    "random_state": 42,
}

# Verification Parameters
VERIFICATION_PARAMETERS = {
    "use_z3": True,  # Whether to use Z3 for fairness/robustness checks
    "violation_threshold": 0.01,  # Threshold for considering a violation significant
}
