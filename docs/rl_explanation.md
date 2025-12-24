# Reinforcement Learning in the Formal Verification Framework

## Overview

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives feedback in the form of rewards or penalties based on its actions, and its goal is to maximize the cumulative reward over time. In this project, RL is used to identify features in a dataset that are most likely to violate fairness or robustness constraints, as verified by an SMT solver (Z3).

---

## Key Components

### 1. Agent
The agent is the decision-maker in the RL system. In this project, the agent is implemented as a `PolicyNetwork`, which is a neural network that takes the current state of the environment as input and outputs a probability distribution over possible actions (features to evaluate).

### 2. Environment
The environment is the system with which the agent interacts. In this project, the environment is defined by the `FairnessEnv` class, which simulates the evaluation of a decision tree model for fairness and robustness violations. The environment provides the agent with observations (states) and rewards based on the agent's actions.

### 3. State
The state represents the current condition of the environment. In `FairnessEnv`, the state is a vector containing two values:
- **Accuracy**: The accuracy of the model on the test dataset.
- **Demographic Parity Gap (DPG)**: A measure of fairness, representing the difference in positive prediction rates between different groups defined by a sensitive attribute.

### 4. Action
An action is a decision made by the agent. In this project, actions correspond to selecting a feature from the dataset to evaluate for fairness or robustness violations. The action space is discrete and corresponds to the indices of the features in the dataset.

### 5. Reward
The reward is the feedback provided by the environment to the agent after it takes an action. In `FairnessEnv`, the reward is calculated as:
\[
\text{reward} = \text{bias\_score} + \lambda \cdot \text{accuracy\_change}
\]
- **Bias Score**: The fraction of samples where a fairness or robustness violation was detected.
- **Accuracy Change**: The change in accuracy when the model's predictions are adjusted for violations.
- **\(\lambda\)**: A hyperparameter that balances the trade-off between detecting bias and maintaining accuracy.

### 6. Policy
The policy is the strategy used by the agent to select actions. In this project, the policy is implemented as a neural network (`PolicyNetwork`) that maps states to action probabilities. The policy is trained using the REINFORCE algorithm, a type of policy gradient method.

---

## How It Works

### Training Process

1. **Initialization**:
   - The `FairnessEnv` is initialized with a dataset and a pre-trained decision tree model.
   - The `PolicyNetwork` is initialized with random weights.

2. **Episode Loop**:
   - For each episode, the environment is reset to a random subset of the test data.
   - The agent observes the initial state (accuracy and DPG) of the environment.
   - The agent selects an action (feature to evaluate) based on the current policy.
   - The environment executes the action, which involves:
     - Using the Z3 solver to check for fairness and robustness violations for each sample in the dataset.
     - Adjusting the model's predictions for samples where violations are detected.
     - Calculating the reward based on the bias score and accuracy change.
   - The agent updates its policy based on the reward received.

3. **Policy Update**:
   - The policy is updated using gradient ascent on the expected cumulative reward. The REINFORCE algorithm is used to compute the gradient of the expected reward with respect to the policy parameters.
   - The loss function for the policy update includes an entropy term to encourage exploration:
     \[
     \text{loss} = -\log(\pi(a|s)) \cdot R - \beta \cdot \text{entropy}(\pi(\cdot|s))
     \]
     where \(\pi(a|s)\) is the probability of taking action \(a\) in state \(s\), \(R\) is the reward, and \(\beta\) is a hyperparameter that controls the strength of the entropy regularization.

4. **Termination**:
   - The training process continues for a fixed number of episodes (`num_episodes`). After training, the policy is evaluated to identify the features that are most likely to violate fairness or robustness constraints.

---

## Integration with Z3 Solver

The RL framework is integrated with the Z3 solver to verify fairness and robustness constraints. The `fair_robust_predict` function in `z3_model.py` is used to check for violations in the neighborhood of each sample in the dataset. The function returns the original prediction and a flag indicating whether a violation was detected. This information is used to calculate the bias score and adjust the model's predictions.

---

## Hyperparameters

The following hyperparameters are used to control the training process:

- **`num_episodes`**: The number of training episodes.
- **`learning_rate`**: The learning rate for the policy network optimizer.
- **`lambda`**: The weight for the accuracy change in the reward calculation.
- **`sample_size`**: The number of samples used for RL training.
- **`alpha`**: The balance between accuracy and fairness in the utility score calculation.

These hyperparameters are defined in the `config.py` file and can be adjusted to optimize the training process.

---

## Summary

The RL framework in this project is designed to identify features that are most likely to violate fairness or robustness constraints in a decision tree model. By integrating RL with formal verification techniques (Z3 solver), the framework provides a powerful tool for evaluating and improving the fairness and robustness of machine learning models. The agent learns to select features that maximize the detection of violations while maintaining model accuracy, making it a valuable tool for bias detection and mitigation.