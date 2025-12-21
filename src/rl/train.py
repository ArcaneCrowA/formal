import os
from datetime import datetime

import torch
import torch.optim as optim
from torch.distributions import Categorical

from config import DATASET_NAME, RL_TRAINING_PARAMETERS
from src.rl.policy_network import PolicyNetwork
from src.rl.rl_environment import FairnessEnv


def train():
    dataset_name = DATASET_NAME
    env = FairnessEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy_net = PolicyNetwork(state_size, action_size)
    optimizer = optim.Adam(
        policy_net.parameters(), lr=RL_TRAINING_PARAMETERS["learning_rate"]
    )

    # Define alpha for the utility score calculation
    alpha = 0.5  # Balance between accuracy and fairness

    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define alpha for the utility score calculation
    alpha = 0.5  # Balance between accuracy and fairness

    log_file = f"logs/training_{dataset_name}_{timestamp}.log"
    with open(log_file, "w") as f:
        f.write(f"Starting training on '{dataset_name}' dataset...\n")
        f.write(f"Number of features (actions): {action_size}\n")
        f.write("-" * 30 + "\n")

    num_episodes = RL_TRAINING_PARAMETERS["num_episodes"]

    # Track cumulative DPG, accuracy, and selection counts for each feature
    feature_dpg = {feature: 0.0 for feature in env.features}
    feature_accuracy = {feature: 0.0 for feature in env.features}
    feature_counts = {feature: 0 for feature in env.features}
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0)

        action_probs = policy_net(state)
        m = Categorical(action_probs)
        action = m.sample()

        next_state, reward, _, _, _ = env.step(action.item())

        # Add entropy for better exploration
        entropy = m.entropy()
        loss = -m.log_prob(action) * reward - 0.03 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track DPG, accuracy, and selection counts for the selected feature
        selected_feature = env.features[action.item()]
        feature_dpg[selected_feature] += next_state[1].item()
        feature_accuracy[selected_feature] += next_state[0].item()
        feature_counts[selected_feature] += 1

        if episode % 100 == 0:
            log_message = (
                f"Episode {episode:4d} | "
                f"Feature: {env.features[action.item()]:20s} | "
                f"Reward: {reward:8.4f} | "
                f"Acc: {next_state[0]:.4f} | "
                f"DPG: {next_state[1]:.4f} | "
                f"Loss: {loss.item():8.4f}\n"
            )
            with open(log_file, "a") as f:
                f.write(log_message)

    # Calculate average DPG and accuracy for each feature
    avg_dpg = {
        feature: feature_dpg[feature] / feature_counts[feature]
        if feature_counts[feature] > 0
        else 0.0
        for feature in env.features
    }
    avg_accuracy = {
        feature: feature_accuracy[feature] / feature_counts[feature]
        if feature_counts[feature] > 0
        else 0.0
        for feature in env.features
    }

    # Sort features by average DPG in descending order
    sorted_features = sorted(avg_dpg.items(), key=lambda x: x[1], reverse=True)

    # Print all non-zero features in order of DPG with additional metrics
    with open(log_file, "a") as f:
        f.write("\nAll Non-Zero Features (Sorted by DPG):\n")
        f.write("Feature                  | Avg DPG  | Acc + Fairness\n")
        f.write("-" * 60 + "\n")
        for i, (feature, dpg) in enumerate(sorted_features, 1):
            if dpg > 0.0:
                # Calculate utility score: U = α·Accuracy + (1−α)·(1−DPG)
                accuracy = avg_accuracy[feature]
                utility_score = alpha * accuracy + (1 - alpha) * (1 - dpg)
                f.write(
                    f"{i}. {feature:20s} | Avg DPG: {dpg:.4f} | Utility Score: {utility_score:.4f}\n"
                )


if __name__ == "__main__":
    train()
