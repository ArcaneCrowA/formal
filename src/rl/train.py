import torch
import torch.optim as optim
from torch.distributions import Categorical

from config import DATASET_NAME, RL_TRAINING_PARAMETERS
from src.rl.policy_network import PolicyNetwork
from src.rl.rl_environment import FairnessEnv


def train():
    # You can switch the dataset here, e.g., "adult", "german", "loan_data"
    dataset_name = DATASET_NAME
    env = FairnessEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy_net = PolicyNetwork(state_size, action_size)
    optimizer = optim.Adam(
        policy_net.parameters(), lr=RL_TRAINING_PARAMETERS["learning_rate"]
    )

    print(f"Starting training on '{dataset_name}' dataset...")
    print(f"Number of features (actions): {action_size}")
    print("-" * 30)

    num_episodes = RL_TRAINING_PARAMETERS["num_episodes"]
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0)

        action_probs = policy_net(state)
        m = Categorical(action_probs)
        action = m.sample()

        next_state, reward, _, _, _ = env.step(action.item())

        # Add entropy for better exploration
        entropy = m.entropy()
        loss = -m.log_prob(action) * reward - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(
                f"Episode {episode:4d} | "
                f"Feature: {env.features[action.item()]:20s} | "
                f"Reward: {reward:8.4f} | "
                f"Acc: {next_state[0]:.4f} | "
                f"DPG: {next_state[1]:.4f} | "
                f"Loss: {loss.item():8.4f}"
            )


if __name__ == "__main__":
    train()
