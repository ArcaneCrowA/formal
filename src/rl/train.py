import torch
import torch.optim as optim
from torch.distributions import Categorical

from config import RL_TRAINING_PARAMETERS
from src.rl.policy_network import PolicyNetwork
from src.rl.rl_environment import FairnessEnv


def train():
    # You can switch the dataset here, e.g., "adult", "german", "loan_data"
    dataset_name = "german"
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

        _, reward, _, _, _ = env.step(action.item())

        loss = -m.log_prob(action) * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(
                f"Episode {episode}, "
                f"Action: {env.features[action.item()]}, "
                f"Reward: {reward:.4f}, "
                f"Loss: {loss.item():.4f}"
            )


if __name__ == "__main__":
    train()
