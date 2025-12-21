import torch
import torch.optim as optim
from torch.distributions import Categorical

from src.rl.policy_network import PolicyNetwork
from src.rl.rl_environment import FairnessEnv


def train():
    dataset_name = "adult"
    # dataset_name = "german"
    # dataset_name = "loan_data"
    env = FairnessEnv(dataset_name=dataset_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy_net = PolicyNetwork(state_size, action_size)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

    print(f"Starting training on '{dataset_name}' dataset...")
    print(f"Number of features (actions): {action_size}")
    print("-" * 30)

    num_episodes = 1000
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
