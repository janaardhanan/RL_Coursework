# Cart_pole program with gym
# pip install gym torch numpy

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Define the DQN architecture
class DQN(nn.Module):
    def __init__(self, input_shape, action_space):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(np.prod(input_shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_space)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def get_epsilon(it, max_epsilon=1.0, min_epsilon=0.01, decay=500):
    return max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon) * it / decay)

def choose_action(state, policy_net, epsilon):
    if random.random() > epsilon:
        with torch.no_grad():
            state = torch.tensor([state], dtype=torch.float32)
            q_values = policy_net(state)
            action = q_values.max(1)[1].item()
    else:
        action = env.action_space.sample()
    return action


def optimize_model(memory, policy_net, target_net, optimizer, batch_size=64, gamma=0.99):
    if len(memory) < batch_size:
        return
    transitions = random.sample(memory, batch_size)
    batch = list(zip(*transitions))

    # Unpack and convert lists directly to tensors
    states = torch.tensor(batch[0], dtype=torch.float32)
    actions = torch.tensor(batch[1], dtype=torch.long)
    rewards = torch.tensor(batch[2], dtype=torch.float32)
    next_states = torch.tensor(batch[3], dtype=torch.float32)
    dones = torch.tensor(batch[4], dtype=torch.float32)

    # Ensure all actions are column vectors
    actions = actions.unsqueeze(-1)

    # Neural network operations
    state_action_values = policy_net(states).gather(1, actions).squeeze(-1)
    next_state_values = target_net(next_states).max(1)[0]
    next_state_values *= (1 - dones)  # Zero out values for done states
    expected_state_action_values = (next_state_values * gamma) + rewards

    # Loss calculation and backpropagation
    loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()




env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = deque(maxlen=10000)
epochs = 1000
sync_freq = 10
batch_size = 64

for epoch in range(epochs):
    state, _ = env.reset()
    total_reward = 0
    epsilon = get_epsilon(epoch)
    while True:
        action = choose_action(state, policy_net, epsilon)
        # Update to unpack all five return values
        next_state, reward, done, truncated, info = env.step(action)
        memory.append((list(state), action, reward, list(next_state), int(done or truncated)))  # Cast 'done' or 'truncated' to int for uniformity
        
        state = next_state
        total_reward += reward
        optimize_model(memory, policy_net, target_net, optimizer, batch_size)
        if done or truncated:  # Handle both done and truncated scenarios
            break
    if epoch % sync_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())
    torch.save(policy_net.state_dict(), f'Cart_pole/models/cartpole_{epoch}.pth')
    print(f"Epoch {epoch}, Total reward: {total_reward}, Epsilon: {epsilon}")

env.close()

# After training, visualize the policy
num_episodes = 5
env_test = gym.make("CartPole-v1", render_mode="human")
for i in range(num_episodes):
    state, _ = env_test.reset()
    done = False
    truncated = False  # Also handle truncation in visualization
    total_reward = 0
    while not done and not truncated:
        # env.render()
        action = choose_action(state, policy_net, epsilon=0)  # Use the trained policy without exploration
        next_state, reward, done, truncated, info = env_test.step(action)
        state = next_state
        total_reward += reward
    print(f"Test Episode {i + 1}: Total Reward = {total_reward}")

env_test.close()