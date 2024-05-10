import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
import time
import csv
from torchviz import make_dot

import matplotlib.pyplot as plt

np.random.seed(100)
torch.manual_seed(100)
if torch.backends.mps.is_available():
    torch.cuda.manual_seed_all(100)
    torch.use_deterministic_algorithms(True)
# Check if a GPU is available and assign a device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def save_to_csv(values, filename="training.csv"):
    with open(filename, 'a') as f_object:
 
        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = csv.writer(f_object)
 
        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow(values)
 
        # Close the file object
        f_object.close()

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

"""class DQN(nn.Module):
    def __init__(self, input_shape, action_space):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(np.prod(input_shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_space)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)"""
    

# Function to get epsilon value based on decay
def get_epsilon(it, max_epsilon=1.0, min_epsilon=0.01, decay=500):
    return max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon) * it / decay)

# Function to choose an action based on the epsilon-greedy policy

def choose_action(state, policy_net, epsilon):
    flag=0
    if random.random() > epsilon:
        with torch.no_grad():
            state = torch.tensor([state], dtype=torch.float32, device=device)
            q_values = policy_net(state)
            action = q_values.max(1)[1].item()
            flag=1
    else:
        action = env.action_space.sample()

    return action,flag




# Initialize environment and parameters
environment= "LunarLander-v2"

env = gym.make(environment)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Create dummy inputs
dummy_input = torch.randn(1, state_dim, device=device)

# Forward pass the dummy input through the policy network to produce an output
policy_output = policy_net(dummy_input)

# Forward pass the dummy input through the target network to produce an output
target_output = target_net(dummy_input)

# Combine the outputs of both networks into one graph
combined_output = torch.cat((policy_output, target_output), dim=0)

# Compute the computational graph
dot = make_dot(combined_output, params=dict(policy_net.named_parameters()))

# Render and save the combined graph as a PNG file
dot.render('ddqn_per_dqn_combined_graph', format='png', quiet=True)
