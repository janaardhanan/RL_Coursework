from pre_req import *


# Optimizer function modified for Double DQN
def optimize_model(memory, policy_net, target_net, optimizer, batch_size=64, gamma=0.99):
    if len(memory) < batch_size:
        return
    transitions = random.sample(memory, batch_size)
    batch = list(zip(*transitions))

    # Unpack and convert lists directly to tensors
    states = torch.tensor(batch[0], dtype=torch.float32, device=device)
    actions = torch.tensor(batch[1], dtype=torch.long, device=device)
    rewards = torch.tensor(batch[2], dtype=torch.float32, device=device)
    next_states = torch.tensor(batch[3], dtype=torch.float32, device=device)
    dones = torch.tensor(batch[4], dtype=torch.float32, device=device)

    # Ensure all actions are column vectors
    actions = actions.unsqueeze(-1)

    # Get the Q-values for the chosen actions in the current states
    state_action_values = policy_net(states).gather(1, actions).squeeze(-1)

    # Double DQN logic
    next_state_actions = policy_net(next_states).max(1)[1].unsqueeze(-1)
    next_state_values = target_net(next_states).gather(1, next_state_actions).squeeze(-1)
    next_state_values *= (1 - dones)  # Zero out values for done states
    expected_state_action_values = (next_state_values * gamma) + rewards

    # Loss calculation and backpropagation
    loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


optimizer = optim.Adam(policy_net.parameters())
memory = deque(maxlen=10000)
epochs = 1000
sync_freq = 10
batch_size = 64

# Training loop
for epoch in range(epochs):
    state, _ = env.reset()
    total_reward = 0
    epsilon = get_epsilon(epoch)
    while True:
        action = choose_action(state, policy_net, epsilon)
        # Update to unpack all five return values
        next_state, reward, done, truncated, info = env.step(action)
        memory.append((list(state), action, reward, list(next_state), int(done or truncated)))  # Cast 'done' or 'truncated' to int
        
        state = next_state
        total_reward += reward
        optimize_model(memory, policy_net, target_net, optimizer, batch_size)
        if done or truncated:
            break
    if epoch % sync_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())
    torch.save(policy_net.state_dict(), f'Cartpole/models/cartpole_ddqn_{epoch}.pth')
    print(f"Epoch {epoch}, Total reward: {total_reward}, Epsilon: {epsilon}")

env.close()

# Evaluation with the trained Double DQN model
num_episodes = 5
env_test = gym.make("CartPole-v1", render_mode="human")
for i in range(num_episodes):
    state, _ = env_test.reset()
    done = False
    truncated = False
    total_reward = 0
    while not done and not truncated:
        action = choose_action(state, policy_net, epsilon=0)
        next_state, reward, done, truncated, info = env_test.step(action)
        state = next_state
        total_reward += reward
    print(f"Test Episode {i + 1}: Total Reward = {total_reward}")

env_test.close()
