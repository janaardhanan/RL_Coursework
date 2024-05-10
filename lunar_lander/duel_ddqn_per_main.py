from duel_ddqn_per_fns import *

csv_filename= "duelling_double_PER_lunar.csv"


# Prioritized Experience Replay Sum Tree
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity

    def get(self, s):
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def total(self):
        return self.tree[0]

# Prioritized Replay Buffer using the SumTree
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.alpha = alpha
        self.tree = SumTree(capacity)

    def add(self, error, sample):
        priority = (abs(error) + 1e-5) ** self.alpha
        self.tree.add(priority, sample)

    def sample(self, batch_size, beta=0.4):
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            s = random.uniform(i * segment, (i + 1) * segment)
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.capacity * sampling_probabilities, -beta)
        is_weights /= is_weights.max()

        return batch, indices, torch.tensor(is_weights, dtype=torch.float32, device=device)

    def update(self, idx, error):
        priority = (abs(error) + 1e-5) ** self.alpha
        self.tree.update(idx, priority)

    def __len__(self):
        return len(self.tree.data) - self.tree.data.count(None)


# Optimizer function modified for Double DQN with PER
def optimize_model(memory, policy_net, target_net, optimizer, batch_size=64, gamma=0.99, beta=0.4):
    if len(memory) < batch_size:
        return 0

    # Sample with prioritization
    transitions, indices, is_weights = memory.sample(batch_size, beta)
    batch = list(zip(*transitions))

    states = torch.tensor(batch[0], dtype=torch.float32, device=device)
    actions = torch.tensor(batch[1], dtype=torch.long, device=device)
    rewards = torch.tensor(batch[2], dtype=torch.float32, device=device)
    next_states = torch.tensor(batch[3], dtype=torch.float32, device=device)
    dones = torch.tensor(batch[4], dtype=torch.float32, device=device)

    actions = actions.unsqueeze(-1)

    state_action_values = policy_net(states).gather(1, actions).squeeze(-1)

    next_state_actions = policy_net(next_states).max(1)[1].unsqueeze(-1)
    next_state_values = target_net(next_states).gather(1, next_state_actions).squeeze(-1)
    next_state_values *= (1 - dones)

    expected_state_action_values = (next_state_values * gamma) + rewards

    errors = expected_state_action_values - state_action_values
    loss = (is_weights * nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values, reduction='none')).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for idx, error in zip(indices, errors.detach().cpu().numpy()):
        memory.update(idx, error)

    return loss.item()

optimizer = optim.Adam(policy_net.parameters())
memory = PrioritizedReplayBuffer(10000)
epochs = 1501
sync_freq = 10
batch_size = 64
gamma = 0.99

# Training loop
for epoch in range(epochs):
    state, _ = env.reset()
    total_reward = 0
    total_loss = 0  # Initialize total loss for the episode
    count_steps = 0  # Count the number of optimization steps

    epsilon = get_epsilon(epoch)
    beta = min(1.0, 0.4 + epoch * (1.0 - 0.4) / epochs)

    while True:
        action,flag = choose_action(state, policy_net, epsilon)
        next_state, reward, done, truncated, info = env.step(action)
        td_error = reward - gamma * (not (done or truncated))
        memory.add(td_error, (list(state), action, reward, list(next_state), int(done or truncated)))

        state = next_state
        total_reward += reward
        loss = optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma, beta)

        total_loss += loss
        count_steps += 1  # Increment the step count whenever model is optimized
        
        step_array=[epoch,count_steps,epsilon,reward,loss,action,flag]
        save_to_csv(step_array,"step_"+csv_filename)
        if done or truncated:
            break

    if epoch % sync_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())
    if epoch %100 ==0:
        torch.save(policy_net.state_dict(), f'lunar_lander/models/duelling_double_PER_lunar_{epoch}.pth')

    torch.save(policy_net.state_dict(), f'lunar_lander/models/duelling_double_PER_lunar_final.pth')

    average_loss = total_loss / count_steps if count_steps != 0 else 0
    print(f"Epoch {epoch}, Total reward: {total_reward}, Epsilon: {epsilon}, average loss: {average_loss}")
    save_to_csv([epoch, total_reward, epsilon, average_loss], csv_filename)

env.close()

# Evaluation with the trained Double DQN + PER model
num_episodes = 5
env_test = gym.make(environment, render_mode="human")
for i in range(num_episodes):
    state, _ = env_test.reset()
    done = False
    truncated = False
    total_reward = 0
    while not done and not truncated:
        action,flag = choose_action(state, policy_net, epsilon=0)
        next_state, reward, done, truncated, info = env_test.step(action)
        state = next_state
        total_reward += reward
    print(f"Test Episode {i + 1}: Total Reward = {total_reward}")

env_test.close()
