def train(env_id, epochs, batch_size, gamma, sync_freq, epsilon_decay, epsilon_min):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = envpool.make(env_id, device=device, num_envs=8)
    policy_net = PolicyNet().to(device)
    target_net = PolicyNet().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters())
    memory = PrioritizedReplayBuffer(10000)
    writer = SummaryWriter()

    for epoch in range(epochs):
        state, _ = env.reset()
        total_reward = 0
        epsilon = get_epsilon(epoch)
        beta = min(1.0, 0.4 + epoch * (1.0 - 0.4) / epochs)

        while True:
            action = choose_action(state, policy_net, epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            td_error = reward - gamma * (not (done or truncated))
            memory.add(td_error, (list(state), action, reward, list(next_state), int(done or truncated)))

            state = next_state
            total_reward += reward
            optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma, beta)
            if done or truncated:
                break

        if epoch % sync_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        writer.add_scalar('Total Reward', total_reward, epoch)
        writer.add_scalar('Epsilon', epsilon, epoch)

    writer.close()
    torch.save(policy_net.state_dict(), f'{env_id}/models/{env_id}_{epoch}.pth')
    print(f"Epoch {epoch}, Total reward: {total_reward}, Epsilon: {epsilon}")

def evaluate(env_id, policy_net, num_episodes=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_id, render_mode="human")
    policy_net.to(device)
    policy_net.eval()

    for i in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        while not done and not truncated:
            action = choose_action(state, policy_net, epsilon=0)
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            total_reward += reward
        print(f"Test Episode {i + 1}: Total Reward = {total_reward}")

    env.close()

if __name__ == "__main__":
    train("LunarLander-v2", epochs=1000, batch_size=64, gamma=0.99, sync_freq=10, epsilon_decay=0.995, epsilon_min=0.01)
    evaluate("LunarLander-v2", policy_net)
