from pre_req import *

num_episodes = 5
model_path= f'lunar_lander/models/PER_lunar_final.pth'

env = gym.make(environment, render_mode="human")

policy_net.load_state_dict(torch.load(model_path))
policy_net.eval()



for i in range(num_episodes):
    state, _ = env.reset()
    done = False
    truncated = False  # Also handle truncation in visualization
    total_reward = 0
    while not done and not truncated:
        # env.render()
        action = choose_action(state, policy_net, epsilon=0)  # Use the trained policy without exploration
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        total_reward += reward
    print(f"Test Episode {i + 1}: Total Reward = {total_reward}")

env.close()