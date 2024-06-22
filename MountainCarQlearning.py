import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def execute_simulation(num_episodes, train_mode=True, show_render=False, apply_reward_shaping=False):

    # Initialize the MountainCar environment
    environment = gym.make('MountainCar-v0', render_mode='human' if show_render else None)

    # Create discrete bins for position and velocity
    position_bins = np.linspace(environment.observation_space.low[0], environment.observation_space.high[0], 40)  # From -1.2 to 0.6
    velocity_bins = np.linspace(environment.observation_space.low[1], environment.observation_space.high[1], 40)  # From -0.07 to 0.07

    # Initialize Q-table
    if train_mode:
        q_table = np.random.uniform(0, 1, (len(position_bins), len(velocity_bins), environment.action_space.n))  # 40x40x3 array
    else:
        q_table = np.load('trained_qtable.npy')
        
    alpha = 0.1  # Learning rate
    gamma = 0.95  # Discount factor

    epsilon = 1  # Initial epsilon value for epsilon-greedy policy
    epsilon_decay = 2 / num_episodes  # Rate of decay for epsilon
    random_gen = np.random.default_rng()  # Random number generator

    rewards_log = np.zeros(num_episodes)  # To store rewards per episode

    # Iterate through episodes
    for episode in range(num_episodes):
        observation = environment.reset()[0]  # Reset environment to initial state
        pos_idx = np.digitize(observation[0], position_bins)
        vel_idx = np.digitize(observation[1], velocity_bins)

        done = False  # To check if the episode has ended
        total_reward = 0
        step_count = 0

        # Run the episode
        while not done and step_count < 1000:
            step_count += 1
            if train_mode and random_gen.random() < epsilon:
                action = environment.action_space.sample()  # Choose random action
            else:
                action = np.argmax(q_table[pos_idx, vel_idx, :])  # Choose best action based on Q-table

            new_observation, reward, done, _, _ = environment.step(action)
            new_pos_idx = np.digitize(new_observation[0], position_bins)
            new_vel_idx = np.digitize(new_observation[1], velocity_bins)

            # Apply reward shaping if enabled
            if apply_reward_shaping:
                reward += 300 * (gamma * abs(new_observation[1]) - abs(observation[1]))

            # Update Q-table if training mode is enabled
            if train_mode:
                q_table[pos_idx, vel_idx, action] += alpha * (
                    reward + gamma * np.max(q_table[new_pos_idx, new_vel_idx, :]) - q_table[pos_idx, vel_idx, action]
                )

            pos_idx = new_pos_idx
            vel_idx = new_vel_idx
            total_reward += reward

        epsilon = max(epsilon - epsilon_decay, 0)  # Decay epsilon value

        rewards_log[episode] = total_reward  # Log the total reward for the episode

        print(f'Episode: {episode}, Reward: {total_reward}, Completed: {done}')

    environment.close()

    # Save Q-table if training
    if train_mode:
        np.save(f"trained_qtable_{num_episodes}.npy", q_table)

    # Calculate and plot mean rewards
    avg_rewards = np.zeros(num_episodes)
    for ep in range(num_episodes):
        avg_rewards[ep] = np.mean(rewards_log[max(0, ep-100):(ep+1)])
    plt.plot(avg_rewards)
    plt.grid()

    # Save plots and data
    if train_mode:
        if apply_reward_shaping:
            plt.savefig(f'plots/mountain_car_shaped_{num_episodes}.png')
            avg_rewards.tofile(f'data/rewards_shaped_{num_episodes}.csv', sep=',')
        else:
            plt.savefig(f'plots/mountain_car_{num_episodes}.png')
            avg_rewards.tofile(f'data/rewards_{num_episodes}.csv', sep=',')

if __name__ == '__main__':
    # execute_simulation(5000, train_mode=True, show_render=False, apply_reward_shaping=False)

    execute_simulation(10, train_mode=False, show_render=True)
