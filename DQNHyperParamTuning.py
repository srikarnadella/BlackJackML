from DQNAgent import DQNAgent
from BlackJackEnvDQN import BlackJackEnv
import itertools

# Define ranges for hyperparameters
param_grid = {
    'hidden_size': [64, 128, 256],
    'lr': [0.001, 0.0001],
    'gamma': [0.99, 0.9],
    'epsilon_start': [1.0, 0.5],
    'epsilon_end': [0.1, 0.05],
    'epsilon_decay': [0.999, 0.995]
}

# Generate all combinations of hyperparameters
combinations = list(itertools.product(*param_grid.values()))

num_episodes = 5000  # Number of episodes for training each combination
num_eval_episodes = 100  # Number of episodes for evaluation

best_reward = -float('inf')
best_params = None

for params in combinations:
    hidden_size, lr, gamma, epsilon_start, epsilon_end, epsilon_decay = params

    env = BlackJackEnv()
    agent = DQNAgent(env, input_size=env.state_space, output_size=len(env.action_space),
                     hidden_size=hidden_size, lr=lr, gamma=gamma)

    # Train the agent
    agent.train_agent(num_episodes=num_episodes, epsilon_start=epsilon_start,
                      epsilon_end=epsilon_end, epsilon_decay=epsilon_decay)

    # Evaluate the agent
    avg_reward = agent.play(num_episodes=num_eval_episodes)

    # Check if this combination is the best so far
    if avg_reward > best_reward:
        best_reward = avg_reward
        best_params = params

print(f"Best parameters: {best_params}, Best average reward: {best_reward}")
