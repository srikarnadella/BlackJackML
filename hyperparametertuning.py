import numpy as np
from itertools import product
from BlackJackAgent import BlackJackAgent 
from BlackJackEnv import BlackJackEnv

env = BlackJackEnv()

# Define ranges for hyperparameters
alpha_values = [0.1, 0.01, 0.001]
gamma_values = [0.9, 0.95, 0.99]
epsilon_values = [1.0, 0.9, 0.8]
epsilon_decay_values = [0.999, 0.995]
min_epsilon_values = [0.1, 0.05]

# Generate all combinations of hyperparameters
combinations = list(product(alpha_values, gamma_values, epsilon_values, epsilon_decay_values, min_epsilon_values))

# Number of combinations to try
num_combinations = 10

# Randomly sample indices to select combinations
indices = np.random.choice(len(combinations), num_combinations, replace=False)
selected_combinations = [combinations[idx] for idx in indices]

# Initialize variables to track best performance
best_average_reward = -float('inf')
best_hyperparams = None

# Loop through each combination
for alpha, gamma, epsilon, epsilon_decay, min_epsilon in selected_combinations:
    agent = BlackJackAgent(alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)
    agent.train(env, num_episodes=500000)  # Adjust num_episodes as needed
    average_reward = agent.play(env, num_episodes=10000)  # Adjust num_episodes as needed

    # Print and store results
    print(f"Hyperparams: alpha={alpha}, gamma={gamma}, epsilon={epsilon}, epsilon_decay={epsilon_decay}, min_epsilon={min_epsilon}")
    print(f"Average reward over 10,000 episodes: {average_reward}\n")

    # Track the best combination
    if average_reward > best_average_reward:
        best_average_reward = average_reward
        best_hyperparams = (alpha, gamma, epsilon, epsilon_decay, min_epsilon)

# Print the best hyperparameters found
print(f"Best hyperparameters: alpha={best_hyperparams[0]}, gamma={best_hyperparams[1]}, epsilon={best_hyperparams[2]}, epsilon_decay={best_hyperparams[3]}, min_epsilon={best_hyperparams[4]}")
print(f"Best average reward: {best_average_reward}")