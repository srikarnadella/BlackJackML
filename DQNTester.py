# main.py

from BlackJackEnvDQN import BlackJackEnv
from DQNAgent import DQNAgent

# Set hyperparameters and instantiate environment
env = BlackJackEnv()
input_size = env.state_space
output_size = len(env.action_space)
hidden_size = 128
lr = 1e-3
gamma = 0.99
batch_size = 128

# Instantiate DQN agent
agent = DQNAgent(env, input_size, output_size, hidden_size, gamma, batch_size, lr)

# Train the agent
agent.train_agent(num_episodes=5000, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.999)

# Evaluate the agent
average_reward = agent.play(num_episodes=100)
print(f"Average reward over 100 episodes: {average_reward}")
