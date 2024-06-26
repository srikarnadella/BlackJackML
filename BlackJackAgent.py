import numpy as np
from BlackJackEnv import BlackJackEnv


class BlackJackAgent:
    def __init__(self, alpha=0.001, gamma=0.9, epsilon=0.9, epsilon_decay=0.999, min_epsilon=0.05):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration-exploitation trade-off
        self.epsilon_decay = epsilon_decay  # Epsilon decay rate
        self.min_epsilon = min_epsilon  # Minimum epsilon value
        self.q_table = {}  # Dictionary to store Q-values

    def state_to_tuple(self, state):
        player_sum, dealer_card, usable_ace = state
        return (player_sum, dealer_card, usable_ace)

    def get_q_value(self, state, action):
        state_tuple = self.state_to_tuple(state)
        return self.q_table.get((state_tuple, action), 0.0)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])  # Explore: random action (0: stand, 1: hit)
        else:
            # Exploit: choose action with highest Q-value
            q_stand = self.get_q_value(state, 0)
            q_hit = self.get_q_value(state, 1)
            return 0 if q_stand > q_hit else 1

    def update_q_table(self, state, action, reward, next_state):
        state_tuple = self.state_to_tuple(state)
        next_state_tuple = self.state_to_tuple(next_state)
        best_next_action = self.choose_action(next_state)
        td_target = reward + self.gamma * self.get_q_value(next_state, best_next_action)
        td_delta = td_target - self.get_q_value(state, action)
        self.q_table[(state_tuple, action)] = self.get_q_value(state, action) + self.alpha * td_delta

    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state

            # Decay epsilon after each episode
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Print progress every 10000 episodes
            if (episode + 1) % 10000 == 0:
                print(f"Episode {episode + 1}/{num_episodes} completed.")

    def play(self, env, num_episodes):
        total_reward = 0

        for _ in range(num_episodes):
            state = env.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                state, reward, done = env.step(action)
                total_reward += reward

        return total_reward / num_episodes

# Create the Blackjack environment
env = BlackJackEnv()

# Create the agent
agent = BlackJackAgent()

# Train the agent
agent.train(env, num_episodes=50000)

# Evaluate the agent
average_reward = agent.play(env, num_episodes=10000)
print(f"Average reward over 10,000 episodes: {average_reward}")
