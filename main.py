import gym
import numpy as np
import random

class BlackjackAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}  # Dictionary to store Q-values

    def state_to_tuple(self, state):
        # Convert the state to a tuple if it's not already
        if not isinstance(state, tuple):
            return tuple(state)
        return state

    def get_q_value(self, state, action):
        # Get Q-value from Q-table or initialize with zeros
        state = self.state_to_tuple(state)
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        state = self.state_to_tuple(state)
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            # Exploit: Choose action with highest Q-value
            actions = [0, 1]  # 0 for stand 1 for hit
            q_values = [self.get_q_value(state, a) for a in actions]
            return actions[np.argmax(q_values)]

    def update_q_table(self, state, action, reward, next_state):
        state = self.state_to_tuple(state)
        next_state = self.state_to_tuple(next_state)
        
        # Calculate TD error
        best_next_action = self.choose_action(next_state)
        td_target = reward + self.gamma * self.get_q_value(next_state, best_next_action)
        td_delta = td_target - self.get_q_value(state, action)

        # Update Q-value for the state-action pair
        self.q_table[(state, action)] = self.get_q_value(state, action) + self.alpha * td_delta

    def train(self, num_episodes):
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state

    def play(self, num_episodes):
        total_reward = 0

        for _ in range(num_episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward

        return total_reward / num_episodes

# Create the blackjack environment
env = gym.make('Blackjack-v1')

# Create the agent
agent = BlackjackAgent(env)

# Train the agent
agent.train(num_episodes=500000)

# Evaluate the agent
average_reward = agent.play(num_episodes=10000)
print(f"Average reward over 10,000 episodes: {average_reward}")
