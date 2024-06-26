# DQNAgent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQNAgent:
    def __init__(self, env, input_size, output_size, hidden_size=128, gamma=0.99, batch_size=64, lr=1e-3):
        self.env = env
        self.input_size = input_size
        self.output_size = output_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Q-networks
        self.policy_net = DQN(input_size, hidden_size, output_size).to(self.device)
        self.target_net = DQN(input_size, hidden_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Replay memory
        self.memory = deque(maxlen=10000)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return self.env.sample_action()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def store_transition(self, transition):
        self.memory.append(transition)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(self.device)
        non_final_next_states = torch.tensor([s for s in batch.next_state if s is not None], dtype=torch.float32).to(self.device)
        state_batch = torch.tensor(batch.state, dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train_agent(self, num_episodes=10000, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.999):
        epsilon = epsilon_start
        rewards = []

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0

            while True:
                action = self.select_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                reward = torch.tensor([reward], dtype=torch.float32).to(self.device)

                if done:
                    next_state = None

                self.store_transition(Transition(state, action, next_state, reward, done))
                state = next_state

                self.train()

                if done:
                    break

            rewards.append(total_reward)

            # Decay epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            # Update target network periodically
            if episode % 100 == 0:
                self.update_target_network()

            if episode % 1000 == 0:
                print(f"Episode {episode}, Average Reward: {np.mean(rewards[-1000:]):.2f}")

        print("Training complete.")

    def play(self, num_episodes=100):
        total_rewards = []

        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0

            while True:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                    q_values = self.policy_net(state_tensor)
                    action = q_values.argmax().item()

                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state

                if done:
                    break

            total_rewards.append(episode_reward)

        avg_reward = np.mean(total_rewards)
        print(f"Average reward over {num_episodes} episodes: {avg_reward}")

        return avg_reward
