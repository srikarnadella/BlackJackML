## Table of Contents
- [NON-DQN](#Non-DQN_Blackjack_Agent)
- [DQN](#Deep_Q-Learning_Network_(DQN)_for_Blackjack)

# Non-DQN Blackjack Agent
## Introduction
This repository contains an implementation of a Blackjack agent using traditional algorithms without deep reinforcement learning (DQN). The agent employs basic strategies and Monte Carlo simulation techniques to make optimal decisions in the game of Blackjack.

## Key Components
### 1. BlackjackAgent.py
This file implements the BlackjackAgent class, which uses Monte Carlo simulation for decision-making in Blackjack:

* Game Representation:
  * State Representation: Each state in Blackjack is represented by a tuple (player_sum, dealer_card, usable_ace), where:
    * player_sum: Total sum of player's cards.
    * dealer_card: Face-up card of the dealer.
    * usable_ace: Boolean indicating whether the player holds an ace that can be counted as 1 or 11.
* Action Space:
  * The agent has a simple action space:
    * 0: Hit (request another card from the dealer).
    * 1: Stand (end the turn and keep the current hand).
### 2. Blackjack Rules and Strategy
* Game Dynamics:
  * Game Rules: Standard Blackjack rules are followed, including card values (2-10 as face value, face cards as 10, Ace as 1 or 11).
  * Game Logic: The agent simulates games using Monte Carlo simulation to estimate the value function of each state-action pair.
* Strategy Determination:
  * Policy:
    * Hit or Stand Decision: The agent decides whether to hit or stand based on a simple heuristic: if the current sum is less than 17, hit; otherwise, stand.
    * Monte Carlo Simulation: To estimate the value of each state-action pair, the agent simulates multiple Blackjack games and averages the outcomes to determine the optimal strategy.
### 3. Mathematics and Theory
* Value Function:
  * The value function 𝑉(𝑠) estimates the expected cumulative reward starting from state 𝑠.
  * Monte Carlo Estimation: The agent uses Monte Carlo methods to estimate 𝑉(𝑠) by averaging returns obtained from simulating multiple games.

* Action Selection:
  * The agent selects actions based on:
    * Heuristic Policy: A simple heuristic guides the agent's decisions (hit below 17, stand otherwise).
    * Monte Carlo Exploration: By simulating games, the agent explores different actions and their outcomes to refine its strategy over time.
### 4. Articles and Resources
* Learning Resources:
    * Reinforcement Learning: An Introduction by Sutton and Barto: This textbook provided foundational knowledge on Monte Carlo methods and their application in Blackjack.
    * Online Blackjack Strategy Guides: Various articles and resources on optimal Blackjack strategies helped inform the heuristic policies and decision-making of the agent.
### What I Learned
* Monte Carlo Methods: Developed proficiency in using Monte Carlo simulation for estimating value functions and making decisions under uncertainty.
* Heuristic Strategies: Explored and implemented basic heuristic strategies for decision-making in games like Blackjack.
* Python Programming: Enhanced Python programming skills, especially in implementing simulations and algorithms for decision-making tasks.
### Future Improvements
* Enhanced Strategy Exploration: Implement more advanced Blackjack strategies, such as card counting or more sophisticated heuristic rules, to improve decision-making accuracy.
* Performance Optimization: Refine simulation methods and optimize algorithms to handle larger state spaces and longer simulation durations efficiently.
* Interactive Visualization: Integrate visualization tools to analyze gameplay and decision-making processes for deeper insights and debugging.
### Conclusion
This project provided practical experience in implementing traditional algorithms for decision-making in games, focusing on Blackjack as a case study. Moving forward, I aim to apply these skills to more complex game environments and explore advanced reinforcement learning techniques to further improve agent performance and versatility.




# Deep Q-Learning Network (DQN) for Blackjack

## Introduction
This repository contains an implementation of a Deep Q-Learning Network (DQN) agent trained to play the game of Blackjack. The DQN agent learns to make optimal decisions in Blackjack using reinforcement learning principles, aiming to maximize cumulative rewards over time.

## Key Components
### 1. DQNAgent.py
This file defines the DQNAgent class, which implements the DQN algorithm. Here’s an overview of its components:

* Neural Network Architecture:
 * __init__: Initializes the neural network with fully connected layers (fc1, fc2) and activation functions (ReLU).
 * forward: Defines the forward pass of the network, applying ReLU activation to hidden layers and outputting Q-values for each action.

* Agent Methods:
 * select_action: Chooses an action based on an epsilon-greedy policy to balance exploration and exploitation.
 * train: Implements the DQN algorithm, including experience replay, target network updates, and loss computation using the Bellman equation.
### 2. BlackjackEnv.py
This file defines the Blackjack environment, encapsulating game rules and state transitions:

* Environment Setup:
 * BlackjackEnv: Initializes the game environment with rules for state representation (player's current sum, dealer's face-up card, and whether the player has a usable ace) and action space (hit or stand).
* Game Dynamics:
 * step: Executes a player action (hit or stand), computes rewards based on game outcomes (win, lose, draw), and transitions to the next state.
### 3. DQNHyperParamTuning.py
This script automates hyperparameter tuning using grid search:

* Grid Search for Hyperparameters:
 * Defines a grid of hyperparameters (learning rate, discount factor, network architecture) to optimize DQN performance.
 * Iterates over combinations, trains agents with varying parameters, and evaluates performance to determine optimal settings.
### 4. Articles and Resources
* Learning Resources:
 * Reinforcement Learning: An Introduction by Sutton and Barto: This textbook provided foundational knowledge on reinforcement learning concepts and algorithms, crucial for understanding DQN.
 * OpenAI Gym Documentation: Reference for creating custom environments and interfacing with reinforcement learning agents using Gym.
 * Medium Articles and GitHub Repositories: Various blog posts and code repositories provided practical implementations and insights into DQN and Blackjack-specific implementations.
### What I Learned
* Deep Q-Learning: Developed a strong understanding of DQN algorithm implementation, including experience replay, target network updates, and epsilon-greedy exploration strategies.
* Neural Network Integration: Learned to integrate neural networks with reinforcement learning frameworks to approximate Q-values and optimize decision-making processes.
* Python and PyTorch Skills: Enhanced proficiency in Python programming and PyTorch library usage for building and training neural networks.
### Future Improvements
* Performance Optimization: Explore techniques like double Q-learning, prioritized experience replay, and dueling network architectures to enhance agent learning efficiency and stability.
* Enhanced Exploration Strategies: Implement more sophisticated exploration-exploitation strategies beyond epsilon-greedy to balance learning speed and policy optimization.
* Visualization and Analysis: Integrate visualization tools to analyze agent behavior and performance over training episodes for deeper insights and debugging.
### Conclusion
This project not only provided hands-on experience in implementing deep reinforcement learning techniques but also strengthened foundational knowledge in neural networks and reinforcement learning algorithms. Moving forward, I aim to apply these skills to more complex environments and contribute to advancements in AI research and application.
