# BlackjackEnv.py

import numpy as np

class BlackJackEnv:
    def __init__(self):
        self.action_space = [0, 1]  # [Stick, Hit]
        self.state_space = 3  # [player_sum, dealer_card, usable_ace]

    def reset(self):
        # Initialize state (player_sum, dealer_card, usable_ace)
        player_sum = np.random.randint(12, 21)  # Random initial player sum between 12 and 20
        dealer_card = np.random.randint(1, 11)  # Random initial dealer card between 1 and 10
        usable_ace = np.random.choice([True, False])  # Random initial usable ace

        self.state = (player_sum, dealer_card, usable_ace)
        return self.state

    def step(self, action):
        player_sum, dealer_card, usable_ace = self.state

        if action == 1:  # Hit
            new_card = np.random.randint(1, 11)
            player_sum += new_card
            if player_sum > 21 and usable_ace:
                player_sum -= 10
                usable_ace = False
            elif player_sum > 21 and not usable_ace:
                return self.state, -1, True, {}  # Player busts

        elif action == 0:  # Stick
            while dealer_card < 17:
                new_card = np.random.randint(1, 11)
                dealer_card += new_card

            if dealer_card > 21 or player_sum > dealer_card:
                return self.state, 1, True, {}  # Player wins
            elif player_sum == dealer_card:
                return self.state, 0, True, {}  # Tie
            else:
                return self.state, -1, True, {}  # Player loses

        self.state = (player_sum, dealer_card, usable_ace)
        return self.state, 0, False, {}  # Continue game

    def sample_action(self):
        return np.random.choice(self.action_space)
