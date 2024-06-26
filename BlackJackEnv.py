import numpy as np

class BlackJackEnv:
    def __init__(self):
        self.reset()
        

    def reset(self):
        # Deck of cards (numbers 1-10 with equal probability)
        self.deck = np.random.choice(range(1, 11), size=52, replace=True)
        self.player_hand = []
        self.dealer_hand = []

        # Deal initial cards
        self.player_hand.append(self.draw_card())
        self.dealer_hand.append(self.draw_card())
        self.player_hand.append(self.draw_card())
        self.dealer_hand.append(self.draw_card())

        return self.get_state()

    def draw_card(self):
        return int(np.random.choice(self.deck))

    def get_state(self):
        # Player's sum, dealer's visible card, usable_ace (True/False)
        return (sum(self.player_hand), self.dealer_hand[0], self.usable_ace())

    def usable_ace(self):
        return 1 in self.player_hand and sum(self.player_hand) + 10 <= 21

    def step(self, action):
        # Action: 0 (stand) or 1 (hit)
        if action == 1:  # Hit: add a card to player's hand
            self.player_hand.append(self.draw_card())
            if sum(self.player_hand) > 21:
                return self.get_state(), -1, True  # Bust, player loses
            else:
                return self.get_state(), 0, False

        # Dealer's turn
        while sum(self.dealer_hand) < 17:
            self.dealer_hand.append(self.draw_card())
        
        # Determine winner
        player_sum = sum(self.player_hand)
        dealer_sum = sum(self.dealer_hand)
        if dealer_sum > 21 or player_sum > dealer_sum:
            return self.get_state(), 1, True  # Player wins
        elif player_sum == dealer_sum:
            return self.get_state(), 0, True  # Tie
        else:
            return self.get_state(), -1, True  # Dealer wins

    def print_hands(self, show_all=False):
        print(f"Player's hand: {self.player_hand}, Sum: {sum(self.player_hand)}")
        if show_all:
            print(f"Dealer's hand: {self.dealer_hand}, Sum: {sum(self.dealer_hand)}")
        else:
            print(f"Dealer's visible card: {self.dealer_hand[0]}")
