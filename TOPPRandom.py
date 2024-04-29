# TESTING=============================================
# Tournament of progressive players (TOPP). The opposite player does random moves
import random
import numpy as np
import torch
import torch.nn.functional as F

from ANET import ANET
from games.TicTacToe import TicTacToe


class TOPPRandom:
    def __init__(self, anet, model_path, num_games, state_manager, board_size, total_actions):
        """
        Initialize the Tournament of Progressive Players (TOPP).
        """
        # Load the model
        self.anet = anet
        self.model_path = model_path
        self.model = self.load_model(model_path)

        # The game
        self.state_manager = state_manager
        self.board_size = board_size
        self.total_actions = total_actions

        # Number of games
        self.num_games = num_games

    def load_model(self, path):
        """
        Load a model.
        """
        self.anet.load_state_dict(torch.load(path))

        # Set model to evaluation mode
        self.anet.eval()

        # Return the loaded anet
        return self.anet

    def play_match(self, model):
        """
        Play a match between two models and return the result.
        """
        # Initialize the game
        self.state_manager = TicTacToe()

        # Play until the game is over
        while not self.state_manager.is_game_over():

            # Player 1 uses the ANET
            if self.state_manager.current_player == 1: 

                # Use the anet to select the action
                action = self.select_anet_action(self.state_manager)

            # Player 2 do random moves
            if self.state_manager.current_player == -1:

                possible_actions = self.state_manager.get_valid_moves()
                action = random.choice(possible_actions)
            
            # Execute the move
            self.state_manager.make_move(action)
        
        # Return the result from player 1's perspective
        return self.state_manager.get_winner()

    def run_tournament(self):
        """
        Run the tournament and print the results.
        """
        # Initialize counters for result
        win, draw, loss = 0, 0, 0

        # Iterate over number of games
        for i in range(self.num_games):

            # Play the match and decide the winner
            winner = self.play_match(self.model)
            if winner == 1:
                win += 1
            elif winner == 0:
                draw += 1
            else:
                loss += 1
            print(i+1)
        
        print("Results against random player:")
        print(f"Wins: {win}, Draws: {draw}, Losses: {loss}")

    def select_anet_action(self, state_manager):
        """
        Select the action using the AI model.
        """

        # Reshape to tensor
        # Use the anet to select the most promising move
        self.anet.eval()
        input = self.anet.prepare_input([state_manager.board], [state_manager.current_player])
        with torch.no_grad():
            predicted_probs, _ = self.anet(input)


        # Ensure the move is valid
        valid_moves = torch.Tensor(state_manager.get_valid_moves_hot_encoded())
        action_probs = F.softmax(predicted_probs, dim=1) * valid_moves

        # Chose the action with highest probability
        action = torch.argmax(action_probs).item()

        # Return the action
        return action

#TESTING==================
topp = TOPPRandom(
    anet = ANET(),
    model_path = 'model_60.pth',
    num_games = 1000,
    state_manager = TicTacToe(),
    board_size = 3,
    total_actions = 9
)
topp.run_tournament()
#TESTING==================
