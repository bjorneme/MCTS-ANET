import random
import numpy as np
import torch
import torch.nn.functional as F

from ANET import ANET
from games.TicTacToe import TicTacToe

# Tournament of progressive players (TOPP)
class TOPP:
    def __init__(self, anet, model_paths, num_games, board_size, total_actions, state_manager = None):
        self.model_paths = model_paths # List of paths to saved models
        self.anet = anet # The Actor network
        self.models = [self.load_model(path) for path in model_paths] # Load models
        self.state_manager = state_manager # The game
        self.board_size = board_size # Board size
        self.total_actions = total_actions # Total possible actions
        self.num_games = num_games # Number of games

    def load_model(self, path):

        # Use the ANETs architecture
        model = ANET()

        # Load the model from the path
        model.load_state_dict(torch.load(path))

        # Set model to evaluation mode
        model.eval()

        # Return the loaded anet
        return model

    def play_match(self, model_1, model_2):

        # Initialize the game
        self.state_manager = TicTacToe()

        # Play until the game is over
        while not self.state_manager.is_game_over():

            # Select the model based on current player
            if self.state_manager.current_player == 1:
                current_model = model_1
            elif self.state_manager.current_player == -1:
                current_model = model_2

            # Select the action
            action = self.select_anet_action(self.state_manager, current_model)

            # Execute the action
            self.state_manager.make_move(action)

        # Return the result from player 1's perspective
        return self.state_manager.get_winner()

    def run_tournament(self):
        """
        Run the tournament and print the results.
        """
        
        # Initialize results
        results = {path: {'wins': 0, 'losses': 0, 'draws': 0} for path in self.model_paths}

        for round in range(self.num_games):
            # Iterate over all the model and play against each other
            for i, model_1 in enumerate(self.models):
                for j, model_2 in enumerate(self.models):

                    # Skip matches against itself
                    if i == j:
                        continue

                    # Play the match
                    winner = self.play_match(model_1, model_2)

                    # Update with the result
                    if winner == 1:
                        results[self.model_paths[i]]['wins'] += 1
                        results[self.model_paths[j]]['losses'] += 1
                    elif winner == -1:
                        results[self.model_paths[i]]['losses'] += 1
                        results[self.model_paths[j]]['wins'] += 1
                    else:
                        results[self.model_paths[i]]['draws'] += 1
                        results[self.model_paths[j]]['draws'] += 1

                    print(round)

        # Print the tournament results
        for path in self.model_paths:
            outcome = results[path]
            print(f"Model {path}: Wins - {outcome['wins']}, Losses - {outcome['losses']}, Draws - {outcome['draws']}")

    def select_anet_action(self, state_manager, current_model):
        """
        Select the action using the AI model.
        """

        # Prepare input
        input = current_model.prepare_input([state_manager.board], [state_manager.current_player])
        
        # Forward pass
        with torch.no_grad():
            predicted_probs = current_model(input)

        # Ensure it is a valid move
        valid_moves = state_manager.get_valid_moves_hot_encoded()
        predicted_probs = F.softmax(predicted_probs, dim=1) * torch.Tensor(valid_moves)

        # Choose top N probable actions. Used such that the model doesnt return the same result each time
        if (sum(valid_moves) > (self.total_actions - self.board_size)):
            N = min(3, sum(valid_moves))

            _, top_indexes = torch.topk(predicted_probs[0], N)

            # Choose a random action from the top N
            action= random.choice(top_indexes.tolist())  

        else:
            # Chose the action with highest probability
            action = torch.argmax(predicted_probs).item() 

        # Return the action
        return action
    
# Simulate the four Topp games
topp = TOPP(
    anet = ANET(),
    model_paths = ['model_0.pth','model_30.pth', 'model_90.pth'],
    num_games = 100,
    board_size = 3,
    total_actions = 9
)
topp.run_tournament()
