import random
import torch
import torch.nn.functional as F


#TOPP: Tournamet of progressive player
class Topp:
    def __init__(self, num_games, model_paths, anet, state_manager):
        # List of model paths
        self.model_paths = model_paths
        self.anet = anet
        self.num_games = num_games
        self.state_manager = state_manager


    def run_tournament(self):
        # Initialize results
        results = {path: {'wins': 0, 'losses': 0, 'draws': 0} for path in self.model_paths}

        for round in range(self.num_games):
            # Iterate over all the model and play against each other
            for i, model_1 in enumerate(self.model_paths):
                for j, model_2 in enumerate(self.model_paths):

                    # Skip matches against itself
                    if i == j:
                        continue

                    # Play the match
                    winner = self.play_match(model_1, model_2)

                    # Update with the result
                    if winner == 1:
                        results[model_1]['wins'] += 1
                    elif winner == -1:
                        results[model_2]['wins'] += 1

                    print(round)

        # Print the tournament results
        for path in self.model_paths:
            outcome = results[path]
            print(f"Model {path}: Wins - {outcome['wins']}")

    def play_match(self, model_1, model_2):
        # Initialize the game
        state_manager = self.state_manager.reset()

        # Play until the game is over
        while not state_manager.is_game_over():

            state_manager.display_board()
                
            # Select the model based on current player
            if state_manager.current_player == 1:
                self.load_model(model_1)
            elif state_manager.current_player == -1:
                self.load_model(model_2)

            # Select the action
            best_move = self.select_action(state_manager)

            # Execute the action
            state_manager.make_move(best_move)

        state_manager.display_board()

        # Return the result:
        return state_manager.get_winner()
    
    def select_action(self, state_manager):
        # Prepare input
        input = self.anet.prepare_input([state_manager.board], [state_manager.current_player])

        # Forward pass
        with torch.no_grad():
            predicted_probs = self.anet(input)

        # Ensure it is a valid move
        valid_moves = state_manager.get_valid_moves_hot_encoded()
        predicted_probs = F.softmax(predicted_probs, dim=1) * torch.Tensor(valid_moves)

        # Choose top N probable actions. Used such that the model doesnt return the same result each time
        if (sum(valid_moves) > 13):
            N = min(3, sum(valid_moves))
            _, top_indexes = torch.topk(predicted_probs[0], N)
            action= random.choice(top_indexes.tolist())  

        else:
            # Chose the action with highest probability
            action = torch.argmax(predicted_probs) 

        # Return the action
        return action


    def load_model(self, path):
        # Load a model.
        self.anet.load_state_dict(torch.load(path))
        self.anet.eval()








