import torch
import torch.nn.functional as F

class PlayGame:
    def __init__(self, anet, state_manager, model_path):
        self.state_manager = state_manager # The game
        self.anet = anet # The AI model
        self.load_model(model_path) # Load the model

    def play_match(self, start):
        # Reset the game to the initial state
        state_manager = self.state_manager.reset()
        if start == False: # Decide starting player
                state_manager.current_player = -1

        # Play until the game is over
        while not state_manager.is_game_over():
            state_manager.display_board()

            # Determine which model to use based on the current player
            if state_manager.current_player == 1:
                move = int(input("Enter your move (number): "))
            else:
                move = self.select_action(state_manager)

            # Execute the move
            state_manager.make_move(move)

        # Print the winner
        state_manager.display_board()
        if state_manager.current_player == -1:
            print("You won!")
        else:
            print("The AI won")

        # Display the final board and return the winner
        return state_manager.get_winner()

    def select_action(self, state_manager):
        # Prepare input for the AI model
        input_tensor = self.anet.prepare_input([state_manager.board], [state_manager.current_player])
        # Predict move probabilities
        with torch.no_grad():
            predicted_probs = self.anet(input_tensor)
        
        # Get valid moves and filter out invalid choices
        valid_moves = state_manager.get_valid_moves_hot_encoded()
        predicted_probs = F.softmax(predicted_probs, dim=1) * torch.tensor(valid_moves)

        # Choose the best valid move
        action = torch.argmax(predicted_probs)

        return action

    def load_model(self, path):
        # Load the AI model
        self.anet.load_state_dict(torch.load(path))
        self.anet.eval()
        return self.anet
