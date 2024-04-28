import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from ANET import ANET
from MCTS import MCTS, MCTSNode
from ReplayBuffer import ReplayBuffer
from games.Hex import Hex
from games.TicTacToe import TicTacToe


class MCTSSystem:
    def __init__(self,anet, model_path = None, optimizer_path = None):
        self.replay_buffer = ReplayBuffer()
        self.anet = anet
        self.num_games = 100
        self.batch_size = 64
        self.save_interval = 10
        self.optimizer = optim.Adam(anet.parameters(), lr=0.001)

        # Load model and optimizer if paths are provided
        self.model_path = model_path
        self.optimizer_path = optimizer_path
        if model_path is not None and optimizer_path is not None:
            self.load_model()


    def self_play(self, episode):

        # Initialize the game and MCTS
        state_manager = TicTacToe()
        root_node = MCTSNode(self.anet, 1.41, state_manager)
        mcts = MCTS(root_node, 3, 9)

        # Alternate the starting player
        if(episode % 2 == 0):
            state_manager.current_player = -1

        # Play until game is over
        while not state_manager.is_game_over():

            # Execute the MCTS search. Get probabilities for possible actions
            action_probs = mcts.run_simulation(2500)
            board_state = copy.deepcopy(state_manager.board)

            # Select the best action based on the action probabilities
            best_action = np.argmax(action_probs)

            print(best_action)

            # Save to a buffer, before loading into replay buffer
            self.replay_buffer.push_to_buffer(board_state, action_probs, state_manager.current_player)

            # Execute the best action
            state_manager.make_move(best_action)

            # Update MCTS root to the new state, and prune
            mcts.update_root(best_action)
            
        # Update win/draw/loss counts based on game outcome
        result = state_manager.get_winner()
        if result == 1:
            self.win += 1
        elif result == 0:
            self.draw += 1
        else:
            self.loss += 1
        print(f"Episode {episode}")

    def run_system(self):
        # For counting win/ draw/ loss
        self.win, self.draw, self.loss = 0, 0, 0

        for ga in range(self.num_games):
            # Step 1: Play games
            self.self_play(ga)

            # Step 2: Train ANET on random minibatches of cases from Replay buffer
            if len(self.replay_buffer.buffer) > 100:
                self.train()

            # Step 3: Save the model
            if ga % self.save_interval == 0:
                self.save_model(ga)

        # Print final results
        print("Final Results:")
        print(f"Wins: {self.win}, Draws: {self.draw}, Losses: {self.loss}")

    def train(self):        
        # Set model to training mode
        anet.train()

        # Sample a batch from Replay Buffer
        board_states, action_probs, players = self.replay_buffer.get_sample(self.batch_size)

        # Prepare board state
        input_state_anet = self.anet.prepare_input(board_states, players)

        # Forward pass
        predicted_probs = self.anet(input_state_anet)

        # Calculate loss
        loss = F.cross_entropy(predicted_probs, torch.Tensor(action_probs))

        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()

        # Print loss
        print(f"Loss: {loss}")


    def save_model(self, model_index):
        # Save the model.
        model_filename = f"model_{model_index}.pth"
        optimizer_filename = f"optimizer_{model_index}.pth"
        
        torch.save(self.anet.state_dict(), model_filename)
        torch.save(self.optimizer.state_dict(), optimizer_filename)
        print(f"Model and optimizer saved: {model_filename}, {optimizer_filename}")

    def load_model(self):
        # Load existing model.
        try:
            self.anet.load_state_dict(torch.load(self.model_path))
            self.optimizer.load_state_dict(torch.load(self.optimizer_path))
            print("Model and optimizer have been successfully loaded.")
        except FileNotFoundError:
            print("No model and optimizer found. Creating a new one.")







anet = ANET()
system = MCTSSystem(anet)
system.run_system()