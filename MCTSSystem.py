import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from ANET import ANET
from MCTS import MCTS, MCTSNode
from ReplayBuffer import ReplayBuffer
from games.Hex import Hex
from games.TicTacToe import TicTacToe


class MCTSSystem:
    def __init__(self,anet):
        self.replay_buffer = ReplayBuffer()
        self.anet = anet
        self.num_games = 100
        self.batch_size = 256

        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(anet.parameters(), lr=0.001)
        self.loss_function = nn.CrossEntropyLoss()

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
            action_probs = mcts.run_simulation(10000)
            board_state = copy.deepcopy(state_manager.board)
            player = copy.deepcopy(state_manager.current_player)

            # Select the best action based on the action probabilities
            best_action = np.argmax(action_probs)

            print(best_action)

            # Save to a buffer, before loading into replay buffer
            self.replay_buffer.push_to_buffer(board_state, action_probs, player)

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
            if len(self.replay_buffer.buffer) > 300:
                self.train()

            # Step 3: Save the model
            if ga % 10 == 0:
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
        print(predicted_probs[0])
        print(action_probs[0])

        # Calculate loss
        loss = self.loss_function(predicted_probs, torch.Tensor(action_probs))

        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()

        # Print loss
        print(f"Loss: {loss}")

    def save_model(self, model_index):
        # Save the model.
        torch.save(self.anet.state_dict(), f"model_{model_index}.pth")
        torch.save(self.optimizer.state_dict(), f"optimizer_{model_index}.pth")
        print(f"Model and optimizer saved.")








anet = ANET()
system = MCTSSystem(anet)
system.run_system()