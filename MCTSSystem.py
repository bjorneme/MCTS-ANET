import copy
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from MCTS import MCTS, MCTSNode
from ReplayBuffer import ReplayBuffer

class MCTSSystem:
    def __init__(self, verbose, anet, state_manager, board_size, total_actions, optimizer, num_games, batch_size, c, mcts_searches, e_greedy_mcts, num_anet_cached, model_path=None, optimizer_path=None):
        # The game
        self.verbose = verbose
        self.state_manager = state_manager
        self.board_size = board_size
        self.total_actions = total_actions
        self.num_games = num_games

        # MCTS
        self.mcts_searches = mcts_searches
        self.c = c
        self.e_greedy_mcts = e_greedy_mcts

        # The Replay Buffer
        self.replay_buffer = ReplayBuffer()

        # Actor network
        self.anet = anet
        self.num_anet_cached = num_anet_cached

        # Initialize optimizer, loss function and batch size
        self.optimizer = optimizer
        self.loss_function = nn.CrossEntropyLoss()
        self.batch_size = batch_size

        # Load model and optimizer if paths are provided
        if model_path is not None and optimizer_path is not None:
            self.load_model(model_path, optimizer_path)

        # List to store loss. Used for plotting
        self.loss_history = []


    def self_play(self, episode):

        # Initialize the game and MCTS
        state_manager = self.state_manager.reset()
        root_node = MCTSNode(self.anet, self.e_greedy_mcts, self.c, state_manager)
        mcts = MCTS(root_node, self.board_size, self.total_actions)

        # Alternate the starting player
        if(episode % 2 == 0):
            state_manager.current_player = -1

        # Play until game is over
        while not state_manager.is_game_over():
            if self.verbose:
                state_manager.display_board()

            # Execute the MCTS search. Get probabilities for possible actions
            action_probs = mcts.run_simulation(self.mcts_searches)
            board_state = copy.deepcopy(state_manager.board)
            player = copy.deepcopy(state_manager.current_player)

            # Select the best action based on the action probabilities
            best_action = np.argmax(action_probs)

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
        if self.verbose:
            state_manager.display_board()
        print(f"Episode {episode}")

    def run_system(self):
        # For counting win/ draw/ loss
        self.win, self.draw, self.loss = 0, 0, 0

        for ga in range(self.num_games + 1):
            # Step 1: Play games
            self.self_play(ga)

            # Step 2: Train ANET on random minibatches of cases from Replay buffer
            if len(self.replay_buffer.buffer) >= self.batch_size and ga != 0:
                self.train()

            # Step 3: Save the model
            save_intervall = self.num_games/int(self.num_anet_cached-1)
            if ga % (save_intervall) == 0:
                self.save_model(ga)


        # Print final results
        print("Final Results:")
        print(f"Wins: {self.win}, Draws: {self.draw}, Losses: {self.loss}")

    def train(self):        
        # Set model to training mode
        self.anet.train()

        # Sample a batch from Replay Buffer
        board_states, action_probs, players = self.replay_buffer.get_sample(self.batch_size)

        # Prepare board state
        input = self.anet.prepare_input(board_states, players)

        # Forward pass
        predicted_probs = self.anet(input)

        # Calculate loss
        loss = self.loss_function(predicted_probs, torch.Tensor(action_probs))
        self.loss_history.append(loss.detach().numpy())

        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()

        # Print loss
        if self.verbose:
            print(f"Loss: {loss}")

    def save_model(self, model_index):
        # Save the model.
        torch.save(self.anet.state_dict(), f"model_{model_index}.pth")
        torch.save(self.optimizer.state_dict(), f"optimizer_{model_index}.pth")
        print(f"Model and optimizer saved.")

    def load_model(self, model_path, optimizer_path):
        # Load existing model.
        try:
            self.anet.load_state_dict(torch.load(model_path))
            self.optimizer.load_state_dict(torch.load(optimizer_path))
            print("Model and optimizer have been successfully loaded.")
        except FileNotFoundError:
            print("No model and optimizer found. Creating a new one.")

    def plot_learning_progress(self):
        # Plot the learning progress.
        plt.plot(self.loss_history)
        plt.title('Plot of the List')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)
        plt.show()
