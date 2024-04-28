import copy
import numpy as np
from MCTS import MCTS, MCTSNode
from ReplayBuffer import ReplayBuffer
from games.Hex import Hex
from games.TicTacToe import TicTacToe


class MCTSSystem:
    def __init__(self):
        self.replay_buffer = ReplayBuffer()

    def self_play(self):
        # For counting win/ draw/ loss
        win, draw, loss = 0, 0, 0

        # Iterate over all the episodes
        for episode in range(1):

            # Initialize the game
            state_manager = TicTacToe()

            # Initialize MCTS with the current game state
            root_node = MCTSNode(1.41, state_manager)
            mcts = MCTS(root_node, 3, 9)

            # Alternate the starting player
            if(episode % 2 == 0):
                state_manager.current_player = -1

            buffer = []

            # Play until game is over
            while not state_manager.is_game_over():

                # Execute the MCTS search. Get probabilities for possible actions
                action_probs = mcts.run_simulation(10000)
                board_state = copy.deepcopy(state_manager.board)

                # Select the best action based on the action probabilities
                best_action = np.argmax(action_probs)

                # Save to a buffer, before loading into replay buffer
                buffer.append((board_state, action_probs, state_manager.current_player))

                # Execute the best action
                state_manager.make_move(best_action)

                # Update MCTS root to the new state, and prune
                mcts.update_root(best_action)
                
            # Update win/draw/loss counts based on game outcome
            result = state_manager.get_winner()
            if result == 1:
                win += 1
            elif result == 0:
                draw += 1
            else:
                loss += 1
            print(f"Episode {episode}")

            # Append game to replay buffer
            for board_state, action_probs, player in buffer:
                self.replay_buffer.push_to_buffer(board_state, action_probs, result, player)

        # Print final results
        print("Final Results:")
        print(f"Wins: {win}, Draws: {draw}, Losses: {loss}")


    def run_system(self):
        # Step 1: Do selfplay
        self.self_play()


system = MCTSSystem()
system.run_system()