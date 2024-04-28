import math
import random

import numpy as np
from games.Hex import Hex
from games.TicTacToe import TicTacToe

# MCTSNode: node in the tree
class MCTSNode:
    def __init__(self, c, current_state, move=None, parent=None):
        self.current_state = current_state # State of the game at this node
        self.move = move # Move that led to this node
        self.parent = parent # Parent to this node
        self.children = [] # This nodes children
        self.visits = 0 # Number of times this node was visited during search
        self.value = 0.0 # Estimate how god this node is
        self.untried_moves = current_state.get_valid_moves() # Moves available and not tried in from this state
        self.c = c # Exploration constant

    def select_best_child(self):
        # Player 1 is maximizing
        if self.current_state.current_player == 1:
            return max(self.children, key=lambda child: child.value / (child.visits + 1) + self.c * math.sqrt(math.log(self.visits) / (1 + child.visits)))
        # Player 2 is minimizing
        else:
            return min(self.children, key=lambda child: child.value / (child.visits + 1) - self.c * math.sqrt(math.log(self.visits) / (1 + child.visits)))

    
    def node_expansion(self):
        # Create a new children
        if self.untried_moves:
            move = random.choice(list(self.untried_moves))
            self.untried_moves.remove(move)
            new_state = self.current_state.copy()
            new_state.make_move(move)
            new_child_node = MCTSNode(self.c,new_state,move, self)
            self.children.append(new_child_node)
            return new_child_node

    def simulation(self):
        simulation_state = self.current_state.copy()

        # Simulate until the game reaches a end state
        while not simulation_state.is_game_over():
            possible_moves = simulation_state.get_valid_moves()
            move = random.choice(possible_moves)
            simulation_state.make_move(move)
            
        # Return the evaluation of the end state
        return simulation_state.get_winner()

    def backpropagate(self, result):
        self.visits += 1
        self.value += result

        # Recursively update parents
        if self.parent: 
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        # Return True if node is fully expanded
        return len(self.untried_moves) == 0

# MCTS: the monte carlo tree
class MCTS:
    def __init__(self, root, board_size, total_actions):
        self.root = root # The root
        self.board_size = board_size # Board size
        self.total_actions = total_actions # Total actions

    def tree_search(self, node):
        # Search until node with untried moves is found
        while node.is_fully_expanded() and node.children != []:
            node = node.select_best_child()

        # Return the node to explore
        return node


    def run_simulation(self, num_search):
        for _ in range(num_search):
            node = self.root

            # Step 1: Selection
            node = self.tree_search(node)

            # Backpropagate if node found is terminal node
            if node.current_state.is_game_over():
                result = node.current_state.get_winner()
                node.backpropagate(result)
            else:
                # Step 2: Expansion
                node.node_expansion()

                # Step 3: Simulation
                result = node.simulation()

                # Step 4: Backpropagation
                node.backpropagate(result)

        # Return action probabilities
        return self.get_action_probabilities()
    
    def get_action_probabilities(self):
        # Calculate action probabilities for each move
        action_probs = np.zeros(self.total_actions)
        total_visits = sum(child.visits for child in self.root.children)
        for child in self.root.children:
            action_probs[self.pos_to_index(child.move)] = child.visits/ total_visits
        return action_probs
    
    def update_root(self, best_action):
        # Update MCTS root to the new state, and prune
        for child in self.root.children:
            if self.pos_to_index(child.move) == best_action:
                self.root = child
                self.root.parent = None

    def pos_to_index(self, pos):
        # If pos is represented as 2D board. Return index in 1D format
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            return pos[0] * self.board_size + pos[1]
        return pos
