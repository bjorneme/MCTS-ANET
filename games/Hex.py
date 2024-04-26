import copy
import numpy as np

from games.DisjointSet import DisjointSet

# Class contining the logic for Hex
class Hex:
    def __init__(self, board_size):
        # Initialize the board and starting player
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1

        # Initialize DisjointSet and top and bottom virtual nodes
        self.disjointset = DisjointSet(board_size**2 + 2)
        self.top_virtual = board_size ** 2
        self.bottom_virtual = board_size ** 2 + 1

        # Directions it is allowed to connect
        self.directions = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]

    def pos_to_index(self, row, col):
        # Convert [row, col] to index
        return row*self.board_size + col
    
    def make_move(self, move):
        row, col = move
        index = self.pos_to_index(row, col)

        # Check is the move is valid
        if self.board[row, col] != 0:
            raise ValueError("Invalid move: position already taken!")
        
        # Play the move
        self.board[row, col] = self.current_player

        # Loop through all possible directions to check for neighbours
        for delta_row, delta_col in self.directions:

            # Calculate row and coloum for neighbour cells
            neighbor_row = row + delta_row
            neighbor_col = col + delta_col

            # Check that neighbour cell is within the board
            if 0 <= neighbor_row < self.board_size and 0 <= neighbor_col < self.board_size:

                # Check if neighbour cell is occupied by same player
                if self.board[neighbor_row][neighbor_col] == self.current_player:

                    # Convert the position of neighbour cell to index
                    neighbor_index = self.pos_to_index(neighbor_row, neighbor_col)

                    # Connect the current cell with neighbour cell in DisjointSet
                    self.disjointset.union(index, neighbor_index)

        # Connect to virtual nodes. Player 1 tries to connect from top to bottom
        if self.current_player == 1:

            # If move is on the top row, connect to top virtual node
            if row == 0:
                self.disjointset.union(index, self.top_virtual)

            # If move is on the bottom row, connect to bottom virtual node
            if row == self.board_size - 1:
                self.disjointset.union(index, self.bottom_virtual)

        # Player -1 tries to connect from left to right
        if self.current_player == -1:

            # If move is on the left column, connect to top virtual node
            if col == 0:
                self.disjointset.union(index, self.top_virtual)

            # If move is on the right column, connect to bottom virtual node
            if col == self.board_size - 1:
                self.disjointset.union(index, self.bottom_virtual)

        # Switch players
        self.current_player *= -1

    def get_valid_moves(self):
        valid_moves = []

        # Loop through each row and column on the board
        for row in range(self.board_size):
            for col in range(self.board_size):

                # Check if the current cell is empty
                if self.board[row][col] == 0:

                    # If the cell is empty, append its coordinates [row, col] to the list of valid moves
                    valid_moves.append([row,col])

        # Return all valid moves
        return valid_moves

    def check_win(self):
        # Check if any player has connected their edges
        if self.disjointset.find(self.top_virtual) == self.disjointset.find(self.bottom_virtual):
            
            # Return the winner
            return self.current_player * -1
        
        # If no winner found, return 0
        return 0
    
    def is_game_over(self):
        # The game is over of there is no winner or no valid moves
        if self.check_win() != 0 or not self.get_valid_moves():
            return True
        return False
    
    def display_board(self):
        # Print the board state
        symbols = {0: '.', 1: 'X', -1: 'O'}

        # Loop through each row in the board
        for row_index, row in enumerate(self.board):

            # Add a space extra for each row
            row_display = ' ' * row_index

            # Add symbols for each cell in the row
            for cell in row:
                row_display += symbols[cell] + " "

            # Print the formatted row
            print(row_display)

    def copy(self):
        # Create a deep copy of the game state
        new_game = Hex(self.board_size)
        new_game.board = np.copy(self.board)
        new_game.current_player = self.current_player
        new_game.disjointset = copy.deepcopy(self.disjointset)
        return new_game
    
    def evaluate_win_loss_for_player(self, player):
        # Evaluate if the specified player has won or lost
        if self.is_game_over():

            # Return 1 if player has won
            if self.check_win() == player:
                return 1
            
            # Return -1 if player has lost
            elif self.check_win() == -player:
                return -1