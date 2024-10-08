import copy
import numpy as np

from games.DisjointSet import DisjointSet

# Class contining the logic for Hex
class Hex:
    def __init__(self, board_size):
        # Initialize the board and starting player
        self.board_size = board_size
        self.board = self.get_initial_board()
        self.current_player = 1

        # Initialize DisjointSet and top and bottom virtual nodes
        self.disjointset = DisjointSet(board_size**2 + 2)
        self.top_virtual = board_size ** 2
        self.bottom_virtual = board_size ** 2 + 1

        # Directions it is allowed to connect
        self.directions = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]

    def get_initial_board(self):
        return np.zeros((self.board_size, self.board_size), dtype=int)

    def pos_to_index(self, row, col):
        # Convert [row, col] to index
        return row*self.board_size + col
    
    def make_move(self, move):
        if isinstance(move, (list, tuple)) and len(move) == 2:
            row, col = move
        else:
            # Calculate coordinates
            row = move // self.board_size
            col = move % self.board_size
            
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
    
    def get_valid_moves_hot_encoded(self):
        # Return one hot encoded valid moves
        return [1 if cell == 0 else 0 for row in self.board for cell in row]


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
        # Symbols representing players
        symbols = {0: '.', 1: 'X', -1: 'O'}

        # Width printout
        width = 2 * self.board_size - 1

        # Create an output list of lists, initialized with spaces
        output = [[' ' for _ in range(width)] for _ in range(width)]

        # Map each element in the board to the new output list
        for i in range(self.board_size):
            for j in range(self.board_size):
                # Calculate new positions
                new_row = i + j
                new_col = self.board_size - 1 + i - j
                
                # Use symbols representing players
                output[new_row][new_col] = symbols[self.board[i][j]]

        # Print the rotated matrix
        for line in output:
            print(' '.join(line))

        print()

    def copy(self):
        # Create a deep copy of the game state
        new_game = Hex(self.board_size)
        new_game.board = np.copy(self.board)
        new_game.current_player = self.current_player
        new_game.disjointset = copy.deepcopy(self.disjointset)
        return new_game
    
    def get_winner(self):
        if self.is_game_over():
            # If player 1 wins. Return 1
            if self.check_win() == 1:
                return 1
            # If player -1 wins. Return -1
            if self.check_win() == -1:
                return -1
            return 0
        
    def reset(self):
        return Hex(self.board_size)