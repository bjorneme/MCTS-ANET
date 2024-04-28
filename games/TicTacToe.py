import numpy as np

# Class contining the logic for TicTacToe
class TicTacToe:
    def __init__(self):
        # Initialize the board and starting player
        self.board_size = 3
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1

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


        # Check is the move is valid
        if self.board[row, col] != 0:
            raise ValueError("Invalid move: position already taken")
        
        # Play the move
        self.board[row, col] = self.current_player

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
        # Check rows, columns, and diagonals for a win
        for i in range(self.board_size):
            if abs(sum(self.board[i, :])) == self.board_size:
                return self.board[i, 0]
            if abs(sum(self.board[:, i])) == self.board_size:
                return self.board[0, i]
        
        # Check diagonals
        if abs(sum(self.board.diagonal())) == self.board_size:
            return self.board[0, 0]
        if abs(sum(np.fliplr(self.board).diagonal())) == self.board_size:
            return self.board[0, self.board_size-1]
        
        return 0
    
    def is_game_over(self):
        return self.check_win() != 0 or not self.get_valid_moves()

    
    def display_board(self):
        # Print the board state
        symbols = {0: '.', 1: 'X', -1: 'O'}

        # Loop through each row in the board
        for row in self.board:
            print(" ".join(symbols[val] for val in row))

    def copy(self):
        # Create a deep copy of the game state
        new_game = TicTacToe()
        new_game.board = np.copy(self.board)
        new_game.current_player = self.current_player
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
