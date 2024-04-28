import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ANET(nn.Module):
    def __init__(self):
        super(ANET, self).__init__()

        # layers
        self.fc1 = nn.Linear(27, 64)
        self.fc2 = nn.Linear(64, 9)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def prepare_input(self, batch_states, players):
        new_batch_states = []

        for board_state, player in zip(batch_states, players):
            # Matrix 1: Positions occupied by Player 1
            player_1 = [[1 if cell == 1 else 0 for cell in row] for row in board_state]

            # Matrix 2: Positions occupied by Player -1
            player_2 = [[1 if cell == -1 else 0 for cell in row] for row in board_state]

            # Matrix 3: Current player's turn. 1 if player 1 and 0 if player -1
            current_turn = [[int(player == 1) for _ in row] for row in board_state]

            # Combine all states of a sample into a single list, flatten the list
            flattened_list = [cell for submatrix in [player_1, player_2, current_turn] for row in submatrix for cell in row]
            new_batch_states.append(flattened_list)

            # TODO: Change here for CNN

        # Return the prepared batch
        return torch.Tensor(new_batch_states)