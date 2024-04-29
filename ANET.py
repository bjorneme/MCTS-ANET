import torch
import torch.nn as nn

class ANET(nn.Module):
    def __init__(self):
        super(ANET, self).__init__()

        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(27, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Policy head (output probabilities for 9 actions)
        self.policy_head = nn.Linear(128, 9)

        # Value head (output a single value estimate)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared_layers(x)

        # Compute policy output
        policy = self.policy_head(x)

        # Compute value output and apply tanh activation
        value = torch.tanh(self.value_head(x))

        return policy, value

    
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