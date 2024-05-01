import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ANET(nn.Module):
    def __init__(self, config_path):
        super(ANET, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.resblock1 = ResidualBlock(64)
        self.resblock2 = ResidualBlock(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.resblock3 = ResidualBlock(128)
        self.resblock4 = ResidualBlock(128)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 49)  # 49 is the output size for a 7x7 Hex board

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.global_avg_pool(x)

        x = x.view(x.size(0), -1)  # Flatten the layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
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

            # Flatten for MLP
            if False:
                state = [cell for submatrix in [player_1, player_2, current_turn] for row in submatrix for cell in row]
            # Three channels for CNN
            elif True:
                state = [player_1, player_2, current_turn]
            
            new_batch_states.append(state)

        # Return the prepared batch
        return torch.Tensor(new_batch_states)