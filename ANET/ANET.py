import json
import torch
import torch.nn as nn
from Anet.ResidualBlock import ResidualBlock

class ANET(nn.Module):
    def __init__(self, config_path):
        super(ANET, self).__init__()
        self.layers = self.create_layers(config_path)

    def create_layers(self, config_path):
        with open(config_path, 'r') as file:
            self.config = json.load(file)

        # Create the layers
        layers = []
        for layer in self.config["layers"]:
            layer_type = layer["type"]

            # Linear layer
            if layer_type == "Linear":
                layers.append(nn.Linear(layer["input"], layer["output"]))

            # Batch norm 1D
            elif layer_type == "BatchNorm1d":
                layers.append(nn.BatchNorm1d(layer["num_features"]))

            # Batch norm 2D
            elif layer_type == "BatchNorm2d":
                layers.append(nn.BatchNorm2d(layer["num_features"]))

            # Activation functions
            elif layer_type == "ReLU":
                layers.append(nn.ReLU())
            elif layer_type == "Sigmoid":
                layers.append(nn.Sigmoid())
            elif layer_type == "Tanh":
                layers.append(nn.Tanh())

            # Convolutional layer
            elif layer_type == "Conv2d":
                layers.append(nn.Conv2d(layer["input"], layer["output"], layer["kernel"], layer["stride"], layer["padding"]))

            # Add Residual block
            elif layer_type == "ResidualBlock":
                layers.append(ResidualBlock(layer["channels"]))

            # Max Pooling
            elif layer_type == "MaxPool2d":
                layers.append(nn.MaxPool2d(layer["kernel"], stride=layer.get("stride", 2)))

            # Dropout layer
            elif layer_type == "Dropout":
                layers.append(nn.Dropout(layer["p"]))

            # Flatten layer
            elif layer_type == "Flatten":
                layers.append(nn.Flatten())

        return nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass
        x = self.layers(x)
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
            if self.config["type_network"] == "MLP":
                state = [cell for submatrix in [player_1, player_2, current_turn] for row in submatrix for cell in row]
            # Three channels for CNN
            elif self.config["type_network"] == "CNN":
                state = [player_1, player_2, current_turn]
            
            new_batch_states.append(state)

        # Return the prepared batch
        return torch.Tensor(new_batch_states)