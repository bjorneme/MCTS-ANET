import torch
import torch.nn as nn
import torch.optim as optim

class ANET(nn.Module):
    def __init__(self):
        super(ANET, self).__init__()
        # Common layers
        self.fc1 = nn.Linear(9, 64)  # Assuming a flattened 3x3 Tic Tac Toe board
        self.fc2 = nn.Linear(64, 64)

        # Policy head
        self.policy_head = nn.Linear(64, 9)  # Output 9 action probabilities

        # Value head
        self.value_head = nn.Linear(64, 1)  # Output a single value estimate

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        policy = torch.softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))  # Output from -1 to 1 (win-loss scale)

        return policy, value