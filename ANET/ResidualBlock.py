import torch.nn as nn
import torch.nn.functional as F

# Residual Block: Used to avoid vanishing gradient and allow for deeper architectures
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        # The residual block
        self.res_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        # Forward pass
        return F.relu(self.res_block(x) + x)