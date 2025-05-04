import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A basic residual block with two convolutional layers and a shortcut connection.
    Used to enable deeper networks without vanishing gradients, inspired by ResNet.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut to match dimensions if needed
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        Forward pass of the residual block.
        Applies two convolutional layers and adds the shortcut.
        """
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class StrongChessNet(nn.Module):
    """
    A deep convolutional neural network for chess, based on ResNet-style blocks.
    Outputs a policy over all legal moves and a scalar value estimate for board evaluation.
    """
    def __init__(self):
        super().__init__()

        # Input: (18, 8, 8)
        self.initial = nn.Conv2d(18, 64, kernel_size=3, padding=1)

        # Residual blocks with increasing channels and downsampling
        self.res1 = ResidualBlock(64, 128, stride=2)   # 8x8 → 4x4
        self.res2 = ResidualBlock(128, 256, stride=2)  # 4x4 → 2x2
        self.res3 = ResidualBlock(256, 512, stride=1)  # keep 2x2
        self.res4 = ResidualBlock(512, 256, stride=1)  # compress
        self.res5 = ResidualBlock(256, 128, stride=1)  # compress again

        self.dropout = nn.Dropout(p=0.3)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2 * 2 * 2, 4672)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2 * 2, 256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Forward pass of the network.
        Returns:
            policy: a 4672-dimensional vector of raw Q-values or logits.
            value: a scalar in [-1, 1] representing position favourability.
        """
        x = F.relu(self.initial(x))   # (64, 8, 8)
        x = self.res1(x)             # (128, 4, 4)
        x = self.res2(x)             # (256, 2, 2)
        x = self.res3(x)             # (512, 2, 2)
        x = self.res4(x)             # (256, 2, 2)
        x = self.res5(x)             # (128, 2, 2)

        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
