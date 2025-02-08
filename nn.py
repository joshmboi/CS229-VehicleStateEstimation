import torch
import torch.nn as nn


class NN(nn.Module):
    """
    Creates Simple Neural Network.

    Fields:
    input_dim - number of inputs
    output_dim - number of outputs
    """

    def __init__(self, input_dim, output_dim):
        super(NN, self).__init__()

        # Three fully connected layers for network
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        """
        Computes forward pass of model.

        Inputs:
        x - input vector
        """
        f1 = torch.relu(self.fc1(x))
        f2 = torch.relu(self.fc2(f1))
        return self.fc3(f2)
