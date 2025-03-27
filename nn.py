import torch.nn as nn


class NN(nn.Module):
    """
    Creates Simple Neural Network.

    Fields:
    input_dim (int) - number of inputs
    hidden_dim (int) - number of neurons in hidden layer
    output_dim (int) - number of outputs
    activation (func) - activation function
    """

    def __init__(self, input_dim, hidden_dim, output_dim, activation):
        super(NN, self).__init__()

        # Three fully connected layers for network
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = activation

    def forward(self, x):
        """
        Computes forward pass of model.

        Inputs:
        x - input vector
        """
        f1 = self.activation(self.fc1(x))
        return self.fc2(f1)
