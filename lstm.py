import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    Creates Long Short Term Memory Network.

    Fields:
    input_dim (int) - number of inputs
    hidden_dim (int) - number of neurons in hidden layer for lstm
    output_dim (int) - number of outputs
    num_layers (int) - number of layers in lstm
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTM, self).__init__()

        # sets hyperparams
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)

        # output
        out, _ = self.lstm(x, (h0, c0))

        # last timestep
        return self.fc(out[:, -1, :])
