import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class VehicleDataset(Dataset):
    def __init__(self, state_file, control_file):
        # Read state in and normalize with z-scores
        state = pd.read_csv(state_file, dtype=np.float32)
        state_norm = torch.tensor(
            StandardScaler().fit_transform(state),
            dtype=torch.float32
        )

        # Read control in and normalize with z-scores
        control = pd.read_csv(control_file, dtype=np.float32)
        control_norm = torch.tensor(
            StandardScaler().fit_transform(control),
            dtype=torch.float32
        )

        # Set values for data and labels
        # We are mapping each (state + control) to the next state
        self.data = torch.cat((state_norm, control_norm), dim=1)[:-1]
        self.labels = state_norm[1:]

        # Store feature names
        self.features = list(state.columns) + list(control.columns)

    def io_size(self):
        return self.data.shape[1], self.labels.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.labels[i]
