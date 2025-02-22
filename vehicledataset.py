import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class VehicleDataset(Dataset):
    def __init__(self, data_files):
        self.data = None
        self.labels = None

        for file_pair in data_files:
            # Read state in and normalize with z-scores
            state = pd.read_csv(file_pair[0], dtype=np.float32)
            state_tensor = torch.tensor(
                state.to_numpy(),
                dtype=torch.float32
            )

            # Read control in and normalize with z-scores
            control = pd.read_csv(file_pair[1], dtype=np.float32)
            control_tensor = torch.tensor(
                control.to_numpy(),
                dtype=torch.float32
            )

            # Set values for data and labels
            # We are mapping each (state + control) to the next state
            data = torch.cat((state_tensor, control_tensor), dim=1)[:-1]
            labels = control_tensor[1:]

            # Concatenate to overall data and labels
            if self.data is not None:
                self.data = torch.cat((self.data, data), dim=0)
                self.labels = torch.cat((self.labels, labels), dim=0)
            else:
                self.data = data
                self.labels = labels

        # Means and Standard Deviations for future reconstruction of values
        self.data_means = torch.mean(self.data, dim=0)
        self.data_std_dev = torch.std(self.data, dim=0)
        self.labels_means = torch.mean(self.labels, dim=0)
        self.labels_std_dev = torch.std(self.labels, dim=0)

        # Normalized data and labels
        self.data = (
            self.data - self.data_means
        ) / (self.data_std_dev + 1e-8)
        self.labels = (
            self.labels - self.labels_means
        ) / (self.labels_std_dev + 1e-8)

        # Store feature names
        self.features = list(state.columns) + list(control.columns)

    def io_size(self):
        return self.data.shape[1], self.labels.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.labels[i]
