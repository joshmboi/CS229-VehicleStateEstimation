import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class VehicleDataset(Dataset):
    def __init__(self, data_files, seq_len=1, is_lstm=False):
        self.data = None
        self.labels = None
        self.seq_len = seq_len
        self.is_lstm = is_lstm

        for file_pair in data_files:
            # Read state in
            state = pd.read_csv(file_pair[0], dtype=np.float32)
            state_tensor = torch.tensor(
                state.to_numpy(),
                dtype=torch.float32
            )

            # Read control in
            control = pd.read_csv(file_pair[1], dtype=np.float32)
            control_tensor = torch.tensor(
                control.to_numpy(),
                dtype=torch.float32
            )

            # Combine state and control
            combined = torch.cat((state_tensor, control_tensor), dim=1)

            # Create sequences and concatenate to overall data and labels
            # We are mapping sequence of (state + control) to the next state
            for i in range(len(combined) - self.seq_len):
                seq = combined[i:i + self.seq_len]

                # unsqueeze if it is lstm
                if self.is_lstm:
                    seq = seq.unsqueeze(0)

                label = state_tensor[i + self.seq_len]
                if self.data is None:
                    self.data = seq
                    self.labels = label.unsqueeze(0)
                else:
                    self.data = torch.cat(
                        (self.data, seq), dim=0
                    )
                    self.labels = torch.cat(
                        (self.labels, label.unsqueeze(0)), dim=0
                    )

        # Store feature names
        self.features = list(state.columns) + list(control.columns)

    def io_size(self):
        # return diff shape if lstm
        if self.is_lstm:
            return self.data.shape[2], self.labels.shape[1]

        return self.data.shape[1], self.labels.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.labels[i]
