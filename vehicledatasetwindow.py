import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class VehicleDatasetWindow(Dataset):
    def __init__(self, data_files, window_size=1):
        # Load your data (states and controls) the same way you do now
        # Suppose you end up with arrays (or tensors) full_state, full_control
        # of shape (N, state_dim) and (N, control_dim).
        self.window_size = window_size
        
        # 1) Load the raw data
        #    e.g. full_state, full_control = load_data(...)
        full_state = None
        full_control = None
        for file_pair in data_files:
            # Read state in and normalize with z-scores
            state = pd.read_csv(file_pair[0], dtype=np.float32)
            full_state = torch.tensor(
                state.to_numpy(),
                dtype=torch.float32
            )

            # Read control in and normalize with z-scores
            control = pd.read_csv(file_pair[1], dtype=np.float32)
            full_control = torch.tensor(
                control.to_numpy(),
                dtype=torch.float32
            )
        
        # 2) Build input-output pairs
        self.inputs = []
        self.targets = []
        
        for i in range(window_size, len(full_state) - 1):
            # Gather the last `window_size` states/controls
            windowed_states = []
            windowed_controls = []
            for w in range(window_size):
                idx = i - w  # or i - w - 1, depending on indexing choice
                windowed_states.append(full_state[idx])
                windowed_controls.append(full_control[idx])
            
            # Flatten them out (or keep them separate if using an RNN)
            windowed_states = torch.cat(windowed_states, dim=0)
            windowed_controls = torch.cat(windowed_controls, dim=0)
            
            # Combine state+control for your final input
            # [S1, C1, S2, C2, S3, C3, S4, C4] for example
            X = torch.cat([windowed_states, windowed_controls], dim=0)
            
            # Next-step state as the target
            y = full_state[i+1]  # or the difference, your choice
            
            self.inputs.append(X)
            self.targets.append(y)
        
        self.inputs = torch.stack(self.inputs)
        self.targets = torch.stack(self.targets)

    def io_size(self):
        return self.data.shape[1], self.labels.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.labels[i]
