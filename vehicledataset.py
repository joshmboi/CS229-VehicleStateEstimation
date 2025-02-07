import pandas as pd
import numpy as np
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset


class VehicleDataset(Dataset):
    def __init__(self, state_file, control_file):
        self.data = pd.concat([
            pd.read_csv(state_file, dtype=np.float32),
            pd.read_csv(control_file, dtype=np.float32)
        ], axis=1)[:-1].reset_index(drop=True).transpose()
        self.labels = pd.read_csv(
            state_file,
            dtype=np.float32,
        )[1:].reset_index(drop=True).transpose()
        self.transform = v2.Compose([v2.ToDtype(torch.float32, scale=True)])

    def io_size(self):
        return self.data.shape[0], self.labels.shape[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data_item = torch.tensor(
            self.data.iloc[:, i].values,
            dtype=torch.float32
        )
        label_item = torch.tensor(
            self.labels.iloc[:, i].values,
            dtype=torch.float32
        )
        return self.transform(data_item), self.transform(label_item)
