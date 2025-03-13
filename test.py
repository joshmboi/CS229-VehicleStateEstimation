import json
import torch
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np

from vehicledataset import VehicleDataset

# set surface and version
surf = "wet"
ver = 4

# get params
params = None
with open(f"./models/{surf}_params.json", "r") as f:
    params = json.load(f)

# load models
nn_model = torch.load(f"./models/nn_{surf}.pth", weights_only=False)
nn_model.eval()
# xgb_model = xgb.Booster().load_model(f"./models/xgb_{surf}.ubj")

# data files for trial
data_files = [
    (f"data/{surf}_data_state{ver}.csv", f"data/{surf}_data_control{ver}.csv"),
]

# load trail data
trial_dataset = VehicleDataset(data_files)

# normalize data
trial_data = torch.stack([example[0] for example in trial_dataset])
trial_labels = torch.stack([example[1] for example in trial_dataset])

mean_data = torch.tensor(params["mean_data"])
std_dev_data = torch.tensor(params["std_dev_data"])
mean_labels = torch.tensor(params["mean_labels"])
std_dev_labels = torch.tensor(params["std_dev_labels"])

trial_data = (trial_data - mean_data) / std_dev_data
trial_labels = (trial_labels - mean_labels) / std_dev_labels

# seed input and neural net outputs array
trial_input = trial_data[0]
nn_outputs = trial_input.unsqueeze(0)

with torch.no_grad():
    for i in range(len(trial_data) - 1):
        trial_input = torch.cat(
            (nn_model(trial_input), trial_data[i + 1][10:]),
        )
        # print(torch.Tensor.size(nn_outputs))
        # print(torch.Tensor.size(trial_input.unsqueeze(0)))
        nn_outputs = torch.cat((nn_outputs, trial_input.unsqueeze(0)), dim=0)

t_5ms = np.linspace(0, 0.005 * len(trial_data), len(trial_data))
x_ind = trial_dataset.features.index("vxCG_mps")
y_ind = trial_dataset.features.index("vyCG_mps")
yaw_rate_ind = trial_dataset.features.index("yawRate_radps")

trial_x = trial_data[:, x_ind] * std_dev_data[x_ind] + mean_data[x_ind]
trial_y = trial_data[:, y_ind] * std_dev_data[y_ind] + mean_data[y_ind]
trial_yaw_rate = trial_data[:, yaw_rate_ind] * std_dev_data[yaw_rate_ind] + mean_data[yaw_rate_ind]

nn_x = nn_outputs[:, x_ind] * std_dev_data[x_ind] + mean_data[x_ind]
nn_y = nn_outputs[:, y_ind] * std_dev_data[y_ind] + mean_data[y_ind]
nn_yaw_rate = nn_outputs[:, yaw_rate_ind] * std_dev_data[yaw_rate_ind] + mean_data[yaw_rate_ind]


plt.figure()
plt.plot(t_5ms, trial_x, linestyle="-")
plt.plot(t_5ms, nn_x, linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("X-velocity (m/s)")
plt.title(f"Predictions of NN from Initial State for {surf.capitalize()}")
plt.legend(["Actual Trial Data", "NN Output"])

plt.figure()
plt.plot(t_5ms, trial_y, linestyle="-")
plt.plot(t_5ms, nn_y, linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Y-velocity (m/s)")
plt.title(f"Predictions of NN from Initial State for {surf.capitalize()}")
plt.legend(["Actual Trial Data", "NN Output"])

plt.figure()
plt.plot(t_5ms, trial_yaw_rate, linestyle="-")
plt.plot(t_5ms, nn_yaw_rate, linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Yaw Rate (rad/s)")
plt.title(f"Predictions of NN from Initial State for {surf.capitalize()}")
plt.legend(["Actual Trial Data", "NN Output"])

plt.show()
