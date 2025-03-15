import json
import torch
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np

from vehicledataset import VehicleDataset

# set surface and version
surf = "ice"
ver = 3
window_size = 1

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
        
orig_dim = trial_data.size(1)

cur_window = []
for i in range(window_size):
    cur_window.append(trial_data[i])   # each is shape [orig_dim]
cur_window = torch.cat(cur_window, dim=0)

nn_outputs = []
with torch.no_grad():
    # We start at index = window_size-1 because our initial window covers steps 0..3
    for i in range(window_size - 1, len(trial_data)):
        # Forward pass. The net was trained with input_dim = 4*orig_dim
        # so we must pass a (1 x 4*orig_dim) tensor.
        out = nn_model(cur_window.unsqueeze(0))  # shape (1, output_dim)
        out = out.squeeze(0)                     # shape (output_dim,)

        # Save the output for future plotting or analysis
        nn_outputs.append(out)

        # hift window for the next iteration (if i+1 < len(trial_data))
        if i + 1 < len(trial_data):
            # Drop the oldest block of orig_dim, add the next row of trial_data
            cur_window = torch.cat([
                cur_window[orig_dim:],       # remove the first orig_dim
                trial_data[i + 1]           # add new step
            ], dim=0)
            
t_5ms = np.linspace(0, 0.005 * len(trial_data), len(trial_data))
x_ind = trial_dataset.features.index("vxCG_mps")
y_ind = trial_dataset.features.index("vyCG_mps")
yaw_rate_ind = trial_dataset.features.index("yawRate_radps")

trial_x = trial_data[:, x_ind] * std_dev_data[x_ind] + mean_data[x_ind]
trial_y = trial_data[:, y_ind] * std_dev_data[y_ind] + mean_data[y_ind]
trial_yaw_rate = trial_data[:, yaw_rate_ind] * std_dev_data[yaw_rate_ind] + mean_data[yaw_rate_ind]

nn_x = []
nn_y = []
nn_yaw_rate = []

for out in nn_outputs:
    real_x = out[x_ind] * std_dev_data[x_ind] + mean_data[x_ind]
    real_y = out[y_ind] * std_dev_data[y_ind] + mean_data[y_ind]
    real_yaw = out[yaw_rate_ind] * std_dev_data[yaw_rate_ind] + mean_data[yaw_rate_ind]

    nn_x.append(real_x.item())
    nn_y.append(real_y.item())
    nn_yaw_rate.append(real_yaw.item())


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
