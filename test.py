import os
import json
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from vehicledataset import VehicleDataset


class Tester:
    def __init__(self, surf, space, ver=4):
        # set number of threads to 1 for XGBoost
        os.environ["OMP_NUM_THREADS"] = "1"

        self.surf = surf
        self.space = space
        self.ver = ver

        self.get_params()
        self.load_models()

    def get_params(self, seq_len=1, is_lstm=False):
        """
        Get mean and std dev of each parameter.

        Params:
        seq_len (int) - sequence length for parameter file

        Returns:
        params (dict) - dict of means and std devs
        """

        # get reduced state params
        params = None
        with open(
            f"./models/{self.surf}_params_{self.space}_{seq_len}.json", "r"
        ) as f:
            params = json.load(f)

        return params

    def load_models(self):
        """
        Load in models.
        """

        # load models
        self.nn_model = torch.load(
            f"./models/nn_{self.surf}_{self.space}.pth"
        )
        self.nn_model.eval()

        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(
            f"./models/xgb_{self.surf}_{self.space}.json"
        )

        self.lstm_model = torch.load(
            f"./models/lstm_{self.surf}_{self.space}.pth"
        )
        self.lstm_model.eval()

    def get_data(self, seq_len=1, is_lstm=False):
        """
        Load in data for sample trajectory and reduced state.

        Params:
        seq_len (int) - sequence length
        is_lstm (bool) - if is for lstm

        Returns:
        traj_dataset (dataset) - actual trajectory dataset
        model_dataset (dataset) - model dataset
        traj (list) - list of trajectory data and labels
        model (list) - list of model data and labels
        data (list) - list of data mean and std_dev
        labels (list) - list of labels mean and std_dev
        state_size (int) - size of the state
        """

        # get params for seq_len
        params = self.get_params(seq_len, is_lstm)

        # data files for sample trajectory
        data_files = [
            (
                f"data/{self.surf}_data_state{self.ver}.csv",
                f"data/{self.surf}_data_control{self.ver}.csv"
            )
        ]

        # load trajectory data
        traj_dataset = VehicleDataset(data_files, seq_len, is_lstm)

        # datafiles for reduced state model
        data_files_model = [
            (
                f"data/{self.surf}_data_state_{self.space}_{self.ver}.csv",
                f"data/{self.surf}_data_control_{self.space}_{self.ver}.csv"
            ),
        ]

        # load model data
        model_dataset = VehicleDataset(data_files_model, seq_len, is_lstm)

        # stack trajectory data
        traj_data = torch.stack(
            [example[0] for example in traj_dataset]
        )
        traj_labels = torch.stack(
            [example[1] for example in traj_dataset]
        )

        # normalize data
        data = torch.stack([example[0] for example in model_dataset])
        labels = torch.stack([example[1] for example in model_dataset])

        mean_data = torch.tensor(params["mean_data"])
        std_dev_data = torch.tensor(params["std_dev_data"])
        mean_labels = torch.tensor(params["mean_labels"])
        std_dev_labels = torch.tensor(params["std_dev_labels"])

        model_data = (data - mean_data) / std_dev_data
        model_labels = (labels - mean_labels) / std_dev_labels

        # get size of the state
        state_size = labels.shape[1]

        return (
            traj_dataset, model_dataset,
            [traj_data, traj_labels],
            [model_data, model_labels],
            [mean_data, std_dev_data],
            [mean_labels, std_dev_labels],
            state_size
        )

    def nn_pred(self):
        """
        Predict with neural network.

        Returns:
        nn_outputs (tensor) - tensor of outputs
        """

        # load in data
        _, _, traj, model, _, _, state_size = self.get_data()

        # seed input and neural net outputs array
        nn_input = model[0][0]
        nn_outputs = nn_input.unsqueeze(0)

        with torch.no_grad():
            for i in range(len(traj[0]) - 1):
                nn_input = torch.cat(
                    (
                        self.nn_model(nn_input),
                        model[0][i + 1][state_size:]
                    ),
                )
                nn_outputs = torch.cat(
                    (nn_outputs, nn_input.unsqueeze(0)), dim=0
                )

        return nn_outputs

    def xgb_pred(self):
        """
        Predict with xgb regressor.

        Returns:
        xgb_outputs (tensor) - tensor of outputs
        """

        # load in data
        _, _, traj, model, _, _, state_size = self.get_data()

        # seed input and xgb outputs array
        xgb_input = model[0][0].numpy()
        xgb_outputs = [xgb_input.copy()]

        for i in range(len(traj[0]) - 1):
            # convert to dmat and predict
            dinput = xgb.DMatrix(xgb_input.reshape(1, -1))

            xgb_output = self.xgb_model.predict(dinput)
            xgb_input = np.concatenate(
                (
                    xgb_output,
                    model[0][i + 1][state_size:]
                    .unsqueeze(0).numpy()
                ),
                axis=1
            )
            xgb_outputs.append(xgb_input.copy())

        xgb_outputs = torch.tensor(np.vstack(xgb_outputs))

        return xgb_outputs

    def lstm_pred(self, seq_len=10):
        """
        Predict with lstm model.

        Params:
        seq_len (int) - length of sequence

        Returns:
        lstm_outputs (tensor) - tensor of outputs
        """

        # load in data
        _, _, traj, model, _, _, state_size = self.get_data(seq_len, True)

        # seed input and neural net outputs array
        lstm_input = model[0][0].unsqueeze(0)
        lstm_outputs = lstm_input

        with torch.no_grad():
            for i in range(len(traj[0]) - 1):
                lstm_output = torch.hstack(
                    (
                        self.lstm_model(lstm_input),
                        model[0][i + 1][seq_len - 1][state_size:].unsqueeze(0)
                    )
                )
                lstm_input = torch.cat(
                    (lstm_input, lstm_output.unsqueeze(0)), dim=1
                )[:, -seq_len:, :]
                lstm_outputs = torch.cat(
                    (lstm_outputs, lstm_input), dim=0
                )

        return lstm_outputs

    def plot_traj(self, filename=None, seq_len=10):
        """
        Plot actual and model predicted trajectories
        """
        # load in data
        t_dset, m_dset, traj, model, data, labels, _ = self.get_data()
        _, _, _, _, data_seq, labels_seq, _ = self.get_data(seq_len, True)

        # get proper params

        # get x and y position of actual trajectory
        x_ind = t_dset.features.index("posE_m")
        y_ind = t_dset.features.index("posN_m")
        traj_yaw_ind = t_dset.features.index("yawAngle_rad")

        vx_ind = m_dset.features.index("vxCG_mps")
        vy_ind = m_dset.features.index("vyCG_mps")
        yaw_ind = m_dset.features.index("yawRate_radps")

        traj_x = traj[0][:len(traj[0]) - seq_len, x_ind]
        traj_y = traj[0][:len(traj[0]) - seq_len, y_ind]
        traj_yaw = traj[0][:len(traj[0]) - seq_len, traj_yaw_ind]

        nn_outputs = self.nn_pred()
        xgb_outputs = self.xgb_pred()
        lstm_outputs = self.lstm_pred(seq_len)

        nn_vx = (
                nn_outputs[:, vx_ind] *
                data[1][vx_ind] +
                data[0][vx_ind]
        )
        nn_vy = (
                nn_outputs[:, vy_ind] *
                data[1][vy_ind] +
                data[0][vy_ind]
        )
        nn_yaw_rate = (
                nn_outputs[:, yaw_ind] *
                data[1][yaw_ind] +
                data[0][yaw_ind]
        )

        xgb_vx = (
                xgb_outputs[:, vx_ind] *
                data[1][vx_ind] +
                data[0][vx_ind]
        )
        xgb_vy = (
                xgb_outputs[:, vy_ind] *
                data[1][vy_ind] +
                data[0][vy_ind]
        )
        xgb_yaw_rate = (
                xgb_outputs[:, yaw_ind] *
                data[1][yaw_ind] +
                data[0][yaw_ind]
        )

        lstm_vx = (
                lstm_outputs[:, 0, vx_ind] *
                data_seq[1][0, vx_ind] +
                data_seq[0][0, vx_ind]
        )
        lstm_vy = (
                lstm_outputs[:, 0, vy_ind] *
                data_seq[1][0, vy_ind] +
                data_seq[0][0, vy_ind]
        )
        lstm_yaw_rate = (
                lstm_outputs[:, 0, yaw_ind] *
                data_seq[1][0, yaw_ind] +
                data_seq[0][0, yaw_ind]
        )

        nn_x = np.copy(traj_x)
        nn_y = np.copy(traj_y)
        nn_yaw = np.copy(traj_yaw)

        xgb_x = np.copy(traj_x)
        xgb_y = np.copy(traj_y)
        xgb_yaw = np.copy(traj_yaw)

        lstm_x = np.copy(traj_x)
        lstm_y = np.copy(traj_y)
        lstm_yaw = np.copy(traj_yaw)

        for t in range(1, len(traj_x)):
            nn_yaw[t] = nn_yaw[t - 1] + nn_yaw_rate[t - 1] * 0.01
            nn_x[t] = (
                    nn_x[t - 1] - nn_vx[t - 1] * 0.01 * math.sin(nn_yaw[t - 1])
                    - nn_vy[t - 1] * 0.01 * math.cos(nn_yaw[t - 1])
            )
            nn_y[t] = (
                    nn_y[t - 1] + nn_vx[t - 1] * 0.01 * math.cos(nn_yaw[t - 1])
                    - nn_vy[t - 1] * 0.01 * math.sin(nn_yaw[t - 1])
            )

            xgb_yaw[t] = xgb_yaw[t - 1] + xgb_yaw_rate[t - 1] * 0.01
            xgb_x[t] = (
                    xgb_x[t - 1] - xgb_vx[t - 1] * 0.01 * math.sin(xgb_yaw[t - 1])
                    - xgb_vy[t - 1] * 0.01 * math.cos(xgb_yaw[t - 1])
            )
            xgb_y[t] = (
                    xgb_y[t - 1] + xgb_vx[t - 1] * 0.01 * math.cos(xgb_yaw[t - 1])
                    - xgb_vy[t - 1] * 0.01 * math.sin(xgb_yaw[t - 1])
            )

            lstm_yaw[t] = lstm_yaw[t - 1] + lstm_yaw_rate[t - 1] * 0.01
            lstm_x[t] = (
                    lstm_x[t - 1] - lstm_vx[t - 1] * 0.01 * math.sin(lstm_yaw[t - 1])
                    - lstm_vy[t - 1] * 0.01 * math.cos(lstm_yaw[t - 1])
            )
            lstm_y[t] = (
                    lstm_y[t - 1] + lstm_vx[t - 1] * 0.01 * math.cos(lstm_yaw[t - 1])
                    - lstm_vy[t - 1] * 0.01 * math.sin(lstm_yaw[t - 1])
            )

        plt.plot(traj_x, traj_y, linestyle="-")
        plt.plot(nn_x, nn_y, linestyle="--")
        plt.plot(xgb_x, xgb_y, linestyle=":")
        plt.plot(lstm_x, lstm_y, linestyle="-.")
        plt.xlabel("Position (m)")
        plt.ylabel("Position (m)")
        plt.title(
            f"Predictions from Initial State for {self.surf.capitalize()}"
        )
        plt.legend(["Actual Trial Data", "NN Output", "XGB Output"])

        if filename:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close()
