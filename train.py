import os
import json
import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, TensorDataset
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from nn import NN
from lstm import LSTM
from vehicledataset import VehicleDataset


class Trainer():
    def __init__(self, surf, space):
        """
        Params:
        surf (str): surface
        space (str): filename extension
        """

        # set only one thread for XGBoost
        os.environ["OMP_NUM_THREADS"] = "1"

        self.surf = surf
        self.space = space

        # set models to None
        self.nn_model = None
        self.xgb_model = None
        self.lstm_model = None

    def get_datasets(self, seq_len=1, is_lstm=False):
        """
        Get datasets related to surface.

        Params:
        seq_len (int) - length of sequence
        is_lstm (bool) - if it is lstm

        Returns:
        io (list) - list with input and output size
        train (list) - list with train data and labels
        test (list) - list with test data and labels
        val (list) - list with val data and labels
        features (list) - list of features
        """
        data_files = [
            (
                f"data/{self.surf}_data_state_{self.space}_1.csv",
                f"data/{self.surf}_data_control_{self.space}_1.csv"
            ),
            (
                f"data/{self.surf}_data_state_{self.space}_2.csv",
                f"data/{self.surf}_data_control_{self.space}_2.csv"
            ),
            (
                f"data/{self.surf}_data_state_{self.space}_3.csv",
                f"data/{self.surf}_data_control_{self.space}_3.csv"
            ),
            (
                f"data/{self.surf}_data_state_{self.space}_4.csv",
                f"data/{self.surf}_data_control_{self.space}_4.csv"
            ),
        ]

        # load dataset
        dataset = VehicleDataset(data_files, seq_len, is_lstm)

        # get input size and output size from the data
        input_size, output_size = dataset.io_size()

        # train, test, val ratios
        train_ratio = 0.64
        test_ratio = 0.2

        dataset_size = len(dataset)
        train_size = int(train_ratio * dataset_size)
        test_size = int(test_ratio * dataset_size)
        val_size = dataset_size - train_size - test_size

        # split dataset
        train_dataset, test_dataset, val_dataset = random_split(
            dataset,
            [train_size, test_size, val_size]
        )

        # get tensors of random split data
        train_data = torch.stack([example[0] for example in train_dataset])
        train_labels = torch.stack([example[1] for example in train_dataset])
        test_data = torch.stack([example[0] for example in test_dataset])
        test_labels = torch.stack([example[1] for example in test_dataset])
        val_data = torch.stack([example[0] for example in val_dataset])
        val_labels = torch.stack([example[1] for example in val_dataset])

        # mean and std dev of training data
        # dont want mean and std dev w test data bc data leakage
        mean_data = torch.mean(train_data, dim=0)
        std_dev_data = torch.std(train_data, dim=0)
        mean_labels = torch.mean(train_labels, dim=0)
        std_dev_labels = torch.std(train_labels, dim=0)

        # save mean and standard deviation for future use
        params_file = f"./models/{self.surf}"
        params_file += f"_params_{self.space}_{seq_len}.json"
        params = {
            "mean_data": mean_data.tolist(),
            "std_dev_data": std_dev_data.tolist(),
            "mean_labels": mean_labels.tolist(),
            "std_dev_labels": std_dev_labels.tolist()
        }

        # write to json file
        with open(params_file, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=4)

        # normalize all data
        train_data = (train_data - mean_data) / std_dev_data
        test_data = (test_data - mean_data) / std_dev_data
        val_data = (val_data - mean_data) / std_dev_data

        train_labels = (train_labels - mean_labels) / std_dev_labels
        test_labels = (test_labels - mean_labels) / std_dev_labels
        val_labels = (val_labels - mean_labels) / std_dev_labels

        return (
            [input_size, output_size],
            [train_data, train_labels],
            [test_data, test_labels],
            [val_data, val_labels],
            dataset.features
        )

    def train_nn(
        self, h_dim=32, act="ReLU", lr=0.001,
        batch_size=64, num_epochs=50
    ):
        """
        Train NN.

        Params:
        hdim (int) - number of neurons in hidden layer
        act (string) - activation function of hidden layer
        lr (int) - learning rate of neural network
        batch_size (int) - batch size
        num_epochs (int) - number of epochs to train for

        Returns:
        nn_train_losses (list) - list of training losses over epochs
        nn_val_losses (list) list of validation losses over epochs
        """

        # load in data
        io, train, test, val, features = self.get_datasets()

        # recreate datasets
        train_dataset = TensorDataset(train[0], train[1])
        test_dataset = TensorDataset(test[0], test[1])
        val_dataset = TensorDataset(val[0], val[1])

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True
        )

        # define nn_model loss function and optimizing
        self.nn_model = NN(io[0], h_dim, io[1], getattr(nn, act)())

        loss_fn = torch.nn.MSELoss()
        optimizer = optim.Adam(self.nn_model.parameters(), lr=lr)

        nn_train_losses, nn_val_losses = [], []
        t = tqdm(range(num_epochs), desc="Training...")

        for epoch in t:
            # train
            train_loss = 0
            for input, target in train_loader:
                # zero grads
                optimizer.zero_grad()

                # forward, loss, backprop
                output = self.nn_model(input)
                loss = loss_fn(output, target)
                loss.backward()

                # update params
                optimizer.step()

                train_loss += loss.item()

            # validation
            self.nn_model.eval()
            val_loss = 0
            with torch.no_grad():
                for input, target in val_loader:
                    output = self.nn_model(input)
                    val_loss += loss_fn(output, target).item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            nn_train_losses.append(train_loss)
            nn_val_losses.append(val_loss)

            # Create description string
            desc = f"Epoch {epoch+1}: "
            desc += f"Training Loss: {train_loss:.4f}, "
            desc += f"Validation Loss: {val_loss:.4f}"
            t.set_description(desc)

        # evaluate test set with nn
        self.nn_model.eval()
        test_loss = 0
        with torch.no_grad():
            for input, target in test_loader:
                output = self.nn_model(input)
                test_loss += loss_fn(output, target).item()

        test_loss /= len(test_loader)
        print(f"NN Test Loss (MSE): {test_loss}")

        return nn_train_losses, nn_val_losses

    def train_xgb(self, xgb_params=None, num_rounds=100):
        """
        Train XGB Regressor.

        Params:
        xgb_params (dict) - parameters of xgb regressor
        num_rounds (int) - number of total boosting rounds

        Returns:
        xgb_train_losses (list) - list of training losses over epochs
        xgb_val_losses (list) list of validation losses over epochs
        """

        # load in data
        io, train, test, val, features = self.get_datasets()

        # set xgb_params if None
        if xgb_params is None:
            xgb_params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "nthread": 1
            }

        # create train, val, and test dmatrices
        dtrain = xgb.DMatrix(train[0], label=train[1])
        dval = xgb.DMatrix(val[0], label=val[1])
        dtest = xgb.DMatrix(test[0])

        # train and val evals and track
        evals = [(dtrain, 'train'), (dval, 'eval')]
        evals_result = {}

        # keep track of training and validation losses
        xgb_train_losses, xgb_val_losses = [], []

        # tqdm for tracking and initialize model
        t = tqdm(range(num_rounds), desc="Training...")
        self.xgb_model = None

        for round in t:
            # train
            self.xgb_model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=1,
                evals=evals,
                evals_result=evals_result,
                xgb_model=self.xgb_model,
                verbose_eval=False
            )

            # get losses
            train_loss = evals_result['train']['rmse'][0]
            val_loss = evals_result['eval']['rmse'][0]

            xgb_train_losses.append(train_loss)
            xgb_val_losses.append(val_loss)

            # create description string
            desc = f"Round {round+1}: "
            desc += f"Training RMSE: {train_loss:.4f}, "
            desc += f"Validation RMSE: {val_loss:.4f}"
            t.set_description(desc)

        # evaluate test set with xgb
        pred_labels = self.xgb_model.predict(dtest)
        mse = mean_squared_error(test[1], pred_labels)

        print(f'XGB Test Loss (MSE): {mse}')

        return xgb_train_losses, xgb_val_losses

    def train_lstm(
        self, seq_len=10, h_dim=64, num_layers=2, lr=0.001,
        batch_size=64, num_epochs=50
    ):
        """
        Train LSTM model.

        Params:
        seq_len (int) - length of sequence
        h_dim (int) - number of neurons per layer for lstm
        num_layers (int) - number of layers in lstm layer

        Returns:
        xgb_train_losses (list) - list of training losses over epochs
        xgb_val_losses (list) list of validation losses over epochs
        """

        # load in data
        io, train, test, val, features = self.get_datasets(seq_len, True)

        # recreate datasets
        train_dataset = TensorDataset(train[0], train[1])
        test_dataset = TensorDataset(test[0], test[1])
        val_dataset = TensorDataset(val[0], val[1])

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True
        )

        # initialize model, loss, and optimizer
        self.lstm_model = LSTM(io[0], h_dim, io[1], num_layers)

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=lr)

        lstm_train_losses, lstm_val_losses = [], []
        t = tqdm(range(num_epochs), desc="Training...")

        # Training loop
        for epoch in t:
            # train
            train_loss = 0
            for input, target in train_loader:
                # zero grads
                optimizer.zero_grad()

                # forward, loss, backprop
                output = self.lstm_model(input)
                loss = loss_fn(output, target)
                loss.backward()

                # update params
                optimizer.step()

                train_loss += loss.item()

            # validation
            self.lstm_model.eval()
            val_loss = 0
            with torch.no_grad():
                for input, target in val_loader:
                    output = self.lstm_model(input)
                    val_loss += loss_fn(output, target).item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            lstm_train_losses.append(train_loss)
            lstm_val_losses.append(val_loss)

            # Create description string
            desc = f"Epoch {epoch+1}: "
            desc += f"Training Loss: {train_loss:.4f}, "
            desc += f"Validation Loss: {val_loss:.4f}"
            t.set_description(desc)

        # evaluate test set with nn
        self.lstm_model.eval()
        test_loss = 0
        with torch.no_grad():
            for input, target in test_loader:
                output = self.lstm_model(input)
                test_loss += loss_fn(output, target).item()

        test_loss /= len(test_loader)
        print(f"LSTM Test Loss (MSE): {test_loss}")

        return lstm_train_losses, lstm_val_losses

    def test_mean(self):
        """
        Mean value predictor testing.
        """

        # load in data
        io, train, test, val, features = self.get_datasets()

        # get test dataset to compare similarly
        test_dataset = TensorDataset(test[0], test[1])
        test_loader = DataLoader(
            test_dataset, batch_size=64, shuffle=True
        )

        mean_loss = 0
        mean_val = torch.mean(train[1], axis=0)
        with torch.no_grad():
            for input, target in test_loader:
                output = mean_val.expand_as(target)
                mean_loss += torch.n.MSELoss()(output, target).item()

        mean_loss /= len(test_loader)
        print(f"Mean Loss: {mean_loss}")

    def save_models(self):
        """
        Save the models if they exist.
        """
        # save models
        if self.nn_model:
            torch.save(
                self.nn_model, f"./models/nn_{self.surf}_{self.space}.pth"
            )
        if self.xgb_model:
            self.xgb_model.save_model(
                f"./models/xgb_{self.surf}_{self.space}.json"
            )
        if self.lstm_model:
            torch.save(
                self.lstm_model, f"./models/lstm_{self.surf}_{self.space}.pth"
            )

    def nn_feature_importance(self):
        """
        Print out all features and their importance.
        """

        # get nn_model weights from first layer
        weights = self.nn_model.fc1.weight.data.abs().numpy()

        # rank feature importance
        importance = dict(zip(self.features, weights.flatten()))

        # sort from most to least important
        sorted_features = sorted(
            importance.items(), key=lambda x: x[1], reverse=True
        )

        for feature, importance in sorted_features:
            print(f"{feature}: {importance:.4f}")

    def plot_losses(self, model, num_iters, train_losses, val_losses):
        """
        Plot training losses and validation losses over number of
        epochs or number of boosting rounds

        Params:
        num_iters (list) - number of iterations of training
        train_losses (list) - training losses
        val_losses (list) - validation losses
        """

        # plot losses
        plt.plot(num_iters, train_losses)
        plt.plot(num_iters, val_losses)
        plt.legend(["Training Error", "Validation Error"])
        plt.xlabel("Epochs")
        plt.ylabel("Mean Squared Error")
        plt.title(
            f"Losses of {model} Training on {self.surf.capitalize()} Data"
        )
        plt.show()
