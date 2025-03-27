import os
import json
import optuna
import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, TensorDataset
from torch import nn
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from nn import NN
from vehicledataset import VehicleDataset

# set surface
surf = "dry"

# set hyperparams
batch_size = 64
num_epochs = 50

# set only one thread for XGBoost
os.environ["OMP_NUM_THREADS"] = "1"


def get_datasets(surf):
    data_files = [
        (
            f"data/{surf}_data_state_red_1.csv",
            f"data/{surf}_data_control_red_1.csv"
        ),
        (
            f"data/{surf}_data_state_red_2.csv",
            f"data/{surf}_data_control_red_2.csv"
        ),
        (
            f"data/{surf}_data_state_red_3.csv",
            f"data/{surf}_data_control_red_3.csv"
        ),
        (
            f"data/{surf}_data_state_red_4.csv",
            f"data/{surf}_data_control_red_4.csv"
        ),
    ]

    # load dataset
    dataset = VehicleDataset(data_files)

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
    params_file = f"./models/{surf}_params.json"
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


io, train, test, val, features = get_datasets(surf)

# recreate datasets
train_dataset = TensorDataset(train[0], train[1])
test_dataset = TensorDataset(test[0], test[1])
val_dataset = TensorDataset(val[0], val[1])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


def nn_obj(trial):
    hidden_dim1 = trial.suggest_int('hidden_dim1', 16, 128)
    hidden_dim2 = trial.suggest_int('hidden_dim2', 16, 128)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    activation_name1 = trial.suggest_categorical(
        'activation',
        ['ReLU', 'Tanh', 'Sigmoid']
    )
    activation_name2 = trial.suggest_categorical(
        'activation',
        ['ReLU', 'Tanh', 'Sigmoid']
    )
    activation1 = getattr(nn, activation_name1)()
    activation2 = getattr(nn, activation_name2)()

    # Model, loss, optimizer
    model = NN(
        io[0], hidden_dim1, hidden_dim2, io[1], activation1, activation2
    )
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        for input, target in train_loader:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

    # Validation
    mse = 0
    model.eval()
    with torch.no_grad():
        for input, target in test_loader:
            output = model(input)
            mse += loss_fn(output, target).item()
    mse /= len(test_loader)
    return mse


nn_study = optuna.create_study(direction='minimize')
nn_study.optimize(nn_obj, n_trials=50)

# create train, val, and test dmatrices
dtrain = xgb.DMatrix(train[0], label=train[1])
dval = xgb.DMatrix(val[0], label=val[1])
dtest = xgb.DMatrix(test[0])


def xgb_obj(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'nthread': 1,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float(
            'learning_rate', 0.01, 0.3, log=True
        ),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }

    evals = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=evals,
        verbose_eval=False
    )

    pred_labels = model.predict(dtest)
    mse = mean_squared_error(test[1], pred_labels)

    return mse


xgb_study = optuna.create_study(direction='minimize')
xgb_study.optimize(xgb_obj, n_trials=50)

# best hyperparameters
print("Best nn hyperparameters:", nn_study.best_params)
print("Best xgb hyperparameters:", xgb_study.best_params)
