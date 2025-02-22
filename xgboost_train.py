import torch
import xgboost as xgb
from torch.utils.data import random_split
from sklearn.metrics import root_mean_squared_error

from vehicledataset import VehicleDataset

data_files = [
    ("data/dry_data_state1.csv", "data/dry_data_control1.csv"),
    ("data/dry_data_state2.csv", "data/dry_data_control2.csv"),
    ("data/dry_data_state3.csv", "data/dry_data_control3.csv"),
    ("data/dry_data_state4.csv", "data/dry_data_control4.csv"),
]

# load dataset
dataset = VehicleDataset(data_files)

# get input size and output size from the data
input_size, output_size = dataset.io_size()
print(dataset.features)

# train, test, val ratios
train_ratio = 0.64
test_ratio = 0.2
val_ratio = 0.16

dataset_size = len(dataset)
train_size = int(train_ratio * dataset_size)
val_size = int(val_ratio * dataset_size)
test_size = dataset_size - train_size - val_size

# split dataset
train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
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

# normalize all data
train_data = (train_data - mean_data) / std_dev_data
test_data = (test_data - mean_data) / std_dev_data
val_data = (val_data - mean_data) / std_dev_data

train_labels = (train_labels - mean_labels) / std_dev_labels
test_labels = (test_labels - mean_labels) / std_dev_labels
val_labels = (val_labels - mean_labels) / std_dev_labels

# create train, val, and test dmatrices
dtrain = xgb.DMatrix(train_data, label=train_labels)
dval = xgb.DMatrix(val_data, label=val_labels)
dtest = xgb.DMatrix(test_data)

# train and val evals
evals = [(dtrain, 'train'), (dval, 'eval')]

# num boosting rounds
num_round = 100

# xgb params
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.1,
    "max_depth": 6,
    "silent": 1,
    "n_estimators": 100
}

# train xgb model
model = xgb.train(
    xgb_params,
    dtrain,
    num_round,
    evals=evals,
    early_stopping_rounds=10
)

pred_labels = model.predict(dtest)

mse = root_mean_squared_error(test_labels, pred_labels)

print(f'Root Mean Squared Error (MSE): {mse}')
