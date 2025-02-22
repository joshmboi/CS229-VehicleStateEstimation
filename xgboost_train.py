import json
import torch
from tqdm import tqdm
import xgboost as xgb
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from sklearn.metrics import mean_squared_error

from vehicledataset import VehicleDataset

data_files = [
    ("data/ice_data_state1.csv", "data/ice_data_control1.csv"),
    ("data/ice_data_state2.csv", "data/ice_data_control2.csv"),
    ("data/ice_data_state3.csv", "data/ice_data_control3.csv"),
    ("data/ice_data_state4.csv", "data/ice_data_control4.csv"),
]

# load dataset
dataset = VehicleDataset(data_files)

# get input size and output size from the data
input_size, output_size = dataset.io_size()

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

# train and val evals and track
evals = [(dtrain, 'train'), (dval, 'eval')]
evals_result = {}

# keep track of training and validation losses
train_losses = []
val_losses = []

# num boosting rounds
num_rounds = 100

# xgb params
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.1,
    "max_depth": 6,
}

# tqdm for tracking and initialize model
t = tqdm(range(num_rounds), desc="Training...")
model = None

for round in t:
    # train
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1,
        evals=evals,
        evals_result=evals_result,
        xgb_model=model,
        verbose_eval=False
    )

    # get losses
    train_loss = evals_result['train']['rmse'][0]
    val_loss = evals_result['eval']['rmse'][0]

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # create description string
    desc = f"Round {round+1}: "
    desc += f"Training RMSE: {train_loss:.4f}, "
    desc += f"Validation RMSE: {val_loss:.4f}"
    t.set_description(desc)

pred_labels = model.predict(dtest)

mse = mean_squared_error(test_labels, pred_labels)

print(f'Mean Squared Error (MSE): {mse}')

# save model
model.save_model("./models/xgb_ice.ubj")

params_file = "./models/xgb_ice_params.json"
params = {
        "mean_data": mean_data.tolist(),
        "std_dev_data": std_dev_data.tolist(),
        "mean_labels": mean_labels.tolist(),
        "std_dev_labels": std_dev_labels.tolist()
}

# write to json file
with open(params_file, "w", encoding="utf-8") as f:
    json.dump(params, f, ensure_ascii=False, indent=4)

plt.plot(range(100), train_losses)
plt.plot(range(100), val_losses)
plt.legend(["Training Error", "Validation Error"])
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Errors of XGB While Training on Ice Data")
plt.show()
