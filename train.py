import json
import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from nn import NN
from vehicledataset import VehicleDataset

# set surface
surf = "wet"

# set hyperparams
lr = 0.001
batch_size = 64
num_epochs = 50


def get_datasets(surf):
    data_files = [
        (f"data/{surf}_data_state1.csv", f"data/{surf}_data_control1.csv"),
        (f"data/{surf}_data_state2.csv", f"data/{surf}_data_control2.csv"),
        (f"data/{surf}_data_state3.csv", f"data/{surf}_data_control3.csv"),
        (f"data/{surf}_data_state4.csv", f"data/{surf}_data_control4.csv"),
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

# define nn_model loss function and optimizing
nn_model = NN(io[0], io[1])
loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=lr)

nn_train_losses, nn_val_losses = [], []
t = tqdm(range(num_epochs), desc="Training...")

for epoch in t:
    # train
    train_loss = 0
    for input, target in train_loader:
        # zero grads
        optimizer.zero_grad()

        # forward, loss, backprop
        output = nn_model(input)
        loss = loss_fn(output, target)
        loss.backward()

        # update params
        optimizer.step()

        train_loss += loss.item()

    # validation
    nn_model.eval()
    val_loss = 0
    with torch.no_grad():
        for input, target in val_loader:
            output = nn_model(input)
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

# create train, val, and test dmatrices
dtrain = xgb.DMatrix(train[0], label=train[1])
dval = xgb.DMatrix(val[0], label=val[1])
dtest = xgb.DMatrix(test[0])

# train and val evals and track
evals = [(dtrain, 'train'), (dval, 'eval')]
evals_result = {}

# keep track of training and validation losses
xgb_train_losses, xgb_val_losses = [], []

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

    xgb_train_losses.append(train_loss)
    xgb_val_losses.append(val_loss)

    # create description string
    desc = f"Round {round+1}: "
    desc += f"Training RMSE: {train_loss:.4f}, "
    desc += f"Validation RMSE: {val_loss:.4f}"
    t.set_description(desc)

# evaluate test set with nn
nn_model.eval()
test_loss = 0
with torch.no_grad():
    for input, target in test_loader:
        output = nn_model(input)
        test_loss += loss_fn(output, target).item()

test_loss /= len(test_loader)
print(f"NN Test Loss (MSE): {test_loss}")

# evaluate test set with xgb
pred_labels = model.predict(dtest)
mse = mean_squared_error(test[1], pred_labels)

print(f'XGB Test Loss (MSE): {mse}')

mean_loss = 0
mean_val = torch.mean(train[1], axis=0)
with torch.no_grad():
    for input, target in test_loader:
        output = mean_val.expand_as(target)
        mean_loss += loss_fn(output, target).item()

mean_loss /= len(test_loader)
print(f"Mean Loss: {mean_loss}")

# save models
torch.save(nn_model, f"./models/nn_{surf}.pth")
model.save_model(f"./models/xgb_{surf}.json")

# get nn_model weights from first layer
weights = nn_model.fc1.weight.data.abs().numpy()

# rank feature importance
importance = dict(zip(features, weights.flatten()))

# sort from most to least important
sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

for feature, importance in sorted_features[:5]:
    print(f"{feature}: {importance:.4f}")

plt.plot(range(1, 51), nn_train_losses)
plt.plot(range(1, 51), nn_val_losses)
plt.legend(["Training Error", "Validation Error"])
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title(f"Errors of NN While Training on {surf.capitalize()} Data")
plt.show()

plt.plot(range(1, 101), xgb_val_losses)
plt.plot(range(1, 101), xgb_train_losses)
plt.legend(["Training Error", "Validation Error"])
plt.xlabel("Rounds")
plt.ylabel("Root Mean Squared Error")
plt.title(f"Rounds of XGB While Training on {surf.capitalize()} Data")
plt.show()
