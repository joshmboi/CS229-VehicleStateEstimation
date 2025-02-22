import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from nn import NN
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
print(dataset.features)

# set hyperparams
lr = 0.001
batch_size = 64
num_epochs = 50

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

# recreate datasets
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)
val_dataset = TensorDataset(val_data, val_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# define model loss function and optimizing
model = NN(input_size, output_size)
loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

train_losses, val_losses = [], []
t = tqdm(range(num_epochs), desc="Training...")

for epoch in t:
    # train
    train_loss = 0
    for input, target in train_loader:
        # zero grads
        optimizer.zero_grad()

        # forward, loss, backward
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()

        # update params
        optimizer.step()

        train_loss += loss.item()

    # validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for input, target in val_loader:
            output = model(input)
            val_loss += loss_fn(output, target).item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Create description string
    desc = f"Epoch {epoch+1}: "
    desc += f"Training Loss: {train_loss:.4f}, "
    desc += f"Validation Loss: {val_loss:.4f}"
    t.set_description(desc)

model.eval()
test_loss = 0
with torch.no_grad():
    for input, target in test_loader:
        output = model(input)
        test_loss += loss_fn(output, target).item()

test_loss /= len(test_loader)
print(f"Test Loss: {test_loss}")

mean_loss = 0
mean_val = torch.mean(train_labels, axis=0)
with torch.no_grad():
    for input, target in test_loader:
        output = mean_val.expand_as(target)
        mean_loss += loss_fn(output, target).item()

mean_loss /= len(test_loader)
print(f"Mean Loss: {mean_loss}")

torch.save(model, "./models/ice")

# get model weights from first layer
weights = model.fc1.weight.data.abs().numpy()

# rank feature importance
features = dataset.features
importance = dict(zip(features, weights.flatten()))

# sort from most to least important
sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

for feature, importance in sorted_features[:5]:
    print(f"{feature}: {importance:.4f}")

plt.plot(range(1, 51), train_losses)
plt.plot(range(1, 51), val_losses)
plt.legend(["Training Error", "Validation Error"])
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Errors of Model While Training on Ice Data")
plt.show()
