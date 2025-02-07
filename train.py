import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

from nn import NN
from vehicledataset import VehicleDataset

# load dataset
dataset = VehicleDataset("state.csv", "control.csv")

input_size, output_size = dataset.io_size()
print(input_size, output_size)

# set hyperparams
lr = 0.001
batch_size = 64
num_epochs = 30

# train, test, val ratios
train_ratio = 0.64
test_ratio = 0.2
val_ratio = 0.16

dataset_size = len(dataset)
train_size = int(train_ratio * dataset_size)
val_size = int(val_ratio * dataset_size)
test_size = dataset_size - train_size - val_size

# split dataset
train_data, val_data, test_data = random_split(
    dataset,
    [train_size, val_size, test_size]
)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

# define model loss function and optimizing
model = NN(input_size, output_size)
loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

train_losses, val_losses = [], []

for epoch in range(num_epochs):
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
    print(f"Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")

model.eval()
test_loss = 0
with torch.no_grad():
    for input, target in test_loader:
        output = model(input)
        test_loss += loss_fn(output, target).item()

test_loss /= len(test_loader)
print(f"Test Loss: {test_loss}")
