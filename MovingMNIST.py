from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from convlstm import Seq2Seq
from torch.utils.data import DataLoader

import io
# import imageio
# from ipywidgets import widgets, HBox

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
# Load Data as Numpy Array
MovingMNIST = np.load('mnist_test_seq.npy').transpose(1, 0, 2, 3)

# Shuffle Data
np.random.shuffle(MovingMNIST)

# Train, Test, Validation splits
train_data = MovingMNIST[:80]
val_data = MovingMNIST[80:90]
test_data = MovingMNIST[90:100]

def collate(batch):

    # Add channel dim, scale pixels between 0 and 1, send to GPU
    batch = torch.tensor(batch).unsqueeze(1)
    batch = batch / 255.0
    batch = batch.to(device)

    # Randomly pick 10 frames as input, 11th frame is target
    rand = np.random.randint(10,20)
    return batch[:,:,rand-10:rand], batch[:,:,rand]


# Training Data Loader
train_loader = DataLoader(train_data, shuffle=True,
                        batch_size=16, collate_fn=collate)

# Validation Data Loader
val_loader = DataLoader(val_data, shuffle=True,
                        batch_size=16, collate_fn=collate)

# Get a batch
input, _ = next(iter(val_loader))

# Reverse process before displaying
input = input.cpu().numpy() * 255.0


# The input video frames are grayscale, thus single channel
model = Seq2Seq(num_channels=1, num_kernels=64,
kernel_size=(3, 3), padding=(1, 1), activation="relu",
frame_size=(64, 64), num_layers=3).to(device)

optim = Adam(model.parameters(), lr=1e-4)

# Binary Cross Entropy, target pixel values either 0 or 1
criterion = nn.BCELoss(reduction='sum')

num_epochs = 1

# Initializing in a separate cell, so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fradar_trainer_{}'.format(timestamp))

for epoch in range(1, num_epochs + 1):

    train_loss = 0
    model.train()
    for batch_num, (input, target) in enumerate(train_loader, 1):
        output = model(input)
        loss = criterion(output.flatten(), target.flatten())
        loss.backward()
        optim.step()
        optim.zero_grad()
        train_loss += loss.item()
        print(f"the train loss is {train_loss}")
    train_loss /= len(train_loader.dataset)

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for input, target in val_loader:
            output = model(input)
            loss = criterion(output.flatten(), target.flatten())
            val_loss += loss.item()
    val_loss /= len(val_loader.dataset)

    print("Epoch:{} Training Loss:{:.2f} Validation Loss:{:.2f}\n".format(
        epoch, train_loss, val_loss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': train_loss, 'Validation': val_loss},
                       epoch + 1)
    writer.flush()


def collate_test(batch):
    # Last 10 frames are target
    target = np.array(batch)[:, 10:]

    # Add channel dim, scale pixels between 0 and 1, send to GPU
    batch = torch.tensor(batch).unsqueeze(1)
    batch = batch / 255.0
    batch = batch.to(device)
    return batch, target


# Test Data Loader
test_loader = DataLoader(test_data, shuffle=True,
                         batch_size=3, collate_fn=collate_test)

# Get a batch
batch, target = next(iter(test_loader))

# Initialize output sequence
output = np.zeros(target.shape, dtype=np.uint8)

# Loop over timesteps
for timestep in range(target.shape[1]):
    input = batch[:, :, timestep:timestep + 10]
    output[:, timestep] = (model(input).squeeze(1).cpu() > 0.5) * 255.0
#test_loss= criterion(torch.from_numpy(output).flatten(), torch.from_numpy(target).flatten())