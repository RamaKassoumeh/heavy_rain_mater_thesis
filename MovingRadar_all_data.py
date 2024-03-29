from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from RadarImageDataset import RadarImageDataset

from convlstm import Seq2Seq
from torch.utils.data import DataLoader
import h5py
import os
import glob
from PIL import Image
import io

from torchvision import transforms

# import imageio
# from ipywidgets import widgets, HBox
radar_data_folder_path = '../RadarData/'
# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
# Load Data as Numpy Array
# MovingMNIST = np.load('mnist_test_seq.npy').transpose(1, 0, 2, 3)

mean = 0.11872672
std = 0.194747557

# make transforms
transform = transforms.Compose([
    transforms.ToTensor()
    , transforms.Normalize((0.11872672,), (0.194747557,))
])


train_dataset = RadarImageDataset(
    img_dir='../RadarData/',
    transform=transform
)

validate_data = RadarImageDataset(
    img_dir='../RadarData_validate/',
    transform=transform
)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

def plot_images(image_list, row, col,epoch,batch_num,name):
    fig, axes = plt.subplots(row, col, figsize=(12, 6))
    for i in range(row):
        for j in range(col):
            image=image_list[i * col + j]
            image = image.detach().cpu().numpy()
            image = image * 136.7
            image = image.astype(np.uint8)
            axes[i, j].imshow(image)
            axes[i, j].axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust spacing between subplots
    isExist = os.path.exists(f"output/image_radar_trainer_128_128_30M_{timestamp}")
    if not isExist:
        os.mkdir(f"output/image_radar_trainer_128_128_30M_{timestamp}") 

    plt.savefig(f"output/image_radar_trainer_128_128_30M_{timestamp}/{name}_{epoch}_{batch_num}")

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=8,
    shuffle=True
)

validate_loader = DataLoader(
    dataset=validate_data,
    batch_size=8,
    shuffle=True
)


# Get a batch
input, _ = next(iter(validate_loader))


# The input video frames are grayscale, thus single channel
model = Seq2Seq(num_channels=1, num_kernels=64,
                kernel_size=(3, 3), padding=(1, 1), activation="relu",
                frame_size=(128, 128), num_layers=3).to(device)

optim = Adam(model.parameters(), lr=1e-4)

# Binary Cross Entropy, target pixel values either 0 or 1
criterion = nn.BCELoss(reduction='sum')
# criterion = nn.MSELoss()
num_epochs = 10


# Initializing in a separate cell, so we can easily add more epochs to the same run

writer = SummaryWriter('runs/radar_trainer_128_128_30M_BCE_{}'.format(timestamp))

for epoch in range(1, num_epochs + 1):

    train_loss = 0
    model.train()
    for batch_num, (input, target) in enumerate(train_dataloader, 1):
        output = model(input)
        loss = criterion(output.flatten(), target.flatten())
        loss.backward()
        optim.step()
        optim.zero_grad()
        train_loss += loss.item()
        # print(f"the train loss is {train_loss}")
        print(f"batch number={batch_num} in epoch {epoch}")
        # plot_images([input[0,0,input.shape[2]-6],input[0,0,input.shape[2]-5],input[0,0,input.shape[2]-4],input[0,0,input.shape[2]-3],input[0,0,input.shape[2]-2],input[0,0,input.shape[2]-1] ,target[0][0],output[0][0]], 2, 4,epoch,batch_num,'train')


    train_loss /= len(train_dataloader.dataset)
    train_loss /= 128
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for input, target in validate_loader:
            output = model(input)
            loss = criterion(output.flatten(), target.flatten())
            val_loss += loss.item()
    plot_images([input[0,0,input.shape[2]-6],input[0,0,input.shape[2]-5],input[0,0,input.shape[2]-4],input[0,0,input.shape[2]-3],input[0,0,input.shape[2]-2],input[0,0,input.shape[2]-1] ,target[0][0],output[0][0]], 2, 4,epoch,batch_num,'validate')
    val_loss /= len(validate_loader.dataset)
    val_loss /= 128
    print("Epoch:{} Training Loss:{:.2f} Validation Loss:{:.2f}\n".format(
        epoch, train_loss, val_loss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': train_loss, 'Validation': val_loss},
                       epoch)
    writer.flush()

def collate_test(batch):
    # Last 10 frames are target
    # target = np.array(batch)[:, 36:]
    #
    # # Add channel dim, scale pixels between 0 and 1, send to GPU
    # batch = torch.tensor(batch).unsqueeze(1)
    # batch = batch / 136.7
    # batch = batch.to(device)
    # return batch, target
    # Add channel dim, scale pixels between 0 and 1, send to GPU
    batch = torch.tensor(batch).unsqueeze(1)
    # batch = (batch - mean) / std
    batch = batch / 136.7
    batch = batch.to(device)

    # Randomly pick 6 frames as input (0.5 hours), 11th frame is target
    rand = np.random.randint(6, 288)
    return batch[:, :, rand - 6:rand], batch[:, :, rand]

# # Test Data Loaderm
# test_loader = DataLoader(test_data, shuffle=True,
#                          batch_size=3, collate_fn=collate_test)

# # Get a batch
# batch, target = next(iter(test_loader))

# # Initialize output sequence
# output = np.zeros(target.shape, dtype=np.uint8)

# # Loop over timesteps
# # for timestep in range(target.shape[1]):
# #     input = batch[:, :, timestep:timestep + 36]
# #     output[:, timestep] = model(input).squeeze(1).detach().cpu().numpy()
# # test_loss = criterion(torch.from_numpy(output).float().flatten(), torch.from_numpy(target).float().flatten())
# # print(f"the test loss is {test_loss}")
# test_loss=0
# i=0
# for input, target in test_loader:
#     output = model(input)
#     loss = criterion(output.flatten(), target.flatten())
#     test_loss += loss.item()
#     # save the output and the target as image side by side
#     plot_images([input[0,0,input.shape[2]-6],input[0,0,input.shape[2]-5],input[0,0,input.shape[2]-4],input[0,0,input.shape[2]-3],input[0,0,input.shape[2]-2],input[0,0,input.shape[2]-1] ,target[0][0],output[0][0]], 2, 4,epoch,batch_num,'test')
#     i=i+1

# test_loss /= len(test_loader.dataset)
# test_loss /= train_loader.dataset.shape[2]

# print(f"the test loss is {test_loss}")

