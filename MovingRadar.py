from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

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
transform = transforms.Compose([
    transforms.ToTensor()
    , transforms.Normalize((0.11872672,), (0.194747557,))
])

radar_dataset = []
count_zeros=0
total_count=0
# Walk through all directories and files
for radar_folders in sorted(os.listdir(radar_data_folder_path)):
    # Construct the full path to the folder
    folder_path = os.path.join(radar_data_folder_path, radar_folders)

    # Check if the path is a directory
    if os.path.isdir(folder_path):
        print(f"\nProcessing folder: {folder_path}")
        # use glob.glob to read all files in the folder order by name
        radar_files = sorted(glob.glob(os.path.join(folder_path, '*.scu')))
        # Walk through all directories and files inside the current folder
        radar_day_file = []
        # Process each file in the current directory
        for radar_file in radar_files:
            # Construct the full path to the file
            # radar_file = os.path.join(folder_path, filename)

            with h5py.File(radar_file, 'a') as file:
                a_group_key = list(file.keys())[0]
                dataset_DXk = file.get(a_group_key)
                ds_arr = dataset_DXk.get('image')[:]  # the image data in an array of floats
                ds_arr = np.where(ds_arr == -999, 0, ds_arr)
                # Convert the 2D array to a PIL Image
                image = Image.fromarray((ds_arr * 255).astype(np.uint8))
                # resized_image = image.resize((128, 128))
                resized_image = image.resize((128, 128))
                
                # plt.show()
                # plt.imshow(image)
                # plt.show()
                # Convert the resized image back to a 2D NumPy array
                resized_image_array = np.array(resized_image)
        
                radar_day_file.append(resized_image_array)
                file.close()
        radar_dataset.append(radar_day_file)

# # Flatten the list of arrays
# flattened_data = np.concatenate(radar_dataset)

# # Create histogram
# plt.figure(figsize=(8, 6))

# plt.hist(flattened_data.flatten(), bins=30, alpha=0.5, color='b')
# plt.savefig('radar hist all')

# flattened_data = flattened_data[flattened_data > 0]

# plt.clf()
# plt.hist(flattened_data.flatten(), bins=30, alpha=0.5, color='b')
# plt.savefig('radar hist non zero')


radar_dataset = np.stack(radar_dataset, axis=0)

#
# input_files = np.expand_dims(input_files, axis=-1)
# dataset = radar_dataset

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

def plot_images2(image1, image2,image3,epoch,batch_num,name):
    image1 = image1.detach().cpu().numpy()
    image1 = image1 * 136.7
    image1 = image1.astype(np.uint8)

    image2 = image2.detach().cpu().numpy()
    image2 = image2 * 136.7
    image2 = image2.astype(np.uint8)

    image3 = image3.detach().cpu().numpy()
    image3 = image3 * 136.7
    image3 = image3.astype(np.uint8)
    fig, axes = plt.subplots(1, 3)  # Create a subplot with 1 row and 2 columns

    # Plot the first image
    axes[0].imshow(image1)  # Assuming grayscale image
    axes[0].axis('off')  # Turn off axis
    axes[0].set_title('input')  # Set title

    # Plot the second image
    axes[1].imshow(image2)  # Assuming grayscale image
    axes[1].axis('off')  # Turn off axis
    axes[1].set_title('output')  # Set title

    # Plot the second image
    axes[2].imshow(image3)  # Assuming grayscale image
    axes[2].axis('off')  # Turn off axis
    axes[2].set_title('target')  # Set title
    isExist = os.path.exists(f"output/images_{timestamp}")
    if not isExist:
        os.mkdir(f"output/images_{timestamp}") 

    plt.savefig(f"output/images_{timestamp}/{name}_{epoch}_{batch_num}")

# np.random.shuffle(radar_dataset)
# Split into train and validation sets using indexing to optimize memory.
indexes = np.arange(radar_dataset.shape[0])
# Define the proportions for each set
train_ratio = 0.6  # 60% for training
val_ratio = 0.2  # 20% for validation
test_ratio = 0.2  # 20% for testing

# Calculate the split points
train_split = int(train_ratio * len(indexes))
val_split = int((train_ratio + val_ratio) * len(indexes))

# Split the indices
train_indexes = indexes[:train_split]
val_indexes = indexes[train_split:val_split]
test_indexes = indexes[val_split:]

# Use the indices to extract the corresponding data for each set
train_data = radar_dataset[train_indexes]
# train_data = torch.stack([transform(item) for item in train_data])


val_data = radar_dataset[val_indexes]
# val_data = torch.stack([transform(item) for item in val_data])

test_data = radar_dataset[test_indexes]


# test_data = torch.stack([transform(item) for item in test_data])

# # Shuffle Data
# np.random.shuffle(MovingMNIST)
#
# # Train, Test, Validation splits
# train_data = MovingMNIST[:80]
# val_data = MovingMNIST[80:90]
# test_data = MovingMNIST[90:100]


def collate(batch):
    # Add channel dim, scale pixels between 0 and 1, send to GPU
    batch = torch.tensor(batch).unsqueeze(1)
    # batch = (batch - mean) / std
    batch = batch / 136.7
    batch = batch.to(device)

    # Randomly pick 6 frames as input (0.5 hours), 11th frame is target
    rand = np.random.randint(6, 288)
    return batch[:, :, rand - 6:rand], batch[:, :, rand]


# Training Data Loader
train_loader = DataLoader(train_data, shuffle=True,
                          batch_size=4 ,collate_fn=collate)

# Validation Data Loader
val_loader = DataLoader(val_data, shuffle=True,
                        batch_size=4, collate_fn=collate)

# Get a batch
input, _ = next(iter(val_loader))

# Reverse process before displaying
input = input.cpu().numpy() * 255.0

# The input video frames are grayscale, thus single channel
model = Seq2Seq(num_channels=1, num_kernels=64,
                kernel_size=(3, 3), padding=(1, 1), activation="relu",
                frame_size=(128, 128), num_layers=3).to(device)

optim = Adam(model.parameters(), lr=1e-4)

# Binary Cross Entropy, target pixel values either 0 or 1
# criterion = nn.BCELoss(reduction='sum')
criterion = nn.MSELoss()
num_epochs = 20


# Initializing in a separate cell, so we can easily add more epochs to the same run

writer = SummaryWriter('runs/radar_trainer_128_128_30M_MSE{}'.format(timestamp))

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
        # print(f"the train loss is {train_loss}")
        
        plot_images([input[0,0,input.shape[2]-6],input[0,0,input.shape[2]-5],input[0,0,input.shape[2]-4],input[0,0,input.shape[2]-3],input[0,0,input.shape[2]-2],input[0,0,input.shape[2]-1] ,target[0][0],output[0][0]], 2, 4,epoch,batch_num,'train')


    train_loss /= len(train_loader.dataset)
    train_loss /= train_loader.dataset.shape[2]
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for input, target in val_loader:
            output = model(input)
            loss = criterion(output.flatten(), target.flatten())
            val_loss += loss.item()
            plot_images([input[0,0,input.shape[2]-6],input[0,0,input.shape[2]-5],input[0,0,input.shape[2]-4],input[0,0,input.shape[2]-3],input[0,0,input.shape[2]-2],input[0,0,input.shape[2]-1] ,target[0][0],output[0][0]], 2, 4,epoch,batch_num,'validate')
    val_loss /= len(val_loader.dataset)
    val_loss /= train_loader.dataset.shape[2]
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

# Test Data Loaderm
test_loader = DataLoader(test_data, shuffle=True,
                         batch_size=3, collate_fn=collate_test)

# Get a batch
batch, target = next(iter(test_loader))

# Initialize output sequence
output = np.zeros(target.shape, dtype=np.uint8)

# Loop over timesteps
# for timestep in range(target.shape[1]):
#     input = batch[:, :, timestep:timestep + 36]
#     output[:, timestep] = model(input).squeeze(1).detach().cpu().numpy()
# test_loss = criterion(torch.from_numpy(output).float().flatten(), torch.from_numpy(target).float().flatten())
# print(f"the test loss is {test_loss}")
test_loss=0
i=0
for input, target in test_loader:
    output = model(input)
    loss = criterion(output.flatten(), target.flatten())
    test_loss += loss.item()
    # save the output and the target as image side by side
    plot_images([input[0,0,input.shape[2]-6],input[0,0,input.shape[2]-5],input[0,0,input.shape[2]-4],input[0,0,input.shape[2]-3],input[0,0,input.shape[2]-2],input[0,0,input.shape[2]-1] ,target[0][0],output[0][0]], 2, 4,epoch,batch_num,'test')
    i=i+1

test_loss /= len(test_loader.dataset)
test_loss /= train_loader.dataset.shape[2]

print(f"the test loss is {test_loss}")

