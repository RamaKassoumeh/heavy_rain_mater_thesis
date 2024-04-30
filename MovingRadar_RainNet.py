from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from RadarFilterImageDataset import RadarFilterImageDataset
from RadarFilterRainNetDataset import RadarFilterRainNetDataset

from RainNet import RainNet
from plotting import plot_images

from convlstm import Seq2Seq
from torch.utils.data import DataLoader
import h5py
import os
import glob
from PIL import Image
import io



from torchvision import transforms
import numpy as np

# import imageio
# from ipywidgets import widgets, HBox
radar_data_folder_path = '../RadarData/'
# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
# Load Data as Numpy Array
# MovingMNIST = np.load('mnist_test_seq.npy').transpose(1, 0, 2, 3)
min_value=0
max_value=200
mean=0.21695
std=0.9829
# make transforms
def custom_transform1(x):
    # Use PyTorch's where function to apply the transformation element-wise
    return torch.where(x >= 0, x + 1, x)
def custom_transform2(x):
    # Use PyTorch's where function to apply the transformation element-wise
    return torch.where(x < 0, 0, x)


transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x.unsqueeze(0))  ,# Add a new dimension at position 0
    # transforms.Lambda(lambda x: x.cuda()) , # send data to cuda
    # transforms.Normalize(mean=[mean,],
    #                          std=[std,],)
    # transforms.Lambda(lambda x: (x-min_value)/(max_value-min_value)),
    # transforms.Lambda(lambda x: torch.log2(x+1))
    transforms.Lambda(custom_transform1) ,
    transforms.Lambda(custom_transform2) ,
    # transforms.Lambda(lambda x: torch.log(x+1)),
     transforms.Lambda(lambda x:  (torch.log(x+1) / torch.log(torch.tensor(max_value))).float()),
    # transforms.Lambda(lambda x: x.float())
    
])

def invert_custom_transform1(x):
    # Use PyTorch's where function to apply the transformation element-wise
    return torch.where(x > -0, x-1, x)
def invert_custom_transform2(x):
    # Use PyTorch's where function to apply the transformation element-wise
    return torch.where(x <= -0, -999, x) 

inverseTransform= transforms.Compose([
    # transforms.Lambda(lambda x: x.unsqueeze(0))  ,# Add a new dimension at position 0
    # transforms.Lambda(lambda x: x.cuda()) , # send data to cuda
    # transforms.Normalize(mean=[-mean/std,],
                            #  std=[1/std,])
    # transforms.Lambda(lambda x: torch.exp(x)-1),
    transforms.Lambda(lambda x: torch.pow(max_value, x)-1),
    transforms.Lambda(invert_custom_transform2) ,
    transforms.Lambda(invert_custom_transform1) ,
    
    # transforms.Lambda(lambda x: (x*(max_value - min_value))+min_value)
    # transforms.Lambda(lambda x: x) 
])

train_dataset = RadarFilterRainNetDataset(
    img_dir='../RadarData/',
    transform=transform,
    inverse_transform=inverseTransform
)

validate_data = RadarFilterRainNetDataset(
    img_dir='../RadarData_validate/',
    transform=transform,
    inverse_transform=inverseTransform
)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=16,
    shuffle=True
)

validate_loader = DataLoader(
    dataset=validate_data,
    batch_size=16,
    shuffle=True
)
class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Compute the difference between predicted and true values
        diff = y_pred - y_true
        # Compute log(cosh(x)) element-wise
        loss = torch.log(torch.cosh(diff))
        # Compute the mean loss over all examples
        loss = torch.mean(loss)
        return loss

# Get a batch
# input, _ = next(iter(validate_loader))

model=RainNet()

model=torch.nn.DataParallel(model)
model.cuda()
# optim = Adam(model.parameters(), lr=1e-4)
optim = Adam(model.parameters(), lr=3e-4)
# Define learning rate scheduler
# scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[10,6,4], gamma=0.1)
# Binary Cross Entropy, target pixel values either 0 or 1
# criterion = nn.BCELoss(reduction='sum')
criterion = LogCoshLoss()
num_epochs = 10

folder_name='radar_trainer_30M_RainNet_512_size_log_200'
# Initializing in a separate cell, so we can easily add more epochs to the same run

writer = SummaryWriter(f'runs/{folder_name}_{timestamp}')

for epoch in range(1, num_epochs + 1):

    train_loss = 0
    acc=0
    total =0
    model.train()
    for batch_num, (input, target) in enumerate(train_dataloader, 1):
        optim.zero_grad()
        output = model(input)
        loss = criterion(output.flatten(), target.flatten())
        loss.backward()
        optim.step()
        # optim.zero_grad()
        train_loss += loss.item()
        # acc += (output.flatten() -target.flatten()<=0.01).sum().item()
        # total += target.size(0)
        # print(f"the accurecy is {acc}")
        # print(f"the train loss is {train_loss}")
        print(f"batch number={batch_num} in epoch {epoch}")
        if batch_num%100 ==0:
            target=inverseTransform(target)
            input=inverseTransform(input)
            output=inverseTransform(output)
            # plot_images([input[0,0,input.shape[2]-1],input[0,0,input.shape[2]-2],input[0,0,input.shape[2]-3],input[0,0,input.shape[2]-4],input[0,0,input.shape[2]-5],input[0,0,input.shape[2]-6] ,target[0][0],output[0][0]], 2, 4,epoch,batch_num,'train',folder_name)
            plot_images([input[0,input.shape[1]-1],input[0,input.shape[1]-2],input[0,input.shape[1]-1],input[0,input.shape[1]-2],input[0,input.shape[1]-3],input[0,input.shape[1]-4] ,target[0,0],output[0,0]], 2, 4,epoch,batch_num,'train',folder_name)
    # print('Accuracy of the network : %.2f %%' % (100 * acc / total))

    train_loss /= len(train_dataloader.dataset)
    # acc /= len(train_dataloader.dataset)
    # train_loss /= 128
    print(f"the train loss is {train_loss}")
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for input, target in validate_loader:
            output = model(input)
            loss = criterion(output.flatten(), target.flatten())
            val_loss += loss.item()
    target=inverseTransform(target)
    input=inverseTransform(input)
    output=inverseTransform(output)
    # plot_images([input[0,0,input.shape[2]-1],input[0,0,input.shape[2]-2],input[0,0,input.shape[2]-3] ,input[0,0,input.shape[2]-4] ,input[0,0,input.shape[2]-5] ,input[0,0,input.shape[2]-6]  ,target[0][0] ,output[0][0] ], 2, 4,epoch,batch_num,'validate',folder_name)
    plot_images([input[0,input.shape[1]-1],input[0,input.shape[1]-2],input[0,input.shape[1]-1],input[0,input.shape[1]-2],input[0,input.shape[1]-3],input[0,input.shape[1]-4] ,target[0,0],output[0,0]], 2, 4,epoch,batch_num,'validate',folder_name)

    val_loss /= len(validate_loader.dataset)
    # val_loss /= 128
    print(f"the validate loss is {val_loss}")
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

