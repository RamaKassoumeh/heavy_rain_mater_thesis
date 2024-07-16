from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
# from RadarFilterImageDataset import RadarFilterImageDataset
from RadarFilterRainNet3DDataset import RadarFilterRainNetDataset
# from RadarFilterRainNetDataset import RadarFilterRainNetDataset

from RainNet3D import RainNet
from model import train_model
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
from sklearn.metrics import confusion_matrix

# import imageio
# from ipywidgets import widgets, HBox
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
     transforms.Lambda(lambda x:  (torch.log(x+1) / torch.log(torch.tensor(max_value+1))).float()),
    # transforms.Lambda(lambda x: x.float())
    
])

# # Declaring the variable by using square kernels and equal stride
# c = nn.Conv3d(18, 35, 5, stride=1,padding="same")

# # Describing the input and output variables
# input = torch.randn(22, 18, 12, 52, 102)
# output = c(input)

# # Print output
# print(output) 


def invert_custom_transform1(x):
    # Use PyTorch's where function to apply the transformation element-wise
    return torch.where(x > -0.1, x-1, x)
def invert_custom_transform2(x):
    # Use PyTorch's where function to apply the transformation element-wise
    return torch.where(x <= -0.1, -999, x) 

inverseTransform= transforms.Compose([
    # transforms.Lambda(lambda x: x.unsqueeze(0))  ,# Add a new dimension at position 0
    # transforms.Lambda(lambda x: x.cuda()) , # send data to cuda
    # transforms.Normalize(mean=[-mean/std,],
                            #  std=[1/std,])
    # transforms.Lambda(lambda x: torch.exp(x)-1),
    transforms.Lambda(lambda x: torch.pow(max_value+1, x)-1),
    transforms.Lambda(invert_custom_transform2) ,
    transforms.Lambda(invert_custom_transform1) ,
    transforms.Lambda(invert_custom_transform2) ,
    # transforms.Lambda(lambda x: (x*(max_value - min_value))+min_value)
    transforms.Lambda(lambda x: x) 
])

train_dataset = RadarFilterRainNetDataset(
    img_dir='../RadarData_18/',
    transform=transform,
    inverse_transform=inverseTransform
)

validate_data = RadarFilterRainNetDataset(
    img_dir='../RadarData_validate_18/',
    transform=transform,
    inverse_transform=inverseTransform
)

test_data = RadarFilterRainNetDataset(
    img_dir='../RadarData_test_18/',
    transform=transform,
    inverse_transform=inverseTransform
)


model=RainNet()
file_name='radar_trainer_30M_RainNet_288_size_log_200_normalize_3d_2018'
train_model(train_dataset,validate_data,test_data,inverseTransform,model)