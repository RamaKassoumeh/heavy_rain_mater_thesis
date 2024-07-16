from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from RadarFilterImageDataset import RadarFilterImageDataset
from RadarFilterRainNetSatelliteDataset import RadarFilterRainNetSatelliteDataset

from RainNet_Satellite import RainNet
from model import train_model

from torch.utils.data import DataLoader

import ast

from torchvision import transforms
import numpy as np

# read data from file
with open("analyse_satellite_IQR.txt", 'r') as file:
    lines = file.readlines()

data = {}
for line in lines:
    key, value = line.split(':', 1)
    key = key.strip()
    value = value.strip()
    value = value.replace('array', '')  # Remove 'array' to make it a valid dictionary
    data[key] = ast.literal_eval(value)

# Extracting the dictionaries
iqr_values = data.get("IQR values", {})
bands_min_values = data.get("Min values", {})
bands_max_values = data.get("Max values", {})
outliers_count_percentage_values = data.get("outliers count percentage values", {})

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


def normalize_Satellite(x):
    for i in range(x.size(0)):
        key=list(bands_min_values.keys())[i]
        x[i] = (x[i]-bands_min_values[key])/(bands_max_values[key]-bands_min_values[key])
    return x

sat_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(normalize_Satellite),
    ])

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

train_dataset = RadarFilterRainNetSatelliteDataset(
    img_dir='/raid/heavyrain_dataset/RadarData_18/',
    sat_dir='/raid/heavyrain_dataset/SatelliteData',
    transform=transform,
    inverse_transform=inverseTransform,
    sat_transform=sat_transform,
    random_satellite=False
)

validate_data = RadarFilterRainNetSatelliteDataset(
    img_dir='/raid/heavyrain_dataset/RadarData_validate_18/',
    sat_dir='/raid/heavyrain_dataset/SatelliteData',
    transform=transform,
    inverse_transform=inverseTransform,
    sat_transform=sat_transform,
    random_satellite=False
)

test_data = RadarFilterRainNetSatelliteDataset(
    img_dir='/raid/heavyrain_dataset/RadarData_test_18/',
    sat_dir='/raid/heavyrain_dataset/SatelliteData',
    transform=transform,
    inverse_transform=inverseTransform,
    sat_transform=sat_transform,
    random_satellite=False
)

model=RainNet()
file_name='radar_trainer_30M_RainNet_Sat_288_size_log_200_normalize_3d_sat_bigger'

train_model(train_dataset,validate_data,test_data,inverseTransform,model)