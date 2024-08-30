import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
parparent = os.path.dirname(parent)
sys.path.append(current)
sys.path.append(parent)
sys.path.append(parparent)

import ast
from datetime import datetime
import re
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from src.plotting.plotting import plot_images

from torch.utils.data import DataLoader, Subset
import os

import numpy as np
from sklearn.metrics import confusion_matrix

from tqdm import tqdm

from tests.test_metrics import calculate_metrics,categories_threshold
from torchvision import transforms

from dataloader.RadarFilterRainNet3DDataset import RadarFilterRainNetDataset

from models.RainNet3D import RainNet

import model_RainNet

min_value=0
max_value=200

# read data from file
with open(f"{parparent}/src/analyse/analyse_satellite_IQR.txt", 'r') as file:
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


def custom_transform1(x):
    # Use PyTorch's where function to apply the transformation element-wise
    return torch.where(x >= 0, x + 1, x)
def custom_transform2(x):
    # Use PyTorch's where function to apply the transformation element-wise
    return torch.where(x < 0, 0, x)

radar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(custom_transform1) ,
    transforms.Lambda(custom_transform2) ,
    transforms.Lambda(lambda x:  (torch.log(x+1) / torch.log(torch.tensor(max_value+2))).float()),    
])

def custom_undefined_transform(x):
    mask_undefined = (x < 0)
    x[mask_undefined] = 0
    x=(torch.log(x+1) / torch.log(torch.tensor(max_value+1))).float()
    x[mask_undefined] = -1
    return x

radar_undefined_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(custom_undefined_transform)
])



def invert_custom_transform1(x):
    # Use PyTorch's where function to apply the transformation element-wise
    return torch.where(x >= 0, x-1, x)
def invert_custom_transform2(x):
    # Use PyTorch's where function to apply the transformation element-wise
    return torch.where(x < 0, -999, x) 
radar_inverseTransform= transforms.Compose([
    transforms.Lambda(lambda x: torch.pow(max_value+2, x)-1),
    # transforms.Lambda(invert_custom_transform2) ,
    transforms.Lambda(invert_custom_transform1) ,
    transforms.Lambda(invert_custom_transform2) ,
    transforms.Lambda(lambda x: x) 
])

def inverse_custom_undefined_transform(x):
    mask_undefined = (x < 0)
    x[mask_undefined] = 0
    x= torch.pow(max_value+1, x)-1
    x[mask_undefined] = -999
    return x

radar_undefined_inverse_transform = transforms.Compose([
    transforms.Lambda(inverse_custom_undefined_transform),
    transforms.Lambda(lambda x: x) 
])


radar_without_undefined_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x:  (torch.log(x+1) / torch.log(torch.tensor(max_value+1))).float()),    
])

radar_without_undefined_inverse_transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.pow(max_value+1, x)-1),
    transforms.Lambda(lambda x: x) 
])


def normalize_Satellite(x):
    for i in range(x.size(0)):
        key=list(bands_min_values.keys())[i]
        x[i] = (x[i]-bands_min_values[key])/(bands_max_values[key]-bands_min_values[key])
    return x

satellite_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(normalize_Satellite),
    ])

def denormalize_Satellite(x):
    for i in range(x.size(0)):
        key=list(bands_min_values.keys())[i]
        x[i] = x[i]*(bands_max_values[key]-bands_min_values[key])+bands_min_values[key]
    return x

satellite_inverseTransform = transforms.Compose([
    transforms.Lambda(denormalize_Satellite),
    ])

def train_model(train_dataset,validate_data,model,file_name,inverse_trans,batch_size,run_name=None,advance_time=5):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # for batch_num, (input, target) in enumerate(tqdm(train_dataloader), 1):
    #     continue

    validate_dataloader = DataLoader(
        dataset=validate_data,
        batch_size=400,
        shuffle=True
    )
    # for batch_num, (input, target) in enumerate(tqdm(validate_dataloader), 1):
    #     continue
    # test_loader = DataLoader(
    #     dataset=test_data,
    #     batch_size=25,
    #     shuffle=True
    # )
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
        
    class LogCoshThresholdLoss(nn.Module):
        def __init__(self,lower_threshold,upper_threshold):
            super(LogCoshThresholdLoss, self).__init__()
            self.lower_threshold=lower_threshold.cuda()
            self.upper_threshold=upper_threshold.cuda()

        def forward(self, y_pred, y_true):
            # get data between threasholds

            # Create a boolean mask for values between the thresholds
            mask = (y_true >= self.lower_threshold) & (y_true < self.upper_threshold)

            # Use the boolean mask to filter the tensor
            y_true_filtered_values = y_true[mask[0,0]]
            y_pred_filtered_values = y_pred[mask[0,0]]
            # Compute the difference between predicted and true values
            diff = y_pred_filtered_values - y_true_filtered_values
            # Compute log(cosh(x)) element-wise
            loss = torch.log(torch.cosh(diff))
            # Compute the mean loss over all examples``
            loss = torch.mean(loss)
            return loss

    # Get a batch
    # input, _ = next(iter(validate_dataloader))
    no_param=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters in the model is {no_param}")
    model=torch.nn.DataParallel(model)
    model.cuda()
    optim = Adam(model.parameters(), lr=1e-4)
    # Define learning rate scheduler
    # scheduler_increase = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=10)
    scheduler= torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[10,30,40], gamma=0.1)
    num_epochs = 35
    criterion = LogCoshLoss()
    # Initializing in a separate cell, so we can easily add more epochs to the same run
    csi_values = {category: [] for category in categories_threshold.keys()}
    fss_values = {category: [] for category in categories_threshold.keys()}
    tp_values_array = {category: [] for category in categories_threshold.keys()}
    fp_values_array = {category: [] for category in categories_threshold.keys()}
    fn_values_array = {category: [] for category in categories_threshold.keys()}
    # Calculate fss for each category across all images
    numerator_values_array = {category: [] for category in categories_threshold.keys()}
    denominator_values_array = {category: [] for category in categories_threshold.keys()}
    if run_name is None:
        run_name=f'{file_name}_{timestamp}'
    run_file_name = f'{parparent}/runs/{run_name}'
    
    start_epoch=1
    model_file_path=f'{parparent}/models_file'
    checkpoint_path =f'{model_file_path}/{file_name}_model_checkpoint_1.pth'
    # List all files in the given directory
    files = os.listdir(f'{model_file_path}')
    pattern_str=f'{file_name}_model_checkpoint_(\d+)\.pth$'
    # Regular expression to match files named in the format t<number>
    pattern = re.compile(pattern_str)
    
    # Extract numbers from the file names
    file_numbers = [int(pattern.match(f).group(1)) for f in files if pattern.match(f)]
    file_numbers.sort()
    for file_number in file_numbers:
        checkpoint_path = f'{model_file_path}/{file_name}_model_checkpoint_{file_number}.pth'
    # Load the checkpoint if it exists
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            if 'run_name' in checkpoint:
                run_name = checkpoint['run_name']
                run_file_name = f'{parparent}/runs/{run_name}_2'
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"Resumed from checkpoint at epoch {epoch}")
            writer = SummaryWriter(run_file_name)
        
            model.eval()
            csi_values.clear()
            fss_values.clear()
            tp_values_array.clear()
            fp_values_array.clear()
            fn_values_array.clear()
            numerator_values_array.clear()
            denominator_values_array.clear()
            # Intinalize CSI for each category across all images
            csi_values = {category: [] for category in categories_threshold.keys()}
            fss_values = {category: [] for category in categories_threshold.keys()}
            tp_values_array = {category: [] for category in categories_threshold.keys()}
            fp_values_array = {category: [] for category in categories_threshold.keys()}
            fn_values_array = {category: [] for category in categories_threshold.keys()}
            # Calculate fss for each category across all images
            numerator_values_array = {category: [] for category in categories_threshold.keys()}
            denominator_values_array = {category: [] for category in categories_threshold.keys()}
            rmse_values = []
            with torch.no_grad():
                for batch_num, (input, target) in enumerate(tqdm(validate_dataloader), 1):
                    output = model(input)
                    # output=torch.round(output, decimals=3)
                    output_flatten=output.flatten()
                    target_flatten=target.flatten()
                    target_img=inverse_trans(target)
                    output_img=inverse_trans(output)
                    mse,tp_values, fp_values, fn_values,numerator,denominator=calculate_metrics(target_img,output_img)
                    rmse = np.sqrt(mse)
                    for category in categories_threshold.keys():
                        # csi_values_array[category].append(csi[category])
                        tp_values_array[category].append(tp_values[category])
                        fp_values_array[category].append(fp_values[category])
                        fn_values_array[category].append(fn_values[category])
                        # fss_values_array[category].append(fss[category])
                        numerator_values_array[category].append(numerator[category])
                        denominator_values_array[category].append(denominator[category])
                    # Append RMSE to list
                    rmse_values.append(rmse)

            # Log the learning rate
            current_lr = optim.param_groups[0]['lr']
            writer.add_scalars('Learning Rate', {'learning rate':current_lr}, epoch)
    
            csi_means = {category: np.sum(tp_values_array[category])/np.sum(tp_values_array[category]+fp_values_array[category]+ fn_values_array[category]) for category in categories_threshold.keys()}
            # average_fss = {category: np.nanmean(fss_values_array[category]) for category in categories_threshold.keys()}
            fss_means = {category: 1- (np.sum(numerator_values_array[category])/np.sum(denominator_values_array[category])) for category in categories_threshold.keys()}
            
            average_rmse = np.mean(rmse_values)
            writer.add_scalars(f'CSI values',csi_means,epoch)
            writer.add_scalars(f'FSS values',fss_means,epoch)
            writer.add_scalars(f'MSE values',{'MSE':average_rmse},epoch)
            writer.flush()
            writer.close()
            print(f'Checkpoint saved at epoch {epoch}')


radar_transform=model_RainNet.radar_transform
radar_inverse_transform=model_RainNet.radar_inverseTransform
# train_dataset = RadarFilterRainNetSatelliteDataset(
#     img_dir='/raid/heavyrain_dataset/RadarData_summer_18_19_min_15/',
#     sat_dir='/raid/heavyrain_dataset/SatelliteData_summer_18_19/',
#     transform=model_RainNet.radar_transform,
#     inverse_transform=radar_inverse_transform,
#     sat_transform=model_RainNet.satellite_transform,
#     random_satellite=False,
#     lead_time=15
# )

# validate_data = RadarFilterRainNetSatelliteDataset(
#     img_dir='/raid/heavyrain_dataset/RadarData_summer_20_min_15/',
#     sat_dir='/raid/heavyrain_dataset/SatelliteData_summer_20/',
#     transform=model_RainNet.radar_transform,
#     inverse_transform=radar_inverse_transform,
#     sat_transform=model_RainNet.satellite_transform,
#     random_satellite=False,
#     lead_time=15
# )

train_dataset = RadarFilterRainNetDataset(
    img_dir='/raid/heavyrain_dataset/RadarData_summer_18_19_min_30/',
    transform=model_RainNet.radar_transform,
    inverse_transform=radar_inverse_transform,
    lead_time=30
)

validate_data = RadarFilterRainNetDataset(
    img_dir='/raid/heavyrain_dataset/RadarData_summer_20_min_30/',
    transform=model_RainNet.radar_transform,
    inverse_transform=radar_inverse_transform,
    lead_time=30
)

modelRainnet=RainNet()
file_name='radar_trainer_30M_RainNet_3d_Log_summer_30_min'
train_model(train_dataset,validate_data,modelRainnet,file_name,radar_inverse_transform,batch_size=100,advance_time=30)