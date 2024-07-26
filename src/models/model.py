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

def train_model(train_dataset,validate_data,test_data,model,file_name,batch_size,run_name=None):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=8
    )
    # for batch_num, (input, target) in enumerate(tqdm(train_dataloader), 1):
    #     continue

    validate_dataloader = DataLoader(
        dataset=validate_data,
        batch_size=25,
        shuffle=True
    )
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
    scheduler_decrease = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[10,30,40], gamma=0.1)
    num_epochs = 50
    criterion = LogCoshLoss()
    # Initializing in a separate cell, so we can easily add more epochs to the same run
    # Calculate CSI for each category across all images
    csi_values = {category: [] for category in categories_threshold.keys()}

    # Calculate fss for each category across all images
    fss_values = {category: [] for category in categories_threshold.keys()}
    if run_name is None:
        run_name=f'{file_name}_{timestamp}'
    run_file_name = f'{parparent}/runs/{run_name}'
    
    start_epoch=1
    model_file_path=f'{parparent}/models_file'
    checkpoint_path =f'{model_file_path}/{file_name}_model_checkpoint_2.pth'
    # List all files in the given directory
    files = os.listdir(f'{model_file_path}')
    pattern_str=f'{file_name}_model_checkpoint_(\d+)\.pth$'
    # Regular expression to match files named in the format t<number>
    pattern = re.compile(pattern_str)
    
    # Extract numbers from the file names
    file_numbers = [int(pattern.match(f).group(1)) for f in files if pattern.match(f)]
    
    if file_numbers:
        # Return the maximum file number
        max_number= max(file_numbers)
        checkpoint_path = f'{model_file_path}/{file_name}_model_checkpoint_{max_number}.pth'
    # Load the checkpoint if it exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'run_name' in checkpoint:
            run_name = checkpoint['run_name']
            run_file_name = f'{parparent}/runs/{run_name}'
        print(f"Resumed from checkpoint at epoch {start_epoch}")
    writer = SummaryWriter(run_file_name)
    for epoch in range(start_epoch, num_epochs + 1):

        train_loss = 0
        model.train() 
        for batch_num, (input, target) in enumerate(tqdm(train_dataloader), 1):
            optim.zero_grad()
            output = model(input)
            output_flatten=output.flatten()
            target_flatten=target.flatten()
            
            # loss_undefined_rain = criterion_undefined_rain(output_flatten, target_flatten)
            # loss_light_rain = criterion_light_rain(output_flatten, target_flatten)
            # loss_moderate_rain = criterion_moderate_rain(output_flatten, target_flatten)
            # loss_heavy_rain = criterion_heavy_rain(output_flatten, target_flatten)
            # loss=0.1*loss_undefined_rain+0.2*loss_light_rain+0.3*loss_moderate_rain+0.4*loss_heavy_rain
            loss = criterion(output_flatten, target_flatten)
            loss.backward()
            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            train_loss += loss.item()
            if batch_num%50 ==0:
                target=radar_inverseTransform(target)
                input=radar_inverseTransform(input)
                output=radar_inverseTransform(output)
                # plot_images([input[0,0,input.shape[2]-1],input[0,0,input.shape[2]-2],input[0,0,input.shape[2]-3],input[0,0,input.shape[2]-4],input[0,0,input.shape[2]-5],input[0,0,input.shape[2]-6] ,target[0][0],output[0][0]], 2, 4,epoch,batch_num,'train',file_name)
                plot_images([input[0,input.shape[1]-1],input[0,input.shape[1]-2],input[0,input.shape[1]-3],input[0,input.shape[1]-4],input[0,input.shape[1]-5],input[0,input.shape[1]-6] ,target[0,0],output[0,0]], 2, 4,epoch,batch_num,'train',file_name)

        train_loss /= len(train_dataloader.dataset)
        print(f"the train loss is {train_loss}")
        val_loss = 0
        model.eval()
        csi_values.clear()
        fss_values.clear()
        # Intinalize CSI for each category across all images
        csi_values = {category: [] for category in categories_threshold.keys()}
        fss_values = {category: [] for category in categories_threshold.keys()}
        rmse_values = []
        with torch.no_grad():
            for batch_num, (input, target) in enumerate(tqdm(validate_dataloader), 1):
                output = model(input)
                # output=torch.round(output, decimals=3)
                output_flatten=output.flatten()
                target_flatten=target.flatten()
                loss = criterion(output_flatten, target_flatten)
                val_loss += loss.item()
                actual_img=radar_inverseTransform(target)
                predicted_img=radar_inverseTransform(output)
                mse,csi,fss=calculate_metrics(actual_img,predicted_img)
                rmse = np.sqrt(mse)
                # Append RMSE to list
                rmse_values.append(rmse)
                for category in categories_threshold.keys():
                    csi_values[category].append(csi[category])
                    fss_values[category].append(fss[category])
                if batch_num%50 ==0:
                    target=radar_inverseTransform(target)
                    input=radar_inverseTransform(input)
                    output=radar_inverseTransform(output)
                    plot_images([input[0,input.shape[1]-1],input[0,input.shape[1]-2],input[0,input.shape[1]-3],input[0,input.shape[1]-4],input[0,input.shape[1]-5],input[0,input.shape[1]-6] ,target[0,0],output[0,0]], 2, 4,epoch,batch_num,'validate',file_name)
               
        val_loss /= len(validate_dataloader.dataset)
        print(f"the validate loss is {val_loss}")
        print("Epoch:{} Training Loss:{:.4f} Validation Loss:{:.4f}\n".format(
            epoch, train_loss, val_loss))

        # Log the running loss averaged per batch for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        {'Training': train_loss, 'Validation': val_loss},
                        epoch)
        # Log the learning rate
        current_lr = optim.param_groups[0]['lr']
        writer.add_scalars('Learning Rate', {'learning rate':current_lr}, epoch)
        
        # Save checkpoint every 2 epochs
        checkpoint_path =f'{model_file_path}/{file_name}_model_checkpoint_{epoch}.pth'
        csi_means = {category: np.nanmean(csi_values[category]) for category in categories_threshold.keys()}
        fss_means = {category: np.nanmean(fss_values[category]) for category in categories_threshold.keys()}
        average_rmse = np.mean(rmse_values)
        writer.add_scalars(f'CSI values',csi_means,epoch)
        writer.add_scalars(f'FSS values',fss_means,epoch)
        writer.add_scalars(f'MSE values',{'MSE':average_rmse},epoch)
        writer.flush()
        if epoch % 2 == 0:
            torch.save({
                    'epoch': epoch,
                    'run_name': run_name,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                }, checkpoint_path)
            print(f'Checkpoint saved at epoch {epoch}')
        scheduler_decrease.step()

    # Save the model's state dictionary
    torch.save(model.state_dict(), f'{model_file_path}/{file_name}_model.pth')

    # Save the optimizer's state dictionary if needed
    torch.save(optim.state_dict(), f'{model_file_path}/{file_name}_optimizer.pth')

    # Calculate RMSE for each image
    rmse_values = []

    output_file_path = f'{parparent}/results/{file_name}_results.txt'  # Specify the file path where you want to save the results

    # save the results
    with open(output_file_path, 'w') as file:
        file.write(f"test on file {file_name}\n")
        file.write(f"\nAverage RMSE across all images: {average_rmse}\n")
        average_csi = {category: np.nanmean(csi_values[category]) for category in categories_threshold.keys()}
        average_fss = {category: np.nanmean(fss_values[category]) for category in categories_threshold.keys()}

        # Display the results
        print("Average CSI for each category across all images:")
        for category, avg_csi in average_csi.items():
            print(f"{category}: {avg_csi}")
            file.write(f"\nAverage CSI for category: {category}: {avg_csi}\n")
        # Display the results
        print("Average FSS for each category across all images:")
        for category, avg_fss in average_fss.items():
            print(f"{category}: {avg_fss}")
            file.write(f"\nAverage FSS for category: {category}: {avg_fss}\n")
        file.close()