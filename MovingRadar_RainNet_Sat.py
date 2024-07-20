from datetime import datetime
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from RadarFilterImageDataset import RadarFilterImageDataset
from RadarFilterRainNetSatelliteDataset import RadarFilterRainNetSatelliteDataset

from RainNet_Satellite import RainNet
from plotting import plot_images,plot_image

from convlstm import Seq2Seq
from torch.utils.data import DataLoader, Subset
import h5py
import os
import glob
from PIL import Image
import io

import ast

from torchvision import transforms
import numpy as np
from sklearn.metrics import confusion_matrix

from tqdm import tqdm

from test_metrics import calculate_metrics,categories_threshold

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


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=10,
    shuffle=True
)

validate_loader = DataLoader(
    dataset=validate_data,
    batch_size=5,
    shuffle=True
)
test_sample_count=len(validate_loader)*validate_loader.batch_size*10/100,
test_loader = DataLoader(
    dataset=test_data,
    batch_size=5,
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
# input, _ = next(iter(validate_loader))

model=RainNet()
no_param=sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"number of parameters in the model is {no_param}")
model=torch.nn.DataParallel(model)
model.cuda()
# optim = Adam(model.parameters(), lr=1e-4)
optim = Adam(model.parameters(), lr=0.1)
# Define learning rate scheduler
# scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[10,6,4], gamma=0.1)
# Binary Cross Entropy, target pixel values either 0 or 1
# criterion = nn.BCELoss(reduction='sum')
# -999 -0
# criterion_undefined_rain = LogCoshThresholdLoss(transform(np.array([[-999]])),transform(np.array([[0]])))
# # 0-2.5
# criterion_light_rain = LogCoshThresholdLoss(transform(np.array([[0]])),transform(np.array([[2.5]])))
# # 2.5 - 7.5
# criterion_moderate_rain = LogCoshThresholdLoss(transform(np.array([[2.5]])),transform(np.array([[7.5]])))
# # 73.5-200
# criterion_heavy_rain = LogCoshThresholdLoss(transform(np.array([[7.5]])),transform(np.array([[201]])))
num_epochs = 50
criterion = LogCoshLoss()
file_name='radar_trainer_30M_RainNet_Sat_288_size_log_200_normalize_3d_sat_bigger'
# Initializing in a separate cell, so we can easily add more epochs to the same run
# Calculate CSI for each category across all images
csi_values = {category: [] for category in categories_threshold.keys()}

# Calculate fss for each category across all images
fss_values = {category: [] for category in categories_threshold.keys()}
writer = SummaryWriter(f'runs/{file_name}_{timestamp}')
start_epoch=1
checkpoint_path =f'models/{file_name}_model_checlpoint.pth'
# Load the checkpoint if it exists
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed from checkpoint at epoch {start_epoch}")

for epoch in range(start_epoch, num_epochs + 1):

    train_loss = 0
    acc=0
    total =0
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
        optim.step()
        train_loss += loss.item()
        if batch_num%100 ==0:
            target=inverseTransform(target)
            input=inverseTransform(input)
            output=inverseTransform(output)
            # plot_images([input[0,0,input.shape[2]-1],input[0,0,input.shape[2]-2],input[0,0,input.shape[2]-3],input[0,0,input.shape[2]-4],input[0,0,input.shape[2]-5],input[0,0,input.shape[2]-6] ,target[0][0],output[0][0]], 2, 4,epoch,batch_num,'train',file_name)
            plot_images([input[0,input.shape[1]-1],input[0,input.shape[1]-2],input[0,input.shape[1]-3],input[0,input.shape[1]-4],input[0,input.shape[1]-5],input[0,input.shape[1]-6] ,target[0,0],output[0,0]], 2, 4,epoch,batch_num,'train',file_name)

    train_loss /= len(train_dataloader.dataset)
    print(f"the train loss is {train_loss}")
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for batch_num, (input, target) in enumerate(tqdm(validate_loader), 1):
            output = model(input)
            output_flatten=output.flatten()
            target_flatten=target.flatten()
            # loss_undefined_rain = criterion_undefined_rain(output_flatten, target_flatten)
            # loss_light_rain = criterion_light_rain(output_flatten, target_flatten)
            # loss_moderate_rain = criterion_moderate_rain(output_flatten, target_flatten)
            # loss_heavy_rain = criterion_heavy_rain(output_flatten, target_flatten)
            # loss=0.1*loss_undefined_rain+0.2*loss_light_rain+0.3*loss_moderate_rain+0.4*loss_heavy_rain
            loss = criterion(output_flatten, target_flatten)
            val_loss += loss.item()
            if batch_num%100 ==0:
                target=inverseTransform(target)
                input=inverseTransform(input)
                output=inverseTransform(output)
                # plot_images([input[0,0,input.shape[2]-1],input[0,0,input.shape[2]-2],input[0,0,input.shape[2]-3] ,input[0,0,input.shape[2]-4] ,input[0,0,input.shape[2]-5] ,input[0,0,input.shape[2]-6]  ,target[0][0] ,output[0][0] ], 2, 4,epoch,batch_num,'validate',file_name)
                plot_images([input[0,input.shape[1]-1],input[0,input.shape[1]-2],input[0,input.shape[1]-3],input[0,input.shape[1]-4],input[0,input.shape[1]-5],input[0,input.shape[1]-6] ,target[0,0],output[0,0]], 2, 4,epoch,batch_num,'validate',file_name)
            if epoch%5==0 and batch_num%10 ==0:
                mse,csi,fss=calculate_metrics(target,output)
                for category in categories_threshold.keys():
                    csi_values[category].append(csi[category])
                    fss_values[category].append(fss[category])
    if epoch%5==0:
        mse,csi,fss=calculate_metrics(target,output)
        for category in categories_threshold.keys():
                csi_values[category].append(csi[category])
                fss_values[category].append(fss[category])
        csi_means = {category: np.nanmean(csi_values[category]) for category in categories_threshold.keys()}
        fss_means = {category: np.nanmean(fss_values[category]) for category in categories_threshold.keys()}
        writer.add_scalars(f'CSI values',csi_means,epoch)
        writer.add_scalars(f'FSS values',fss_means,epoch)



    val_loss /= len(validate_loader.dataset)
    # val_loss /= 128
    print(f"the validate loss is {val_loss}")
    print("Epoch:{} Training Loss:{:.2f} Validation Loss:{:.2f}\n".format(
        epoch, train_loss, val_loss))

    # Log the running loss averaged per batch for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': train_loss, 'Validation': val_loss},
                       epoch)
    writer.flush()
    # Save checkpoint every 5 epochs
    if epoch% 5 == 0:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
            }, checkpoint_path)
        print(f'Checkpoint saved at epoch {epoch + 1}')

# Save the model's state dictionary
torch.save(model.state_dict(), f'models/{file_name}_model.pth')

# Save the optimizer's state dictionary if needed
torch.save(optim.state_dict(), f'models/{file_name}_optimizer.pth')

# test phase

# Define the rain categories and thresholds
categories_threshold={'undefined':(-999, 0),'light rain':(0, 2.5), 'moderate rain':(2.5, 7.5), 'heavy rain':(7.5, 200)}# Function to categorize pixel values based on thresholds
def categorize_pixel(value, thresholds, categories):
    for i, (lower, upper) in enumerate(thresholds):
        if lower <= value < upper:
            return categories[i]
    return categories[-1]  # Return the last category as default

# Function to calculate CSI for a single category
def calculate_cat_csi(predicted, actual, category):
    actual_label=actual.detach().cpu().numpy().astype(int)
    predicted_label=predicted.detach().cpu().numpy().astype(int)
    # Calculate confusion matrix
    cm = confusion_matrix((actual_label>= categories_threshold[category][0]).astype(int) & (actual_label< categories_threshold[category][1]).astype(int), (predicted_label>= categories_threshold[category][0]).astype(int) & (predicted_label< categories_threshold[category][1]).astype(int), labels=[0, 1])
    # Check the shape of the confusion matrix
    # Check the shape of the confusion matrix
    if cm.shape == (2, 2):
        # Unpack the confusion matrix values
        tn, fp, fn, tp = cm.ravel()

        # Calculate CSI using the formula
        if tp + fp + fn == 0:
            # Handle the case where TP + FP + FN is zero
            csi = 1 
        else:
            # Calculate CSI
            csi = tp / (tp + fp + fn)
    else:
        # In case the confusion matrix is not 2x2, handle appropriately
        csi = 0 
    # Calculate CSI
    return csi


# Function to calculate CSI for a single category
def calculate_csi(predicted, actual, category):
    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(actual == category, predicted == category).ravel()
    # Calculate CSI
    csi = tp / (tp + fp + fn)
    return csi
# Calculate RMSE for each image
rmse_values = []

# Calculate CSI for each category across all images
csi_values = {category: [] for category in categories_threshold.keys()}
output_file_path = file_name+'_results.txt'  # Specify the file path where you want to save the results

model.eval()
with torch.no_grad():
    for batch_num, (input, target) in enumerate(tqdm(test_loader), 1):
        output = model(input)
        actual_img=inverseTransform(target)
        predicted_img=inverseTransform(output)
        if batch_num%100 ==0:
            input=inverseTransform(input)
            # plot_images([input[0,0,input.shape[2]-1],input[0,0,input.shape[2]-2],input[0,0,input.shape[2]-3],input[0,0,input.shape[2]-4],input[0,0,input.shape[2]-5],input[0,0,input.shape[2]-6] ,target[0][0],output[0][0]], 2, 4,epoch,batch_num,'train',file_name)
            plot_images([input[0,input.shape[1]-1],input[0,input.shape[1]-2],input[0,input.shape[1]-3],input[0,input.shape[1]-4],input[0,input.shape[1]-5],input[0,input.shape[1]-6] ,actual_img[0,0],predicted_img[0,0]], 2, 4,1,batch_num,'test',file_name)
        
         # Calculate the squared differences between actual and predicted values
        squared_differences = (actual_img - predicted_img) ** 2
    
        # Calculate the mean of the squared differences
        mean_squared_error = np.mean(squared_differences.detach().cpu().numpy())
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error)
        
        # Append RMSE to list
        rmse_values.append(rmse)
        # Flatten the images to 1D arrays for comparison
        actual_flat = actual_img.flatten()
        predicted_flat = predicted_img.flatten()
        
        # Categorize pixel values
        # actual_categorized = np.array([categorize_pixel(value, thresholds, categories) for value in actual_flat])
        # predicted_categorized = np.array([categorize_pixel(value, thresholds, categories) for value in predicted_flat])
        
        # Calculate CSI for each category
        for category in categories_threshold.keys():
            # csi = calculate_csi(predicted_categorized, actual_categorized, category)
            csi=calculate_cat_csi(predicted_flat, actual_flat, category)
            csi_values[category].append(csi)
        print(f"test batch number={batch_num}")



# Calculate the average RMSE across all images
average_rmse = np.mean(rmse_values)

# Display the results
print(f"Average RMSE across all images: {average_rmse}")
with open(f'results/{output_file_path}', 'w') as file:
    file.write(f"\nAverage RMSE across all images: {average_rmse}\n")

    # Calculate the average CSI for each category across all images
    average_csi = {category: np.mean(csi_values[category]) for category in categories_threshold.keys()}

    # Display the results
    print("Average CSI for each category across all images:")
    for category, avg_csi in average_csi.items():
        print(f"{category}: {avg_csi}")
        file.write(f"\nAverage CSI for category: {category}: {avg_csi}\n")
    file.close()

