from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from RadarFilterRainNetSatelliteDataset import RadarFilterRainNetSatelliteDataset

from RainNet_Satellite import RainNet
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
import seaborn as sns

import ast

decimal_places = 3

# Multiply the tensor by 10^decimal_places
factor = 10 ** decimal_places

file_name='radar_trainer_30M_RainNet_Sat_288_size_log_200_normalize_3d_sat'

model=RainNet()
model=torch.nn.DataParallel(model)
model.cuda()
model.load_state_dict(torch.load(file_name+'_model.pth'), strict=False)
# from ipywidgets import widgets, HBox
radar_data_folder_path = '../RadarData_test_18/'
# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
# Load Data as Numpy Array
# MovingMNIST = np.load('mnist_test_seq.npy').transpose(1, 0, 2, 3)
min_value=0
max_value=200
mean=0.21695
std=0.9829

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


# # make transforms
# def custom_transform1(x):
#     # Use PyTorch's where function to apply the transformation element-wise
#     return torch.where(x >= 0, x + 1, x)
# def custom_transform2(x):
#     # Use PyTorch's where function to apply the transformation element-wise
#     return torch.where(x < 0, 0, x)


# transform = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Lambda(lambda x: x.unsqueeze(0))  ,# Add a new dimension at position 0
#     # transforms.Lambda(lambda x: x.cuda()) , # send data to cuda
#     # transforms.Normalize(mean=[mean,],
#     #                          std=[std,],)
#     # transforms.Lambda(lambda x: (x-min_value)/(max_value-min_value)),
#     # transforms.Lambda(lambda x: torch.log2(x+1))
#     transforms.Lambda(custom_transform1) ,
#     transforms.Lambda(custom_transform2) ,
#     # transforms.Lambda(lambda x: torch.log(x+1)),
#      transforms.Lambda(lambda x:  (torch.log(x+1) / torch.log(torch.tensor(max_value))).float()),
#     # transforms.Lambda(lambda x: x.float())
#     transforms.Lambda(lambda x: torch.round(x*factor)/factor) 

    
# ])
# def invert_custom_transform1(x):
#     # Use PyTorch's where function to apply the transformation element-wise
#     return torch.where(x > 0, x-1, x)
# def invert_custom_transform2(x):
#     # Use PyTorch's where function to apply the transformation element-wise
#     return torch.where(x <= 0, -999, x) 

# inverseTransform= transforms.Compose([
#     # transforms.Lambda(lambda x: x.unsqueeze(0))  ,# Add a new dimension at position 0
#     # transforms.Lambda(lambda x: x.cuda()) , # send data to cuda
#     # transforms.Normalize(mean=[-mean/std,],
#                             #  std=[1/std,])
#     # transforms.Lambda(lambda x: torch.exp(x)-1),
#     transforms.Lambda(lambda x: torch.pow(max_value, x)-1),
#     transforms.Lambda(invert_custom_transform2) ,
#     transforms.Lambda(invert_custom_transform1) ,
#     # transforms.Lambda(invert_custom_transform2) ,
#     # transforms.Lambda(lambda x: (x*(max_value - min_value))+min_value)
#     transforms.Lambda(lambda x: torch.round(x*factor)/factor) 
# ])

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

def normalize_Satellite(x):
    for i in range(x.size(0)):
        key=list(bands_min_values.keys())[i]
        x[i] = (x[i]-bands_min_values[key])/(bands_max_values[key]-bands_min_values[key])
    return x

sat_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(normalize_Satellite),
    ])

test_data = RadarFilterRainNetSatelliteDataset(
    img_dir='../RadarData_test_18/',
    sat_dir='../SatelliteData',
    transform=transform,
    inverse_transform=inverseTransform,
    sat_transform=sat_transform,
    random_satellite=True
)
test_loader = DataLoader(
    dataset=test_data,
    batch_size=15,
    shuffle=False
)

# test phase
# Define the rain categories and thresholds
# categories = ['undefined','light rain', 'moderate rain', 'heavy rain','violent rain']
# thresholds = [(-999, -0.1),(-0.1, 2.5), (2.5, 15), (15, 30), (30, 200)]  # Adjust based on your data


categories_threshold={'undefined':(-999, 0),'light rain':(0, 2.5), 'moderate rain':(2.5, 7.5), 'heavy rain':(7.5, 50),'Violent rain':(50, 201)}# Function to categorize pixel values based on thresholds
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

def check_spetial_residual(actual_img,predicted_img):
    return (actual_img - predicted_img) ** 2
    

# Function to calculate CSI for a single category
def calculate_csi(predicted, actual, category):
    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(actual == category, predicted == category).ravel()
    # Calculate CSI
    csi = tp / (tp + fp + fn)
    return csi



def calculate_fractional_coverage(grid, lower_threshold, upper_threshold, neighborhood_size):
    """
    Calculate the fractional coverage of grid points exceeding the given threshold
    within a specified neighborhood size.

    Parameters:
    grid (np.ndarray): 2D array of precipitation values.
    lower_threshold (float): Precipitation lower threshold.
    upper_threshold (float): Precipitation upper threshold.
    neighborhood_size (int): Size of the neighborhood to consider.

    Returns:
    np.ndarray: Fractional coverage for each grid point.
    """
    grid = grid.cpu().numpy()  # Move tensor to CPU and convert to numpy array
    padded_grid = np.pad(grid, pad_width=neighborhood_size // 2, mode='constant', constant_values=0)
    # fractional_coverage = np.zeros_like(grid, dtype=float)

    # for i in range(grid.shape[2]):
    #     for j in range(grid.shape[3]):
    #         neighborhood = padded_grid[:,:,i:i + neighborhood_size, j:j + neighborhood_size]
    #         exceed_count = np.sum((neighborhood >= lower_threshold) & (neighborhood < upper_threshold))
    #         total_points = neighborhood.size
    #         fractional_coverage[i, j] = exceed_count / total_points

    fractional_coverage = np.zeros_like(grid, dtype=float)
    width=grid.shape[2]
    height=grid.shape[3]
    if len(grid.shape)==5:
        width=grid.shape[3]
        height=grid.shape[4]
    pad_size = neighborhood_size // 2
    for b in range(grid.shape[0]):
        for t in range(grid.shape[1]):
            padded_grid = np.pad(grid[b, t,0], pad_width=pad_size, mode='constant', constant_values=0)
            for i in range(width):
                for j in range(height):
                    neighborhood = padded_grid[i:i+neighborhood_size, j:j+neighborhood_size]
                    exceed_count = np.sum((neighborhood >= lower_threshold) & (neighborhood < upper_threshold))
                    total_points = neighborhood.size
                    fractional_coverage[b, t,0, i, j] = exceed_count / total_points
    

    return fractional_coverage


def calculate_fss(observed, forecasted, lower_threshold, upper_threshold, neighborhood_size):
    """
    Calculate the Fractional Skill Score (FSS) for the given observed and forecasted precipitation grids
    using a specified threshold and neighborhood size.

    Parameters:
    observed (np.ndarray): 2D array of observed precipitation values.
    forecasted (np.ndarray): 2D array of forecasted precipitation values.
    lower_threshold (float): Precipitation lower threshold.
    upper_threshold (float): Precipitation upper threshold.
    neighborhood_size (int): Size of the neighborhood to consider.

    Returns:
    float: Fractional Skill Score (FSS).
    """
    # Calculate fractional coverage for observed and forecasted grids
    observed_fractional_coverage = calculate_fractional_coverage(observed, lower_threshold, upper_threshold,
                                                                 neighborhood_size)
    forecasted_fractional_coverage = calculate_fractional_coverage(forecasted, lower_threshold, upper_threshold,
                                                                   neighborhood_size)

    # Calculate Mean Squared Error (MSE) between fractional coverages
    # mse = np.mean((observed_fractional_coverage - forecasted_fractional_coverage) ** 2)

    # # Calculate reference MSE (MSE for no-skill forecast)
    # # Assuming no-skill forecast is just the mean of the observed fractional coverage
    # reference_mse = np.mean(observed_fractional_coverage ** 2)

    # # Calculate FSS
    # if reference_mse ==0:
    #     return 0
    # fss = 1 - (mse / reference_mse)

    # Calculate the numerator and denominator for FSS
    numerator = np.sum((forecasted_fractional_coverage - observed_fractional_coverage) ** 2)
    denominator = np.sum(forecasted_fractional_coverage ** 2 + observed_fractional_coverage ** 2)
    if numerator==denominator ==0:
        return 1
    elif denominator==0:
        return 0
    # Calculate FSS
    fss = 1 - (numerator / denominator)

    return fss


# Calculate RMSE for each image
rmse_values = []

# Calculate CSI for each category across all images
csi_values = {category: [] for category in categories_threshold.keys()}

# Calculate fss for each category across all images
fss_values = {category: [] for category in categories_threshold.keys()}
output_file_path = file_name+'_test_results.txt'  # Specify the file path where you want to save the results
spatial_errors = []
neighborhood_size=3
model.eval()
with torch.no_grad():
    for batch_num, (input, target) in enumerate(tqdm(test_loader), 1):
        output = model(input)
        actual_img=inverseTransform(target)
        predicted_img=inverseTransform(output)
        
        if batch_num%100 ==0:
            input=inverseTransform(input)
            # plot_images([input[0,0,input.shape[2]-1],input[0,0,input.shape[2]-2],input[0,0,input.shape[2]-3],input[0,0,input.shape[2]-4],input[0,0,input.shape[2]-5],input[0,0,input.shape[2]-6] ,target[0][0],output[0][0]], 2, 4,epoch,batch_num,'train',folder_name)
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
            fss = calculate_fss(actual_img, predicted_img, categories_threshold[category][0],
                        categories_threshold[category][1], neighborhood_size)
            fss_values[category].append(fss)
        #print(f"test batch number={batch_num}")
        if batch_num%10 ==0:
            predicted_img=invert_custom_transform2(predicted_img)
            spatial_error = check_spetial_residual(actual_img,predicted_img)
            spatial_errors.append(spatial_error)


# Calculate the average RMSE across all images
average_rmse = np.mean(rmse_values)

# Display the results
print(f"Average RMSE across all images: {average_rmse}")
with open(output_file_path, 'w') as file:
    file.write(f"\nAverage RMSE across all images: {average_rmse}\n")

    # Calculate the average CSI for each category across all images
    average_csi = {category: np.mean(csi_values[category]) for category in categories_threshold.keys()}
    # Calculate the average FSS for each category across all images
    average_fss = {category: np.mean(fss_values[category]) for category in categories_threshold.keys()}

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

# # Create Heatmaps
# for i, spatial_error in enumerate(spatial_errors):
#     plt.figure()
#     sns.heatmap(spatial_error[0,0].detach().cpu().numpy(), cmap='viridis')
#     plt.title(f'Spatial Error Heatmap (Image {i + 1})')
#     plt.savefig(f"output/Errors/Spatial Error Heatmap (Image {i + 1})")
#     plt.close()

# # Create Contour Plots
# for i, spatial_error in enumerate(spatial_errors):
#     plt.figure()
#     contour = plt.contour(spatial_error[0,0].detach().cpu().numpy(), cmap='viridis')
#     plt.title(f'Spatial Error Contour Plot (Image {i + 1})')
#     plt.colorbar(contour)
#     plt.savefig(f"output/Errors/Spatial Error Contour (Image {i + 1})")
#     plt.close()