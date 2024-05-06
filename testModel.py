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
from sklearn.metrics import confusion_matrix


folder_name='radar_trainer_30M_RainNet_512_size_log_200_multi_loss'

model=RainNet()
model=torch.nn.DataParallel(model)
model.cuda()
model.load_state_dict(torch.load(folder_name+'_model.pth'), strict=False)
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
    return torch.where(x > 0, x-1, x)
def invert_custom_transform2(x):
    # Use PyTorch's where function to apply the transformation element-wise
    return torch.where(x <= 0, -999, x) 

inverseTransform= transforms.Compose([
    # transforms.Lambda(lambda x: x.unsqueeze(0))  ,# Add a new dimension at position 0
    # transforms.Lambda(lambda x: x.cuda()) , # send data to cuda
    # transforms.Normalize(mean=[-mean/std,],
                            #  std=[1/std,])
    # transforms.Lambda(lambda x: torch.exp(x)-1),
    transforms.Lambda(lambda x: torch.pow(max_value, x)-1),
    transforms.Lambda(invert_custom_transform2) ,
    transforms.Lambda(invert_custom_transform1) ,
    # transforms.Lambda(invert_custom_transform2) ,
    # transforms.Lambda(lambda x: (x*(max_value - min_value))+min_value)
    # transforms.Lambda(lambda x: x) 
])

test_data = RadarFilterRainNetDataset(
    img_dir='../RadarData_test/',
    transform=transform,
    inverse_transform=inverseTransform
)
test_loader = DataLoader(
    dataset=test_data,
    batch_size=16,
    shuffle=False
)

# test phase
# Define the rain categories and thresholds
# categories = ['undefined','light rain', 'moderate rain', 'heavy rain','violent rain']
# thresholds = [(-999, -0.1),(-0.1, 2.5), (2.5, 15), (15, 30), (30, 200)]  # Adjust based on your data


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
output_file_path = folder_name+'_results.txt'  # Specify the file path where you want to save the results

model.eval()
with torch.no_grad():
    for batch_num, (input, target) in enumerate(test_loader, 1):
        output = model(input)
        actual_img=inverseTransform(target)
        predicted_img=inverseTransform(output)
        if batch_num%100 ==0:
            input=inverseTransform(input)
            # plot_images([input[0,0,input.shape[2]-1],input[0,0,input.shape[2]-2],input[0,0,input.shape[2]-3],input[0,0,input.shape[2]-4],input[0,0,input.shape[2]-5],input[0,0,input.shape[2]-6] ,target[0][0],output[0][0]], 2, 4,epoch,batch_num,'train',folder_name)
            plot_images([input[0,input.shape[1]-1],input[0,input.shape[1]-2],input[0,input.shape[1]-3],input[0,input.shape[1]-4],input[0,input.shape[1]-5],input[0,input.shape[1]-6] ,actual_img[0,0],predicted_img[0,0]], 2, 4,1,batch_num,'test',folder_name)
        
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
with open(output_file_path, 'w') as file:
    file.write(f"\nAverage RMSE across all images: {average_rmse}\n")

    # Calculate the average CSI for each category across all images
    average_csi = {category: np.mean(csi_values[category]) for category in categories_threshold.keys()}

    # Display the results
    print("Average CSI for each category across all images:")
    for category, avg_csi in average_csi.items():
        print(f"{category}: {avg_csi}")
        file.write(f"\nAverage CSI for category: {category}: {avg_csi}\n")
    file.close()

